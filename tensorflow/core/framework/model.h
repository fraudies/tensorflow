/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_MODEL_H_

#include <list>
#include <memory>
#include <string>
#include <thread>  // (b/114492873): move this include into core/platform
#include <utility>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace model {

// Represents thread-safe state that can be shared between an input pipeline and
// the performance model.
struct SharedState {
 public:
  explicit SharedState(int64 value, std::shared_ptr<mutex> mu,
                       std::shared_ptr<condition_variable> cond_var)
      : value(value), mu(std::move(mu)), cond_var(std::move(cond_var)) {}

  std::shared_ptr<mutex> mu;
  std::shared_ptr<condition_variable> cond_var;
  int64 value;

  // Identifies the minimum value of the parameter.
  int64 min;

  // Identifies the maximum value of the parameter.
  int64 max;

  // Shared state of the parameter.
  std::shared_ptr<SharedState> state;
};

std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         int64 min, int64 max);

// Abstract representation of a TensorFlow input pipeline node. It collects
// information about inputs to this node, processing time spent executing the
// node logic, number of elements produced by the node, various other
// information (e.g. batch size or execution parallelism).
//
// Developers of tf.data transformations are not expected to interact with
// this class directly. Boiler plate code for creating the abstract
// representation of the input pipeline and collecting common information has
// been added to the implementation of `DatasetBase` and `DatasetBaseIterator`
// respectively.
//
// In addition, `DatasetBaseIterator` provides wrappers that can be used for
// transformation-specific information collection. The `SetMetadata` wrapper
// can be used to pass arbitrary metadata to the modeling framework, while the
// `StartWork` and `StopWork` wrappers should be used to correctly account for
// processing time of multi-threaded transformation that yield the CPU; such
// transformations should invoke `StartWork()` when a transformation thread
// starts executing (e.g. when created or woken up) and `StopWork()` when a
// transformation thread stops executing (e.g. when returning or waiting).
class Node {
 public:
  // Arguments for `Node` constructor.
  struct Args {
    int64 id;
    string name;
    std::shared_ptr<Node> output;
  };

  using Factory = std::function<std::shared_ptr<Node>(Args)>;

  explicit Node(Args args)
      : id_(args.id), name_(args.name), output_(args.output.get()) {}

  // Increments the bytes buffered by the given delta.
  void add_buffered_bytes(int64 delta) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    buffered_bytes_ += delta;
  }

  // Adds an input.
  void add_input(std::shared_ptr<Node> node) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.push_back(node);
  }

  // Increments the aggregate processing time by the given delta.
  void add_processing_time(int64 delta) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    processing_time_ += delta;
  }

  // Returns the number of bytes stored in this node's buffer.
  int64 buffered_bytes() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return buffered_bytes_;
  }

  // Indicates whether the node has tunable parameters.
  bool has_tunable_parameters() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    for (const auto& pair : parameters_) {
      if (pair.second->state->tunable) return true;
    }
    return false;
  }

  // Returns the unique node ID.
  int64 id() const LOCKS_EXCLUDED(mu_) { return id_; }

  // Returns the node inputs.
  std::list<std::shared_ptr<Node>> inputs() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return inputs_;
  }

  // Returns a longer node name that is guaranteed to be unique.
  string long_name() const { return strings::StrCat(name_, "(id:", id_, ")"); }

  // Returns the node name.
  const string& name() const { return name_; }

  // Returns the number of elements produced by the node.
  int64 num_elements() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return num_elements_;
  }

  // Returns the node output.
  Node* output() const { return output_; }

  // Returns the aggregate processing time.
  int64 processing_time() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return processing_time_;
  }

  // Records that the node produced an element.
  void record_element() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    num_elements_++;
  }

  // Records that a node thread has started executing.
  void record_start(int64 time_nanos) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    work_start_[std::this_thread::get_id()] = time_nanos;
  }

  // Records that a node thread has stopped executing.
  void record_stop(int64 time_nanos) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    std::thread::id tid = std::this_thread::get_id();
    auto iter = work_start_.find(tid);
    if (iter != work_start_.end()) {
      processing_time_ += time_nanos - iter->second;
      work_start_.erase(iter);
    } else {
      LOG(WARNING)
          << "Encountered a stop event that was not preceded by a start event.";
    }
  }

  // Removes an input.
  void remove_input(std::shared_ptr<Node> input) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.remove(input);
  }

  // Collects tunable parameters in the subtree rooted in this node.
  void CollectTunableParameters(
      std::map<string, std::shared_ptr<Parameter>>* parameters) const
      LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    for (auto& pair : parameters_) {
      if (pair.second->state->tunable) {
        parameters->insert(std::make_pair(long_name(), pair.second));
      }
    }
    for (auto& input : inputs_) {
      input->CollectTunableParameters(parameters);
    }
  }

  // Returns the per-element output time for this node.
  int64 OutputTime(std::vector<int64>* input_times) const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return OutputTimeLocked(input_times);
  }

  // Returns the per-element processing time spent in the subtree rooted in
  // this node.
  int64 ProcessingTime() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return ProcessingTimeLocked();
  }

  // Returns a copy of this node, making a deep copy of its inputs and a
  // shallow copy of its tunable parameters.
  //
  // The purpose for this method is to allow the model optimization logic to
  // operate over immutable state while allowing concurrent model updates.
  std::shared_ptr<Node> Snapshot(std::shared_ptr<Node> output)
      LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    std::shared_ptr<Node> result = Clone(output);
    {
      mutex_lock l2(result->mu_);
      result->buffered_bytes_ = buffered_bytes_;
      result->processing_time_ = processing_time_;
      result->num_elements_ = num_elements_;
      result->parameters_ = parameters_;
    }
    for (auto& input : inputs_) {
      result->add_input(input->Snapshot(result));
    }
    return result;
  }

 protected:
  // Creates a clone of this node.
  virtual std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const
      SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the per-element processing time spent in this node.
  int64 NanosPerElementLocked() const SHARED_LOCKS_REQUIRED(mu_) {
    if (num_elements_ == 0) {
      return 0;
    }
    return static_cast<int64>(static_cast<double>(processing_time_) /
                              static_cast<double>(num_elements_));
  }

  // Returns the sum of per-element output time for the inputs of this node.
  int64 OutputTimeForInputs(std::vector<int64>* input_times) const
      SHARED_LOCKS_REQUIRED(mu_) {
    int64 sum = 0;
    for (auto& input : inputs_) {
      sum += input->OutputTime(input_times);
    }
    return sum;
  }

  // Returns the per-element output time for this node.
  virtual int64 OutputTimeLocked(std::vector<int64>* input_times) const
      SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the sum of per-element processing time for the inputs of this node.
  //
  // TODO(jsimsa): use processing time history as a prior for future inputs
  int64 ProcessingTimeForInputs() const SHARED_LOCKS_REQUIRED(mu_) {
    int64 sum = 0;
    for (auto& input : inputs_) {
      sum += input->ProcessingTime();
    }
    return sum;
  }

  // Returns the per-element processing time spent in the subtree rooted in
  // this node.
  virtual int64 ProcessingTimeLocked() const SHARED_LOCKS_REQUIRED(mu_) = 0;

  mutable mutex mu_;
  const int64 id_;
  const string name_;
  int64 buffered_bytes_ GUARDED_BY(mu_) = 0;
  int64 processing_time_ GUARDED_BY(mu_) = 0;
  int64 num_elements_ GUARDED_BY(mu_) = 0;
  std::map<std::thread::id, int64> work_start_ GUARDED_BY(mu_);
  std::map<string, std::shared_ptr<Parameter>> parameters_ GUARDED_BY(mu_);
  std::list<std::shared_ptr<Node>> inputs_ GUARDED_BY(mu_);

  // The reference to the output node is not owned so that deletion of a
  // node results in recursive deletion of the subtree rooted in the node.
  Node* const output_;
};

// Abstract representation of a TensorFlow input pipeline that can be used
// for collecting runtime information and optimizing performance. It collects
// runtime information about execution of the input pipeline that is used to
// create a performance model, which is in turn used to identify optimal values
// of tunable parameters.
//
// Developers of tf.data transformations are not expected to interact with this
// class directly. Boiler plate code for creating the abstract representation of
// the input pipeline and collecting runtime information has been added to the
// implementation of `DatasetBase` and `DatasetBaseIterator` respectively.
class Model {
 public:
  using NodeHook = std::function<void(std::shared_ptr<Node>)>;

  // Creates a new model.
  //
  // The `remove_node_hook` argument can be used to specify functionality that
  // should be invoked before a node is removed from the model. The hook can be
  // used for dependency injection -- to allow the model to invoke functionality
  // from modules that it could not depend on statically.
  Model(NodeHook remove_node_hook)
      : collect_resource_usage_(false),
        remove_node_hook_(std::move(remove_node_hook)) {
    DCHECK(remove_node_hook_ != nullptr);
  }

  // Adds a constant parameter for the given node.
  void AddConstantParameter(const string& node_name,
                            const string& parameter_name, int64 value)
      LOCKS_EXCLUDED(mu_);

  // Adds a node with the given name and given output (identified by name).
  void AddNode(const string& name, const string& output_name)
      LOCKS_EXCLUDED(mu_);

  // Increments the processing time for the given node..
  void AddProcessingTime(const string& name, int64 delta) LOCKS_EXCLUDED(mu_);

  // Adds a tunable parameter for the given node.
  void AddTunableParameter(const string& node_name,
                           const string& parameter_name,
                           std::shared_ptr<SharedState> value, int64 min,
                           int64 max) LOCKS_EXCLUDED(mu_);

  // Runs optimization.
  void Optimize(int64 cpu_budget) LOCKS_EXCLUDED(mu_);

  // Records that a node has produced an element.
  void RecordElement(const string& name) LOCKS_EXCLUDED(mu_);

  // Returns the number of elements that the input pipeline has produced.
  int64 NumElements(const string& name) LOCKS_EXCLUDED(mu_);

  // Records that the given node has started work. If `stop_output` is set, it
  // also records that the output of the given node has stopped work.
  void RecordStart(const string& name, bool stop_output) LOCKS_EXCLUDED(mu_);

  // Records that the given node has stopped work. If `stop_output` is set, it
  // also records that the output of the given node has started work.
  void RecordStop(const string& name, bool start_output) LOCKS_EXCLUDED(mu_);

  // Removes the given node.
  void RemoveNode(const string& name) LOCKS_EXCLUDED(mu_);

 private:
  // Collects tunable parameters in the tree rooted in the given node, returning
  // a mapping from a (unique) node name to a tunable parameter.
  std::map<string, std::shared_ptr<Parameter>> CollectTunableParameters(
      std::shared_ptr<Node> node);

    // Adds an input.
    void add_input(std::shared_ptr<Node> node) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      inputs_.push_back(node);
    }

    // Increments the aggregate processing time by the given delta.
    void add_processing_time(int64 delta) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      processing_time_ += delta;
    }

    // Adds a tunable parameter.
    void add_tunable_param(const string& name,
                           std::shared_ptr<SharedState> state, int64 min,
                           int64 max) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      tunable_params_[name] =
          std::make_shared<Tunable>(std::move(state), min, max);
    }

    // Returns the unique node ID.
    int64 id() LOCKS_EXCLUDED(mu_) { return id_; }

    // Returns the node inputs.
    std::list<std::shared_ptr<Node>> inputs() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return inputs_;
    }

    // Returns the node name.
    const string& name() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return name_;
    }

    // Returns the number of elements produced by the node.
    int64 num_elements() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return num_elements_;
    }

    // Returns the node output.
    std::shared_ptr<Node> output() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return output_;
    }

    // Records that the node produced an element.
    void record_element() LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      num_elements_++;
    }

    // Records that a node thread has started executing.
    void record_start() LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      work_start_[std::this_thread::get_id()] = Env::Default()->NowNanos();
    }

    // Records that a node thread has stopped executing.
    void record_stop() LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      std::thread::id tid = std::this_thread::get_id();
      auto start_time = gtl::FindOrNull(work_start_, tid);
      DCHECK(start_time)
          << "Encountered a stop event that was not preceded by a start event.";
      if (start_time) {
        processing_time_ += Env::Default()->NowNanos() - *start_time;
        work_start_.erase(tid);
      }
    }

    // Removes an input.
    void remove_input(std::shared_ptr<Node> input) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      inputs_.remove(input);
    }

    // Set the node output.
    void set_output(std::shared_ptr<Node> output) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      output_ = output;
    }

    // Collects tunable parameters in the subtree rooted in this node.
    void CollectTunables(std::vector<std::shared_ptr<Tunable>>* tunables)
        LOCKS_EXCLUDED(mu_);

    // Returns the per-element output time for this node.
    int64 OutputTime(std::vector<int64>* input_times) LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return OutputTimeLocked(input_times);
    }

    // Returns the per-element processing time spent in the subtree rooted in
    // this node.
    int64 ProcessingTime() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return ProcessingTimeLocked();
    }

   private:
    enum class Type {
      BATCH = 0,
      CACHE,
      CONCATENATE,
      FILTER,
      FLAT_MAP,
      INTERLEAVE,
      MAP,
      MAP_AND_BATCH,
      PADDED_BATCH,
      PARALLEL_INTERLEAVE,
      PARALLEL_INTERLEAVE_V2,
      PARALLEL_MAP,
      PREFETCH,
      REPEAT,
      SHUFFLE,
      SKIP,
      TAKE,
      ZIP,
      UNKNOWN,
    };

    // Gets a value of the given parameter (tunable or constant).
    int64 GetParameterValue(const string& name) SHARED_LOCKS_REQUIRED(mu_);

    // Returns the per-element processing time spent in this node.
    int64 NanosPerElement() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return NanosPerElementLocked();
    }

    int64 NanosPerElementLocked() SHARED_LOCKS_REQUIRED(mu_) {
      if (num_elements_ == 0) {
        return 0;
      }
      return (int64)((double)processing_time_ / (double)num_elements_);
    }

    int64 OutputTimeLocked(std::vector<int64>* input_times)
        SHARED_LOCKS_REQUIRED(mu_);

    int64 OutputTimeForInputs(std::vector<int64>* input_times)
        SHARED_LOCKS_REQUIRED(mu_) {
      int64 sum = 0;
      for (auto input : inputs_) {
        sum += input->OutputTime(input_times);
      }
      return sum;
    }

    int64 ProcessingTimeLocked() SHARED_LOCKS_REQUIRED(mu_);

    // Returns the per-element processing time spent in the inputs of this node.
    int64 ProcessingTimeForInputs() SHARED_LOCKS_REQUIRED(mu_) {
      int64 sum = 0;
      for (auto input : inputs_) {
        sum += input->ProcessingTime();
      }
      return sum;
    }

    Type TypeFromName(const string& name) SHARED_LOCKS_REQUIRED(mu_) {
      if (name_ == "Batch") {
        return Type::BATCH;
      }
      if (str_util::EndsWith(name_, "Cache")) {
        return Type::CACHE;
      }
      if (name_ == "Concatenate") {
        return Type::CONCATENATE;
      }
      if (name_ == "Filter") {
        return Type::FILTER;
      }
      if (name_ == "FlatMap") {
        return Type::FLAT_MAP;
      }
      if (name_ == "Interleave") {
        return Type::INTERLEAVE;
      }
      if (name_ == "Map") {
        return Type::MAP;
      }
      if (name_ == "MapAndBatch" || name_ == "NumaMapAndBatch") {
        return Type::MAP_AND_BATCH;
      }
      if (name_ == "PaddedBatch") {
        return Type::PADDED_BATCH;
      }
      if (name_ == "ParallelInterleave") {
        return Type::PARALLEL_INTERLEAVE;
      }
      if (name_ == "ParallelInterleaveV2") {
        return Type::PARALLEL_INTERLEAVE_V2;
      }
      if (name_ == "ParallelMap") {
        return Type::PARALLEL_MAP;
      }
      if (name_ == "Prefetch") {
        return Type::PREFETCH;
      }
      if (str_util::EndsWith(name_, "Repeat")) {
        return Type::REPEAT;
      }
      if (name_ == "Shuffle") {
        return Type::SHUFFLE;
      }
      if (str_util::EndsWith(name_, "Skip")) {
        return Type::SKIP;
      }
      if (str_util::EndsWith(name_, "Take")) {
        return Type::TAKE;
      }
      if (name_ == "Zip") {
        return Type::ZIP;
      }
      return Type::UNKNOWN;
    }

    mutex mu_;
    const int64 id_;
    const string name_;
    const Type type_;
    int64 processing_time_ GUARDED_BY(mu_) = 0;
    int64 num_elements_ GUARDED_BY(mu_) = 0;
    std::map<std::thread::id, int64> work_start_ GUARDED_BY(mu_);
    std::map<string, int64> constant_params_ GUARDED_BY(mu_);
    // Tunables are shared with the model during optimization.
    std::map<string, std::shared_ptr<Tunable>> tunable_params_ GUARDED_BY(mu_);
    std::list<std::shared_ptr<Node>> inputs_ GUARDED_BY(mu_);
    std::shared_ptr<Node> output_ GUARDED_BY(mu_);
  };

  std::vector<std::shared_ptr<Node::Tunable>> CollectTunables()
      SHARED_LOCKS_REQUIRED(mu_);

  int64 OutputTime() SHARED_LOCKS_REQUIRED(mu_);

  int64 ProcessingTime() SHARED_LOCKS_REQUIRED(mu_);

  // Used for coordination between different input pipeline threads. Exclusive
  // access is required only when adding or removing nodes. Concurrent access to
  // existing nodes is protected by a node mutex.
  mutex mu_;
  int64 id_counter_ GUARDED_BY(mu_) = 1;
  std::shared_ptr<Node> output_ GUARDED_BY(mu_);
  std::map<string, std::shared_ptr<Node>> lookup_table_ GUARDED_BY(mu_);

  // Indicates whether the modeling framework should collect resource usage
  // (e.g. CPU, memory). The logic for collecting this information assumes that
  // the collection is not repeatedly disabled and enabled. As a consequence,
  // the implementation starts collecting resource usage when it encounters a
  // tunable parameter (because the information is used for for tuning the value
  // of the parameter) and never stops.
  std::atomic<bool> collect_resource_usage_;

  // A hook invoked immediately before a node is removed from the model.
  const NodeHook remove_node_hook_;
};

}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
