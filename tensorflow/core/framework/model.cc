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

#include "tensorflow/core/framework/model.h"

#include <memory>

#include "absl/time/clock.h"

namespace tensorflow {
namespace data {
namespace model {

std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         int64 min, int64 max) {
  return std::make_shared<Parameter>(name, state, min, max);
}

namespace {

// Given the average time between output events (`output_time`), the average
// time between input events (`input_time`) and the buffer size, the method
// computes the expected time an input event will have to wait.
//
// The wait time is approximated as the product of the probability the buffer
// will be empty and the time it takes to produce an element into the buffer.
//
// The formula used for computing the probability is derived by modeling the
// problem as an M/M/1/K queue
// (https://en.wikipedia.org/wiki/Birth%E2%80%93death_process#M/M/1/K_queue).
int64 ComputeWaitTime(int64 output_time, int64 input_time, int64 buffer_size) {
  if (output_time == 0 || input_time == 0) {
    return output_time;
  }
  if (input_time == output_time) {
    const double p_buffer_empty = 1.0L / static_cast<double>(buffer_size + 1);
    return p_buffer_empty * output_time;
  }
  const double alpha = 1.0L / static_cast<double>(input_time);
  const double beta = 1.0L / static_cast<double>(output_time);
  const double p_buffer_empty =
      (1.0L - beta / alpha) /
      (1.0L - std::pow((beta / alpha), static_cast<double>(buffer_size + 1)));
  return p_buffer_empty * output_time;
}

// The first input of InterleaveMany corresponds to the input dataset whose
// elements are used to create the (derived) input datasets whose elements are
// interleaved as output.
//
// TODO(jsimsa): model the first input
class InterleaveMany : public Node {
 public:
  using Node::Node;

  virtual ~InterleaveMany() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<InterleaveMany>(
        Args{id_, name_, std::move(output)});
  }

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (inputs_.size() <= 1) {
      return NanosPerElementLocked();
    }
    default:
      return;
  }
}

int64 Model::Node::GetParameterValue(const string& name) {
  if (auto* tunable_param = gtl::FindOrNull(tunable_params_, name)) {
    return (*tunable_param)->value;
  }
  return constant_params_[name];
}

int64 Model::Node::ProcessingTimeLocked() {
  switch (type_) {
    case Type::BATCH:
    case Type::MAP_AND_BATCH:
    case Type::PADDED_BATCH: {
      int64 batch_size = GetParameterValue("batch_size");
      return NanosPerElementLocked() + batch_size * ProcessingTimeForInputs();
    }
    case Type::FILTER: {
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      std::shared_ptr<Node> input = inputs_.front();
      double ratio = 0.0L;
      if (num_elements_ > 0) {
        ratio = static_cast<double>(input->num_elements()) /
                static_cast<double>(num_elements_);
      }
      return NanosPerElementLocked() +
             static_cast<int64>(ratio *
                                static_cast<double>(ProcessingTimeForInputs()));
    }
    case Type::FLAT_MAP:
    case Type::INTERLEAVE:
    case Type::PARALLEL_INTERLEAVE:
    case Type::PARALLEL_INTERLEAVE_V2: {
      // TODO(jsimsa): model the first input
      // TODO(jsimsa): use processing time history as a prior for future inputs
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 processing_time =
          ProcessingTimeForInputs() - inputs_.front()->ProcessingTime();
      return NanosPerElementLocked() +
             static_cast<double>(processing_time) /
                 static_cast<double>(inputs_.size() - 1);
    }
    case Type::CACHE:
    case Type::CONCATENATE:
    case Type::MAP:
    case Type::PARALLEL_MAP:
    case Type::PREFETCH:
      // TODO(jsimsa): use processing time history as a prior for future inputs
    case Type::REPEAT:
    case Type::SHUFFLE:
    case Type::SKIP:
    case Type::TAKE:
    case Type::ZIP: {
      return NanosPerElementLocked() + ProcessingTimeForInputs();
    }
    int64 output_time =
        static_cast<double>(OutputTimeForInputs(input_times) -
                            inputs_.front()->OutputTime(input_times)) /
        static_cast<double>(inputs_.size() - 1) / parallelism;
    return ComputeWaitTime(NanosPerElementLocked() + output_time,
                           old_input_time, parallelism);
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    if (inputs_.size() <= 1) {
      return NanosPerElementLocked();
  }
}

int64 Model::Node::OutputTimeLocked(std::vector<int64>* input_times) {
  switch (type_) {
    case Type::BATCH:
    case Type::PADDED_BATCH: {
      double batch_size = GetParameterValue("batch_size");
      int64 old_value = (*input_times)[input_times->size() - 1];
      (*input_times)[input_times->size() - 1] = static_cast<int64>(
          static_cast<double>(old_value + NanosPerElementLocked()) /
          batch_size);
      auto cleanup = gtl::MakeCleanup([input_times, old_value]() {
        (*input_times)[input_times->size() - 1] = old_value;
      });
      return NanosPerElementLocked() +
             batch_size * OutputTimeForInputs(input_times);
    }
    case Type::FILTER: {
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      std::shared_ptr<Node> input = inputs_.front();
      double ratio = 0.0L;
      if (num_elements_ > 0) {
        ratio = static_cast<double>(input->num_elements()) /
                static_cast<double>(num_elements_);
        int64 old_value = (*input_times)[input_times->size() - 1];
        (*input_times)[input_times->size() - 1] = static_cast<int64>(
            static_cast<double>(old_value + NanosPerElementLocked()) / ratio);
        auto cleanup = gtl::MakeCleanup([input_times, old_value]() {
          (*input_times)[input_times->size() - 1] = old_value;
        });
      }
      return NanosPerElementLocked() +
             static_cast<int64>(
                 static_cast<double>(OutputTimeForInputs(input_times)) * ratio);
    }
    case Type::FLAT_MAP:
    case Type::INTERLEAVE: {
      // TODO(jsimsa): model the first input
      // TODO(jsimsa): use cycle length metadata instead of `inputs_.size() - 1`
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 delta =
          static_cast<int64>(static_cast<double>(NanosPerElementLocked()) *
                             static_cast<double>(inputs_.size() - 1));
      (*input_times)[input_times->size() - 1] += delta;
      auto cleanup = gtl::MakeCleanup([input_times, delta]() {
        (*input_times)[input_times->size() - 1] -= delta;
      });
      int64 output_time = OutputTimeForInputs(input_times) -
                          inputs_.front()->OutputTime(input_times);
      return NanosPerElementLocked() +
             static_cast<double>(output_time) /
                 static_cast<double>(inputs_.size() - 1);
    }
    case Type::MAP_AND_BATCH: {
      double batch_size = GetParameterValue("batch_size");
      double parallelism = GetParameterValue("parallelism");
      int64 delta =
          static_cast<int64>(static_cast<double>(NanosPerElementLocked()) /
                             (batch_size * parallelism));
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 output_time = static_cast<int64>(
          static_cast<double>(NanosPerElementLocked()) / parallelism +
          batch_size * OutputTimeForInputs(input_times));
      return std::max(0LL,
                      output_time - input_times->at(input_times->size() - 2));
    }
    case Type::PARALLEL_INTERLEAVE: {
      // TODO(jsimsa): model the first input
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 delta = static_cast<double>(NanosPerElementLocked()) *
                    static_cast<double>(inputs_.size() - 1);
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 inputs_output_time = OutputTimeForInputs(input_times) -
                                 inputs_.front()->OutputTime(input_times);
      double parallelism = GetParameterValue("parallelism");
      int64 output_time =
          static_cast<double>(NanosPerElementLocked()) / parallelism;
      return ComputeWaitTime(output_time, input_times->back(), parallelism);
    }
    int64 old_input_time = input_times->back();
    int64 new_input_time = static_cast<int64>(
        static_cast<double>(NanosPerElementLocked()) / ratio_ / parallelism);
    input_times->push_back(new_input_time);
    auto cleanup =
        gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
    int64 output_time = static_cast<int64>(
        static_cast<double>(NanosPerElementLocked()) / parallelism +
        ratio_ * OutputTimeForInputs(input_times));
    return ComputeWaitTime(output_time, old_input_time, parallelism);
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    return NanosPerElementLocked() + ratio_ * ProcessingTimeForInputs();
  }

 private:
  const double ratio_;
};

class UnknownRatio : public Node {
 public:
  using Node::Node;

  virtual ~UnknownRatio() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<UnknownRatio>(Args{id_, name_, std::move(output)});
  }

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (num_elements_ == 0 || inputs_.empty() ||
        inputs_.front()->num_elements() == 0) {
      return NanosPerElementLocked();
    }
    case Type::PARALLEL_MAP: {
      double parallelism =
          std::min(port::NumSchedulableCPUs(),
                   static_cast<int>(GetParameterValue("parallelism")));
      int64 delta = static_cast<int64>(
          static_cast<double>(NanosPerElementLocked()) / parallelism);
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 output_time =
          static_cast<double>(NanosPerElementLocked()) / parallelism +
          OutputTimeForInputs(input_times);
      return std::max(0LL,
                      output_time - input_times->at(input_times->size() - 2));
    }
    case Type::PREFETCH: {
      int64 delta = NanosPerElementLocked();
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      return std::max(0LL, NanosPerElementLocked() +
                               OutputTimeForInputs(input_times) -
                               input_times->at(input_times->size() - 2));
    }
    case Type::CACHE:
    case Type::CONCATENATE:
    case Type::MAP:
    case Type::REPEAT:
    case Type::SHUFFLE:
    case Type::SKIP:
    case Type::TAKE:
    case Type::ZIP: {
      int64 delta = NanosPerElementLocked();
      (*input_times)[input_times->size() - 1] += delta;
      auto cleanup = gtl::MakeCleanup([input_times, delta]() {
        (*input_times)[input_times->size() - 1] -= delta;
      });
      return NanosPerElementLocked() + OutputTimeForInputs(input_times);
    }
    default:
      return NanosPerElementLocked();
  }
}

void Model::AddConstantParameter(const string& node_name,
                                 const string& parameter_name, int64 value) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, node_name);
  if (node) {
    (*node)->add_constant_param(parameter_name, value);
  }
}

void Model::AddNode(const string& name, const string& output_name) {
  // The name captures the sequence of iterators joined by `::`. We use the full
  // sequence as the key in the lookup table, but only the last element of the
  // sequence as the name node.
  std::vector<string> tokens =
      str_util::Split(name, ':', str_util::SkipEmpty());
  // The output name might contain an index. We need to strip it to make it
  // possible for the model to successfully identify the output node.
  string sanitized_output_name = output_name;
  if (str_util::EndsWith(output_name, "]")) {
    sanitized_output_name = output_name.substr(0, output_name.rfind('['));
  }
  std::shared_ptr<Node> output;
  mutex_lock l(mu_);
  auto it = lookup_table_.find(sanitized_output_name);
  if (it != lookup_table_.end()) {
    output = it->second;
  }
  std::shared_ptr<Node> node(new Node(id_counter_++, tokens.back(), output));
  if (!output_) {
    output_ = node;
  }
  if (output) {
    VLOG(3) << "Adding " << node->long_name() << " as input for "
            << output->long_name();
    output->add_input(node);
  } else {
    VLOG(3) << "Adding " << node->long_name();
  }
  lookup_table_.insert(std::make_pair(name, node));
}

void Model::AddProcessingTime(const string& name, int64 delta) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    (*node)->add_processing_time(delta);
  }
}

void Model::AddTunableParameter(const string& node_name,
                                const string& parameter_name,
                                std::shared_ptr<SharedState> state, int64 min,
                                int64 max) {
  tf_shared_lock l(mu_);
  auto node = *gtl::FindOrNull(lookup_table_, node_name);
  DCHECK(node);
  node->add_tunable_param(parameter_name, std::move(state), min, max);
}

// The optimization algorithm starts by setting all tunable parallelism
// parameters to 1. It then repeatedly identifies the parameter whose increase
// in parallelism decreases the output time the most. This process is repeated
// until all parameters reach their maximum values or the projected output time
// is less than or equal to the processing time needed to produce an element
// divided by CPU budget.
void Model::Optimize(int64 cpu_budget) {
  std::vector<std::shared_ptr<Model::Node::Tunable>> tunables;
  {
    tf_shared_lock lock(mu_);
    snapshot = output_->Snapshot(nullptr);
  }
  VLOG(2) << "Starting optimization of tunable parameters";
  const int64 processing_time = ProcessingTime(snapshot);
  auto parameters = CollectTunableParameters(snapshot);
  for (auto& pair : parameters) {
    pair.second->value = 1;
  }
  while (true) {
    const int64 output_time = OutputTime(snapshot);
    bool all_max = true;
    for (auto& pair : parameters) {
      if (pair.second->value < pair.second->max) {
        all_max = false;
        break;
      }
    }
    if (output_time < processing_time / cpu_budget || all_max) {
      break;
    }
    int64 best_delta = -1;
    Parameter* best_parameter = nullptr;
    for (auto& pair : parameters) {
      if (pair.second->value == pair.second->max) {
        continue;
      }
      pair.second->value++;
      int64 new_output_time = OutputTime(snapshot);
      int64 delta = output_time - new_output_time;
      if (delta < 0) {
        VLOG(3) << "Increasing the parallelism of tunable parameter "
                << pair.first << " resulted in slowdown (before=" << output_time
                << ", after=" << new_output_time
                << "). This should never happen because the latency "
                   "should be monotonic w.r.t. to parallelism.";
      }
      if (delta > best_delta) {
        best_delta = delta;
        best_parameter = pair.second.get();
      }
      pair.second->value--;
    }
    if (!best_parameter) {
      // This should never happen because we are using a model snapshot and
      // the output time is monotonically decreasing w.r.t. parallelism.
      LOG(WARNING) << "Failed to find a tunable parameter that would "
                      "decrease the output time, aborting the current "
                      "optimization attempt.";
      return;
    }
  }
  VLOG(2) << "Number of tunable parameters: " << parameters.size();
  for (auto& pair : parameters) {
    auto& parameter = pair.second;
    VLOG(2) << "Setting tunable parameter " << pair.first << " to "
            << parameter->value;
    mutex_lock l(*parameter->state->mu);
    parameter->state->value = parameter->value;
    parameter->state->cond_var->notify_all();
  }
}

void Model::RecordElement(const string& name) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    (*node)->record_element();
  }
}

int64 Model::NumElements(const string& name) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    return (*node)->num_elements();
  }
  return 0;
}

void Model::RecordStart(const string& name, bool stop_output) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (collect_resource_usage_ && node) {
    int64 now_nanos = absl::GetCurrentTimeNanos();
    if (stop_output && (*node)->output()) {
      (*node)->output()->record_stop();
    }
    (*node)->record_start();
  }
}

void Model::RecordStop(const string& name, bool start_output) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (collect_resource_usage_ && node) {
    int64 now_nanos = absl::GetCurrentTimeNanos();
    (*node)->record_stop(now_nanos);
    if (start_output && (*node)->output()) {
      (*node)->output()->record_start();
    }
  }
}

void Model::RemoveNode(const string& name) {
  mutex_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    if ((*node)->output()) {
      (*node)->output()->remove_input(*node);
    }
    VLOG(3) << "Removing " << (*node)->long_name();
    remove_node_hook_(*node);
  }
  lookup_table_.erase(name);
}

std::map<string, std::shared_ptr<Parameter>> Model::CollectTunableParameters(
    std::shared_ptr<Node> node) {
  std::map<string, std::shared_ptr<Parameter>> parameters;
  node->CollectTunableParameters(&parameters);
  return parameters;
}

int64 Model::OutputTime() {
  std::vector<int64> input_times(1, 0);
  return output_->OutputTime(&input_times);
}

int64 Model::ProcessingTime() { return output_->ProcessingTime(); }

}  // namespace model
}  // namespace data
}  // namespace tensorflow
