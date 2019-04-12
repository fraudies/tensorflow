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

#include "tensorflow/core/grappler/optimizers/function_optimizer.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {
namespace grappler {
namespace {

// WARNING: Code in this file implicitly assumes that function input and output
// arguments are plain tensors (tensor lists are not supported). Function inputs
// and outputs are always expanded to a single placeholder or output tensor.
// With this assumption, the calling node's input/output ports always match
// function input/output arguments.
//
// This is guaranteed by the implementation of MakeGrapplerFunctionItem.

// Mark functions that were created as a result of function specialization.
constexpr char kGrapplerSpecializedFuncAttr[] = "_GrapplerSpecializedFunc";

constexpr char kNoInlineAttr[] = "_noinline";

// Name of the node that will have control edges from function input nodes, and
// also used as a new destination for incoming control edges.
constexpr char kInputsReadyNodeName[] = "inputs_ready";

// Name of the node that will have control edges from function control output
// nodes, and also used as a new source of outgoing control edges. This node
// will guarantee that all side-effects inside function body will be executed
// after function inlining.
constexpr char kSideEffectsExecutedNodeName[] = "side_effects_executed";

bool AttrIsTrue(const FunctionDef& func, const string& attr) {
  return func.attr().count(attr) != 0 && func.attr().at(attr).b();
}

bool MarkedSpecialized(const FunctionDef& func) {
  return AttrIsTrue(func, kGrapplerSpecializedFuncAttr);
}

bool MarkedNoInline(const FunctionDef& func) {
  return AttrIsTrue(func, kNoInlineAttr);
}

// Find unique name for the specialized function. Collision can happen if
// specialized function is instantiated for the nodes with the same name (e.g.
// inside function body of two different functions).
string UniqueSpecializedFunctionName(const FunctionDef& func,
                                     const NodeDef& func_node,
                                     const FunctionLibraryDefinition& flib) {
  using str_util::StringReplace;
  using strings::StrCat;

  string specialized_name = StrCat(func.signature().name(), "_specialized_for_",
                                   StringReplace(func_node.name(), "/", "_",
                                                 /*replace_all*/ true));
  string unique_name = specialized_name;

// Check if func_node has function attribute with a function name matching
// FunctionDef signature.
bool IsIndirectFunctionCall(const FunctionDef& func, const NodeDef& func_node) {
  if (!IsPartitionedCall(func_node) && !IsStatefulPartitionedCall(func_node)) {
    return false;
  }

  auto* func_attr = AttrSlice(func_node).Find(kFuncAttrName);
  return func_attr != nullptr && func_attr->has_func() &&
         func_attr->func().name() == func.signature().name();
}

AttrSlice FunctionInstantiationAttributes(const FunctionDef& func,
                                          const NodeDef& func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    return AttrSlice(func_node);

  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* func_attr = AttrSlice(func_node).Find(kFuncAttrName);
    return AttrSlice(&func_attr->func().attr());

  } else {
    LOG(WARNING) << "Can't resolve function instantiation attributes: "
                 << SummarizeNodeDef(func_node);
    return AttrSlice();
  }
  return unique_name;
}

// Specialized function instantiation type parameters, body parameters, and
// const inputs.
struct FunctionSpecializationSignature {
  // Currently we do not support functions with tensor lists as inputs or
  // outputs, so caller node input/output ports always match function
  // input/output arguments.
  using InputPort = int;
  using OutputPort = int;

  string func_name;
  bool is_in_fetch_set;
  absl::flat_hash_set<OutputPort> active_outputs;
  absl::flat_hash_map<string, DataType> type_parameters;
  absl::flat_hash_map<string, AttrValue> body_parameters;
  absl::flat_hash_map<InputPort, string> const_inputs;

  bool operator==(const FunctionSpecializationSignature& other) const {
    bool equals = func_name == other.func_name &&
                  is_in_fetch_set == other.is_in_fetch_set &&
                  active_outputs == other.active_outputs &&
                  type_parameters == other.type_parameters &&
                  const_inputs == other.const_inputs;

    if (!equals) return false;

    // Equality is not defined for AttrValue.
    if (body_parameters.size() != other.body_parameters.size()) return false;

    for (const auto& lhs : body_parameters) {
      auto it = other.body_parameters.find(lhs.first);
      if (it == other.body_parameters.end()) return false;
      if (!FastAreAttrValuesEqual(lhs.second, (*it).second)) return false;
    }

    return true;
  }

  template <typename H>
  friend H AbslHashValue(H h, const FunctionSpecializationSignature& s) {
    H base = H::combine(std::move(h), s.func_name, s.is_in_fetch_set);

    // First pre-compute hashes for all values in collections with
    // non-deterministic iteration order.
    std::vector<uint64> hashes;
    hashes.reserve(s.active_outputs.size()         //
                   + s.type_parameters.size() * 2  //
                   + s.body_parameters.size() * 2  //
                   + s.const_inputs.size() * 2);

    absl::c_transform(s.active_outputs, std::back_inserter(hashes),
                      hash<OutputPort>());

    using TypeParam = std::pair<const string, DataType>;
    absl::c_for_each(s.type_parameters, [&hashes](const TypeParam& type_param) {
      AttrValue attr_value;
      attr_value.set_type(type_param.second);
      hashes.push_back(Hash64(type_param.first));
      hashes.push_back(AttrValueHash(attr_value));
    });

    using BodyParam = std::pair<const string, AttrValue>;
    absl::c_for_each(s.body_parameters, [&hashes](const BodyParam& body_param) {
      hashes.push_back(Hash64(body_param.first));
      hashes.push_back(FastAttrValueHash(body_param.second));
    });

    using ConstInput = std::pair<const InputPort, string>;
    absl::c_for_each(s.const_inputs, [&hashes](const ConstInput& const_input) {
      hashes.push_back(hash<InputPort>()(const_input.first));
      hashes.push_back(Hash64(const_input.second));
    });

    // Combine all pre-computed hashes in a deterministic order.
    absl::c_sort(hashes);
    return H::combine_contiguous(std::move(base), hashes.data(), hashes.size());
  }
};

struct FunctionSpecialization {
  string specialized_func_name;
  // Names of the tensors that were pushed down into the function body.
  absl::flat_hash_set<string> const_inputs;
  // Control dependencies of pushed down const inputs have to be attached to
  // function caller node.
  absl::flat_hash_set<string> control_deps;
  // Output tensors (ports) that consumed by other nodes in the graph or in a
  // GrapplerItem fetch set.
  absl::flat_hash_set<int> active_outputs;
  // Mapping from original function output port to the output port of
  // specialized function. If function specialization changes the number of
  // function outputs it's required to update all node consumers.
  std::vector<std::pair<int, int>> output_mapping;
};

// Function optimizer context initialized once for each optimization pass, and
// it uses the latest available graph (for the first iteration it will be the
// GrapplerItem.graph, for next iterations it will be the output of previous
// function optimizer pass).
class FunctionOptimizerContext {
 public:
  explicit FunctionOptimizerContext(const GrapplerItem& item,
                                    RewriterConfig::Toggle opt_level,
                                    const GraphDef& graph)
      : item_(&item),
        opt_level_(opt_level),
        function_library_(OpRegistry::Global(), graph.library()),
        truly_const_nodes_(InferTrulyConstNodes(item, graph)),
        graph_view_(&graph) {}

  const GrapplerItem& item() const { return *item_; }

  const int graph_version() const { return item_->graph.versions().producer(); }

  RewriterConfig::Toggle opt_level() const { return opt_level_; }

  const FunctionLibraryDefinition& function_library() const {
    return function_library_;
  }

  FunctionLibraryDefinition* mutable_function_library() {
    return &function_library_;
  }

  FunctionLibraryRuntime* mutable_function_library_runtime() {
    InitializeFunctionLibraryRuntime();
    return flr_;
  }

  const absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>&
  tensor_mapping() const {
    return tensor_mapping_;
  }

  const absl::flat_hash_map<string, std::vector<string>>& control_overrides()
      const {
    return control_overrides_;
  }

  const GraphView& graph_view() const { return graph_view_; }

  const DeviceSet* devices() const {
    // Create fake devices lazily only if we need a DeviceSet.
    if (available_devices_.empty() && !item_->devices().empty()) {
      for (const string& name : item_->devices()) {
        auto device = absl::make_unique<FakeDevice>(name);
        available_device_set_.AddDevice(device.get());
        available_devices_.push_back(std::move(device));
      }
    }
    return &available_device_set_;
  }

  bool IsFetchNode(const string& node_name) const {
    return absl::c_any_of(item_->fetch, [&](const string& fetch) {
      return ParseTensorName(fetch).node() == node_name;
    });
  }

  bool IsKeepOp(const string& node_name) const {
    return absl::c_any_of(item_->keep_ops, [&](const string& keep_node) {
      return keep_node == node_name;
    });
  }

  bool IsInlinedFunction(const string& name) const {
    return inlined_functions_.count(name) > 0;
  }

  bool IsTrulyConst(const string& name) const {
    return TrulyConstNode(name) != nullptr;
  }

  const NodeDef* TrulyConstNode(const string& name) const {
    return gtl::FindWithDefault(truly_const_nodes_, name, nullptr);
  }

  // Find inlining candidate by name. Return nullptr if not found.
  const FunctionDef* FindInlinedFunction(const string& name) const {
    return gtl::FindWithDefault(inlined_functions_, name, nullptr);
  }

  const FunctionSpecialization* FindFunctionSpecialization(
      const FunctionSpecializationSignature& sig) const {
    return gtl::FindOrNull(specialized_functions_, sig);
  }

  void AddSpecializedFunction(const FunctionSpecializationSignature& sig,
                              const FunctionSpecialization& specialized_func) {
    specialized_functions_.emplace(sig, specialized_func);
  }

  void AddTensorMapping(const SafeTensorId& from, const SafeTensorId& to) {
    DCHECK(from.index() != Graph::kControlSlot)
        << "Tensor mapping must be from regular tensor";
    DCHECK(to.index() != Graph::kControlSlot)
        << "Tensor mapping must be to regular tensor";

    auto inserted = tensor_mapping_.insert({from, to});
    DCHECK(inserted.second)
        << "Failed to insert duplicated tensor mapping: "
        << "from=" << from.ToString() << " to=" << to.ToString();
  }

  void AddTensorMapping(const string& func_node,
                        const FunctionSpecialization& specialized_func) {
    for (const auto& pair : specialized_func.output_mapping) {
      int from_idx = pair.first;
      int to_idx = pair.second;
      if (from_idx != to_idx) {
        SafeTensorId from_tensor(func_node, from_idx);
        SafeTensorId to_tensor(func_node, to_idx);
        AddTensorMapping(from_tensor, to_tensor);
      }
    }
  }

  void AddControlOverrides(const NodeDef& func_node,
                           const std::vector<string>& control_overrides) {
    VLOG(4) << "Add control overrides: from=" << func_node.name() << " to: ["
            << absl::StrJoin(control_overrides, ", ") << "]";

    control_overrides_[func_node.name()].reserve(control_overrides.size());
    for (const string& control_override : control_overrides) {
      control_overrides_[func_node.name()].push_back(control_override);
    }
    return false;
  }

 private:
  static absl::flat_hash_map<string, const NodeDef*> InferTrulyConstNodes(
      const GrapplerItem& item, const GraphDef& graph) {
    absl::flat_hash_set<absl::string_view> feed_nodes;
    for (const auto& feed : item.feed) {
      feed_nodes.insert(feed.first);
    }

    absl::flat_hash_map<string, const NodeDef*> const_nodes;
    for (const NodeDef& node : graph.node()) {
      if (IsConstant(node) && !feed_nodes.contains(node.name())) {
        const_nodes[node.name()] = &node;
      }
    }

    return const_nodes;
  }

  void InitializeFunctionLibraryRuntime() {
    if (!flr_) {
      Env* env = Env::Default();
      DeviceAttributes attr;
      attr.set_name("/device:CPU:0");
      attr.set_device_type("CPU");
      Device* device = new FakeCPUDevice(env, attr);
      device_mgr_.reset(new DeviceMgr({device}));
      OptimizerOptions optimizer_opts;
      optimizer_opts.set_do_function_inlining(true);
      process_flr_.reset(new ProcessFunctionLibraryRuntime(
          device_mgr_.get(), env, item_->graph.versions().producer(),
          &function_library_, optimizer_opts));
      flr_ = process_flr_->GetFLR(device_mgr_->ListDevices()[0]->name());
    }
  }

  const GrapplerItem* item_;  // must outlive this object
  RewriterConfig::Toggle opt_level_;

  // Function library constructed from current graph.
  FunctionLibraryDefinition function_library_;

  // These fields initialized lazily only if needed.
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> process_flr_;
  FunctionLibraryRuntime* flr_ = nullptr;

  // List of available `FakedDevices` (lazily initialized, see devices()).
  mutable std::vector<std::unique_ptr<Device>> available_devices_;

  // DeviceSet of fake devices (`FakeDevice`) constructed from
  // item_.devices() (lazily initialized).
  mutable DeviceSet available_device_set_;

  // Nodes that are Const and not in feed.
  absl::flat_hash_map<string, const NodeDef*> truly_const_nodes_;
  // Specialized functions.
  absl::flat_hash_map<FunctionSpecializationSignature,
                      const FunctionSpecialization>
      specialized_functions_;

  // After function inlining and specialization, the optimized graph might be in
  // invalid state, nodes can read from non-existing function call nodes that
  // were inlined, or they can read from output index that is no longer valid
  // after unused outputs pruning.
  //
  // Tensor mapping that has to be applied to the graph after all functions
  // optimizations (invalidated tensor id -> optimized graph tensor id).
  absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>
      tensor_mapping_;

  // When we inline a function into the optimized graph, we no longer have the
  // function call node to anchor control dependencies. Instead we must expand
  // each function call control output edge into multiple control dependencies
  // to all side-effectful ops inside the function body.
  //
  // Invalidated function call node name -> Inlined side-effectful nodes
  absl::flat_hash_map<string, std::vector<string>> control_overrides_;

  // Use graph view to find active outputs of the function caller nodes.
  GraphView graph_view_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionOptimizerContext);
};

// Returns a pointer to the called function definition iff the given node is
// indeed a function call. Otherwise returns nullptr.
const FunctionDef* FindFunctionCall(const FunctionOptimizerContext& ctx,
                                    const NodeDef& node) {
  // Check if a node does indirect function call via PartitionedCallOp.
  if (IsPartitionedCall(node) || IsStatefulPartitionedCall(node)) {
    const AttrValue* func_attr = AttrSlice(node).Find("f");
    return (func_attr != nullptr && func_attr->has_func())
               ? ctx.function_library().Find(func_attr->func().name())
               : nullptr;
  }

  // Check if the function op itself is a function name.
  return ctx.function_library().Find(node.op());
}

absl::flat_hash_set<int> GetActiveOutputs(const NodeDef& node,
                                          const FunctionOptimizerContext& ctx,
                                          int size_hint = 0) {
  absl::flat_hash_set<int> active_outputs;
  active_outputs.reserve(static_cast<size_t>(size_hint));

  // 1. Output can be consumed by the other graph node.
  const auto node_fanout_edges =
      ctx.graph_view().GetFanoutEdges(node, /*include_controlled_edges=*/false);
  for (const GraphView::Edge& edge : node_fanout_edges) {
    active_outputs.insert(edge.src.port_id);
  }

  // 2. Or it can be in a fetch set.
  for (const string& fetch : ctx.item().fetch) {
    TensorId fetch_tensor = ParseTensorName(fetch);
    if (fetch_tensor.node() == node.name()) {
      active_outputs.insert(fetch_tensor.index());
    }
  }

  return active_outputs;
}

bool HasTrulyConstInputs(const NodeDef& node,
                         const FunctionOptimizerContext& ctx) {
  const auto is_truly_const = [&ctx](const string& input) {
    return ctx.IsTrulyConst(NodeName(input));
  };
  return std::any_of(node.input().begin(), node.input().end(), is_truly_const);
}

bool HasUnusedOutputs(const NodeDef& func_node, const FunctionDef& func,
                      const FunctionOptimizerContext& ctx) {
  // Functions with tensor list outputs are not supported right now, so the
  // number of output args is the same as number of possible function caller
  // node outputs.
  int num_outputs = func.signature().output_arg_size();
  const absl::flat_hash_set<int> active_outputs =
      GetActiveOutputs(func_node, ctx, /*size_hind*/ num_outputs);

  return active_outputs.size() != num_outputs;
}

// Return trimmed FunctionDefLibrary with functions that are reachable from
// the optimized graph.
FunctionDefLibrary PruneFunctionLibrary(const FunctionLibraryDefinition& flib,
                                        const GraphDef& optimized_graph) {
  FunctionLibraryDefinition pruned_flib =
      flib.ReachableDefinitions(optimized_graph);

      // 2. AttrValue.ListValue.func
      if (attr_value.has_list()) {
        for (const auto& func : attr_value.list().func()) {
          add_to_func_queue(func.name());
        }
      }
    }
  };

  // Add all functions that are directly called from the optimized graph.
  const auto& graph_nodes = optimized_graph.node();
  std::for_each(graph_nodes.begin(), graph_nodes.end(), add_node_to_func_queue);

  // Process all reachable functions.
  while (!func_queue.empty()) {
    const FunctionDef* func = func_queue.back();
    func_queue.pop_back();

    const string& func_name = func->signature().name();
    keep_funcs.insert(func_name);

    // Find all the functions called from the function body.
    const auto& func_body = func->node_def();
    std::for_each(func_body.begin(), func_body.end(), add_node_to_func_queue);

    // Check if the function has a registered gradient.
    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty()) add_to_func_queue(grad_func_name);
  }

  FunctionDefLibrary lib;
  for (const string& func_name : keep_funcs) {
    const FunctionDef* func = CHECK_NOTNULL(flib.Find(func_name));
    *lib.add_function() = *func;

    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty()) {
      GradientDef* gd = lib.add_gradient();
      gd->set_function_name(func_name);
      gd->set_gradient_func(grad_func_name);
    }
  }

  VLOG(3) << "Trimmed function library: " << keep_funcs.size() << " functions ("
          << static_cast<int>(keep_funcs.size() - flib.num_functions()) << ")";

  return lib;
}

// Push all constant inputs of an instantiating node into the function body.
Status PushDownConstInputs(const NodeDef& func_node,
                           const FunctionOptimizerContext& ctx,
                           GrapplerFunctionItem* item,
                           absl::flat_hash_set<string>* const_inputs,
                           absl::flat_hash_set<string>* control_deps) {
  // Record node control dependencies in the control_deps set.
  const auto record_control_deps = [&](const NodeDef* const_input) {
    for (int i = const_input->input_size() - 1; i >= 0; --i) {
      const string& input = const_input->input(i);
      if (IsControlInput(input))
        control_deps->insert(input);
      else
        break;
    }
  };

  for (int i = func_node.input_size() - 1; i >= 0; --i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) continue;

    const string node_name = NodeName(input);
    if (ctx.IsTrulyConst(node_name)) {
      VLOG(3) << "Push const into function body: input=" << input;
      const auto* const_input = CHECK_NOTNULL(ctx.TrulyConstNode(node_name));
      const_inputs->insert(input);
      record_control_deps(const_input);
      TF_RETURN_IF_ERROR(ReplaceInputWithConst(*const_input, i, item));
    }
  }

  return Status::OK();
}

// Remove inputs that were pushed into the function body, and attach their
// control dependencies to the function caller node.
void RemovePushedDownConstInputs(const gtl::FlatSet<string>& const_inputs,
                                 const gtl::FlatSet<string>& control_deps,
                                 NodeDef* specialized_func_node) {
  // Nothing to do if it was no const inputs to the function node.
  if (const_inputs.empty()) return;

  // Keep only non-const inputs.
  std::vector<string> keep_inputs;
  const auto& inputs = specialized_func_node->input();
  std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(keep_inputs),
               [&](const string& input) {
                 return const_inputs.find(input) == const_inputs.end();
               });

  specialized_func_node->clear_input();
  for (const auto& keep : keep_inputs) specialized_func_node->add_input(keep);

  // Attach control dependencies of pushed down const input to the caller node.
  if (!specialization.control_deps.empty()) {
    absl::flat_hash_set<string> existing_control_deps;

    for (const string& input : keep_inputs) {
      existing_control_deps.insert(AsControlDependency(NodeName(input)));
    }

    for (const string& ctrl : control_deps) {
      if (existing_control_deps.find(ctrl) == existing_control_deps.end()) {
        VLOG(3) << "Forward control dependency: input=" << ctrl;
        specialized_func_node->add_input(ctrl);
      }
    }
  }
}

Status InitializeFunctionSpecializationSignature(
    const NodeDef& func_node, const FunctionDef& func,
    const AttrValueMap& func_attr, const FunctionOptimizerContext& ctx,
    FunctionSpecializationSignature* sig) {
  DCHECK(sig->const_inputs.empty());
  DCHECK(sig->active_outputs.empty());

  sig->func_name = func.signature().name();
  sig->is_in_fetch_set = ctx.IsFetchNode(func_node.name());
  sig->active_outputs = GetActiveOutputs(func_node, ctx);

  TF_RETURN_IF_ERROR(
      InstantiationTypeParameters(func, func_attr, &sig->type_parameters));
  TF_RETURN_IF_ERROR(
      InstantiationBodyParameters(func, func_attr, &sig->body_parameters));

  for (int i = 0; i < func_node.input_size(); ++i) {
    const string& input = func_node.input(i);
    if (ctx.IsTrulyConst(input)) {
      sig->const_inputs.emplace(i, input);
    }
  }

  return Status::OK();
}

// Create a name for the function specialization. The name of the function, name
// of the node instantiating it, and a Grappler item id should generate unique
// function name. Meta optimizer might create multiple Grappler items for the
// same graph when optimizing functions, but it's guaranteed that they all will
// have unique ids.
string SpecializedFunctionName(const FunctionOptimizerContext& ctx,
                               const FunctionDef& func,
                               const NodeDef& func_node) {
  return absl::Substitute(
      "$0_specialized_for_$1_at_$2", func.signature().name(),
      absl::StrReplaceAll(func_node.name(), {{"/", "_"}}), ctx.item().id);
}

Status SpecializeFunction(const NodeDef& func_node, const FunctionDef& func,
                          FunctionOptimizerContext* ctx,
                          GraphDef* optimized_graph) {
  VLOG(2) << "Specialize function instantiation: "
          << SummarizeNodeDef(func_node);

  const std::unordered_map<string, AttrValue> func_attr(
      func_node.attr().begin(), func_node.attr().end());

  FunctionSpecializationSignature signature;
  TF_RETURN_IF_ERROR(InitializeFunctionSpecializationSignature(
      func_node, func, func_attr, *ctx, &signature));

  // Check if function was already specialized for identical context.
  const FunctionSpecialization* already_specialized =
      ctx->FindFunctionSpecialization(signature);

  if (already_specialized) {
    VLOG(2) << "Function was already specialized in identical context: "
               "specialized_name="
            << already_specialized->specialized_func_name;

    // Add a function call node for the specialized function.
    NodeDef* specialized_func_node = optimized_graph->add_node();
    *specialized_func_node = func_node;
    specialized_func_node->set_op(already_specialized->specialized_func_name);

    RemovePushedDownConstInputs(already_specialized->const_inputs,
                                already_specialized->control_deps,
                                specialized_func_node);

    ctx->AddOutputMapping(specialized_func_node->name(), *already_specialized);

    return Status::OK();
  }

  // Add a new specialized function definition to the library.
  const auto& flib = ctx->function_library();

  // Make a GrapplerFunctionItem and convert it back to FunctionDef after
  // pushing all constant inputs into the function body.
  GrapplerFunctionItem item;
  TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
      func, func_instantiation_attr, flib, ctx->graph_version(), &item));

  // Push const inputs into the function body, and keep track of their control
  // dependencies.
  absl::flat_hash_set<string> const_inputs;
  absl::flat_hash_set<string> control_deps;
  TF_RETURN_IF_ERROR(PushDownConstInputs(func_node, *ctx, &item, &const_inputs,
                                         &control_deps));

  // Remove function outputs that do not have any consumers. We can't safely
  // update outputs for the fetch nodes, so we just skip them.
  std::vector<std::pair<int, int>> output_mapping;
  if (!signature.is_in_fetch_set) {
    int num_func_outputs = item.output_size();

    absl::flat_hash_set<int> remove;
    for (int i = 0; i < num_func_outputs; ++i) {
      if (!signature.active_outputs.count(i)) remove.insert(i);
    }

    TF_RETURN_IF_ERROR(RemoveFunctionOutputs(remove, &item, &output_mapping));
  }

  // TODO(ezhulenev): Push down known input shapes.
  FunctionDef specialized_func;
  TF_RETURN_IF_ERROR(MakeFunctionDef(item, flib, &specialized_func));

  // Find a name for specialized function.
  const string specialized_func_name =
      UniqueSpecializedFunctionName(func, func_node, flib);

  specialized_func.mutable_signature()->set_name(specialized_func_name);
  auto* specialized_attr = specialized_func.mutable_attr();
  (*specialized_attr)[kGrapplerSpecializedFuncAttr].set_b(true);

  // Add specialized function to the library.
  TF_RETURN_IF_ERROR(
      ctx->mutable_function_library()->AddFunctionDef(specialized_func));

  // Add a function call node for the specialized function.
  NodeDef* specialized_func_node = optimized_graph->add_node();
  *specialized_func_node = func_node;
  specialized_func_node->set_op(specialized_func_name);

  // Update specialized node to remove inputs for pushed down consts.
  RemovePushedDownConstInputs(const_inputs, control_deps,
                              specialized_func_node);

  FunctionSpecialization func_specialization = {
      specialized_func_name, const_inputs, control_deps, output_mapping};
  ctx->AddSpecializedFunction(signature, func_specialization);
  ctx->AddOutputMapping(specialized_func_node->name(), func_specialization);

  return Status::OK();
}

// Create an IdentityN node to hook the function inputs to: this ensures that
// they're all evaluated before the evaluation of the function body starts.
NodeDef InlinedFunctionInputsNode(const NodeDef& func_node,
                                  const GrapplerFunctionItem& item) {
  NodeDef inputs;
  inputs.set_name(strings::StrCat(func_node.name(), "/", "inlined_inputs"));
  inputs.set_op("IdentityN");
  inputs.set_device(func_node.device());
  *inputs.mutable_input() = func_node.input();
  AttrValue::ListValue* type_list =
      (*inputs.mutable_attr())["T"].mutable_list();

  for (const InputArgInstantiation& input_arg : item.inputs()) {
    type_list->add_type(input_arg.data_type);
  }

  return inputs;
}

// Create an IdentityN node to hook the function outputs to: this ensures that
// the function body is fully evaluated before its fanout gets scheduled.
NodeDef InlinedFunctionOutputsNode(
    const NodeDef& func_node, const GrapplerFunctionItem& item,
    const absl::flat_hash_map<absl::string_view, absl::string_view>
        output_tensors) {
  NodeDef outputs;
  outputs.set_name(func_node.name());
  outputs.set_op("IdentityN");
  outputs.set_device(func_node.device());
  AttrValue::ListValue* type_list =
      (*outputs.mutable_attr())["T"].mutable_list();

  for (const OutputArgInstantiation& output_arg : item.outputs()) {
    const absl::string_view output_tensor =
        output_tensors.at(output_arg.node_name);
    type_list->add_type(output_arg.data_type);
    outputs.add_input(strings::StrCat(func_node.name(), "/", output_tensor));
  }

  return outputs;
}

Status InlineDirectFunctionCall(const NodeDef& func_node,
                                const FunctionDef& func,
                                const FunctionOptimizerContext& ctx,
                                GraphDef* optimized_graph) {
  VLOG(2) << "Inline direct function call: " << SummarizeNodeDef(func_node);
  TF_RETURN_IF_ERROR(IsInlinableDirectFunctionCall(ctx, func, func_node));

  const std::unordered_map<string, AttrValue> func_attr(
      func_node.attr().begin(), func_node.attr().end());

  GrapplerFunctionItem item;
  Status item_status = MakeGrapplerFunctionItem(func, func_instantiation_attr,
                                                ctx.function_library(),
                                                ctx.graph_version(), &item);

  if (!item_status.ok()) {
    return errors::InvalidArgument("Failed to inline function ", func_node.op(),
                                   " instantiated by ", func_node.name(),
                                   ". Error: ", item_status.error_message());
  }

  // Mapping from input arg node name to function input position.
  absl::flat_hash_map<absl::string_view, int> input_args_idx;
  for (const InputArgInstantiation& input_arg : item.inputs()) {
    const int idx = input_args_idx.size();
    input_args_idx[input_arg.node_name] = idx;
  }

  // Mapping from the '_Retval' node name to the output tensor.
  absl::flat_hash_map<absl::string_view, absl::string_view> output_tensors;
  for (const NodeDef& func_body_node : item.function_body().node()) {
    if (!IsRetval(func_body_node)) continue;
    if (func_body_node.input_size() != 1) {
      return errors::Internal("_Retval node must have single input: ",
                              SummarizeNodeDef(func_body_node));
    }
    output_tensors.emplace(func_body_node.name(), func_body_node.input(0));
  }

  // Hook inlined function inputs to IdentityN node.
  NodeDef* func_inputs = optimized_graph->add_node();
  *func_inputs = InlinedFunctionInputsNode(func_node, item);

  for (NodeDef& func_body_node : *item.mutable_function_body().mutable_node()) {
    const string& node_name = func_body_node.name();

    // Skip function output nodes.
    if (IsRetval(func_body_node)) continue;

    // Turn _Arg nodes added in place of input arguments into identity nodes.
    const auto input_arg_idx = input_args_idx.find(node_name);
    if (input_arg_idx != input_args_idx.end()) {
      CHECK_EQ(0, func_body_node.input_size());
      func_body_node.set_op("Identity");
      func_body_node.mutable_attr()->erase("index");
      func_body_node.mutable_attr()->erase("shape");
      func_body_node.add_input(
          strings::StrCat(func_inputs->name(), ":", input_arg_idx->second));
    } else {
      // Update the input names if any.
      for (string& input : *func_body_node.mutable_input()) {
        input = AddPrefixToNodeName(input, /*prefix=*/func_node.name());
      }
      // If the node has no input, make hook it up to the func_inputs node to
      // ensure it runs in the same frame as the other nodes of the function
      // body.
      if (func_body_node.input_size() == 0) {
        *func_body_node.add_input() = AsControlDependency(func_inputs->name());
      }
    }

    // Add the function node name as a prefix 1) to node name to avoid
    // collisions; 2) to frame name to avoid multiple LoopCond nodes in one
    // frame after inlining.
    const string prefix = strings::StrCat(func_node.name(), "/");
    TF_RETURN_IF_ERROR(
        AddPrefixAndSuffixToNode(prefix, "" /* suffix */, &func_body_node));

    // Make sure the node is placed.
    func_body_node.set_device(func_node.device());

    // Move the node to the main graph.
    optimized_graph->add_node()->Swap(&func_body_node);
  }

  DCHECK(output_tensors.size() == item.output_size())
      << "Each function output must be mapped to an output tensor";

  // Hook inlined function outputs to IdentityN node.
  NodeDef* func_outputs = optimized_graph->add_node();
  *func_outputs = InlinedFunctionOutputsNode(func_node, item, output_tensors);

  return Status::OK();
}

Status InlineSymbolicGradient(const NodeDef& node,
                              FunctionOptimizerContext* ctx,
                              GraphDef* optimized_graph) {
  VLOG(2) << "Inline symbolic gradient: " << SummarizeNodeDef(node);

  GraphDef graph_def;

  // Create a node to anchor the gradient inputs
  NodeDef* inlined_input = graph_def.add_node();
  inlined_input->set_name("FunctionInputs");
  inlined_input->set_op("IdentityN");
  AttrValue::ListValue* type_list =
      (*inlined_input->mutable_attr())["T"].mutable_list();
  for (const auto& type : node.attr().at("Tin").list().type()) {
    type_list->add_type(static_cast<DataType>(type));
  }

  // Add the gradient node
  NodeDef* inlined = graph_def.add_node();
  *inlined = node;
  inlined->clear_input();
  for (int i = 0; i < node.attr().at("Tin").list().type_size(); ++i) {
    inlined->add_input(strings::StrCat(inlined_input->name(), ":", i));
  }

  // Create a node to anchor the gradient outputs
  NodeDef* inlined_output = graph_def.add_node();
  inlined_output->set_name("FunctionOutputs");
  inlined_output->set_op("IdentityN");
  type_list = (*inlined_output->mutable_attr())["T"].mutable_list();
  for (const auto& type : node.attr().at("Tout").list().type()) {
    type_list->add_type(static_cast<DataType>(type));
  }
  for (int i = 0; i < node.attr().at("Tout").list().type_size(); ++i) {
    inlined_output->add_input(strings::StrCat(inlined->name(), ":", i));
  }

  // Convert the graphdef to a graph
  GraphConstructorOptions graph_ctor_opts;
  graph_ctor_opts.allow_internal_ops = true;
  graph_ctor_opts.expect_device_spec = false;
  Graph graph(ctx->function_library());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(graph_ctor_opts, graph_def, &graph));

  FunctionLibraryRuntime* flr = ctx->mutable_function_library_runtime();

  // 1. Inline symbolic gradient node.
  const bool expanded = ExpandInlineFunctions(flr, &graph);
  if (!expanded) {
    return errors::Internal("Failed to expand SymbolicGradient op");
  }

  // TODO(ezhulenev): InlineFunctionBody in common_runtime/function silently
  // fails to inline function into the graph, and leaves the graph unmodified.
  // We check that graph has our symbolic gradient inlined, otherwise we return
  // a error.
  const auto is_symbolic_gradient_op = [&](const Node* node) {
    return node->name() == inlined->name() &&
           node->type_string() == "SymbolicGradient";
  };
  for (Node* node : graph.nodes()) {
    if (is_symbolic_gradient_op(node)) {
      return errors::Internal("Failed to inline symbolic gradient node: ",
                              SummarizeNode(*node));
    }
  }

  // 2. Recursively inline nested function calls.
  int iteration = 0;
  while (ExpandInlineFunctions(flr, &graph)) {
    if (++iteration >= 50) {
      VLOG(2) << "Break symbolic gradient inlining loop at iteration #"
              << iteration;
      break;
    }
  }

  GraphDef inlined_graph_def;
  graph.ToGraphDef(&inlined_graph_def);

  // Add the default values of attributes to the nodes that have been inlined.
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&inlined_graph_def,
                                               *graph.op_registry(), 0, true));

  // Add the inlined nodes to the graph
  for (NodeDef& inlined_node : *inlined_graph_def.mutable_node()) {
    if (inlined_node.name() == "FunctionOutputs") {
      inlined_node.set_name(node.name());
      for (int i = 0; i < inlined_node.input_size(); ++i) {
        inlined_node.set_input(
            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
      }
    } else if (inlined_node.name() == "FunctionInputs") {
      inlined_node.set_name(
          AddPrefixToNodeName(inlined_node.name(), node.name()));
      inlined_node.clear_input();
      for (int i = 0; i < node.input_size(); ++i) {
        inlined_node.add_input(node.input(i));
      }
    } else {
      inlined_node.set_name(
          AddPrefixToNodeName(inlined_node.name(), node.name()));
      for (int i = 0; i < inlined_node.input_size(); ++i) {
        inlined_node.set_input(
            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
      }
      // If the node has no input, hook it up to the function input node to make
      // sure it runs in the same frame as the other nodes of the function body.
      if (inlined_node.input_size() == 0) {
        *inlined_node.add_input() = AsControlDependency(
            AddPrefixToNodeName("FunctionInputs", node.name()));
      }
    }
    inlined_node.set_device(node.device());
    optimized_graph->add_node()->Swap(&inlined_node);
  }

  return Status::OK();
}

// -------------------------------------------------------------------------- //
// Inline indirect functions calls (aka PartitionedCallOp).
//
// When we inline indirect function calls, we instantiate the function body from
// its FunctionDef and caller node attributes, and embed the instantiated graph
// into the "main graph".
//
// In contrast to direct function calls, `PartitionedCallOp` has automatic
// dependency tracking via input/output control edges, and we relax some of the
// constraints that we have for direct function call inlining.
//
// Automatic control dependency rules:
//
// 1) "When a `PartitionedCallOp` function has a resource (DT_RESOURCE data
//    type) input argument it "captures" the mutable resource.  This is
//    implemented by automatically adding a incoming control edge from the
//    previous side-effectful op touching that resource, and an outgoing control
//    edge to the next side-effectful op using the same resource. This
//    serializes the mutations of the resource to make graph execution
//    deterministic.
//
// 2) All stateful ops inside a function body are guaranteed to execute in
//    program order, this is achieved by adding control edges between stateful
//    ops at graph construction time.
//
// 3) Furthermore, all ops accepting the same resource as an input are
//    guaranteed to run in program order. This is also done by adding control
//    edges at graph construction time. The last op touching the resource
//    will have an outgoing control edge to all function return nodes, which
//    will guarantee that all side effects to the resource will happen before
//    function completion.
//
// Function call inlining must preserve side effect visibility:
//
// 1) All side effects to the captured resources, that happened before function
//    call must be visible to the function body nodes using that resources.
// 2) All side effects to the captured resources, that happened inside function
//    body, must be visible to every op/function using that resource after the
//    function call completed.
//
// To guarantee that these properties are preserved after inlining we:
//
// 1) Create "input_control" NoOp. Function call node incoming control edges
//    will be forwarded *to* this node. Function inputs (Identity nodes) will
//    have a control edge *from* this node. If function has no inputs, by
//    construction it must have nodes without inputs in the function body, and
//    in this case these nodes will have a control edge *from* this node.

// 2) Create "output_control" NoOp. All nodes that have incoming control edge
//    *from* the function call node, will be forwarded to this node. Function
//    outputs (Identity nodes) will have a control edge *to* this node. This
//    will guarantee that nodes that have control dependency on the function
//    call, will observe all side-effects (guaranteed by graph construction with
//    automatic control dependencies tracking).
//
// If after function instantiation we find a stateful or a dataset op inside
// the function body, that is not reachable from any of the function outputs (or
// if the function has no outputs), we do not inline it, because we can't
// guarantee that these nodes will be executed in correct order (or executed at
// all) after inlining.
//
// We do not try to add any extra control edges to make sure that all
// side-effectful nodes will be executed, that should be handled at graph
// construction time.

struct MaybeDeadOutput {
  const NodeDef* dead_tensor_src;
  const NodeDef* output_node_dst;
};

// Finds all function outputs that might return a dead tensor. This can happen
// if there is no `Merge` node on the path from the `Switch` node, to the
// function output.
Status MaybeDeadOutputs(const FunctionOptimizerContext& ctx,
                        const GrapplerFunctionItem& item,
                        std::vector<MaybeDeadOutput>* maybe_dead) {
  VLOG(3) << "Find function outputs that might return dead tensors: item.id="
          << item.id;
  DCHECK(maybe_dead->empty()) << "Input argument must be an empty vector";

  std::vector<const NodeDef*> dead_tensor_srcs;
  for (const NodeDef& node : item.graph.node()) {
    if (IsSwitch(node)) {
      VLOG(4) << "Add dead tensors source. Switch node: " << node.name();
      dead_tensor_srcs.push_back(&node);
      continue;
    }

    // Regular (aka 'direct') function call can also produce dead tensors if
    // the function body has mergeless switches.
    const FunctionDef* func = ctx.function_library().Find(node.op());
    if (func != nullptr) {
      GrapplerFunctionItem func_item;
      TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
          *func, FunctionInstantiationAttributes(*func, node),
          ctx.function_library(), ctx.graph_version(), &func_item));

      std::vector<MaybeDeadOutput> func_dead_outputs;
      TF_RETURN_IF_ERROR(MaybeDeadOutputs(ctx, func_item, &func_dead_outputs));

      if (!func_dead_outputs.empty()) {
        VLOG(4) << "Add dead tensors source. Function call: " << node.op()
                << " node=" << node.name();
        dead_tensor_srcs.push_back(&node);
      }
    }
  }

  // If we do not have dead tensor sources in the function body, it's
  // guaranteed that all output tensors can't become dead.
  if (dead_tensor_srcs.empty()) return Status::OK();

  // Names of the function body nodes that return function output values.
  absl::flat_hash_set<absl::string_view> output_nodes;
  for (const auto& output_arg : item.outputs()) {
    output_nodes.insert(output_arg.node_name);
  }

  GraphTopologyView topology_view;
  TF_RETURN_IF_ERROR(topology_view.InitializeFromGraph(item.graph));

  for (const NodeDef* dead_tensor_src : dead_tensor_srcs) {
    DfsTraversal(topology_view, {dead_tensor_src},
                 TraversalDirection::kFollowOutputs,
                 // Stop traversal when reached first `Merge` node.
                 DfsPredicates::Advance(
                     [](const NodeDef* node) { return !IsMerge(*node); }),
                 // If we reached output node, add MaybeDeadOutput edge.
                 DfsCallbacks::PreOrder([&](const NodeDef* node) {
                   if (output_nodes.find(node->name()) != output_nodes.end()) {
                     maybe_dead->push_back({dead_tensor_src, node});
                   }
                 }));
  }

  return Status::OK();
}

// Returns `Status::OK()` iff `node` is an indirect function call of `func`, and
// we know how to inline it into the main graph, otherwise returns and error
// indicating why the function call is not inlinable.
Status IsInlinableIndirectFunctionCall(const FunctionOptimizerContext& ctx,
                                       const FunctionDef& func,
                                       const NodeDef& func_node) {
  // We inline direct function calls above, using different rules.
  if (!IsIndirectFunctionCall(func, func_node)) {
    return errors::InvalidArgument("Unsupported function call type: ",
                                   SummarizeNodeDef(func_node));
  }

  if (MarkedNoInline(func)) {
    return errors::FailedPrecondition(
        "Can't inline function marked with '_noinline': ",
        SummarizeNodeDef(func_node));
  }

  // Function specialization and inlining must be mutually exclusive.
  if (MarkedSpecialized(func)) {
    return errors::FailedPrecondition(
        "Can't inline function created in Grappler function specialization: ",
        SummarizeNodeDef(func_node));
  }

  // We can't inline functions that are in a fetch set, because it would
  // invalidate fetch tensors (function call node fully inlined and doesn't
  // exist in the optimized graph).
  if (ctx.IsFetchNode(func_node.name())) {
    return errors::FailedPrecondition(
        "Can't inline function in a Grappler item fetch set: ",
        SummarizeNodeDef(func_node));
  }

  return Status::OK();
}

// Checks that all side-effects will be executed in well defined order. We do it
// by checking if there is a path from stateful/dataset ops to one of the
// control output nodes.
Status CheckThatSideEffectsWillExecute(
    const FunctionOptimizerContext& ctx,
    const GraphTopologyView& graph_topo_view,
    const absl::flat_hash_set<string> control_output_nodes) {
  // In aggressive mode we just print a warning for side-effectful nodes that
  // might not be executed after inlining.
  const bool aggressive = ctx.opt_level() == RewriterConfig::AGGRESSIVE;

  for (const NodeDef& func_body_node : graph_topo_view.graph()->node()) {
    const bool node_must_execute =
        IsDataset(func_body_node) ||
        IsStateful(func_body_node, &ctx.function_library());

    // If op has DT_RESOURCE argument it will be marked as stateful, though if
    // it only reads from that resource, it's allowed to prune it, because it
    // can't produce any visible side-effects.
    const bool read_only = IsReadVariableOp(func_body_node);

    // _Retval marked as stateful, but we will remove it before inlining.
    const bool retval = IsRetval(func_body_node);

    if (read_only || retval || !node_must_execute) continue;

    VLOG(3) << "Check that node " << func_body_node.name()
            << " will execute after inlining.";
    bool will_execute = false;

    // Check if we reached one of the output nodes.
    const auto callbacks = DfsCallbacks::PreOrder([&](const NodeDef* node) {
      if (control_output_nodes.contains(node->name())) {
        VLOG(4) << "Found a path to control output node: " << node->name();
        will_execute = true;
      }
    });

    // Stop if we already proved that node will execute.
    const auto predicates = DfsPredicates::Enter(
        [&](const NodeDef* node) { return !will_execute; });

    DfsTraversal(graph_topo_view, {&func_body_node},
                 TraversalDirection::kFollowOutputs, predicates, callbacks);

    if (!will_execute) {
      const string error_message = absl::StrCat(
          "Can't guarantee execution of a side-effectful node, that is not "
          "reachable from function outputs. Function body node: ",
          SummarizeNodeDef(func_body_node));

      if (aggressive) {
        LOG(WARNING) << error_message;
      } else {
        return errors::Internal(error_message);
      }
    }
  }

  return Status::OK();
}

Status PlaceInlinedFunctionBody(
    const NodeDef& func_node, const GrapplerFunctionItem& item,
    const absl::flat_hash_map<absl::string_view, int>& input_args_idx,
    FunctionOptimizerContext* ctx, GraphDef* placed_graph_def) {
  // Control flow lowering and Placer works with a Graph object.
  std::unique_ptr<Graph> func_body_graph =
      absl::make_unique<Graph>(ctx->function_library());

  GraphConstructorOptions opts;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(opts, item.graph, func_body_graph.get()));

  // ------------------------------------------------------------------------ //
  // Grappler receives the graph after PRE_PLACEMENT, Placer, and POST_PLACEMENT
  // passes, so each node has a valid device assignment. Also V2 control
  // flow ops (functional If and While) should have been lowered to V1 control
  // flow (Switch and Merge nodes). To keep the graph valid for execution we
  // must assign device to every inlined graph node, and also lower the control
  // flow.

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = &func_body_graph;
  opt_options.flib_def = ctx->mutable_function_library();

  // TODO(ezhulenev): Should we run full PRE_PLACEMENT pass here? And
  // POST_PLACEMENT after placer?
  LowerFunctionalOpsPass pass(/*lower_function_calls=*/false,
                              /*keep_lowered_nodes_fetchable=*/false);
  TF_RETURN_IF_ERROR(pass.Run(opt_options));

  // ------------------------------------------------------------------------ //
  // Before placing the function body nodes we pin input arguments to the
  // same device as their corresponding input nodes.

  for (Node* func_body_node : func_body_graph->nodes()) {
    const auto input_arg_idx = input_args_idx.find(func_body_node->name());

    if (input_arg_idx != input_args_idx.end()) {
      const int input_idx = input_arg_idx->second;
      const GraphView::OutputPort output_port =
          ctx->graph_view().GetRegularFanin({&func_node, input_idx});

      const string& input_device = output_port.node->device();

      if (!input_device.empty()) {
        VLOG(3) << "Pin inlined function input node '" << func_body_node->name()
                << "' to the '" << output_port.node->device() << "' device.";
        func_body_node->set_requested_device(output_port.node->device());
      } else {
        VLOG(3) << "Inlined function input node '" << func_body_node->name()
                << "' device is undefined.";
      }
    }
  }

  // ------------------------------------------------------------------------ //
  // After placing nodes corresponding to the function inputs, we need to assign
  // device placements to all other function body nodes.

  const DeviceSet* devices = ctx->devices();

  if (devices->devices().empty()) {
    // If there are no devices available for placer, we do not place function
    // body nodes. This happens when Grappler optimizing function library, or
    // when graph optimized "offline", without active runtime session, for
    // example as a part of batch job for graph analysis/optimization.
    // GrapplerItem instantiated from a function library doesn't have to be
    // fully placed after all optimization, it will be placed by the function
    // library runtime before execution.
    VLOG(3) << "Do not place instantiated function body.";
  } else {
    // If we are running in an active runtime session, Grappler will get the
    // graph after initial placing is done, and we should have devices for the
    // placer.
    VLOG(3) << "Run placer for instantiated function body. Devices: ["
            << absl::StrJoin(
                   devices->devices(), ", ",
                   [](string* out, const Device* d) { out->append(d->name()); })
            << "]";

    // Use function caller node device as a default for placer.
    const Device* default_device =
        devices->FindDeviceByName(func_node.device());

    Placer placer(func_body_graph.get(), devices, default_device);
    TF_RETURN_IF_ERROR(placer.Run());
  }

  // Convert Graph back to the placed GraphDef.
  func_body_graph->ToGraphDef(placed_graph_def);

  return Status::OK();
}

Status InlineIndirectFunctionCall(const NodeDef& func_node,
                                  const FunctionDef& func,
                                  FunctionOptimizerContext* ctx,
                                  GraphDef* optimized_graph) {
  VLOG(2) << "Inline indirect function call: " << SummarizeNodeDef(func_node);
  VLOG(4) << "Inlined function definition: " << DebugString(func);
  TF_RETURN_IF_ERROR(IsInlinableIndirectFunctionCall(*ctx, func, func_node));

  const AttrSlice func_instantiation_attr =
      FunctionInstantiationAttributes(func, func_node);

  GrapplerFunctionItem item;
  Status item_status = MakeGrapplerFunctionItem(func, func_instantiation_attr,
                                                ctx->function_library(),
                                                ctx->graph_version(), &item);

  if (!item_status.ok()) {
    return errors::InvalidArgument("Failed to inline function ", func_node.op(),
                                   " instantiated by ", func_node.name(),
                                   ". Error: ", item_status.error_message());
  }

  // `PartitionedCallOp` invokes functions with `allow_dead_tensors = true` to
  // reset dead flag, and return default initialized tensors instead of a dead
  // tensors. There is no way to express this in a regular Tensorflow graph, so
  // we choose not to inline if a function can have dead tensors as an output
  // position. In practice `mergeless switches` should not exists in a function
  // body, because tf-eager will only use v2 control flow ops.
  std::vector<MaybeDeadOutput> maybe_dead_outputs;
  TF_RETURN_IF_ERROR(MaybeDeadOutputs(*ctx, item, &maybe_dead_outputs));
  if (!maybe_dead_outputs.empty()) {
    struct MaybeDeadOutputFormatter {
      void operator()(string* out, const MaybeDeadOutput& md) const {
        absl::StrAppend(out, SummarizeNodeDef(*md.dead_tensor_src));
      }
    };
    return errors::FailedPrecondition(
        "Can't inline function with dead outputs. Dead tensor sources (size = ",
        maybe_dead_outputs.size(), "): ",
        absl::StrJoin(maybe_dead_outputs, "\n", MaybeDeadOutputFormatter()));
  }

  GraphView::InputPort control_input_port =
      ctx->graph_view().GetInputPort(func_node.name(), Graph::kControlSlot);
  GraphView::OutputPort control_output_port =
      ctx->graph_view().GetOutputPort(func_node.name(), Graph::kControlSlot);

  // Nodes that have side effects to the captured resources.
  std::vector<string> happens_before;
  absl::c_transform(
      ctx->graph_view().GetFanin(control_input_port),
      std::back_inserter(happens_before),
      [](const GraphView::OutputPort port) { return port.node->name(); });

  VLOG(3) << "Happens before set (size = " << happens_before.size()
          << "): " << absl::StrJoin(happens_before, ", ");

  // Nodes that must observe side effects to the captured resources.
  std::vector<string> happens_after;
  absl::c_transform(
      ctx->graph_view().GetFanout(control_output_port),
      std::back_inserter(happens_after),
      [](const GraphView::InputPort port) { return port.node->name(); });

  VLOG(3) << "Happens after set (size = " << happens_after.size()
          << "): " << absl::StrJoin(happens_after, ", ");

  // Regular (data) inputs to the function call.
  std::vector<SafeTensorId> inputs;
  for (const string& input : func_node.input()) {
    SafeTensorId tensor_id = ParseTensorName(input);
    if (tensor_id.index() == Graph::kControlSlot) break;
    inputs.push_back(tensor_id);
  }

  // Mapping from input argument node to function input position.
  absl::flat_hash_map<absl::string_view, int> input_args_idx;
  for (const InputArgInstantiation& input_arg : item.inputs()) {
    const int idx = input_args_idx.size();
    input_args_idx[input_arg.node_name] = idx;
  }

  const string prefix = strings::StrCat(func_node.name(), "/");

  // ------------------------------------------------------------------------ //
  // IMPORTANT: Actual inputs will be added to the following nodes at the very
  // last stage, because we don't want to have invalid edges in a function body
  // graph (control edges that depend on the nodes in the "outer" optimized
  // graph).

  // If one of the function inputs is a dead tensor, we must not execute any of
  // the function body nodes, and let the dead tensor flag propagate through the
  // inlined function body. We add NoOp inputs_ready node, and add control edges
  // to it from all input nodes. Inlined function arguments (Identity nodes)
  // will have a control dependency on it.
  //
  // TODO(ezhulenev): We do not need to provide this guarantee for ALL nodes in
  // the function body. We must only ensure that we do not generate observable
  // side effects.
  //
  // If the function call node has incoming control edges, we will update them
  // to use this node as destination, to ensure side-effects execution order.
  NodeDef* inputs_ready_node = nullptr;
  if (func_node.input_size() > 0) {
    inputs_ready_node = item.graph.add_node();
    inputs_ready_node->set_op("NoOp");
    inputs_ready_node->set_name(kInputsReadyNodeName);
  }

  // All nodes that have a control edge from the function call node, will be
  // updated to have a control edge from 'side_effects_executed_node`. This node
  // will have control edges from all function control outputs (see
  // `control_ret` in FunctionDef). This a "barrier" that guarantees that all
  // ops with side effects in the function body were executed
  //
  // If the function call node has no outgoing control edges, it means that no
  // one is interested in the function side-effect affecting captured resources.
  //
  // If node is in keep_ops set, it means that it must execute. This could
  // happen if the graph is an instantiation of a function with control output.
  NodeDef* side_effects_executed_node = nullptr;
  if (!happens_after.empty() || ctx->IsKeepOp(func_node.name())) {
    side_effects_executed_node = item.graph.add_node();
    side_effects_executed_node->set_op("NoOp");
    side_effects_executed_node->set_name(kSideEffectsExecutedNodeName);
  }

  // If function executed only for the regular data outputs, it's totally safe
  // to prune side-effects. If side-effects order is important, it must be
  // captured at graph construction time via control edges.
  if (item.control_output_size() > 0 && happens_after.empty()) {
    VLOG(2) << "Function has control outputs and empty happens after set.";
  }

  // ------------------------------------------------------------------------ //
  // If we have a node inside the function body without inputs (e.g. Const), we
  // must attach a control dependency to it, to make sure that if a function
  // call happens inside a loop, the node will be evaluated in correct frame.
  //
  // If the function call node has no inputs and no control dependencies, it
  // means that it can't be a function call inside a loop, and we can safely
  // insert that node without inputs into the main graph.
  //
  // TODO(ezhulenev): Use FrameMap (see grappler/utils/frame.h) to find out if
  // the function is called inside a loop.
  std::vector<string> empty_inputs_hook;
  if (inputs_ready_node != nullptr) {
    empty_inputs_hook.push_back(inputs_ready_node->name());
  }

  // ------------------------------------------------------------------------ //
  // Grappler called after PRE_PLACEMENT and PLACEMENT passes, so we have to
  // make sure that after inlining all nodes will have valid device assignment.

  GraphDef placed_graph_def;
  TF_RETURN_IF_ERROR(PlaceInlinedFunctionBody(func_node, item, input_args_idx,
                                              ctx, &placed_graph_def));

  // ------------------------------------------------------------------------ //
  // Mapping from the '_Retval' node name to the output tensor. We build this
  // mapping after the placement, because we might have inlined some of the
  // functional If/While nodes (see a call to LowerFunctionalOpsPass).
  absl::flat_hash_map<string, string> output_tensors;

  for (const NodeDef& func_body_node : placed_graph_def.node()) {
    if (!IsRetval(func_body_node)) continue;
    if (func_body_node.input_size() != 1) {
      return errors::Internal("_Retval node must have single input: ",
                              SummarizeNodeDef(func_body_node));
    }
    output_tensors.emplace(func_body_node.name(), func_body_node.input(0));
  }

  // ------------------------------------------------------------------------ //
  // After all nodes placed we need to prepare them for inlining into the
  // optimized graph: turn placeholders into identities, update nodes
  // connectivity, etc...

  const auto inlined_node_name = [&func_node](const string& name) -> string {
    return AddPrefixToNodeName(name, /*prefix=*/func_node.name());
  };

  for (NodeDef& func_body_node : *placed_graph_def.mutable_node()) {
    const string& node_name = func_body_node.name();

    // Turn _Arg nodes added in place of input arguments into identity nodes.
    const auto input_arg_idx = input_args_idx.find(node_name);
    if (input_arg_idx != input_args_idx.end()) {
      DCHECK_EQ(0, func_body_node.input_size());
      func_body_node.set_op("Identity");
      func_body_node.mutable_attr()->erase("index");
      func_body_node.mutable_attr()->erase("shape");
      const int input_idx = input_arg_idx->second;
      func_body_node.add_input(inputs[input_idx].ToString());

      // Add a control dependency on 'inputs_ready' node, to guarantee that all
      // inputs are alive and all side-effects executed before function body.
      if (inputs_ready_node) {
        func_body_node.add_input(
            AsControlDependency(inlined_node_name(inputs_ready_node->name())));
      }
    } else {
      // Update inputs of the regular function body nodes.
      for (string& input : *func_body_node.mutable_input()) {
        input = inlined_node_name(input);
      }

      // Check if we need to ensure node execution in correct loop frame.
      bool node_needs_empty_inputs_hook =
          // We have a node to hook and node has no inputs.
          !empty_inputs_hook.empty() && func_body_node.input_size() == 0 &&
          // Inputs ready node will always have edge from main graph. If
          // function call has no regular and control inputs, we will not add
          // inputs_ready node to the function body graph.
          node_name != kInputsReadyNodeName &&
          // The node acting as a return barrier for execution of side effects
          // might not have any inputs (in case function has no control outputs,
          // but we still added it because of non-empty happens-after set), so
          // we must make sure it's executed in correct frame.
          (node_name != kSideEffectsExecutedNodeName ||
           item.control_output_size() == 0);

      if (node_needs_empty_inputs_hook) {
        *func_body_node.add_input() =
            AsControlDependency(inlined_node_name(empty_inputs_hook[0]));
      }
    }

    // Add the function node name as a prefix 1) to node name to avoid
    // collisions; 2) to frame name to avoid multiple LoopCond nodes in one
    // frame after inlining.
    TF_RETURN_IF_ERROR(
        AddPrefixAndSuffixToNode(prefix, /*suffix=*/"", &func_body_node));

    // After inlining into the optimized graph, NodeDef must have all attributes
    // defined, which is not required for a node in a FunctionDef.
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(
        ctx->function_library().LookUpOpDef(func_body_node.op(), &op_def));
    AddDefaultsToNodeDef(*op_def, &func_body_node);
  }

  // ------------------------------------------------------------------------ //
  // Check that after inlining all side-effects will be executed in well defined
  // order. We do it by checking if there is a path from stateful/dataset ops to
  // one of the control output nodes.

  // Names of the inlined control output nodes.
  absl::flat_hash_set<string> inlined_control_output_nodes;
  for (const ControlOutput& control_output : item.control_outputs()) {
    inlined_control_output_nodes.insert(
        inlined_node_name(control_output.node_name));
  }

  // Construct a graph topology view for DFS traversals (skip invalid edges for
  // input nodes connected to nodes in the optimized graph).
  GraphTopologyView placed_topo_view(/*skip_invalid_edges=*/true);
  TF_RETURN_IF_ERROR(placed_topo_view.InitializeFromGraph(placed_graph_def));
  TF_RETURN_IF_ERROR(CheckThatSideEffectsWillExecute(
      *ctx, placed_topo_view, inlined_control_output_nodes));

  // ------------------------------------------------------------------------ //
  // Move all the nodes to the optimized graph after successful preprocessing.

  if (inputs_ready_node != nullptr) {
    string inlined_node = inlined_node_name(inputs_ready_node->name());
    absl::optional<int> node_idx = placed_topo_view.GetNodeIndex(inlined_node);

    absl::flat_hash_set<string> input_nodes;
    for (const string& input : func_node.input()) {
      SafeTensorId tensor = ParseTensorName(input);

      // Input node might have been a function call that was already inlined.
      auto it = ctx->tensor_mapping().find(tensor);
      while (it != ctx->tensor_mapping().end()) {
        tensor = it->second;
        it = ctx->tensor_mapping().find(tensor);
      }

      if (input_nodes.insert(tensor.node()).second) {
        placed_graph_def.mutable_node(*node_idx)->add_input(
            AsControlDependency(tensor.node()));
      }
    }
  }

  if (side_effects_executed_node != nullptr) {
    string inlined_node = inlined_node_name(side_effects_executed_node->name());
    absl::optional<int> node_idx = placed_topo_view.GetNodeIndex(inlined_node);

    // Add control edges from all control output nodes.
    for (const string& node_name : inlined_control_output_nodes) {
      placed_graph_def.mutable_node(*node_idx)->add_input(
          AsControlDependency(node_name));
    }

    // Forward all control dependencies in the optimized graph to the new node.
    ctx->AddControlOverrides(func_node, {inlined_node});
  }

  for (NodeDef& func_body_node : *placed_graph_def.mutable_node()) {
    // We bypass _Retval nodes and fetch tensors from `retval.input(0)`.
    if (IsRetval(func_body_node)) continue;
    optimized_graph->add_node()->Swap(&func_body_node);
  }

  // Indirect function call is fully inlined into the optimized graph, and we do
  // not copy the original function call node, so we have to setup tensor
  // mapping from old output tensors, to the outputs of inlined nodes.
  int output_idx = 0;
  for (const OutputArgInstantiation& output : item.outputs()) {
    const string& output_tensor = output_tensors.at(output.node_name);

    const SafeTensorId from_tensor(func_node.name(), output_idx++);
    const SafeTensorId to_tensor = ParseTensorName(output_tensor);

    const SafeTensorId inlined_to_tensor =
        SafeTensorId(absl::StrCat(func_node.name(), "/", to_tensor.node()),
                     to_tensor.index());

    ctx->AddTensorMapping(from_tensor, inlined_to_tensor);
  }

  // If function call node was in keep_ops set, it means that we need to keep a
  // node with the same name in the optimized graph. We forward all data
  // consumers to inlined nodes, and we verify that the node is not in a fetch
  // set, so it's safe to assume that the function call node is only required
  // for a control edge source.
  if (ctx->IsKeepOp(func_node.name())) {
    VLOG(4) << "Add NoOp for inlined function in keep ops set.";
    NodeDef* keep_func_node = optimized_graph->add_node();
    keep_func_node->set_op("NoOp");
    keep_func_node->set_name(func_node.name());
    keep_func_node->set_device(func_node.device());
    keep_func_node->add_input(
        AsControlDependency(inlined_node_name(kSideEffectsExecutedNodeName)));
  }

  VLOG(3) << "Successfully inlined indirect function call: "
          << SummarizeNodeDef(func_node);

  return Status::OK();
}

// Restores graph invariants after function specialization and inlining: all
// inputs must be connected to valid nodes.
Status RestoreGraphInvariants(const FunctionOptimizerContext& ctx,
                              GraphDef* optimized_graph) {
  // After function specialization and inlining graph might be in invalid
  // state, and some nodes can read tensors that do not exists anymore in the
  // optimized graph: function call node was fully inlined into the graph, or
  // output index was invalidated by the output pruning.

  if (!ctx.tensor_mapping().empty()) {
    for (NodeDef& node : *optimized_graph->mutable_node()) {
      for (int idx = 0; idx < node.input_size(); ++idx) {
        TensorId input_tensor = ParseTensorName(node.input(idx));
        if (input_tensor.index() == Graph::kControlSlot) break;

        auto mapping = ctx.tensor_mapping().find(input_tensor);
        if (mapping != ctx.tensor_mapping().end()) {
          node.set_input(idx, mapping->second.ToString());
        }
      }
    }
  }

  // Function inlining instantiates function body directly into the optimized
  // graph, and we might end up with control dependencies to the nodes that no
  // longer exist in a graph. We need to apply control overrides to all
  // invalidated nodes, and rewire control dependencies to the control outputs
  // node (it's also possible to rewrite singe control edge into multiple edges
  // to inlined side-effectful nodes).

  if (!ctx.control_overrides().empty()) {
    for (NodeDef& node : *optimized_graph->mutable_node()) {
      // Keep track of new control inputs to the node.
      absl::flat_hash_set<string> add_ctrl_inputs;

      // Remove all invalidated control inputs.
      for (int idx = 0; idx < node.input_size(); /* see below */) {
        // TODO(ezhulenev): Use non-allocating TensorId after migrating
        // `control_overrides()` to absl::flat_hash_set.
        SafeTensorId input_tensor = ParseTensorName(node.input(idx));

        auto overrides = ctx.control_overrides().find(input_tensor.node());
        if (overrides != ctx.control_overrides().end()) {
          // If this happens it's a bug in the function inlining.
          if (input_tensor.index() != Graph::kControlSlot) {
            return errors::Internal(
                "Illegal input edge from inlined function call node");
          }
          // Remove control dependency to the inlined function call node.
          node.mutable_input()->SwapElements(idx, node.input_size() - 1);
          node.mutable_input()->RemoveLast();

          // Keep track of all overrides.
          for (const string& override : overrides->second) {
            add_ctrl_inputs.insert(AsControlDependency(override));
          }
        } else {
          // Go to the next input only if the current one was not invalidated,
          // otherwise we need to check the swapped input as well.
          ++idx;
        }
      }

      // Add overrides to the node inputs.
      for (const string& ctrl_input : add_ctrl_inputs) {
        node.add_input(ctrl_input);
      }
    }
  }

  return Status::OK();
}

}  // namespace

Status FunctionOptimizer::RunFunctionOptimizerPass(
    const GrapplerItem& item, const GraphDef& graph, const int iteration,
    std::unordered_set<string>* skip_nodes, GraphDef* optimized_graph,
    bool* graph_has_unoptimized_function_calls) const {
  VLOG(3) << absl::Substitute(
      "Run function optimizer pass (iteration = $0): grappler_item_id = $1",
      iteration, item.id);

  FunctionOptimizerContext ctx(item, opt_level_, graph);

  bool inline_gradients = options_.enable_symbolic_gradient_inlining;
  bool inline_func = options_.enable_function_inlining;
  bool specialize_func = options_.enable_function_specialization;

  // We will process all the nodes in topological order, to correctly handle
  // inlining of function call chains.
  std::vector<const NodeDef*> topo_ordered_nodes;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(graph, &topo_ordered_nodes));

  for (const NodeDef* node : topo_ordered_nodes) {
    // Each node optimization can modify optimized graph only by adding new
    // nodes, we can check node size to make sure that graph was not modified.
    const int num_nodes_before = optimized_graph->node_size();
    const auto is_graph_modified = [&]() {
      int num_nodes = optimized_graph->node_size();
      DCHECK_GE(num_nodes, num_nodes_before) << "Nodes should not be removed";
      return num_nodes > num_nodes_before;
    };

    // Copy node from the `graph` to the `optimized_graph`.
    const auto copy_node = [&]() { *optimized_graph->add_node() = *node; };

    // If we already failed to optimize this node during one of the previous
    // passes, we just give up, and do not try on more time.
    if (skip_nodes->find(node->name()) != skip_nodes->end()) {
      VLOG(3) << "Skip optimization for node: " << node->name();
      copy_node();
      continue;
    }

// Skip errors if optimized graph was not modified before error happened.
#define TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(...)                     \
  do {                                                             \
    const Status _status = (__VA_ARGS__);                          \
    if (TF_PREDICT_FALSE(!_status.ok() && is_graph_modified()))    \
      return _status;                                              \
    if (TF_PREDICT_FALSE(!_status.ok() && !is_graph_modified())) { \
      VLOG(3) << "Skip error: " << _status.error_message();        \
      skip_nodes->insert(node->name());                            \
      copy_node();                                                 \
    }                                                              \
  } while (0)

    // ---------------------------------------------------------------------- //
    // 1. Inline symbolic gradients into the optimized graph.                 //
    // ---------------------------------------------------------------------- //

    if (IsSymbolicGradient(*node) && inline_gradients) {
      // Inline symbolic gradients only if the corresponding function is not
      // marked as `_noinline`.
      const auto* f_attr = gtl::FindOrNull(node->attr(), "f");
      const string f_name = f_attr != nullptr ? f_attr->func().name() : "";
      const FunctionDef* func = ctx.function_library().Find(f_name);
      if (func && !MarkedNoInline(*func)) {
        TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(
            InlineSymbolicGradient(*node, &ctx, optimized_graph));
        continue;
      } else {
        VLOG(2) << "Skip SymbolicGradient inlining: function=" << f_name;
        skip_nodes->insert(node->name());
      }
    }

    // ---------------------------------------------------------------------- //
    // 2. Inline or specialize function calls.                                //
    // ---------------------------------------------------------------------- //

    // Find if a node is a function call (direct or indirect).
    const FunctionDef* func = FindFunctionCall(ctx, *node);

    if (func != nullptr) {
      const string& func_name = func->signature().name();

      const bool is_direct_func = IsDirectFunctionCall(*func, *node);
      const bool is_indirect_func = IsIndirectFunctionCall(*func, *node);

      // 2a. Inline direct function call if it's inlinable.
      if (inline_func && is_direct_func) {
        Status inlinable = IsInlinableDirectFunctionCall(ctx, *func, *node);
        if (inlinable.ok()) {
          TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(
              InlineDirectFunctionCall(*node, *func, ctx, optimized_graph));
          continue;
        } else {
          VLOG(2) << inlinable.error_message();
          skip_nodes->insert(node->name());
        }
      }

      // 2b. Inline indirect function call if it's inlinable.
      if (inline_func && is_indirect_func) {
        Status inlinable = IsInlinableIndirectFunctionCall(ctx, *func, *node);
        if (inlinable.ok()) {
          TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(
              InlineIndirectFunctionCall(*node, *func, &ctx, optimized_graph));
          continue;
        } else {
          VLOG(2) << inlinable.error_message();
          skip_nodes->insert(node->name());
        }
      }

      // 2b. Specialize it to it's instantiation context if can't be inlined,
      // and it has something worth specializing.
      bool specialization_worthy = IsParametrized(*func) ||
                                   HasTrulyConstInputs(*node, ctx) ||
                                   HasUnusedOutputs(*node, *func, ctx);

      // Do not specialize if function has custom gradient.
      const string grad_func = ctx.function_library().FindGradient(func_name);

      if (specialize_func && grad_func.empty() && specialization_worthy) {
        // TODO(ezhulenev): Specialize function call if input has a known shape.
        // Specialize function body for its instantiation attributes and inputs.
        TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(
            SpecializeFunction(*node, *func, &ctx, optimized_graph));
        continue;
      } else {
        VLOG(2) << "Skip function specialization: " << func->signature().name();
        skip_nodes->insert(node->name());
      }
    }

    // If we reached this point, node was not handled by any of the stages
    // (inline, specialize), simply copy the node to the optimized graph.
    copy_node();

#undef TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED
  }

  TF_RETURN_IF_ERROR(RestoreGraphInvariants(ctx, optimized_graph));

  // Preserve the graph version.
  *optimized_graph->mutable_versions() = graph.versions();

  // Prune unreachable function from the library.
  if (options_.enable_trim_function_library) {
    *optimized_graph->mutable_library() =
        PruneFunctionLibrary(ctx.function_library(), *optimized_graph);
  } else {
    *optimized_graph->mutable_library() = ctx.function_library().ToProto();
  }

  // Before returning we check if after single optimization pass we have more
  // unoptimized function calls.
  *graph_has_unoptimized_function_calls = false;
  for (const NodeDef& node : optimized_graph->node()) {
    // Check if we can inline symbolic gradient.
    if (IsSymbolicGradient(node) && inline_gradients &&
        skip_nodes->count(node.name()) == 0) {
      *graph_has_unoptimized_function_calls = true;
      break;
    }

    // Check if after inlining we have unoptimized function calls.
    const FunctionDef* func = FindFunctionCall(ctx, node);
    if (func != nullptr && !MarkedSpecialized(*func) &&
        skip_nodes->count(node.name()) == 0) {
      *graph_has_unoptimized_function_calls = true;
      break;
    }
  }

  return Status::OK();
}

Status FunctionOptimizer::Optimize(Cluster*, const GrapplerItem& item,
                                   GraphDef* optimized_graph) {
  // Nothing to do here.
  if (item.graph.library().function_size() == 0) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  // Do not retry failed function inlining or specialization.
  std::unordered_set<string> skip_nodes;
  bool graph_has_unoptimized_function_calls = false;

  // We'll keep running function optimizer pass until we inlined and optimized
  // all function call nodes.
  int iteration = 0;
  constexpr int kMaxIterations = 50;

  // 1. Run first optimizer pass with GrapplerItem.graph.
  TF_RETURN_IF_ERROR(RunFunctionOptimizerPass(
      item, item.graph, 0, &skip_nodes, optimized_graph,
      &graph_has_unoptimized_function_calls));

  // 2. If after function inlining we have unoptimized function calls, we have
  // to run function optimization pass one more time.
  while (graph_has_unoptimized_function_calls) {
    if (iteration++ > kMaxIterations) {
      VLOG(1) << "Break function optimizer loop at iteration #" << iteration;
      break;
    }

    GraphDef workspace_graph;
    workspace_graph.Swap(optimized_graph);

    TF_RETURN_IF_ERROR(RunFunctionOptimizerPass(
        item, workspace_graph, iteration, &skip_nodes, optimized_graph,
        &graph_has_unoptimized_function_calls));
  }

  return Status::OK();
}

void FunctionOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                 const GraphDef& optimized_graph,
                                 double result) {
  // Nothing to do for FunctionOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
