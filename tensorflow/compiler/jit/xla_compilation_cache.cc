/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"

#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

constexpr int64 XlaCompilationCache::kDefaultCompilationThreshold;

XlaCompilationCache::XlaCompilationCache(xla::LocalClient* client,
                                         DeviceType device_type)
    : client_(client), device_type_(std::move(device_type)) {}

XlaCompilationCache::~XlaCompilationCache() {
  // Ensure any use of our programs have completed by waiting for all stream
  // executors to complete.
  for (auto* executor : client_->backend().stream_executors()) {
    bool ok = executor->SynchronizeAllActivity();
    if (!ok) {
      LOG(ERROR) << "Error synchronizing activity while waiting for all "
                    "programs to complete";
    }
  }
  // TODO(b/110813685): Think about the program ownership model. Programs are
  // currently owned by the compilation cache which means we must wait for
  // program completion in the destructor. There are multiple compilation caches
  // around, which complicates things a little. Perhaps having programs be
  // shared_ptrs (an invasive change) would make the model easier to reason
  // about?
}

string XlaCompilationCache::DebugString() const {
  return "XLA JIT compilation cache";
}

// Compute a string signature which encodes the shapes of the
// arguments in the supplied list.
string XlaCompilationCache::Signature::HumanString() const {
  string result = name;
  for (const auto& a : arg_shapes) {
    absl::StrAppend(&result, ",", DataTypeString(a.first));
    absl::StrAppend(&result, " [", absl::StrJoin(a.second, ","), "]");
  }

  for (const auto& v : sig.arg_values) {
    absl::StrAppend(&result, "; ", v.DebugString());
  }
  return result;
}

bool XlaCompilationCache::Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (arg_shapes != other.arg_shapes) return false;

  if (arg_values.size() != other.arg_values.size()) return false;
  for (int i = 0; i < arg_values.size(); ++i) {
    if (arg_values[i].tensor_data() != other.arg_values[i].tensor_data()) {
      return false;
    }
  }
  return true;
}

uint64 XlaCompilationCache::Signature::Hash::operator()(
    const XlaCompilationCache::Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.arg_shapes) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.size()));
    for (int dim : arg.second) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
  }
  for (const auto& arg : signature.arg_values) {
    h = Hash64Combine(
        h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
  }
  return h;
}

xla::StatusOr<XlaCompilationCache::Signature>
XlaCompilationCache::BuildSignature(
    const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args) {
  Signature signature;
  signature.name = Canonicalize(function.name(), AttrSlice(&function.attr()));
  for (const XlaCompiler::Argument& arg : args) {
    switch (arg.kind) {
      case XlaCompiler::Argument::kConstant:
        signature.arg_values.push_back(arg.constant_value);
        break;
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kResource:
        signature.arg_shapes.emplace_back(arg.type, arg.DimensionSizes());
        break;
      default:
        return errors::InvalidArgument(
            "Unhandled argument kind in XlaCompilationCache: ",
            arg.HumanString());
    }
  }

  return Status::OK();
}

}  // namespace

Status XlaCompilationCache::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    std::unique_ptr<xla::LocalExecutable>* executable) {
  VLOG(2) << "Compiling to local executable";

  std::vector<const xla::Shape*> argument_layouts(
      result.xla_input_shapes.size());
  for (int i = 0; i < result.xla_input_shapes.size(); ++i) {
    argument_layouts[i] = &result.xla_input_shapes[i];
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(options.device_ordinal != -1
                                       ? options.device_ordinal
                                       : client_->default_device_ordinal());
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_device_allocator(options.device_allocator);

  auto compile_result =
      client_->Compile(*result.computation, argument_layouts, build_options);
  if (!compile_result.ok()) {
    return compile_result.status();
  }
  *executable = std::move(compile_result.ValueOrDie());
  return Status::OK();
}

Status XlaCompilationCache::Compile(
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    const XlaCompiler::CompileOptions& compile_options,
    CompileMode compile_mode,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  // Set the compile threshold to 1 to implement CompileMode::kStrict.
  int64 compile_threshold =
      compile_mode == CompileMode::kLazy ? kDefaultCompilationThreshold : 1;
  return CompileImpl(options, function, constant_args, variable_args, ctx,
                     compile_options, /*compile_single_op=*/false,
                     /*compile_threshold=*/compile_threshold,
                     out_compilation_result, out_executable);
}

Status XlaCompilationCache::CompileSingleOp(
    const XlaCompiler::Options& options,
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  const NodeDef& def = ctx->op_kernel().def();
  NameAttrList name;
  name.set_name(def.op());
  *name.mutable_attr() = def.attr();
  // Remove the "_class" attribute from the attribute set used to create the
  // compilation cache key. This attribute is information for the colocator
  // and causes false uniqueness between nodes.
  name.mutable_attr()->erase("_class");
  auto compile_op = [&](XlaCompiler* compiler,
                        XlaCompiler::CompilationResult* result) {
    std::vector<DataType> result_dtypes(ctx->num_outputs());
    for (int i = 0; i < result_dtypes.size(); ++i) {
      result_dtypes[i] = ctx->expected_output_dtype(i);
    }
    return compiler->CompileSingleOp(compile_options, ctx->op_kernel().def(),
                                     args, result_dtypes, result);
  };
  return CompileImpl(options, name, args, compile_op,
                     /*compile_threshold=*/absl::nullopt,
                     out_compilation_result, out_executable);
}

Status XlaCompilationCache::CompileImpl(
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    const XlaCompiler::CompileOptions& compile_options, bool compile_single_op,
    int64 compile_threshold,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  DCHECK_NE(out_executable, nullptr);
  VLOG(2) << "XlaCompilationCache::Compile " << DebugString();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "num_inputs=" << ctx->num_inputs()
            << " num_constant_args=" << constant_args.size()
            << " num_variable_args=" << variable_args.size();
    for (int i = 0; i < ctx->num_inputs(); i++) {
      TensorShape shape = ctx->input(i).shape();
      VLOG(2) << i << ": dtype=" << DataTypeString(ctx->input_dtype(i))
              << " present=" << ctx->has_input(i)
              << " shape=" << shape.DebugString();
    }
    for (auto& iterator : variable_args) {
      const OptionalTensor& variable = iterator.second;
      VLOG(2) << "variable present=" << variable.present
              << " type=" << DataTypeString(variable.value.dtype())
              << " shape=" << variable.value.shape().DebugString()
              << " TF arg= " << iterator.first;
    }
    VLOG(2) << "num_outputs = " << ctx->num_outputs();
    for (int i = 0; i < ctx->num_outputs(); i++) {
      VLOG(2) << i << ": dtype=" << ctx->expected_output_dtype(i);
    }
  }

  TF_RET_CHECK(constant_args.size() + variable_args.size() <=
               ctx->num_inputs());

  Signature signature;
  TF_RETURN_IF_ERROR(
      BuildSignature(function, constant_args, variable_args, ctx, &signature));

  VLOG(2) << "Signature: " << SignatureDebugString(signature);
  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    // Find or create a cache entry.
    std::unique_ptr<Entry>& e = cache_[signature];
    if (!e) {
      e.reset(new Entry);
    }
    entry = e.get();
  }

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  mutex_lock entry_lock(entry->mu);
  int64 current_request_count = ++entry->request_count;
  if (!entry->compiled) {
    VLOG(2) << "Compilation cache miss for signature: "
            << SignatureDebugString(signature) << " with request count "
            << current_request_count << " and compile threshold "
            << compile_threshold;
    if (current_request_count < compile_threshold) {
      *out_compilation_result = nullptr;
      *out_executable = nullptr;
      return Status::OK();
    }

    tensorflow::Env* env = tensorflow::Env::Default();
    const uint64 compile_start_us = env->NowMicros();
    // Do the actual JIT compilation without holding the lock (it can take
    // a long time.)
    std::vector<XlaCompiler::Argument> args;
    TF_RETURN_IF_ERROR(
        BuildArguments(constant_args, variable_args, ctx, &args));

    XlaCompiler compiler(options);
    entry->compiled = true;

    if (compile_single_op) {
      entry->compilation_status =
          compiler.CompileSingleOp(compile_options, signature.name, ctx, args,
                                   &entry->compilation_result);
    } else {
      entry->compilation_status = compiler.CompileFunction(
          compile_options, function, args, &entry->compilation_result);
    }
    TF_RETURN_IF_ERROR(entry->compilation_status);
    CHECK_EQ(entry->executable.get(), nullptr);
    entry->compilation_status =
        BuildExecutable(options, entry->compilation_result, &entry->executable);

    const uint64 compile_end_us = env->NowMicros();
    const uint64 compile_time_us = compile_end_us - compile_start_us;
    {
      mutex_lock lock(compile_stats_mu_);
      auto it = compile_stats_.emplace(function.name(), CompileStats{}).first;
      it->second.compile_count++;
      it->second.cumulative_compile_time_us += compile_time_us;
      VLOG(1) << "compiled " << function.name() << " "
              << it->second.compile_count
              << " times, compile time: " << compile_time_us
              << " us, cumulative: " << it->second.cumulative_compile_time_us
              << " us ("
              << tensorflow::strings::HumanReadableElapsedTime(compile_time_us /
                                                               1.0e6)
              << " / "
              << tensorflow::strings::HumanReadableElapsedTime(
                     it->second.cumulative_compile_time_us / 1.0e6)
              << ")";
    }
  }
  TF_RETURN_IF_ERROR(entry->compilation_status);
  *out_compilation_result = &entry->compilation_result;
  *out_executable = entry->executable.get();
  return Status::OK();
}

}  // namespace tensorflow
