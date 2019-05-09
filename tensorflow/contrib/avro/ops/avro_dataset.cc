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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

#include "tensorflow/contrib/avro/utils/avro_reader.h"
#include "tensorflow/contrib/avro/utils/avro_attr_parser.h"

// As boiler plate I used
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/dataset.h  DatasetBase
//
// Example build with headers
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/sql/BUILD
//
// Parse example dataset op implementation
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/parse_example_dataset_op.cc
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_fast_parsing.cc
//
// dataset
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/dataset_ops.cc

// CSV parser
// Op definition: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/experimental_dataset_ops.cc
// Op implementation: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/experimental/csv_dataset_op.cc

// Example parser
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/parsing_ops.cc
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/experimental/parse_example_dataset_op.cc
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_helper.h

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/parsing_ops.py
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/data/ops/dataset_ops.py

// Attr vs Input
// https://www.tensorflow.org/guide/extend/op#attrs

namespace tensorflow {
namespace data {

using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("AvroDataset")
    .Input("filenames: string")
    .Input("reader_schema: string")
    .Input("sparse_keys: Nsparse * string")
    .Input("dense_keys: Ndense * string")
    .Input("dense_defaults: Tdense")
    .Output("handle: variant")
    .Attr("Nsparse: int >= 0")  // Inferred from sparse_keys
    .Attr("Ndense: int >= 0")   // Inferred from dense_keys
    .Attr("sparse_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("Tdense: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")  // Output components will be
                                              // sorted by key (dense_keys and
                                              // sparse_keys combined) here.
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ParseAvroAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      // Output sparse_indices, sparse_values, and sparse_shapes.
      int output_idx = 0;
      for (int i = 0; i < attrs.num_sparse; ++i) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 2));
      }
      for (int i = 0; i < attrs.num_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i = 0; i < attrs.num_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(2));
      }

      // Output dense_shapes.
      for (int i = 0; i < attrs.num_dense; ++i) {
        ShapeHandle dense;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(attrs.dense_shapes[i], &dense));
        c->set_output(output_idx++, dense);
      }

      return Status::OK();
    });


class AvroDatasetOp : public DatasetOpKernel {
 public:
  explicit AvroDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx),
      graph_def_version_(ctx->graph_def_version()) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_types", &sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tdense", &dense_types_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_shapes", &dense_shapes_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {

    // Get filenames
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));
    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    // Get schema
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "reader_schema",
      &reader_schema_));

    // Get keys
    OpInputList sparse_keys;
    OpInputList dense_keys;
    OP_REQUIRES_OK(ctx, ctx->input_list("sparse_keys", &sparse_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_keys", &dense_keys));
    //    CHECK_EQ(dense_keys.size(), attrs_.num_dense);
    //    CHECK_EQ(sparse_keys.size(), attrs_.num_sparse);
    sparse_keys_.resize(sparse_keys.size());
    for (size_t i_sparse = 0; i_sparse < sparse_keys.size(); ++i_sparse) {
      sparse_keys_[i_sparse] = sparse_keys[i_sparse].scalar<string>()();
    }
    dense_keys_.resize(dense_keys.size());
    for (size_t i_dense = 0; i_dense < dense_keys.size(); ++i_dense) {
      dense_keys_[i_dense] = dense_keys[i_dense].scalar<string>()();
    }

    // Get dense default tensors
    OpInputList dense_default_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("dense_defaults", &dense_default_tensors));

    OP_REQUIRES(ctx, dense_default_tensors.size() == dense_keys_.size(),
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_default_tensors.size(), " vs. ", dense_keys_.size()));

    std::vector<Tensor> dense_defaults(dense_default_tensors.begin(),
                                       dense_default_tensors.end());

    for (int d = 0; d < dense_keys_.size(); ++d) {
      const Tensor& def_value = dense_defaults[d];
      OP_REQUIRES(ctx, def_value.dtype() == dense_types_[d],
                  errors::InvalidArgument(
                      "dense_defaults[", d, "].dtype() == ",
                      DataTypeString(def_value.dtype()), " != dense_types_[", d,
                      "] == ", DataTypeString(dense_types_[d])));
    }

    AvroParseConfig config;
    std::map<string, int> key_to_output_index;
    for (int d = 0; d < dense_keys_.size(); ++d) {
      config.dense.push_back({dense_keys_[d], dense_types_[d], dense_shapes_[d],
                              dense_default_tensors[d]});
      auto result = key_to_output_index.insert({dense_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          dense_keys_[d]));
    }

    for (int d = 0; d < sparse_keys_.size(); ++d) {
      config.sparse.push_back({sparse_keys_[d], sparse_types_[d]});
      auto result = key_to_output_index.insert({sparse_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          sparse_keys_[d]));
    }
    int i = 0;
    for (auto it = key_to_output_index.begin(); it != key_to_output_index.end();
         it++) {
      it->second = i++;
    }

    *output = new Dataset(ctx, std::move(filenames), reader_schema_,
                    dense_defaults, sparse_keys_, dense_keys_,
                    std::move(key_to_output_index), std::move(config),
                    sparse_types_, dense_types_,
                    dense_shapes_, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx,
            std::vector<string> filenames,
            string reader_schema,
            std::vector<Tensor> dense_defaults,
            std::vector<string> sparse_keys,
            std::vector<string> dense_keys,
            std::map<string, int> key_to_output_index,
            AvroParseConfig config,
            const DataTypeVector& sparse_types,
            const DataTypeVector& dense_types,
            const std::vector<PartialTensorShape>& dense_shapes,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(std::move(filenames)),
          reader_schema_(std::move(reader_schema)),
          dense_defaults_(std::move(dense_defaults)),
          sparse_keys_(std::move(sparse_keys)),
          dense_keys_(std::move(dense_keys)),
          key_to_output_index_(std::move(key_to_output_index)),
          config_(std::move(config)),
          sparse_types_(sparse_types),
          dense_types_(dense_types),
          dense_shapes_(dense_shapes),
          output_types_(output_types),
          output_shapes_(output_shapes) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Avro")}));
    }

    const DataTypeVector& output_dtypes() const override { return output_types_; }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "AvroDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {

      Node* filenames = nullptr;
      Node* reader_schema = nullptr;

      std::vector<Node*> dense_defaults_nodes;
      dense_defaults_nodes.reserve(dense_defaults_.size());

      for (const Tensor& dense_default : dense_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(dense_default, &node));
        dense_defaults_nodes.emplace_back(node);
      }

      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddScalar(reader_schema_, &reader_schema));

      AttrValue sparse_keys_attr;
      AttrValue dense_keys_attr;
      AttrValue sparse_types_attr;
      AttrValue dense_attr;
      AttrValue dense_shapes_attr;

      b->BuildAttrValue(sparse_keys_, &sparse_keys_attr);
      b->BuildAttrValue(dense_keys_, &dense_keys_attr);
      b->BuildAttrValue(sparse_types_, &sparse_types_attr);
      b->BuildAttrValue(dense_types_, &dense_attr);
      b->BuildAttrValue(dense_shapes_, &dense_shapes_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(this,
                                       {
                                          {0, filenames},
                                          {1, reader_schema},
                                       }, // single tensor inputs
                                       {
                                          {0, dense_defaults_nodes}
                                       }, // list tensor inputs
                                       {
                                          {"sparse_keys", sparse_keys_attr},
                                          {"dense_keys", dense_keys_attr},
                                          {"sparse_types", sparse_types_attr},
                                          {"Tdense", dense_attr},
                                          {"dense_shapes", dense_shapes_attr}
                                       }, // non-tensor inputs, attributes
                                       output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {

        mutex_lock l(mu_);

        // Loops over all files
        do {
          // We are currently processing a file, so try to read the next record.
          if (reader_) {
            Status s = Read(ctx, out_tensors);
            if (s.ok()) {
              *end_of_sequence = false;
              return Status::OK();
            } else if (!errors::IsOutOfRange(s)) {
              return s;
            } else {
              CHECK(errors::IsOutOfRange(s));
              // We have reached the end of the current file, so maybe
              // move on to next file.
              reader_.reset();
              file_.reset();
              ++current_file_index_;
            }
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          // Actually move on to next file.
          // Looks like this cannot request multiple files in parallel. Hmm.
          const string& next_filename =
              dataset()->filenames_[current_file_index_];

          TF_RETURN_IF_ERROR(
            ctx->env()->NewRandomAccessFile(next_filename, &file_));

          uint64 file_size;
          TF_RETURN_IF_ERROR(
            ctx->env()->GetFileSize(next_filename, &file_size));

          reader_.reset(new AvroReader(
              std::move(file_), file_size, next_filename,
              dataset()->reader_schema_, dataset()->config_));
          TF_RETURN_IF_ERROR(reader_->OnWorkStartup());
        } while (true);
      }

     private:
      Status Read(IteratorContext* ctx, std::vector<Tensor>* out_tensors) {

        // TODO(fraudies): Perf; check if we can initialize and re-use the result struct
        AvroResult avro_result;
        TF_RETURN_IF_ERROR((*reader_).Read(&avro_result));

        // Check validity of the result and assign it to the out tensors
        (*out_tensors).resize(dataset()->key_to_output_index_.size());
        for (int d = 0; d < dataset()->dense_keys_.size(); ++d) {
          int output_index = dataset()->key_to_output_index_.at(dataset()->dense_keys_[d]);
          CHECK(avro_result.dense_values[d].dtype() ==
                dataset()->output_dtypes()[output_index])
              << "Got wrong type for AvroDataset return value " << d
              << " (expected "
              << DataTypeString(dataset()->output_dtypes()[output_index]) << ", got "
              << DataTypeString(avro_result.dense_values[d].dtype())
              << ").";
          CHECK(dataset()->output_shapes()[output_index].IsCompatibleWith(
              avro_result.dense_values[d].shape()))
              << "Got wrong shape for AvroDataset return value " << d
              << " (expected "
              << dataset()->output_shapes()[output_index].DebugString() << ", got "
              << avro_result.dense_values[d].shape().DebugString()
              << ").";
          (*out_tensors)[output_index] = avro_result.dense_values[d];
        }
        for (int d = 0; d < dataset()->sparse_keys_.size(); ++d) {
          int output_index = dataset()->key_to_output_index_.at(dataset()->sparse_keys_[d]);
          (*out_tensors)[output_index] =
              Tensor(ctx->allocator({}), DT_VARIANT, {3});
          Tensor& serialized_sparse = (*out_tensors)[output_index];
          auto serialized_sparse_t = serialized_sparse.vec<Variant>();
          serialized_sparse_t(0) = avro_result.sparse_indices[d];
          serialized_sparse_t(1) = avro_result.sparse_values[d];
          serialized_sparse_t(2) = avro_result.sparse_shapes[d];
          CHECK(serialized_sparse.dtype() == dataset()->output_dtypes()[output_index])
              << "Got wrong type for AvroDataset return value " << d
              << " (expected "
              << DataTypeString(dataset()->output_dtypes()[output_index]) << ", got "
              << DataTypeString(serialized_sparse.dtype()) << ").";
          CHECK(output_shapes()[output_index].IsCompatibleWith(
              serialized_sparse.shape()))
              << "Got wrong shape for AvroDataset return value " << d
              << " (expected "
              << dataset()->output_shapes()[output_index].DebugString() << ", got "
              << serialized_sparse.shape().DebugString() << ").";
        }

        return Status::OK();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;

      // `reader_` will borrow the object that `file_` points to, so
      // we must destroy `reader_` before `file_`.
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
      std::unique_ptr<AvroReader> reader_ GUARDED_BY(mu_);
    };                      // class Iterator

    const std::vector<string> filenames_;
    const string reader_schema_;
    const std::vector<Tensor> dense_defaults_;
    const std::vector<string> sparse_keys_;
    const std::vector<string> dense_keys_;
    const std::map<string, int> key_to_output_index_;
    const AvroParseConfig config_;
    const DataTypeVector sparse_types_;
    const DataTypeVector dense_types_;
    const std::vector<PartialTensorShape> dense_shapes_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };  // class Dataset

  const int graph_def_version_;
  std::string reader_schema_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::vector<string> sparse_keys_;
  std::vector<string> dense_keys_;
  DataTypeVector sparse_types_;
  DataTypeVector dense_types_;
  std::vector<PartialTensorShape> dense_shapes_;
};  // class AvroDataset

// Register the kernel implementation for AvroDataset.
REGISTER_KERNEL_BUILDER(Name("AvroDataset").Device(DEVICE_CPU),
                        AvroDatasetOp);

}  // namespace data
}  // namespace tensorflow
