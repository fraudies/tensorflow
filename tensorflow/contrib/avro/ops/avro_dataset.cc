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

// Batching
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/dataset_ops.cc   BatchDataset
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/batch_dataset_op.cc
// https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/data/ops/dataset_ops.py BatchDataset

namespace tensorflow {
namespace data {

using ::tensorflow::shape_inference::ShapeHandle;

// TODO(fraudies): Find a better place for this method
Status CheckValidType(const DataType& dtype) {
  switch (dtype) {
    case DT_INT32:
    case DT_INT64:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_STRING:
    case DT_BOOL:
      return Status::OK();
    default:
      return errors::InvalidArgument("Received invalid input dtype: ",
                                     DataTypeString(dtype),
                                     ". Valid types are float, double, ",
                                     "int64, int32, string, bool.");
  }
}

REGISTER_OP("AvroDataset")
    .Input("filenames: string")
    .Input("batch_size: int64")
    .Input("dense_defaults: Tdense")
    .Output("handle: variant")
    .Attr("reader_schema: string")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("Tdense: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")  // Output components will be
                                              // sorted by key (dense_keys and
                                              // sparse_keys combined) here.
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_dense;
      std::vector<DataType> sparse_types;
      std::vector<DataType> dense_types;
      std::vector<PartialTensorShape> dense_shapes;

      TF_RETURN_IF_ERROR(c->GetAttr("sparse_types", &sparse_types));
      TF_RETURN_IF_ERROR(c->GetAttr("Tdense", &dense_types));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_shapes", &dense_shapes));

      num_dense = dense_types.size();

      // Add input checking
      if (static_cast<size_t>(num_dense) != dense_shapes.size()) {
        return errors::InvalidArgument("len(dense_keys) != len(dense_shapes)");
      }
      if (num_dense > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("num_dense_ too large");
      }
      for (const DataType& type : dense_types) {
        TF_RETURN_IF_ERROR(CheckValidType(type));
      }
      for (const DataType& type : sparse_types) {
        TF_RETURN_IF_ERROR(CheckValidType(type));
      }

      // Log schema if the user provided one at op kernel construction
      string schema;
      TF_RETURN_IF_ERROR(c->GetAttr("reader_schema", &schema));
      if (schema.size()) {
        VLOG(4) << "Avro parser for reader schema\n" << schema;
      } else {
        VLOG(4) << "Avro parser with default schema";
      }

      return shape_inference::ScalarShape(c);
    });


class AvroDatasetOp : public DatasetOpKernel {
 public:
  explicit AvroDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx),
      graph_def_version_(ctx->graph_def_version()) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("reader_schema", &reader_schema_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_keys", &sparse_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_keys", &dense_keys_));

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

    int64 batch_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size",
                                            &batch_size));
    OP_REQUIRES(ctx, batch_size > 0,
                errors::InvalidArgument(
                    "batch_size must be greater than zero."));

    // Get keys
    OpInputList sparse_keys;
    OpInputList dense_keys;

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

    std::map<string, int> key_to_output_index;
    for (int d = 0; d < dense_keys_.size(); ++d) {
      auto result = key_to_output_index.insert({dense_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          dense_keys_[d]));
    }

    for (int d = 0; d < sparse_keys_.size(); ++d) {
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

    *output = new Dataset(ctx, std::move(filenames), batch_size, reader_schema_,
                    dense_defaults, sparse_keys_, dense_keys_,
                    std::move(key_to_output_index),
                    sparse_types_, dense_types_,
                    dense_shapes_, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    // Need to keep batched separate from original values because that state can't be
    // transferred to outside of this class (using output_shapes or node graph def)
    // We will pass that batched information to the parser and to the output shape check
    Dataset(OpKernelContext* ctx,
            std::vector<string> filenames,
            int64 batch_size,
            string reader_schema,
            std::vector<Tensor> dense_defaults,
            std::vector<string> sparse_keys,
            std::vector<string> dense_keys,
            std::map<string, int> key_to_output_index,
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
          sparse_types_(sparse_types),
          dense_types_(dense_types),
          dense_shapes_(dense_shapes),
          dense_shapes_batched_(PrependDimension(dense_shapes, batch_size)),
          output_types_(output_types),
          output_shapes_(output_shapes),
          output_shapes_batched_(SelectAndPrependDimension(output_shapes, batch_size, 0, dense_keys_.size())),
          config_(BuildConfig(batch_size, dense_keys_, dense_types_, dense_shapes_batched_, dense_defaults_,
                              sparse_keys_, sparse_types_)) {

      LOG(INFO) << "Config has " << config_.dense.size() << " dense keys";

    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Avro")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      LOG(INFO) << "Called output shapes";

      return output_shapes_;
    }

    const std::vector<PartialTensorShape>& output_shapes_batched() const {
      return output_shapes_batched_;
    }

    string DebugString() const override { return "AvroDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {

      Node* filenames = nullptr;

      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));

      Node* batch_size = nullptr;

      TF_RETURN_IF_ERROR(b->AddScalar(config_.batch_size, &batch_size));

      std::vector<Node*> dense_defaults_nodes;
      dense_defaults_nodes.reserve(dense_defaults_.size());

      for (const Tensor& dense_default : dense_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(dense_default, &node));
        dense_defaults_nodes.emplace_back(node);
      }

      AttrValue reader_schema_attr;
      AttrValue sparse_keys_attr;
      AttrValue dense_keys_attr;
      AttrValue sparse_types_attr;
      AttrValue dense_attr;
      AttrValue dense_shapes_attr;

      b->BuildAttrValue(reader_schema_, &reader_schema_attr);
      b->BuildAttrValue(sparse_keys_, &sparse_keys_attr);
      b->BuildAttrValue(dense_keys_, &dense_keys_attr);
      b->BuildAttrValue(sparse_types_, &sparse_types_attr);
      b->BuildAttrValue(dense_types_, &dense_attr);
      b->BuildAttrValue(dense_shapes_, &dense_shapes_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(this,
                                       {
                                          {0, filenames},
                                          {1, batch_size}
                                       }, // single tensor inputs
                                       {
                                          {2, dense_defaults_nodes}
                                       }, // list tensor inputs
                                       {
                                          {"reader_schema", reader_schema_attr},
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
          CHECK(dataset()->output_shapes_batched_[output_index].IsCompatibleWith(
              avro_result.dense_values[d].shape()))
              << "Got wrong shape for AvroDataset return value " << d
              << " (expected "
              << dataset()->output_shapes_batched_[output_index].DebugString() << ", got "
              << avro_result.dense_values[d].shape().DebugString()
              << ").";

          LOG(INFO) << "Output tensor for " << dataset()->dense_keys_[d] << " is " << avro_result.dense_values[d].SummarizeValue(3);
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
          CHECK(dataset()->output_shapes_batched_[output_index].IsCompatibleWith(
              serialized_sparse.shape()))
              << "Got wrong shape for AvroDataset return value " << d
              << " (expected "
              << dataset()->output_shapes_batched_[output_index].DebugString() << ", got "
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


    static std::vector<PartialTensorShape> PrependDimension(
      const std::vector<PartialTensorShape>& shapes, int64 dim) {

      std::vector<PartialTensorShape> new_shapes(shapes);
      for (int s = 0; s < shapes.size(); ++s) {
        new_shapes[s] = PartialTensorShape({dim}).Concatenate(new_shapes[s]);

        LOG(INFO) << "New shape " << new_shapes[s];
      }

      return new_shapes;
    }

    static std::vector<PartialTensorShape> SelectAndPrependDimension(
      const std::vector<PartialTensorShape>& shapes, int64 dim, int64 first, int64 num) {

      std::vector<PartialTensorShape> new_shapes(shapes);
      for (int s = first; s < num; ++s) {
        new_shapes[s] = PartialTensorShape({dim}).Concatenate(new_shapes[s]);
      }

      return new_shapes;
    }

    static AvroParseConfig BuildConfig(int64 batch_size,
      const std::vector<string>& dense_keys, const DataTypeVector& dense_types,
      const std::vector<PartialTensorShape>& dense_shapes,
      const std::vector<Tensor>& dense_defaults, const std::vector<string>& sparse_keys,
      const DataTypeVector& sparse_types) {

      AvroParseConfig config;
      // Create the config
      config.batch_size = batch_size;
      for (int d = 0; d < dense_keys.size(); ++d) {
        config.dense.push_back({dense_keys[d], dense_types[d], dense_shapes[d],
                                dense_defaults[d]});
      }
      for (int d = 0; d < sparse_keys.size(); ++d) {
        config.sparse.push_back({sparse_keys[d], sparse_types[d]});
      }

      return config;
    }

    static bool prepend_dimension_for_dense_shapes;

    const std::vector<string> filenames_;
    const string reader_schema_;
    const std::vector<Tensor> dense_defaults_;
    const std::vector<string> sparse_keys_;
    const std::vector<string> dense_keys_;
    const std::map<string, int> key_to_output_index_;
    const DataTypeVector sparse_types_;
    const DataTypeVector dense_types_;
    const std::vector<PartialTensorShape> dense_shapes_;
    const std::vector<PartialTensorShape> dense_shapes_batched_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::vector<PartialTensorShape> output_shapes_batched_;
    const AvroParseConfig config_;
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
};  // class AvroDatasetOp

// Register the kernel implementation for AvroDataset.
REGISTER_KERNEL_BUILDER(Name("AvroDataset").Device(DEVICE_CPU),
                        AvroDatasetOp);

}  // namespace data
}  // namespace tensorflow
