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

#include "tensorflow/contrib/avro/utils/avro_reader.h"

namespace tensorflow {
namespace data {

Status AvroReader::OnWorkStartup() {

  // Allocate memory for the file part
  data_.reset(new (std::nothrow) char[file_size_]);
  if (data_.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to allocate ", file_size_/1024,
                                          " kB on memory in avro reader."));
  }

  // Read the file into the memory file
  StringPiece result;
  TF_RETURN_IF_ERROR((*file_).Read(0, file_size_, &result, data_.get()));

  // Create the memory reader
  TF_RETURN_IF_ERROR(AvroMemReader::Create(&avro_mem_reader_, data_, file_size_, filename_));

  // Create the parser tree
  TF_RETURN_IF_ERROR(avro_parser_tree_.Build(&avro_parser_tree_, CreateKeysAndTypesFromConfig()));

  return Status::OK();
}


Status AvroReader::Read(AvroResult* result) {

  TF_RETURN_IF_ERROR(avro_mem_reader_.ReadNext(avro_value_));

  TF_RETURN_IF_ERROR(avro_parser_tree_.ParseValue(&key_to_value_, std::move(avro_value_)));

  // Get sparse tensors
  size_t n_sparse = config_.sparse.size();
  (*result).sparse_indices.resize(n_sparse);
  (*result).sparse_values.resize(n_sparse);
  (*result).sparse_shapes.resize(n_sparse);

  for (size_t i_sparse = 0; i_sparse < n_sparse; ++i_sparse) {
    const AvroParseConfig::Sparse& sparse = config_.sparse[i_sparse];
    const ValueStoreUniquePtr& value_store = key_to_value_[sparse.feature_name];

    TensorShape value_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseValueShape(&value_shape));
    (*result).sparse_values[i_sparse] = Tensor(allocator_, sparse.dtype, value_shape);

    TensorShape index_shape;
    TF_RETURN_IF_ERROR((*value_store).GetSparseIndexShape(&index_shape));
    (*result).sparse_indices[i_sparse] = Tensor(allocator_, DT_INT64, index_shape);

    TF_RETURN_IF_ERROR((*value_store).MakeSparse(
      &(*result).sparse_values[i_sparse],
      &(*result).sparse_indices[i_sparse]));

    TensorShape size_shape;
    size_shape.AddDim(value_shape.dims());
    (*result).sparse_shapes[i_sparse] = Tensor(allocator_, DT_INT64, size_shape);
    TF_RETURN_IF_ERROR(ShapeToTensor(&(*result).sparse_shapes[i_sparse], value_shape));
  }

  // Get dense tensors
  size_t n_dense = config_.dense.size();
  (*result).dense_values.resize(n_dense);

  for (size_t i_dense = 0; i_dense < n_dense; ++i_dense) {
    const AvroParseConfig::Dense& dense = config_.dense[i_dense];
    const ValueStoreUniquePtr& value_store = key_to_value_[dense.feature_name];

    TensorShape resolved_shape;
    TF_RETURN_IF_ERROR((*value_store).ResolveDenseShape(&resolved_shape, dense.shape,
      dense.default_value.shape()));

    (*result).dense_values[i_dense] = Tensor(allocator_, dense.dtype, resolved_shape);

    TF_RETURN_IF_ERROR((*value_store).MakeDense(&(*result).dense_values[i_dense],
      resolved_shape, dense.default_value));
  }

  return Status::OK();
}

// Assumes tensor has been allocated appropriate space -- not checked
Status AvroReader::ShapeToTensor(Tensor* tensor, const TensorShape& shape) {
  auto tensor_flat = (*tensor).flat<int64>();
  size_t n_dim = shape.dims();
  for (size_t i_dim = 0; i_dim < n_dim; ++i_dim) {
    tensor_flat(i_dim) = shape.dim_size(i_dim);
  }
  return Status::OK();
}

std::vector<std::pair<string, DataType>> AvroReader::CreateKeysAndTypesFromConfig() {

  std::vector<std::pair<string, DataType>> keys_and_types;
  for (const AvroParseConfig::Sparse& sparse : config_.sparse) {
    keys_and_types.push_back({sparse.feature_name, sparse.dtype});
  }
  for (const AvroParseConfig::Dense& dense : config_.dense) {
    keys_and_types.push_back({dense.feature_name, dense.dtype});
  }

  return keys_and_types;
}


}  // namespace data
}  // namespace tensorflow