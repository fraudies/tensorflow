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
#include "tensorflow/contrib/avro/utils/avro_attr_parser.h"

namespace tensorflow {
namespace data {

// Checks for the type to be float, double, int64, int32, string, or bool
// These are compatible with the avro types
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

Status ParseAvroAttrs::FinishInit() {
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
  return Status::OK();
}

} // namespace data
} // namespace tensorflow