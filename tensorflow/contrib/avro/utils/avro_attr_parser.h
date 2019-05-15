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
#ifndef TENSORFLOW_DATA_AVRO_ATTR_PARSER_H_
#define TENSORFLOW_DATA_AVRO_ATTR_PARSER_H_

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_helper.cc
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/example_proto_helper.h

#include <string>
#include <vector>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {
namespace data {

// Parses the attributes passed to AvroDataset.
// REQUIRES: Init must be called after construction.
class ParseAvroAttrs {
 public:
  template <typename ContextType>
  Status Init(ContextType* ctx) {
    TF_RETURN_IF_ERROR(ctx->GetAttr("sparse_types", &sparse_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tdense", &dense_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("dense_shapes", &dense_shapes));

    num_sparse = sparse_types.size();
    num_dense = dense_types.size();

    string schema;
    TF_RETURN_IF_ERROR(ctx->GetAttr("reader_schema", &schema));
    if (schema.size()) {
      VLOG(4) << "Avro parser for reader schema\n" << schema;
    } else {
      VLOG(4) << "Avro parser with default schema";
    }

    return FinishInit();
  }

  int64 num_sparse;
  int64 num_dense;
  std::vector<DataType> sparse_types;
  std::vector<DataType> dense_types;
  std::vector<PartialTensorShape> dense_shapes;

 private:
  Status FinishInit();
};

} // namespace data
} // namespace tensorflow

#endif // AVRO_ATTR_PARSER