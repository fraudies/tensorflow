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
#include "tensorflow/contrib/avro/utils/avro_parser.h"

namespace tensorflow {
namespace data {

AvroParser::AvroParser() : parser_tree(nullptr) { }
AvroParser::~AvroParser() { }

Status AvroParser::Create(AvroParser* parser, const vector<pair<string, DataType>& keys_and_types) {
  ParserTree* parser_tree = new ParserTree();
  TF_RETURN_IF_ERROR(ParserTree::Build(parser_tree, keys_and_types));
  parser_tree_.reset(parser_tree);
  return Status::OK():
}

Status AvroParser::ParseAll(vector<Tensor>* tensors, const avro_value_t& in_value) {
  // May add multiple elements and dimensions to the tensors or only a single value
  // Value buffer keeps track of the required size and supports filling in to transform into a tensor
  // Will check the expected type against the actual type
  // Return tensors in the same order the keys have been provided
  return (*parser_tree).ParseValue(tensors, in_value);
}

}
}