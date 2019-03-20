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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/contrib/avro/utils/avro_parser_tree.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that. Instead these tests only cover the wrapping in the mem readers
namespace tensorflow {
namespace data {

TEST(RegexTest, SplitExpressions) {
  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair("name.first", DT_STRING),
    std::make_pair("friends[2].name.first", DT_STRING),
    std::make_pair("friends[*].name.first", DT_STRING),
    std::make_pair("friends[*].address[*].street", DT_STRING),
    std::make_pair("friends[*].job[*].coworker[*].name.first", DT_STRING),
    std::make_pair("car['nickname'].color", DT_STRING),
    std::make_pair("friends[gender='unknown'].name.first", DT_STRING),
    std::make_pair("friends[name.first=name.last].name.initial", DT_STRING),
    std::make_pair("friends[name.first=@name.first].name.initial", DT_STRING)};
  AvroParserTree parser_tree;
  AvroParserTree::Build(&parser_tree, keys_and_types);
}

}
}