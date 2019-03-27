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
#include "tensorflow/core/lib/core/status_test_util.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that. Instead these tests only cover the wrapping in the mem readers
namespace tensorflow {
namespace data {

TEST(RegexTest, SplitExpressions) {
  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair("friends[2].name.first", DT_STRING),
    std::make_pair("friends[*].address[*].street", DT_STRING),
    std::make_pair("friends[*].job[*].coworker[*].name.first", DT_STRING),
    std::make_pair("car['nickname'].color", DT_STRING),
    std::make_pair("friends[gender='unknown'].name.first", DT_STRING),
    std::make_pair("friends[name.first=name.last].name.initial", DT_STRING),
    std::make_pair("friends[name.first=@name.first].name.initial", DT_STRING)};
  AvroParserTree parser_tree;
  TF_EXPECT_OK(AvroParserTree::Build(&parser_tree, keys_and_types));
  AvroParserSharedPtr root_parser = parser_tree.getRoot();
  NamespaceParser* namespace_parser = dynamic_cast<NamespaceParser*>(root_parser.get());
  EXPECT_TRUE(namespace_parser != nullptr);
  const std::vector<AvroParserSharedPtr>& children((*root_parser).GetChildren());
  EXPECT_EQ(children.size(), 3);
  const string actual((*root_parser).ToString());
  const string expected =
    "|---NamespaceParser(default)\n"
    "|   |---AttributeParser(name)\n"
    "|   |   |---StringValue(default.name.first)\n"
    "|   |---AttributeParser(friends)\n"
    "|   |   |---ArrayAllParser\n"
    "|   |   |   |---StringValue(default.friends.[*].gender)\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---StringValue(default.friends.[*].name.first)\n"
    "|   |   |   |   |---StringValue(default.friends.[*].name.last)\n"
    "|   |   |   |---AttributeParser(address)\n"
    "|   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |---StringValue(default.friends.[*].address.[*].street)\n"
    "|   |   |   |---AttributeParser(job)\n"
    "|   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |---AttributeParser(coworker)\n"
    "|   |   |   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |   |   |   |   |---StringValue(default.friends.[*].job.[*].coworker.[*].name.first)\n"
    "|   |   |---ArrayFilterParser(gender=unknown) with type 0\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---StringValue(default.friends.[gender='unknown'].name.first)\n"
    "|   |   |---ArrayFilterParser(name.first=@name.first) with type 1\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---StringValue(default.friends.[name.first=@name.first].name.initial)\n"
    "|   |   |---ArrayFilterParser(name.first=name.last) with type 1\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---StringValue(default.friends.[name.first=name.last].name.initial)\n"
    "|   |   |---ArrayIndexParser(2)\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---StringValue(default.friends.[2].name.first)\n"
    "|   |---AttributeParser(car)\n"
    "|   |   |---MapKeyParser(nickname)\n"
    "|   |   |   |---StringValue(default.car.['nickname'].color)\n";
  EXPECT_EQ(actual, expected);
}

}
}