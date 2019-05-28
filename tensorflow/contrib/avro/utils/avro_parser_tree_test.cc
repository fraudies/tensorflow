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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/contrib/avro/utils/avro_parser_tree.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that. Instead these tests only cover the wrapping in the mem readers
namespace tensorflow {
namespace data {

TEST(AvroParserTreeTest, BuildParserTree) {
  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair("friends[2].name.first", DT_STRING),
    std::make_pair("friends[*].address[*].street", DT_STRING),
    std::make_pair("friends[*].job[*].coworker[*].name.first", DT_STRING),
    std::make_pair("car['nickname'].color", DT_STRING),
    std::make_pair("friends[gender='unknown'].name.first", DT_STRING),
    std::make_pair("friends[name.first=name.last].name.initial", DT_STRING),
    std::make_pair("friends[name.first=@name.first].name.initial", DT_STRING)};
  AvroParserTree parser_tree; // will use default namespace
  TF_EXPECT_OK(AvroParserTree::Build(&parser_tree, keys_and_types));
  AvroParserSharedPtr root_parser = parser_tree.getRoot();
  NamespaceParser* namespace_parser = dynamic_cast<NamespaceParser*>(root_parser.get());
  EXPECT_TRUE(namespace_parser != nullptr);
  const std::vector<AvroParserSharedPtr>& children((*root_parser).GetChildren());
  EXPECT_EQ(children.size(), 3);
  const string actual(parser_tree.ToString());
  const string expected =
    "|---NamespaceParser(default)\n"
    "|   |---AttributeParser(name)\n"
    "|   |   |---AttributeParser(first)\n"
    "|   |   |   |---StringValue(name.first)\n"
    "|   |---AttributeParser(friends)\n"
    "|   |   |---ArrayAllParser\n"
    "|   |   |   |---AttributeParser(gender)\n"
    "|   |   |   |   |---StringValue(friends[*].gender)\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---AttributeParser(first)\n"
    "|   |   |   |   |   |---StringValue(friends[*].name.first)\n"
    "|   |   |   |   |---AttributeParser(last)\n"
    "|   |   |   |   |   |---StringValue(friends[*].name.last)\n"
    "|   |   |   |---AttributeParser(address)\n"
    "|   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |---AttributeParser(street)\n"
    "|   |   |   |   |   |   |---StringValue(friends[*].address[*].street)\n"
    "|   |   |   |---AttributeParser(job)\n"
    "|   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |---AttributeParser(coworker)\n"
    "|   |   |   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |   |   |   |   |---AttributeParser(first)\n"
    "|   |   |   |   |   |   |   |   |   |---StringValue(friends[*].job[*].coworker[*].name.first)\n"
    "|   |   |---ArrayFilterParser(gender=unknown) with type 0\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---AttributeParser(first)\n"
    "|   |   |   |   |   |---StringValue(friends[gender='unknown'].name.first)\n"
    "|   |   |---ArrayFilterParser(name.first=@name.first) with type 1\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---AttributeParser(initial)\n"
    "|   |   |   |   |   |---StringValue(friends[name.first=@name.first].name.initial)\n"
    "|   |   |---ArrayFilterParser(name.first=name.last) with type 1\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---AttributeParser(initial)\n"
    "|   |   |   |   |   |---StringValue(friends[name.first=name.last].name.initial)\n"
    "|   |   |---ArrayIndexParser(2)\n"
    "|   |   |   |---AttributeParser(name)\n"
    "|   |   |   |   |---AttributeParser(first)\n"
    "|   |   |   |   |   |---StringValue(friends[2].name.first)\n"
    "|   |---AttributeParser(car)\n"
    "|   |   |---MapKeyParser(nickname)\n"
    "|   |   |   |---AttributeParser(color)\n"
    "|   |   |   |   |---StringValue(car['nickname'].color)\n";

  EXPECT_EQ(actual, expected);
}

TEST(AvroParserTreeTest, ParseIntArray) {
  const string int_array_key = "int_array[*]";
  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair(int_array_key, DT_INT32)
  };
  AvroParserTree parser_tree;
  TF_EXPECT_OK(AvroParserTree::Build(&parser_tree, keys_and_types));

  const std::vector<int> int_values = {1, 2, 3, 4};
  const string schema = "{  \"type\":\"record\","
                        "   \"name\":\"values\","
                        "   \"fields\":["
                        "      {"
                        "         \"name\": \"int_array\","
                        "         \"type\":{"
                        "             \"type\": \"array\","
                        "             \"items\": \"int\""
                        "         }"
                        "      }"
                        "   ]"
                        "}";

  avro_value_t value;
  avro_schema_t record_schema = nullptr;
  EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

  avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
  EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

  size_t field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t int_array_field;
  const string int_array_name = "int_array";
  EXPECT_EQ(avro_value_get_by_name(&value, int_array_name.c_str(), &int_array_field, NULL), 0);

  for (int i_value = 0; i_value < int_values.size(); ++i_value) {
    int int_value = int_values[i_value];
    avro_value_t int_field;
    size_t index;

    // Get the field, check index, and set int value
    EXPECT_EQ(avro_value_append(&int_array_field, &int_field, &index), 0);
    EXPECT_EQ(i_value, index);
    EXPECT_EQ(avro_value_set_int(&int_field, int_value), 0);
  }

  std::map<string, ValueStoreUniquePtr> key_to_value;
  std::vector<AvroValueSharedPtr> values;
  values.push_back(std::make_shared<avro_value_t>(value));
  TF_EXPECT_OK(parser_tree.ParseValues(&key_to_value, values));

  auto key_and_value = key_to_value.find(int_array_key);
  // Entry should exist
  EXPECT_FALSE(key_and_value == key_to_value.end());

  // Define shapes
  const TensorShape shape({1, 4});

  // Define expected values
  Tensor expected(DT_INT32, shape);
  auto expected_flat = expected.flat<int>();
  for (int i_value = 0; i_value < int_values.size(); ++i_value) {
    expected_flat(i_value) = int_values[i_value];
  }

  // Define defaults
  Tensor defaults(DT_INT32, shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < int_values.size(); ++i_value) {
    defaults_flat(i_value) = 0;
  }

  // Allocate memory for actual
  Tensor actual(DT_INT32, shape);

  // Make dense tensor from buffer
  TF_EXPECT_OK((*(key_and_value->second)).MakeDense(&actual, shape, defaults));

  // actual and expected must match
  test::ExpectTensorEqual<int>(actual, expected);
}


TEST(AvroParserTreeTest, ParseIntValue) {
  const string int_value_name = "int_value";

  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair(int_value_name, DT_INT32)
  };
  AvroParserTree parser_tree;
  TF_EXPECT_OK(AvroParserTree::Build(&parser_tree, keys_and_types));

  const int int_value = 12;
  const string schema = "{  \"type\":\"record\","
                        "   \"name\":\"values\","
                        "   \"fields\":["
                        "      {"
                        "         \"name\":\"int_value\","
                        "         \"type\":\"int\""
                        "      }"
                        "   ]"
                        "}";

  avro_value_t value;
  avro_schema_t record_schema = nullptr;
  EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

  avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
  EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

  size_t field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t int_field;
  EXPECT_EQ(avro_value_get_by_name(&value, int_value_name.c_str(), &int_field, NULL), 0);
  EXPECT_EQ(avro_value_set_int(&int_field, int_value), 0);

  std::map<string, ValueStoreUniquePtr> key_to_value;
  std::vector<AvroValueSharedPtr> values;
  values.push_back(std::make_shared<avro_value_t>(value));
  TF_EXPECT_OK(parser_tree.ParseValues(&key_to_value, values));

  auto key_and_value = key_to_value.find(int_value_name);
  // Entry should exist
  EXPECT_FALSE(key_and_value == key_to_value.end());

  // Define shapes
  const TensorShape shape({1});

  // Define expected values
  Tensor expected(DT_INT32, shape);
  auto expected_flat = expected.flat<int>();
  expected_flat(0) = int_value;

  // Define defaults
  Tensor defaults(DT_INT32, shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 0;

  // Allocate memory for actual
  Tensor actual(DT_INT32, shape);

  // Make dense tensor from buffer
  TF_EXPECT_OK((*(key_and_value->second)).MakeDense(&actual, shape, defaults));

  // actual and expected must match
  test::ExpectTensorEqual<int>(actual, expected);
}


}
}