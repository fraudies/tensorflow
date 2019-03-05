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

#include <memory>
#include "tensorflow/contrib/avro/utils/avro_parser.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// Tests for avro terminal types
// ------------------------------------------------------------
TEST(AvroParserTest, BoolValueParser) {
  const string key("dummyKey");
  std::vector<bool> field_values = {true, false};
  for (bool field_value : field_values) {
    BoolValueParser parser(key);
    std::map<string, ValueStorePtr> values;
    values.insert(std::make_pair(key, std::unique_ptr<BoolValueBuffer>(new BoolValueBuffer())));
    avro_value_t value;
    avro_generic_boolean_new(&value, field_value);
    TF_EXPECT_OK(parser.ParseValue(&values, value));
    EXPECT_EQ((*reinterpret_cast<BoolValueBuffer*>(values[key].get())).back(), field_value);
  }
}

TEST(AvroParserTest, IntValueParser) {
  const string key("dummyKey");
  std::vector<int> field_values = {std::numeric_limits<int>::min(), -1, 0, 1,
    std::numeric_limits<int>::max()};
  for (int field_value : field_values) {
    IntValueParser parser(key);
    std::map<string, ValueStorePtr> values;
    values.insert(std::make_pair(key, std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));
    avro_value_t value;
    avro_generic_int_new(&value, field_value);
    TF_EXPECT_OK(parser.ParseValue(&values, value));
    EXPECT_EQ((*reinterpret_cast<IntValueBuffer*>(values[key].get())).back(), field_value);
  }
}

TEST(AvroParserTest, StringValueParser) {
  const string key("dummyKey");
  std::vector<string> field_values = {"", "a", "abc", "328983"};
  for (const string& field_value : field_values) {
    StringValueParser parser(key);
    std::map<string, ValueStorePtr> values;
    values.insert(std::make_pair(key, std::unique_ptr<StringValueBuffer>(new StringValueBuffer())));
    avro_value_t value;
    avro_generic_string_new(&value, field_value.c_str());
    TF_EXPECT_OK(parser.ParseValue(&values, value));
    EXPECT_EQ((*reinterpret_cast<StringValueBuffer*>(values[key].get())).back(), field_value);
  }
}


// ------------------------------------------------------------
// Tests for avro intermediary types
// ------------------------------------------------------------
TEST(AttributeParserTest, ResolveValues) {
  // Create the value and fill it with dummy data
  const string name_value = "Karl Gauss";
  const string schema =
          "{"
          "  \"type\": \"record\","
          "  \"name\": \"person\","
          "  \"fields\": ["
          "    { \"name\": \"name\", \"type\": \"string\" },"
          "    { \"name\": \"age\", \"type\": \"int\" }"
          "  ]"
          "}";

	avro_value_t value;
	avro_schema_t record_schema = nullptr;
	EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

	avro_value_iface_t *record_class = avro_generic_class_from_schema(record_schema);
	EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

	size_t  field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 2);

  avro_value_t field;
  EXPECT_EQ(avro_value_get_by_index(&value, 0, &field, NULL), 0);
  EXPECT_EQ(avro_value_set_string(&field, name_value.data()), 0);

  EXPECT_EQ(avro_value_get_by_index(&value, 1, &field, NULL), 0);
  EXPECT_EQ(avro_value_set_int(&field, 139), 0);

  AttributeParser parser("name");
  StringValueParser* p = new StringValueParser("person.name");
  parser.AddChild(std::unique_ptr<StringValueParser>(p));

  std::map<string, ValueStorePtr> parsed_values; // empty on purpose
  std::stack<std::pair<AvroParserPtr, AvroValuePtr> > parser_for_value;

  TF_EXPECT_OK(parser.ResolveValues(&parser_for_value, value, parsed_values));

  // Ensure we have exactly one parser
  EXPECT_EQ(parser_for_value.size(), 1);
  const auto& current = parser_for_value.top();
  AvroParserPtr avro_parser = current.first;
  StringValueParser* string_parser = dynamic_cast<StringValueParser*>(avro_parser.get());
  EXPECT_TRUE(string_parser != nullptr);

  // Check the avro value
  const AvroValuePtr& avro_value = std::move(current.second);
  EXPECT_TRUE(avro_value.get() != nullptr);
  avro_type_t field_type = avro_value_get_type(avro_value.get());

  EXPECT_EQ(field_type, AVRO_STRING);
  const char *actual_str = NULL;
	size_t actual_size = 0;
	EXPECT_EQ(avro_value_get_string(avro_value.get(), &actual_str, &actual_size), 0);
  EXPECT_EQ(name_value.length(), actual_size-1);
  EXPECT_EQ(name_value, string(actual_str, actual_size-1));
}

}
}