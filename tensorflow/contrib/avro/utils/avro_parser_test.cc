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
    std::map<string, ValueStoreUniquePtr> values;
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
    std::map<string, ValueStoreUniquePtr> values;
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
    std::map<string, ValueStoreUniquePtr> values;
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

	avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
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
  parser.AddChild(std::unique_ptr<StringValueParser>(new StringValueParser("person.name")));

  std::map<string, ValueStoreUniquePtr> parsed_values; // empty on purpose
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> > parser_for_value;

  TF_EXPECT_OK(parser.ResolveValues(&parser_for_value, value, parsed_values));

  // Ensure we have exactly one parser
  EXPECT_EQ(parser_for_value.size(), 1);
  const auto& current = parser_for_value.front();
  AvroParserSharedPtr avro_parser = current.first;
  StringValueParser* string_parser = dynamic_cast<StringValueParser*>(avro_parser.get());
  EXPECT_TRUE(string_parser != nullptr);

  // Check the avro value
  const AvroValueSharedPtr avro_value = current.second;
  EXPECT_TRUE(avro_value.get() != nullptr);
  avro_type_t field_type = avro_value_get_type(avro_value.get());

  EXPECT_EQ(field_type, AVRO_STRING);
  const char *actual_str = NULL;
	size_t actual_size = 0;
	EXPECT_EQ(avro_value_get_string(avro_value.get(), &actual_str, &actual_size), 0);
  EXPECT_EQ(name_value.length(), actual_size-1);
  EXPECT_EQ(name_value, string(actual_str, actual_size-1));
}

TEST(ArrayAllParser, ResolveValues) {
  // Create the value and fill it with dummy data
  const string person_socials_key = "person.socials";
  const string person_friends_key = "person.friends";

  const int n_num = 10;
  const std::vector<string> friends_names = {"Karl Gauss", "Kurt Goedel"};
  const string schema =
          "{"
          "  \"type\": \"record\","
          "  \"name\": \"person\","
          "  \"fields\": ["
          "    { \"name\": \"socials\", \"type\": {\"type\": \"array\", \"items\": \"int\"} }"
          "  ]"
          "}";

	avro_value_t value;
	avro_schema_t record_schema = nullptr;
	EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

	avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
	EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

	size_t  field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t socials_field;
  avro_value_t element;
  size_t index;
  EXPECT_EQ(avro_value_get_by_index(&value, 0, &socials_field, NULL), 0);
  for (int i_num = 0; i_num < n_num; ++i_num) {
    EXPECT_EQ(avro_value_append(&socials_field, &element, &index), 0);
    EXPECT_EQ(i_num, index);
    EXPECT_EQ(avro_value_set_int(&element, i_num), 0);
  }

  // Define the parsers for the socials
  AttributeParser socials_parser("socials");
  AvroParserSharedPtr parse_all_items = std::make_shared<ArrayAllParser>();
  AvroValueParserSharedPtr parse_ints = std::make_shared<IntValueParser>(person_socials_key);
  socials_parser.AddChild(parse_all_items);
  (*parse_all_items).AddChild(parse_ints);

  std::map<string, ValueStoreUniquePtr> parsed_values; // empty on purpose
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> > parser_for_value;
  parsed_values.insert(std::make_pair(
    person_socials_key,
    std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));
  TF_EXPECT_OK((*parse_all_items).ResolveValues(&parser_for_value, socials_field, parsed_values));

  // Check the state in the variables resolving values
  EXPECT_EQ(parser_for_value.size(), n_num + 2);

  // Check for begin marker
  ArrayBeginMarkerParser* begin_parser = dynamic_cast<ArrayBeginMarkerParser*>(parser_for_value.front().first.get());
  EXPECT_TRUE(begin_parser != nullptr);
  parser_for_value.pop();

  // Check elements with their values
  for (int i_num = 0; i_num < n_num; ++i_num) {
    const auto& element_pair = parser_for_value.front();
    IntValueParser* int_parser = dynamic_cast<IntValueParser*>(element_pair.first.get());
    EXPECT_TRUE(int_parser != nullptr);

    const AvroValueSharedPtr avro_value = element_pair.second;
    EXPECT_TRUE(avro_value.get() != nullptr);
    avro_type_t field_type = avro_value_get_type(avro_value.get());

    EXPECT_EQ(field_type, AVRO_INT32);
    int actual_int = 0;
    EXPECT_EQ(avro_value_get_int(avro_value.get(), &actual_int), 0);
    EXPECT_EQ(i_num, actual_int);

    parser_for_value.pop();
  }

  // Check for finish marker
  ArrayFinishMarkerParser* finish_parser = dynamic_cast<ArrayFinishMarkerParser*>(parser_for_value.front().first.get());
  EXPECT_TRUE(finish_parser != nullptr);
  parser_for_value.pop();
}

struct Person {
  string name;
  int age;
  Person(const string& name, int age) : name(name), age(age) { }
};

TEST(ArrayFilterParser, ResolveValues) {
  // This test does not cover yet the actual mechanism of parsing out all the values
  // This is the task of the avro parser tree

  // Create the value and fill it with dummy data
  const std::vector<Person> persons = {Person("Carl", 33), Person("Mary", 22), Person("Carl", 12)};
  const string persons_name = "persons";
  const string name_name = "name";
  const string age_name = "age";
  const string persons_name_key = "persons.[*].name";
  const string persons_age_key = "persons.[name='Carl'].age";
  const string schema = "{"
                        "  \"type\":\"record\","
                        "  \"name\":\"contact\","
                        "  \"fields\":["
                        "      {"
                        "         \"name\":\"persons\","
                        "         \"type\":{"
                        "            \"type\":\"array\","
                        "            \"items\":{"
                        "               \"type\":\"record\","
                        "               \"name\":\"person\","
                        "               \"fields\":["
                        "                  {"
                        "                     \"name\":\"name\","
                        "                     \"type\":\"string\""
                        "                  },"
                        "                  {"
                        "                     \"name\":\"age\","
                        "                     \"type\":\"int\""
                        "                  }"
                        "               ]"
                        "            }"
                        "         }"
                        "      }"
                        "   ]"
                        "}";

	avro_value_t value;
	avro_schema_t record_schema = nullptr;
	EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

	avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
	EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

	size_t  field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t persons_field, persons_element, person_name, person_age;
  size_t index;
  EXPECT_EQ(avro_value_get_by_name(&value, persons_name.c_str(), &persons_field, NULL), 0);
  for (const Person& person : persons) {
    EXPECT_EQ(avro_value_append(&persons_field, &persons_element, &index), 0);

    EXPECT_EQ(avro_value_get_by_name(&persons_element, name_name.c_str(), &person_name, NULL), 0);
    EXPECT_EQ(avro_value_set_string(&person_name, person.name.data()), 0);

    EXPECT_EQ(avro_value_get_by_name(&persons_element, age_name.c_str(), &person_age, NULL), 0);
    EXPECT_EQ(avro_value_set_int(&person_age, person.age), 0);
  }

  // persons.[*].name
  AttributeParser persons_parser(persons_name);
  AvroParserSharedPtr parse_names_items = std::make_shared<ArrayAllParser>();
  AvroParserSharedPtr parse_names = std::make_shared<AttributeParser>(name_name);
  (*parse_names_items).AddChild(parse_names);

  // persons.[name='Carl'].age
  AvroParserSharedPtr parse_carls_items = std::make_shared<ArrayFilterParser>(persons_name_key, "Carl", kRhsIsConstant);
  AvroParserSharedPtr parse_ages = std::make_shared<AttributeParser>(age_name);
  (*parse_carls_items).AddChild(parse_ages);

  persons_parser.AddChild(parse_names_items);
  persons_parser.AddChild(parse_carls_items);

  std::unique_ptr<StringValueBuffer> names(new StringValueBuffer());
  (*names).AddByRef("Carl"); (*names).AddByRef("Mary"); (*names).AddByRef("Carl");

  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> > parser_for_value;
  std::map<string, ValueStoreUniquePtr> parsed_values;
  parsed_values.insert(std::make_pair(
    persons_name_key,
    std::move(names)));
  parsed_values.insert(std::make_pair(
    persons_age_key,
    std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));

  TF_EXPECT_OK((*parse_carls_items).ResolveValues(&parser_for_value, persons_field, parsed_values));

  // Being/finish marker are added, and two value parsers for the two matches of 'Carl'
  EXPECT_EQ(parser_for_value.size(), 4);
}

TEST(MapKeyParser, ResolveValues) {
  const string cars_name = "cars";
  const string serials_name = "serials";
  const string serial_key = "302984";
  const string car_in_map_key = "cars['" + serial_key + "']";
  const int serial_value = 32948;
  const string schema = "{"
                        "   \"type\":\"record\","
                        "   \"name\":\"cars\","
                        "   \"fields\":["
                        "      {"
                        "         \"name\":\"serials\","
                        "         \"type\":{"
                        "            \"type\":\"map\","
                        "            \"values\":{"
                        "               \"type\":\"int\""
                        "            }"
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

  avro_value_t serials_field, serial_value_field;
  EXPECT_EQ(avro_value_get_by_name(&value, serials_name.c_str(), &serials_field, NULL), 0);
  size_t value_index;
  int updated = 0;

  EXPECT_EQ(avro_value_add(&serials_field, serial_key.c_str(),
    &serial_value_field, &value_index, &updated), 0);
  EXPECT_EQ(avro_value_set_int(&serial_value_field, serial_value), 0);

  AttributeParser cars_parser(cars_name);
  AvroParserSharedPtr map_key_parser = std::make_shared<MapKeyParser>(serial_key);
  AvroParserSharedPtr map_value_parser = std::make_shared<IntValueParser>(car_in_map_key);
  (*map_key_parser).AddChild(map_value_parser);
  cars_parser.AddChild(map_key_parser);

  std::map<string, ValueStoreUniquePtr> parsed_values; // empty on purpose
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> > parser_for_value;
  parsed_values.insert(std::make_pair(
    car_in_map_key,
    std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));

  TF_EXPECT_OK((*map_key_parser).ResolveValues(&parser_for_value, serials_field, parsed_values));

  // Ensure we have exactly one parser
  EXPECT_EQ(parser_for_value.size(), 1);
  const auto& current = parser_for_value.front();
  AvroParserSharedPtr avro_parser = current.first;
  IntValueParser* int_parser = dynamic_cast<IntValueParser*>(avro_parser.get());
  EXPECT_TRUE(int_parser != nullptr);

  // Check the avro value
  const AvroValueSharedPtr avro_value = current.second;
  EXPECT_TRUE(avro_value.get() != nullptr);
  avro_type_t field_type = avro_value_get_type(avro_value.get());

  int actual_int = 0;
  EXPECT_EQ(field_type, AVRO_INT32);
  EXPECT_EQ(avro_value_get_int(avro_value.get(), &actual_int), 0);
  EXPECT_EQ(serial_value, actual_int);
}


}
}