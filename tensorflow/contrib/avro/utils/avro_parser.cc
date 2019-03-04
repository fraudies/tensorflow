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
#include "tensorflow/contrib/avro/utils/avro_value_parser.h"

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// AvroParser
// ------------------------------------------------------------
AvroParser::AvroParser() {}
AvroParser::~AvroParser() {}

// ------------------------------------------------------------
// AvroValueParser
// ------------------------------------------------------------
AvroValueParser::AvroValueParser(const string& key) : key_(key) { }
AvroValueParser::~AvroValueParser() { }
AvroValueParser::ResolveValues(std::stack<std:pair<AvroParser*, avro_value_t*>>* values,
  const avro_value_t& value) const {
  return Status(errors::Unimplemented("Not implemented for avro value parsers."));
}

// ------------------------------------------------------------
// Concrete implementations of avro value parsers
// ------------------------------------------------------------
BoolValueParser::BoolValueParser(const string& key) : AvroValueParser(key) { }
BoolValueParser::~BoolValueParser() { }
Status BoolValueParser::ParseValue(std::map<string, ValueStore*>* values, const avro_value_t& value) {
  // TODO: Check compatibility between value and type or before calling this method--let's see where it fits better
  int field_value = 0;
  if (avro_value_get_boolean(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract boolean. Error: ",
      avro_strerror()));
  }
  // Assume the key exists
  ValueStore *store = (*values)[key];
  (*store).Add(field_value ? true : false);
  return Status::OK();
}

IntValueParser::IntValueParser(const string& key) : AvroValueParser(key) { }
IntValueParser::~IntValueParser() { }
Status IntValueParser::ParseValue(std::map<string, ValueStore*>* values, const avro_value_t& value) {
  int field_value = 0;
  if (avro_value_get_int(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract int. Error: ",
      avro_strerror()));
  }
  // Assume the key exists
  ValueStore *store = (*values)[key];
  (*store).Add(field_value);
  return Status::OK();
}

StringValueParser::StringValueParser(const string& key) : AvroValueParser(key) { }
StringValueParser::~StringValueParser() { }
Status StringValueParser::ParseValue(std::map<string, ValueStore*>* values, const avro_value_t& value) {
  const char* field_value = nullptr;  // just a pointer to the data not a copy, no need to free this
  size_t field_size = 0;
  if (avro_value_get_string(&value, &field_value, &field_size) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract string. Error: ",
      avro_strerror()));
  }
  // Assume the key exists
  ValueStore *store = (*values)[key];
  (*store).AddByRef(string(field_value, field_size - 1));
  return Status::OK();
}

// ------------------------------------------------------------
// Concrete implementations of value parsers
// ------------------------------------------------------------
Status ArrayAllParser::ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values,
  const avro_value_t& value, const map<string, std::unique_ptr<ValueStore>>& parsed_values) {
  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);
  avro_value_t next_value;
  // TODO(fraudies): Add begin mark to all buffers. Add an iterator to the prefix tree and
  // shuttle it through the parser tree to add all these marks.
  for (i_elements = 0; i_elements < n_elements; ++i_elements) {
    avro_value_get_by_index(&value, i_elements, &next_value, NULL);
    // For all children
    for (const std::unique_ptr<AvroParser>& child : children_) {
      (*values).push_back(make_pair(child.get(), &next_value));
    }
  }
  // TODO(fraudies): Add finish mark to all buffers using the iterator from above
  return Status::OK();
}

ArrayIndexParser::ArrayIndexParser(int index) : index_(index) { }
ArrayIndexParser::~ArrayIndexParser() { }
Status ArrayIndexParser::ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values,
  const avro_value_t& value, const map<string, std::unique_ptr<ValueStore>>& parsed_values) {
  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);
  if (index_ > n_elements || _index < 0) {
    return Status(errors::InvalidArgument("Invalid index ", index_,
      ". Range [", 0, ", ", next_value, ")."));
  }
  avro_value_t next_value;
  // TODO(fraudies): Add begin mark to all buffers. Add an iterator to the prefix tree and
  // shuttle it through the parser tree to add all these marks.
  avro_value_get_by_index(&value, index_, &next_value, NULL);
  // For all children
  for (const std::unique_ptr<AvroParser>& child : children_) {
    (*values).push_back(make_pair(child.get(), &next_value));
  }
  // TODO(fraudies): Add finish mark to all buffers using the iterator from above
  return Status::OK();
}

ArrayFilterParser::ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type)
  : lhs_(lhs), rhs_(rhs), type_(type) { }
ArrayFilterParser::~ArrayFilterParser() { }
Status ArrayFilterParser::ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values,
  const avro_value_t& value, const map<string, std::unique_ptr<ValueStore>>& parsed_values) {
  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);
  avro_value_t next_value;

  bool addValue = false;
  if (type_ == kRhsIsConstant) {
    addValue = (*parsed_values[lhs_]).LatestValueMatches(rhs_);
  } else if (type_ == kRhsIsValue) {
    addValue = (*parsed_values[lhs_]).LatestValuesMatch(*parsed_values[rhs_]);
  } else {
    return Status(errors::Internal("Unknown constant type ", type_));
  }

  if (addValue) {
    // TODO(fraudies): Add begin mark to all buffers. Add an iterator to the prefix tree and
    // shuttle it through the parser tree to add all these marks.
    for (i_elements = 0; i_elements < n_elements; ++i_elements) {
      avro_value_get_by_index(&value, i_elements, &next_value, NULL);
      // For all children
      for (const std::unique_ptr<AvroParser>& child : children_) {
        (*values).push_back(make_pair(child.get(), &next_value));
      }
    }
    // TODO(fraudies): Add finish mark to all buffers using the iterator from above
  }

  return Status::OK();
}

MapKeyParser::MapKeyParser(const string& key) : key_(key) { }
MapKeyParser::~MapKeyParser() { }
Status MapKeyParser::ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values,
  const avro_value_t& value, const map<string, std::unique_ptr<ValueStore>>& parsed_values) {
  // TODO(fraudies): Large overlap between key/attribute parser--only difference is the error
  // TODO(fraudies): Use AvroValuePtr here
  avro_value_t next_value;
  if (avro_value_get_by_name(&value, key_.c_str(), &next_value, NULL) != 0) {
    return errors::InvalidArgument("Unable to find key '", key_, "'.");
  }
  for (const std::unique_ptr<AvroParser>& child : children_) {
    (*values).push_back(make_pair(child.get(), &next_value));
  }
  return Status::OK();
}

AttributeParser::AttributeParser(const string& name) : name_(name) { }
AttributeParser::~AttributeParser() { }
Status AttributeParser::ResolveValues(std::stack<std:pair<AvroParser*, avro_value_t*>>* values,
  const avro_value_t& value, const map<string, std::unique_ptr<ValueStore>>& parsed_values) const {
  // TODO(fraudies): Use AvroValuePtr here
  avro_value_t next_value;
  if (avro_value_get_by_name(&value, name_.c_str(), &next_value, NULL) != 0) {
    return errors::InvalidArgument("Unable to find name '", name_, "'.");
  }
  for (const std::unique_ptr<AvroParser>& child : children_) {
    (*values).push_back(make_pair(child.get(), &next_value));
  }
  return Status::OK();
}

}
}