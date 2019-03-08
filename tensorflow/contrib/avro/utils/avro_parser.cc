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
#include <queue>
#include "tensorflow/contrib/avro/utils/avro_parser.h"

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// AvroParser
// ------------------------------------------------------------
AvroParser::AvroParser() {}
AvroParser::~AvroParser() {}
Status AvroParser::ResolveValues(std::queue<std::pair<AvroParserPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStorePtr>& parsed_values) const {
  return Status(errors::Unimplemented("Not implemented for avro parser."));
}

Status AvroParser::ParseValue(std::map<string, ValueStorePtr>* values,
  const avro_value_t& value) const {
  return Status(errors::Unimplemented("Not implemented for avro parser."));
}

const std::vector<AvroParserPtr>& AvroParser::GetChildren() const {
  return children_;
}

const std::vector<AvroValueParserPtr>& AvroParser::GetFinalDescendents() const {
  // If this parser is terminal there are no final descendents
  if (IsTerminal()) {
    return final_descendents_;
  }

  // Compute the final descendents if we never computed them before
  if (final_descendents_.size() == 0) {
    std::queue<AvroParserPtr> current;
    const std::vector<AvroParserPtr>& children = GetChildren();
    for (const auto& child : children) {
      current.push(child);
    }
    // Helper variable for children of subsequent nodes
    while (!current.empty()) {
      const std::vector<AvroParserPtr>& children = (*current.front()).GetChildren();
      if (children.size() == 0) {
        // TODO(fraudies): Maybe we can design this better avoiding this down-cast
        // Here we assume each child is a value parser
        final_descendents_.push_back(std::dynamic_pointer_cast<AvroValueParser>(current.front()));
      } else {
        for (const auto& child : children) {
          current.push(child);
        }
      }
      current.pop();
    }
  }

  // Return the final descendents
  return final_descendents_;
}


// ------------------------------------------------------------
// AvroValueParser
// ------------------------------------------------------------
AvroValueParser::AvroValueParser(const string& key) : key_(key) { }
AvroValueParser::~AvroValueParser() { }

// ------------------------------------------------------------
// Concrete implementations of avro value parsers
// ------------------------------------------------------------
BoolValueParser::BoolValueParser(const string& key) : AvroValueParser(key) { }
BoolValueParser::~BoolValueParser() { }
Status BoolValueParser::ParseValue(std::map<string, ValueStorePtr>* values, const avro_value_t& value) const {
  // TODO: Check compatibility between value and type or before calling this method--let's see where it fits better
  int field_value = 0;
  if (avro_value_get_boolean(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract boolean. Error: ",
      avro_strerror()));
  }
  // Assumes the key exists
  // TODO(fraudies): Redesign to remove this cast
  (*reinterpret_cast<BoolValueBuffer*>((*values)[key_].get())).Add(field_value ? true : false);
  //(*(*values)[key_]).Add<bool>(field_value ? true : false);
  return Status::OK();
}

IntValueParser::IntValueParser(const string& key) : AvroValueParser(key) { }
IntValueParser::~IntValueParser() { }
Status IntValueParser::ParseValue(std::map<string, ValueStorePtr>* values, const avro_value_t& value) const {
  int field_value = 0;
  if (avro_value_get_int(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract int. Error: ",
      avro_strerror()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<IntValueBuffer*>((*values)[key_].get())).Add(field_value);
  return Status::OK();
}

StringValueParser::StringValueParser(const string& key) : AvroValueParser(key) { }
StringValueParser::~StringValueParser() { }
Status StringValueParser::ParseValue(std::map<string, ValueStorePtr>* values, const avro_value_t& value) const {
  const char* field_value = nullptr;  // just a pointer to the data not a copy, no need to free this
  size_t field_size = 0;
  if (avro_value_get_string(&value, &field_value, &field_size) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract string. Error: ",
      avro_strerror()));
  }
  // Assume the key exists
  (*reinterpret_cast<StringValueBuffer*>((*values)[key_].get())).AddByRef(string(field_value, field_size - 1));
  // (*(*values)[key_]).AddByRef<string>(string(field_value, field_size - 1));
  return Status::OK();
}

ArrayBeginMarkerParser::ArrayBeginMarkerParser(const std::vector<AvroValueParserPtr>& final_descendents)
  : AvroValueParser("BeginMarker"), final_descendents_(final_descendents) { }
Status ArrayBeginMarkerParser::ParseValue(std::map<string, ValueStorePtr>* values,
    const avro_value_t& value) const {

  for (const AvroValueParserPtr& value_parser : final_descendents_) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).BeginMark();
  }

  return Status::OK();
}

ArrayFinishMarkerParser::ArrayFinishMarkerParser(const std::vector<AvroValueParserPtr>& final_descendents)
  : AvroValueParser("FinishMarker"), final_descendents_(final_descendents) { }
Status ArrayFinishMarkerParser::ParseValue(std::map<string, ValueStorePtr>* values,
    const avro_value_t& value) const {

  for (const AvroValueParserPtr& value_parser : final_descendents_) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).FinishMark();
  }

  return Status::OK();
}


// ------------------------------------------------------------
// Concrete implementations of value parsers
// ------------------------------------------------------------
Status ArrayAllParser::ResolveValues(
  std::queue<std::pair<AvroParserPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStorePtr>& parsed_values) const {

  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);
  const std::vector<AvroParserPtr>& children = GetChildren();
  const std::vector<AvroValueParserPtr>& final_descendents = GetFinalDescendents();

  // Add a begin mark to all value buffers under this array
  AvroValueSharedPtr begin_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayBeginMarkerParser>(final_descendents), begin_value));

  // Resolve all the values from the array
  for (size_t i_elements = 0; i_elements < n_elements; ++i_elements) {
    AvroValueSharedPtr next_value(new avro_value_t);
    avro_value_get_by_index(&value, i_elements, next_value.get(), NULL);
    // For all children
    for (const AvroParserPtr& child : children) {
      (*values).push(std::make_pair(child, next_value));
    }
  }

  // Add a finish mark to all value buffers under this array
  AvroValueSharedPtr finish_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayFinishMarkerParser>(final_descendents), finish_value));

  return Status::OK();
}

ArrayIndexParser::ArrayIndexParser(size_t index) : index_(index) { }
ArrayIndexParser::~ArrayIndexParser() { }
Status ArrayIndexParser::ResolveValues(
  std::queue<std::pair<AvroParserPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStorePtr>& parsed_values) const {

  // Check for valid index
  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);
  if (index_ > n_elements || index_ < 0) {
    return Status(errors::InvalidArgument("Invalid index ", index_,
      ". Range [", 0, ", ", n_elements, ")."));
  }

  const std::vector<AvroParserPtr>& children = GetChildren();
  const std::vector<AvroValueParserPtr>& final_descendents = GetFinalDescendents();

  // Add a begin mark to all value buffers under this array
  AvroValueSharedPtr begin_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayBeginMarkerParser>(final_descendents), begin_value));

  AvroValueSharedPtr next_value(new avro_value_t);
  avro_value_get_by_index(&value, index_, next_value.get(), NULL);

  // For all children same next value
  for (const AvroParserPtr& child : children) {
    (*values).push(std::make_pair(child, next_value));
  }

  // Add a finish mark to all value buffers under this array
  AvroValueSharedPtr finish_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayFinishMarkerParser>(final_descendents), finish_value));

  return Status::OK();
}

ArrayFilterParser::ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type)
  : lhs_(lhs), rhs_(rhs), type_(type) { }
ArrayFilterParser::~ArrayFilterParser() { }

Status ArrayFilterParser::ResolveValues(
  std::queue<std::pair<AvroParserPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStorePtr>& parsed_values) const {

  bool addValue = false;
  // Note, here we assume that entries for the lhs and rhs exist in the map!
  if (type_ == kRhsIsConstant) {
    addValue = (*parsed_values.at(lhs_)).LatestValueMatches(rhs_);
  } else if (type_ == kRhsIsValue) {
    addValue = (*parsed_values.at(lhs_)).LatestValuesMatch(*parsed_values.at(rhs_));
  } else {
    return Status(errors::Internal("Unknown constant type ", type_));
  }

  const std::vector<AvroValueParserPtr>& final_descendents = GetFinalDescendents();

  // Add a begin mark to all value buffers under this array
  AvroValueSharedPtr begin_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayBeginMarkerParser>(final_descendents), begin_value));

  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);

  if (addValue) {
    const std::vector<AvroParserPtr>& children = GetChildren();
    // shuttle it through the parser tree to add all these marks.
    for (size_t i_elements = 0; i_elements < n_elements; ++i_elements) {
      AvroValueSharedPtr next_value(new avro_value_t);
      avro_value_get_by_index(&value, i_elements, next_value.get(), NULL);
      // For all children
      for (const AvroParserPtr& child : children) {
        (*values).push(std::make_pair(child, next_value));
      }
    }
  }

  // Add a finish mark to all value buffers under this array
  AvroValueSharedPtr finish_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayFinishMarkerParser>(final_descendents), finish_value));

  return Status::OK();
}

MapKeyParser::MapKeyParser(const string& key) : key_(key) { }
MapKeyParser::~MapKeyParser() { }
Status MapKeyParser::ResolveValues(
  std::queue<std::pair<AvroParserPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStorePtr>& parsed_values) const {

  // TODO(fraudies): Code for key/attribute parser almost identical, except for error message
  AvroValueSharedPtr next_value(new avro_value_t);
  if (avro_value_get_by_name(&value, key_.c_str(), next_value.get(), NULL) != 0) {
    return errors::InvalidArgument("Unable to find key '", key_, "'.");
  }
  const std::vector<AvroParserPtr>& children = GetChildren();
  for (const AvroParserPtr& child : children) {
    (*values).push(std::make_pair(child, next_value));
  }
  return Status::OK();
}

AttributeParser::AttributeParser(const string& name) : name_(name) { }
AttributeParser::~AttributeParser() { }
Status AttributeParser::ResolveValues(
  std::queue<std::pair<AvroParserPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStorePtr>& parsed_values) const {

  AvroValueSharedPtr next_value(new avro_value_t);
  if (avro_value_get_by_name(&value, name_.c_str(), next_value.get(), NULL) != 0) {
    return errors::InvalidArgument("Unable to find name '", name_, "'.");
  }
  const std::vector<AvroParserPtr>& children = GetChildren();
  for (const AvroParserPtr& child : children) {
    // TODO: Check if need to do more for memory management
    (*values).push(std::make_pair(child, next_value));
  }
  return Status::OK();
}

}
}