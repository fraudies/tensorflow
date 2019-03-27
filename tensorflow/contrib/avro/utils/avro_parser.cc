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
#include <sstream>
#include "tensorflow/contrib/avro/utils/avro_parser.h"

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// AvroParser
// ------------------------------------------------------------
AvroParser::AvroParser() {}
AvroParser::~AvroParser() {}
Status AvroParser::ResolveValues(std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStoreUniquePtr>& parsed_values) const {
  return Status(errors::Unimplemented("Not implemented for avro parser."));
}

Status AvroParser::ParseValue(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {
  return Status(errors::Unimplemented("Not implemented for avro parser."));
}

const std::vector<AvroParserSharedPtr>& AvroParser::GetChildren() const {
  return children_;
}

const std::vector<AvroValueParserSharedPtr>& AvroParser::GetFinalDescendents() const {
  // If this parser is terminal there are no final descendents
  if (IsTerminal()) {
    return final_descendents_;
  }

  // Compute the final descendents if we never computed them before
  if (final_descendents_.size() == 0) {
    std::queue<AvroParserSharedPtr> current;
    const std::vector<AvroParserSharedPtr>& children = GetChildren();
    for (const auto& child : children) {
      current.push(child);
    }
    // Helper variable for children of subsequent nodes
    while (!current.empty()) {
      const std::vector<AvroParserSharedPtr>& children = (*current.front()).GetChildren();
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

string AvroParser::ChildrenToString(int level) const {
  std::stringstream ss;
  for (const auto child : children_) {
    ss << (*child).ToString(level + 1);
  }
  return ss.str();
}

string AvroParser::LevelToString(int level) const {
  std::stringstream ss;
  for (int l = 0; l < level; ++l) {
    ss << "|   ";
  }
  return ss.str();
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
Status BoolValueParser::ParseValue(std::map<string, ValueStoreUniquePtr>* values, const avro_value_t& value) const {
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
string BoolValueParser::ToString(int level) const {
  return LevelToString(level) + "|---BoolValue(" + key_ + ")\n";
}

IntValueParser::IntValueParser(const string& key) : AvroValueParser(key) { }
IntValueParser::~IntValueParser() { }
Status IntValueParser::ParseValue(std::map<string, ValueStoreUniquePtr>* values, const avro_value_t& value) const {
  int field_value = 0;
  if (avro_value_get_int(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract int. Error: ",
      avro_strerror()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<IntValueBuffer*>((*values)[key_].get())).Add(field_value);
  return Status::OK();
}
string IntValueParser::ToString(int level) const {
  return LevelToString(level) + "|---IntValue(" + key_ + ")\n";
}

StringValueParser::StringValueParser(const string& key) : AvroValueParser(key) { }
StringValueParser::~StringValueParser() { }
Status StringValueParser::ParseValue(std::map<string, ValueStoreUniquePtr>* values, const avro_value_t& value) const {
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
string StringValueParser::ToString(int level) const {
  return LevelToString(level) + "|---StringValue(" + key_ + ")\n";
}

ArrayBeginMarkerParser::ArrayBeginMarkerParser(const std::vector<AvroValueParserSharedPtr>& final_descendents)
  : AvroValueParser("BeginMarker"), final_descendents_(final_descendents) { }
Status ArrayBeginMarkerParser::ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const {

  for (const AvroValueParserSharedPtr& value_parser : final_descendents_) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).BeginMark();
  }

  return Status::OK();
}
string ArrayBeginMarkerParser::ToString(int level) const {
  return LevelToString(level) + "|---ArrayBeginMarkerParser\n";
}

ArrayFinishMarkerParser::ArrayFinishMarkerParser(const std::vector<AvroValueParserSharedPtr>& final_descendents)
  : AvroValueParser("FinishMarker"), final_descendents_(final_descendents) { }
Status ArrayFinishMarkerParser::ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const {

  for (const AvroValueParserSharedPtr& value_parser : final_descendents_) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).FinishMark();
  }

  return Status::OK();
}
string ArrayFinishMarkerParser::ToString(int level) const {
  return LevelToString(level) + "|---ArrayFinishMarkerParser\n";
}

// ------------------------------------------------------------
// Concrete implementations of value parsers
// ------------------------------------------------------------
Status ArrayAllParser::ResolveValues(
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStoreUniquePtr>& parsed_values) const {

  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  const std::vector<AvroValueParserSharedPtr>& final_descendents(GetFinalDescendents());

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
    for (const AvroParserSharedPtr& child : children) {
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
string ArrayAllParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayAllParser" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

ArrayIndexParser::ArrayIndexParser(size_t index) : index_(index) { }
ArrayIndexParser::~ArrayIndexParser() { }
Status ArrayIndexParser::ResolveValues(
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStoreUniquePtr>& parsed_values) const {

  // Check for valid index
  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);
  if (index_ > n_elements || index_ < 0) {
    return Status(errors::InvalidArgument("Invalid index ", index_,
      ". Range [", 0, ", ", n_elements, ")."));
  }

  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  const std::vector<AvroValueParserSharedPtr>& final_descendents(GetFinalDescendents());

  // Add a begin mark to all value buffers under this array
  AvroValueSharedPtr begin_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayBeginMarkerParser>(final_descendents), begin_value));

  AvroValueSharedPtr next_value(new avro_value_t);
  avro_value_get_by_index(&value, index_, next_value.get(), NULL);

  // For all children same next value
  for (const AvroParserSharedPtr& child : children) {
    (*values).push(std::make_pair(child, next_value));
  }

  // Add a finish mark to all value buffers under this array
  AvroValueSharedPtr finish_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayFinishMarkerParser>(final_descendents), finish_value));

  return Status::OK();
}
string ArrayIndexParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayIndexParser(" << index_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

ArrayFilterParser::ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type)
  : lhs_(lhs), rhs_(rhs), type_(type) { }
ArrayFilterParser::~ArrayFilterParser() { }
Status ArrayFilterParser::ResolveValues(
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStoreUniquePtr>& parsed_values) const {

  if (type_ != kRhsIsConstant && type_ != kRhsIsValue) {
    return Status(errors::Internal("Unknown constant type ", type_));
  }

  const std::vector<AvroValueParserSharedPtr>& final_descendents = GetFinalDescendents();

  // Add a begin mark to all value buffers under this array
  AvroValueSharedPtr begin_value(new avro_value_t);
  (*values).push(
    std::make_pair(
      std::make_shared<ArrayBeginMarkerParser>(final_descendents), begin_value));

  size_t n_elements = 0;
  avro_value_get_size(&value, &n_elements);

  const std::vector<AvroParserSharedPtr>& children(GetChildren());

  for (size_t i_elements = 0; i_elements < n_elements; ++i_elements) {

    size_t reverse_index = n_elements - i_elements;

    if (type_ == kRhsIsConstant
          && (*parsed_values.at(lhs_)).ValueMatchesAtReverseIndex(rhs_, reverse_index)
      || type_ == kRhsIsValue
          && (*parsed_values.at(lhs_)).ValuesMatchAtReverseIndex(*parsed_values.at(rhs_), reverse_index)) {

      AvroValueSharedPtr next_value(new avro_value_t);
      avro_value_get_by_index(&value, i_elements, next_value.get(), NULL);
      // For all children
      for (const AvroParserSharedPtr& child : children) {
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
string ArrayFilterParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayFilterParser(" << lhs_ << "=" << rhs_ << ") with type ";
  ss << type_ << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}


MapKeyParser::MapKeyParser(const string& key) : key_(key) { }
MapKeyParser::~MapKeyParser() { }
Status MapKeyParser::ResolveValues(
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStoreUniquePtr>& parsed_values) const {

  // TODO(fraudies): Code for key/attribute parser almost identical, except for error message
  AvroValueSharedPtr next_value(new avro_value_t);
  if (avro_value_get_by_name(&value, key_.c_str(), next_value.get(), NULL) != 0) {
    return errors::InvalidArgument("Unable to find key '", key_, "'.");
  }
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    (*values).push(std::make_pair(child, next_value));
  }
  return Status::OK();
}
string MapKeyParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---MapKeyParser(" << key_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}


AttributeParser::AttributeParser(const string& name) : name_(name) { }
AttributeParser::~AttributeParser() { }
Status AttributeParser::ResolveValues(
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStoreUniquePtr>& parsed_values) const {

  AvroValueSharedPtr next_value(new avro_value_t);
  if (avro_value_get_by_name(&value, name_.c_str(), next_value.get(), NULL) != 0) {
    return errors::InvalidArgument("Unable to find name '", name_, "'.");
  }
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    (*values).push(std::make_pair(child, next_value));
  }
  return Status::OK();
}
string AttributeParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---AttributeParser(" << name_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}


NamespaceParser::NamespaceParser(const string& name) : name_(name) { }
NamespaceParser::~NamespaceParser() { }
Status NamespaceParser::ResolveValues(
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
  const avro_value_t& value,
  const std::map<string, ValueStoreUniquePtr>& parsed_values) const {

  // TODO(fraudies): Check namespace match
  AvroValueSharedPtr next_value(new avro_value_t);
  avro_value_copy_ref(next_value.get(), &value);

  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    (*values).push(std::make_pair(child, next_value));
  }
  return Status::OK();
}
string NamespaceParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---NamespaceParser(" << name_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}


}
}