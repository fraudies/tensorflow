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
AvroParser::AvroParser(const string& key) : key_(key) { }

const std::vector<AvroParserSharedPtr>& AvroParser::GetChildren() const {
  return children_;
}

const std::vector<AvroParserSharedPtr>& AvroParser::GetFinalDescendents() const {
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
      if ((*current.front()).IsTerminal()) {
        // TODO(fraudies): Improve design to avoid cast; at least check outcome of cast
        final_descendents_.push_back(current.front());
      } else {
        const std::vector<AvroParserSharedPtr>& children = (*current.front()).GetChildren();
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
// Concrete implementations of avro value parsers
// ------------------------------------------------------------
BoolValueParser::BoolValueParser(const string& key) : AvroParser(key) { }
Status BoolValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

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

LongValueParser::LongValueParser(const string& key) : AvroParser(key) { }
Status LongValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  long field_value = 0;
  if (avro_value_get_long(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract int. Error: ",
      avro_strerror()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<LongValueBuffer*>((*values)[key_].get())).Add(field_value);
  return Status::OK();
}
string LongValueParser::ToString(int level) const {
  return LevelToString(level) + "|---LongValue(" + key_ + ")\n";
}

IntValueParser::IntValueParser(const string& key) : AvroParser(key) { }
Status IntValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

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

DoubleValueParser::DoubleValueParser(const string& key) : AvroParser(key) { }
Status DoubleValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  double field_value = 0;
  if (avro_value_get_double(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract int. Error: ",
      avro_strerror()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<DoubleValueBuffer*>((*values)[key_].get())).Add(field_value);
  return Status::OK();
}
string DoubleValueParser::ToString(int level) const {
  return LevelToString(level) + "|---DoubleValue(" + key_ + ")\n";
}

FloatValueParser::FloatValueParser(const string& key) : AvroParser(key) { }
Status FloatValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  float field_value = 0;
  if (avro_value_get_float(&value, &field_value) != 0) {
    return Status(errors::InvalidArgument("For '", key_, "' could not extract int. Error: ",
      avro_strerror()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<FloatValueBuffer*>((*values)[key_].get())).Add(field_value);
  return Status::OK();
}
string FloatValueParser::ToString(int level) const {
  return LevelToString(level) + "|---FloatValue(" + key_ + ")\n";
}

StringOrBytesValueParser::StringOrBytesValueParser(const string& key) : AvroParser(key) { }
Status StringOrBytesValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  const char* char_field_value = nullptr;  // just a pointer to the data not a copy, no need to free this
  size_t field_size = 0;
  if (avro_value_get_string(&value, &char_field_value, &field_size) == 0) {
    // Assume the key exists
    (*reinterpret_cast<StringValueBuffer*>((*values)[key_].get())).AddByRef(string(char_field_value, field_size - 1));
    // (*(*values)[key_]).AddByRef<string>(string(field_value, field_size - 1));
    return Status::OK();
  }

  const void* bytes_field_value = nullptr;
  if (avro_value_get_bytes(&value, &bytes_field_value, &field_size) == 0) {
    // Assume the key exists
    (*reinterpret_cast<StringValueBuffer*>((*values)[key_].get())).AddByRef(string((const char*)bytes_field_value, field_size));
    // (*(*values)[key_]).AddByRef<string>(string(field_value, field_size - 1));
    return Status::OK();
  }

  return Status(errors::InvalidArgument("For '", key_, "' could not extract string. Error: ",
    avro_strerror()));
}
string StringOrBytesValueParser::ToString(int level) const {
  return LevelToString(level) + "|---StringOrBytesValue(" + key_ + ")\n";
}

// ------------------------------------------------------------
// Concrete implementations of value parsers
// ------------------------------------------------------------
ArrayAllParser::ArrayAllParser() : AvroParser("") { }
Status ArrayAllParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  size_t n_elements = 0;

  if (avro_value_get_size(&value, &n_elements) != 0) {
    return errors::InvalidArgument("Unable to find size for array due to error: ",
      avro_strerror());
  }

  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  const std::vector<AvroParserSharedPtr>& final_descendents(GetFinalDescendents());

  // Add a begin mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).BeginMark();
  }

  // Resolve all the values from the array
  for (size_t i_elements = 0; i_elements < n_elements; ++i_elements) {
    AvroValueSharedPtr next_value(new avro_value_t);

    if (avro_value_get_by_index(&value, i_elements, next_value.get(), NULL) != 0) {
      return errors::InvalidArgument("Unable to find value for index '", i_elements,
        "' in array due to error: ", avro_strerror());
    }
    // For all children
    for (const AvroParserSharedPtr& child : children) {
      TF_RETURN_IF_ERROR((*child).Parse(values, *next_value));
    }
  }

  // Add a finish mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).FinishMark();
  }

  return Status::OK();
}
string ArrayAllParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayAllParser" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

ArrayIndexParser::ArrayIndexParser(size_t index) : AvroParser(""), index_(index) { }
Status ArrayIndexParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  // Check for valid index
  size_t n_elements = 0;

  if (avro_value_get_size(&value, &n_elements) != 0) {
    return errors::InvalidArgument("Unable to find size for array index due to error: ",
      avro_strerror());
  }

  if (index_ > n_elements || index_ < 0) {
    return Status(errors::InvalidArgument("Invalid index ", index_,
      ". Range [", 0, ", ", n_elements, ")."));
  }

  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  const std::vector<AvroParserSharedPtr>& final_descendents(GetFinalDescendents());

  // Add a begin mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).BeginMark();
  }

  AvroValueSharedPtr next_value(new avro_value_t);
  if (avro_value_get_by_index(&value, index_, next_value.get(), NULL) != 0) {
    return errors::InvalidArgument("Unable to find value for index '", index_, "' due to error: ",
      avro_strerror());
  }

  // For all children same next value
  for (const AvroParserSharedPtr& child : children) {
    TF_RETURN_IF_ERROR((*child).Parse(values, *next_value));
  }

  // Add a finish mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).FinishMark();
  }

  return Status::OK();
}
string ArrayIndexParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayIndexParser(" << index_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

ArrayFilterParser::ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type)
  : AvroParser(""), lhs_(lhs), rhs_(rhs), type_(type) { }
Status ArrayFilterParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  if (type_ != kRhsIsConstant && type_ != kRhsIsValue) {
    return Status(errors::Internal("Unknown constant type ", type_));
  }

  const std::vector<AvroParserSharedPtr>& final_descendents = GetFinalDescendents();

  // Add a begin mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).BeginMark();
  }

  size_t n_elements = 0;

  if (avro_value_get_size(&value, &n_elements) != 0) {
    return errors::InvalidArgument("Unable to find size for array filter due to error: ",
      avro_strerror());
  }

  const std::vector<AvroParserSharedPtr>& children(GetChildren());

  for (size_t i_elements = 0; i_elements < n_elements; ++i_elements) {

    size_t reverse_index = n_elements - i_elements;

    if ( ((type_ == kRhsIsConstant)
           && (*(*values).at(lhs_)).ValueMatchesAtReverseIndex(rhs_, reverse_index))
      || ((type_ == kRhsIsValue)
          && (*(*values).at(lhs_)).ValuesMatchAtReverseIndex(*(*values).at(rhs_), reverse_index)) ) {

      AvroValueSharedPtr next_value(new avro_value_t);
      if (avro_value_get_by_index(&value, i_elements, next_value.get(), NULL) != 0) {
        return errors::InvalidArgument("Unable to find value for index '", i_elements, "' due to error: ",
          avro_strerror());
      }

      // For all children
      for (const AvroParserSharedPtr& child : children) {
        TF_RETURN_IF_ERROR((*child).Parse(values, *next_value));
      }
    }
  }

  // Add a finish mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).FinishMark();
  }

  return Status::OK();
}
string ArrayFilterParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayFilterParser(" << lhs_ << "=" << rhs_ << ") with type ";
  ss << type_ << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}


MapKeyParser::MapKeyParser(const string& key) : AvroParser(""), key_(key) { }
Status MapKeyParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  // TODO(fraudies): Code for key/attribute parser almost identical, except for error message
  AvroValueSharedPtr next_value(new avro_value_t);
  if (avro_value_get_by_name(&value, key_.c_str(), next_value.get(), NULL) != 0) {
    return errors::InvalidArgument("Unable to find key '", key_, "'.");
  }
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    TF_RETURN_IF_ERROR((*child).Parse(values, *next_value));
  }
  return Status::OK();
}
string MapKeyParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---MapKeyParser(" << key_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}


AttributeParser::AttributeParser(const string& name) : AvroParser(""), name_(name) { }
Status AttributeParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  AvroValueSharedPtr next_value(new avro_value_t);
  if (avro_value_get_by_name(&value, name_.c_str(), next_value.get(), NULL) != 0) {
    return errors::InvalidArgument("Unable to find name '", name_, "'.");
  }
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    TF_RETURN_IF_ERROR((*child).Parse(values, *next_value));
  }
  return Status::OK();
}
string AttributeParser::ToString(int level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---AttributeParser(" << name_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}


NamespaceParser::NamespaceParser(const string& name) : AvroParser(""), name_(name) { }
Status NamespaceParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
  const avro_value_t& value) const {

  // TODO(fraudies): Check namespace match
  AvroValueSharedPtr next_value(new avro_value_t);
  avro_value_copy_ref(next_value.get(), &value);

  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    TF_RETURN_IF_ERROR((*child).Parse(values, *next_value));
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