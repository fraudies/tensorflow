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
#ifndef TENSORFLOW_DATA_AVRO_PARSER_H_
#define TENSORFLOW_DATA_AVRO_PARSER_H_

#include <avro.h>
#include <vector>
#include <map>
#include <queue>
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/contrib/avro/utils/avro_value.h"
#include "tensorflow/contrib/avro/utils/value_buffer.h"

namespace tensorflow {
namespace data {

class AvroParser; // forward declare for pointer definition
class AvroValueParser;

using AvroParserUniquePtr = std::unique_ptr<AvroParser>;
using AvroParserSharedPtr = std::shared_ptr<AvroParser>;
using AvroValueParserSharedPtr = std::shared_ptr<AvroValueParser>;


class AvroParser {
public:
  AvroParser();

  // Default implementation returns error
  virtual Status ResolveValues(
    std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
    const avro_value_t& value,
    const std::map<string, ValueStoreUniquePtr>& parsed_values) const;

  // Default implementation returns error
  virtual Status ParseValue(std::map<string, ValueStoreUniquePtr>* values, const avro_value_t& value) const;

  inline virtual bool UsesParsedValues() const { return false; }
  inline bool IsTerminal() const { return children_.size() == 0; }
  inline void AddChild(const AvroParserSharedPtr& child) { children_.push_back(child); }

  // public for testing
  const std::vector<AvroParserSharedPtr>& GetChildren() const;
  virtual string ToString(int level = 0) const = 0;
protected:
  const std::vector<AvroValueParserSharedPtr>& GetFinalDescendents() const;
  void AddBeginMarkersToFinalDescendents(std::map<string, ValueStoreUniquePtr>* parsed_values) const;
  void AddFinishMarkersToFinalDescendents(std::map<string, ValueStoreUniquePtr>* parsed_values) const;
  string ChildrenToString(int level) const;
  string LevelToString(int level) const;
private:
  std::vector<AvroParserSharedPtr> children_;
  mutable std::vector<AvroValueParserSharedPtr> final_descendents_; // computed upon first call and then cached
};


class AvroValueParser : public AvroParser {
public:
  AvroValueParser(const string& key);
  inline const string& GetKey() const { return key_; }
protected:
  //static Status CheckKey(const std::map<string, std::unique_ptr<ValueStore> >& values);
  string key_;
};

// -------------------------------------------------------------------------------------------------
// Terminal types
// -------------------------------------------------------------------------------------------------
class BoolValueParser : public AvroValueParser {
public:
  BoolValueParser(const string& key);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class LongValueParser : public AvroValueParser {
public:
  LongValueParser(const string& key);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};


class IntValueParser : public AvroValueParser {
public:
  IntValueParser(const string& key);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class DoubleValueParser : public AvroValueParser {
public:
  DoubleValueParser(const string& key);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class FloatValueParser : public AvroValueParser {
public:
  FloatValueParser(const string& key);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class StringValueParser : public AvroValueParser {
public:
  StringValueParser(const string& key);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class ArrayBeginMarkerParser : public AvroValueParser {
public:
  ArrayBeginMarkerParser(const std::vector<AvroValueParserSharedPtr>& final_descendents);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
private:
  const std::vector<AvroValueParserSharedPtr> final_descendents_;
};

class ArrayFinishMarkerParser : public AvroValueParser {
public:
  ArrayFinishMarkerParser(const std::vector<AvroValueParserSharedPtr>& final_descendents);
  Status ParseValue(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
private:
  const std::vector<AvroValueParserSharedPtr> final_descendents_;
};

// -------------------------------------------------------------------------------------------------
// Intermediary types
// -------------------------------------------------------------------------------------------------
class ArrayAllParser : public AvroParser {
public:
  Status ResolveValues(
    std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
    const avro_value_t& value,
    const std::map<string, ValueStoreUniquePtr>& parsed_values) const override;
  virtual string ToString(int level = 0) const;
};

class ArrayIndexParser : public AvroParser {
public:
  ArrayIndexParser(size_t index);
  Status ResolveValues(
    std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
    const avro_value_t& value,
    const std::map<string, ValueStoreUniquePtr>& parsed_values) const override;
  virtual string ToString(int level = 0) const;
private:
  size_t index_;
};

enum ArrayFilterType { kRhsIsConstant, kRhsIsValue };

class ArrayFilterParser : public AvroParser {
public:
  ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type);
  inline virtual bool UsesParsedValues() const { return true; }
  Status ResolveValues(
    std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
    const avro_value_t& value,
    const std::map<string, ValueStoreUniquePtr>& parsed_values) const override;
  virtual string ToString(int level = 0) const;
private:
  string lhs_;
  string rhs_;
  ArrayFilterType type_;
};

class MapKeyParser : public AvroParser {
public:
  MapKeyParser(const string& key);
  Status ResolveValues(
    std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
    const avro_value_t& value,
    const std::map<string, ValueStoreUniquePtr>& parsed_values) const override;
  virtual string ToString(int level = 0) const;
private:
  string key_;
};

class AttributeParser : public AvroParser {
public:
  AttributeParser(const string& name);
  // check that the in_value is of type record
  // check that an attribute with name exists
  // get the the attribute for the name and return it in the vector as single element
  Status ResolveValues(
    std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
    const avro_value_t& value,
    const std::map<string, ValueStoreUniquePtr>& parsed_values) const override;
  virtual string ToString(int level = 0) const;
private:
  string name_;
};

class NamespaceParser : public AvroParser {
public:
  NamespaceParser(const string& name);
  // checks namespace of value against given namespace
  // - if matches passes avro value to all it's child parsers
  // - if does not match returns error with actual and expected namespace
  Status ResolveValues(
    std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> >* values,
    const avro_value_t& value,
    const std::map<string, ValueStoreUniquePtr>& parsed_values) const override;
  virtual string ToString(int level = 0) const;
private:
  string name_;
};


}
}

#endif // TENSORFLOW_DATA_AVRO_PARSER_H_
