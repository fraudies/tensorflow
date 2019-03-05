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
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/contrib/avro/utils/avro_value.h"
#include "tensorflow/contrib/avro/utils/value_buffer.h"

namespace tensorflow {
namespace data {

class AvroParser; // forward declare for pointer definition
using AvroParserPtr = std::shared_ptr<AvroParser>;

class AvroParser {
public:
  AvroParser();
  virtual ~AvroParser();

  // Default implementation returns error
  virtual Status ResolveValues(std::stack<std::pair<AvroParserPtr, AvroValuePtr> >* values,
    const avro_value_t& value, const std::map<string, ValueStorePtr>& parsed_values) const;

  // Default implementation returns error
  virtual Status ParseValue(std::map<string, ValueStorePtr>* values, const avro_value_t& value) const;

  inline virtual bool UsesParsedValues() const { return false; }
  inline bool IsTerminal() const { return children_.size() == 0; }
  inline void AddChild(const AvroParserPtr& child) { children_.push_back(child); }
protected:
  std::vector<AvroParserPtr> children_;
};


class AvroValueParser : public AvroParser {
public:
  AvroValueParser(const string& key);
  virtual ~AvroValueParser();
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
  virtual ~BoolValueParser();
  Status ParseValue(std::map<string, ValueStorePtr>* values, const avro_value_t& value) const override;
};

class IntValueParser : public AvroValueParser {
public:
  IntValueParser(const string& key);
  virtual ~IntValueParser();
  Status ParseValue(std::map<string, ValueStorePtr>* values, const avro_value_t& value) const override;
};

class StringValueParser : public AvroValueParser {
public:
  StringValueParser(const string& key);
  virtual ~StringValueParser();
  Status ParseValue(std::map<string, ValueStorePtr>* values, const avro_value_t& value) const override;
};

// -------------------------------------------------------------------------------------------------
// Intermediary types
// -------------------------------------------------------------------------------------------------
class ArrayAllParser : public AvroParser {
protected:
  Status ResolveValues(std::stack<std::pair<AvroParserPtr, AvroValuePtr> >* values, const avro_value_t& value,
    const std::map<string, ValueStorePtr>& parsed_values) const override;
};

class ArrayIndexParser : public AvroParser {
public:
  ArrayIndexParser(size_t index);
  virtual ~ArrayIndexParser();
  Status ResolveValues(std::stack<std::pair<AvroParserPtr, AvroValuePtr> >* values, const avro_value_t& value,
    const std::map<string, ValueStorePtr>& parsed_values) const override;
private:
  size_t index_;
};

enum ArrayFilterType { kRhsIsConstant, kRhsIsValue };

class ArrayFilterParser : public AvroParser {
public:
  ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type);
  virtual ~ArrayFilterParser();
  inline virtual bool UsesParsedValues() const { return true; }
  Status ResolveValues(std::stack<std::pair<AvroParserPtr, AvroValuePtr> >* values,
    const avro_value_t& value, const std::map<string, ValueStorePtr>& parsed_values) const override;
private:
  string lhs_;
  string rhs_;
  ArrayFilterType type_;
};

class MapKeyParser : public AvroParser {
public:
  MapKeyParser(const string& key);
  ~MapKeyParser();
  Status ResolveValues(std::stack<std::pair<AvroParserPtr, AvroValuePtr> >* values, const avro_value_t& value,
    const std::map<string, ValueStorePtr>& parsed_values) const override;
private:
  string key_;
};

class AttributeParser : public AvroParser {
public:
  AttributeParser(const string& name);
  virtual ~AttributeParser();
  // check that the in_value is of type record
  // check that an attribute with name exists
  // get the the attribute for the name and return it in the vector as single element
  Status ResolveValues(std::stack<std::pair<AvroParserPtr, AvroValuePtr> >* values, const avro_value_t& value,
    const std::map<string, ValueStorePtr>& parsed_values) const override;
private:
  string name_;
};

}
}

#endif // TENSORFLOW_DATA_AVRO_PARSER_H_
