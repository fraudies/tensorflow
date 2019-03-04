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
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/contrib/avro/utils/value_buffer.h"

namespace tensorflow {
namespace data {

class AvroParser {
public:
  AvroParser();
  virtual ~AvroParser();
  virtual Status ResolveValues(std::stack<std:pair<AvroParser*, avro_value_t*>>* values, const avro_value_t& value) const = 0;
  inline virtual bool UsesParsedValues() const { return false; }
  inline bool IsTerminal() const { return children_.size() == 0; }
  inline void AddChild(const std::unique_ptr<AvroParser>& child) { children_.push_back(std::move(child)); }
private:
  std::vector<std::unique_ptr<AvroParser>> children_;
};

class AvroValueParser : public AvroParser {
public:
  AvroValueParser(const string& key);
  virtual ~AvroValueParser();
  // returns error since we can't resolve values for avro value parser and below
  Status ResolveValues(std::stack<std:pair<AvroParser*, avro_value_t*>>* values, const avro_value_t& value,
    const map<string, std::unique_ptr<ValueStore>>& parsed_values) const;

  virtual Status ParseValue(std::map<string, std::unique_ptr<ValueStore>>* values, const avro_value_t& value) const = 0;

private:
  //static Status CheckKey(const std::map<string, std::unique_ptr<ValueStore>>& values);
  string key_;
};

// -------------------------------------------------------------------------------------------------
// Terminal types
// -------------------------------------------------------------------------------------------------
class BoolValueParser : public AvroValueParser {
public:
  BoolValueParser(const string& key);
  virtual ~BoolValueParser();
protected:
  Status ParseValue(std::map<string, ValueStore*>* values, const avro_value_t& value) override;
};

class IntValueParser : public AvroValueParser {
public:
  IntValueParser(const string& key);
  virtual ~IntValueParser();
protected:
  Status ParseValue(std::map<string, ValueStore*>* values, const avro_value_t& value) override;
};

class StringValueParser : public AvroValueParser {
public:
  StringValueParser(const string& key);
  virtual ~StringValueParser();
protected:
  Status ParseValue(std::map<string, ValueStore*>* values, const avro_value_t& value) override;
};

// -------------------------------------------------------------------------------------------------
// Intermediary types
// -------------------------------------------------------------------------------------------------
class ArrayAllParser : public AvroParser {
protected:
  Status ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values, const avro_value_t& value,
    const map<string, std::unique_ptr<ValueStore>>& parsed_values) override;
};

class ArrayIndexParser : public AvroParser {
public:
  ArrayIndexParser(int index);
  virtual ~ArrayIndexParser();
protected:
  Status ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values, const avro_value_t& value,
    const map<string, std::unique_ptr<ValueStore>>& parsed_values) override;
private:
  int index_;
};

enum ArrayFilterType { kRhsIsConstant, kRhsIsValue };

class ArrayFilterParser : public AvroParser {
public:
  ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type);
  virtual ~ArrayFilterParser();
protected:
  inline virtual bool UsesParsedValues() const { return true; }
  Status ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values,
    const avro_value_t& value, const map<string, std::unique_ptr<ValueStore>>& parsed_values) override;
private:
  string lhs_;
  string rhs_;
  ArrayFilterType type_;
};

class MapKeyParser : public AvroParser {
public:
  MapKeyParser(const string& key);
  ~MapKeyParser();
protected:
  Status ResolveValues(std::stack<std:pair<AvroValueParser*, avro_value_t*>>* values, const avro_value_t& value,
    const map<string, std::unique_ptr<ValueStore>>& parsed_values) override;
private:
  string key_;
};

class AttributeParser : public AvroParser {
public:
  AttributeParser(const string& name);
  virtual ~AttributeParser();
protected:
  // check that the in_value is of type record
  // check that an attribute with name exists
  // get the the attribute for the name and return it in the vector as single element
  Status ResolveValues(std::stack<std:pair<AvroParser*, avro_value_t*>>* values, const avro_value_t& value,
    const map<string, std::unique_ptr<ValueStore>>& parsed_values) const override;
private:
  string name_;
};

}
}

#endif // TENSORFLOW_DATA_AVRO_PARSER_H_
