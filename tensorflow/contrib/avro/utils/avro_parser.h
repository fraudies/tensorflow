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

using AvroParserUniquePtr = std::unique_ptr<AvroParser>;
using AvroParserSharedPtr = std::shared_ptr<AvroParser>;


class AvroParser {
public:
  AvroParser(const string& key);

  // may also read from parsed values if filtering
  virtual Status Parse(std::map<string, ValueStoreUniquePtr>* parsed_values,
    const avro_value_t& value) const = 0;

  inline void AddChild(const AvroParserSharedPtr& child) { children_.push_back(child); }

  // public for testing
  const std::vector<AvroParserSharedPtr>& GetChildren() const;
  virtual string ToString(int level = 0) const = 0;
  inline const string& GetKey() const { return key_; }
protected:
  const std::vector<AvroParserSharedPtr>& GetFinalDescendents() const;
  string ChildrenToString(int level) const;
  string LevelToString(int level) const;
  string key_;
private:
  inline bool IsTerminal() const { return children_.size() == 0; }
  std::vector<AvroParserSharedPtr> children_;

  // computed upon first call and then cached
  mutable std::vector<AvroParserSharedPtr> final_descendents_;
};

class BoolValueParser : public AvroParser {
public:
  BoolValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class LongValueParser : public AvroParser {
public:
  LongValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};


class IntValueParser : public AvroParser {
public:
  IntValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class DoubleValueParser : public AvroParser {
public:
  DoubleValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class FloatValueParser : public AvroParser {
public:
  FloatValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class StringValueParser : public AvroParser {
public:
  StringValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class ArrayAllParser : public AvroParser {
public:
  ArrayAllParser();
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
};

class ArrayIndexParser : public AvroParser {
public:
  ArrayIndexParser(size_t index);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
private:
  size_t index_;
};

enum ArrayFilterType { kRhsIsConstant, kRhsIsValue };

class ArrayFilterParser : public AvroParser {
public:
  ArrayFilterParser(const string& lhs, const string& rhs, ArrayFilterType type);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
private:
  string lhs_;
  string rhs_;
  ArrayFilterType type_;
};

class MapKeyParser : public AvroParser {
public:
  MapKeyParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
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
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
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
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
private:
  string name_;
};


}
}

#endif // TENSORFLOW_DATA_AVRO_PARSER_H_
