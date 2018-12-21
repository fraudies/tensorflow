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
#include <avro.h>
#include <vector>
#include "tensorflow/contrib/avro/utils/value_buffer.h"

namespace tensorflow {
namespace data {

class ValueParser {
public:
  ValueParser();
  virtual ~ValueParser();
  // If is terminal calls ParseValue
  // Otherwise add to the key and call parse on all children for all values
  // Once done with all values end the dimension on the value buffer
  Status Parse(const string& key, const avro_value_t& avro_value);
protected:
  // Default implementation returns error and does not alter the out_value
  virtual Status ResolveValues(vector<avro_value_t*>* out_values, const avro_value_t& in_value);
  // Default implementation does not do anything
  virtual Status ParseValue(map<string, ValueBuffer*>* values, const avro_value_t& value);
  vector<std::unique_ptr<ValueParser>> children_;
};

// TODO(fraudies): Cleanup hierarchy a bit; note that the part that is currently ugly is the
// difference between a child node and the other nodes, in terms of which method to call
// and the other method is not implemented in the other case

// -------------------------------------------------------------------------------------------------
// Terminal types
// -------------------------------------------------------------------------------------------------
class BoolValueParser : public ValueParser {
public:
  BoolValueParser();
  virtual ~BoolValueParser();
protected:
  Status ParseValue(map<string, ValueBuffer*>* values, const avro_value_t& value) override;
};

class IntValueParser : public ValueParser {
public:
  IntValueParser();
  virtual ~IntValueParser();
protected:
  Status ParseValue(map<string, ValueBuffer*>* values, const avro_value_t& value) override;
};


// -------------------------------------------------------------------------------------------------
// Intermediary types
// -------------------------------------------------------------------------------------------------
class ArrayAllParser : public ValueParser {
protected:
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value) override;
};

class ArrayIndexParser : public ValueParser {
public:
  ArrayIndexParser(int index);
  virtual ~ArrayIndexParser();
protected:
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value) override;
private:
  int index_;
};

class ArrayFilterParser : public ValueParser {
public:
  ArrayFilterParser(const string& lhs, const string& rhs);
  virtual ~ArrayFilterParser();
protected:
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value) override;
private:
  string lhs_;
  string rhs_;
};

class MapAllParser : public ValueParser {
public:
  MapAllParser();
  virtual ~MapAllParser();
protected:
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value) override;
}

class MapKeyParser : public ValueParser {
public:
  MapAllParser(const string& key);
  ~MapAllParser();
protected:
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value) override;
private:
  string key_;
};

class AttributeParser : public ValueParser {
public:
  AttributeParser(const string& name);
  virtual ~AttributeParser();
protected:
  // check that the in_value is of type record
  // check that an attribute with name exists
  // get the the attribute for the name and return it in the vector as single element
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value) override;
private:
  string name_;
};


class ParserTree {
public:
  ParserTree();
  // creates all the correct parser nodes with
  static Status Build(ParserTree* parser_tree, const vector<pair<string, DataType>>& keys_and_types);
  Status ParseValue(vector<Tensor>* tensors, const avro_value_t& in_value);
private:
  Build(ValueParser* father, const vector<std::shared_ptr<TreeNode>>& children);
  static Status GetUniqueKeys(unordered_set<string>* keys, const vector<pair<string, DataType>>& keys_and_types);
  static Status CreateValueParser(std::unique_ptr<ValueParser>& value_parser, const string& infix);
  static Status CreateValueParser(std::unique_ptr<ValueParser>& value_parser, DataType data_type);
  static bool IsFilter(string* lhs, string* rhs, const string& key);
  static bool IsArrayAll(const string& infix);
  static bool IsArrayIndex(int* index, const string& infix);
  static bool IsArrayFilter(string* lhs, string* rhs, const string& infix);
  static bool IsMapAll(const string& infix);
  static bool IsMapKey(string* key, const string& infix);
  static bool IsAttribute(const string& infix);
  static bool IsStringConstant(string* constant, const string& infix);
  Status InitValues(const vector<string>& keys, const vector<DataType>& expected_types);
  std::unique_ptr<ValueParser> root_;
  vector<string> keys_; // we need this here to preserve the order in the parse value method
  map<string, std::unique_ptr<ValueBuffer>> values_; // will also be used to get the data type for the node
};

}
}
