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
#include "tensorflow/contrib/avro/utils/avro_tensor.h"

namespace tensorflow {
namespace data {

// Encapsulates all the logic to build a prefix tree
// Encapsulates all the logic to build a prefix tree
class TreeNode {
public:
  TreeNode();
  virtual ~TreeNode(const string& prefix);
  void GetChildren(vector<std::shared_ptr<TreeNode>>* children) const;
  void GetPrefix(string* prefix) const;
  void GetName(string* name, const char separator) const; // returns the full name using the separator
private:
  string prefix_;
  std::shared_ptr<TreeNode> father_;
  vector<std::shared_ptr<TreeNode>> children_;
};

class PrefixTree {
public:
  PrefixTree(const string& root_name = "");
  virtual ~PrefixTree();
  void GetRootPrefix(string* root_prefix) const;
  // Assumes tree != nullptr
  static void Build(PrefixTree* tree, const vector<vector<string>>& prefixes);
private:
  std::unique_ptr<TreeNode> root;
};


class ValueParser {
public:
  ValueParser(const string& name);
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
  string name_;
  //set<avro_type_t> expected_types_; // this is a set because of unions, currently used for nulls
  vector<std::unique_ptr<ValueParser>> children_;
};

class StringValueParser : public ValueParser {
protected:
  // check that the type matches string
  // add ref to value buffer
  Status ParseValue(map<string, ValueBuffer*>* values, const avro_value_t& value);
};

class IntValueParser : public ValueParser {
protected:
  // check that the type matches string
  // add value to buffer
  Status ParseValue(map<string, ValueBuffer*>* values, const avro_value_t& value);
};

class AttributeParser : public ValueParser {
protected:
  // check that the in_value is of type record
  // check that an attribute with name exists
  // get the the attribute for the name and return it in the vector as single element
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value);
};

class ArrayParser : public ValueParser {
protected:
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value);
};

class FilterArrayParser : public ArrayParser {
protected:
  Status ResolveValue(vector<avro_value_t*>* out_values, const avro_value_t& in_value);
};

class ParserTree {
public:
  ParserTree();
  // creates all the correct parser nodes with
  static Status Build(ParserTree* parser_tree, const vector<string>& keys, const vector<DataType>& expected_types);
  Status ParseValue(vector<Tensor>* tensors, const avro_value_t& in_value);
private:
  Build(ValueParser* father, const vector<std::shared_ptr<TreeNode>>& children);
  Status InitValues(const vector<string>& keys, const vector<DataType>& expected_types);
  std_unique_ptr<ValueParser> root_;
  map<string, std::unique_ptr<ValueBuffer>> values_;
};

class AvroParser {
public:
  // Ensure the following
  // 0. Check for duplicates in the definition of the keys
  // 1. @ from filters are first
  // 2. If we have a filter and the [*] expression does not appear, add it with resolving the lhs 2nd
  // 3. Add all other keys
  // Use the expected type to decide which value parser node to add
  static Status Create(AvroParser* parser, const vector<pair<string, DataType>>& keys_and_types);
  AvroParser();
  virtual ~AvroParser();
  // May add multiple elements and dimensions to the tensors or only a single value
  // Avro tensor keeps track of the required size and supports filling in to transform into a tensor
  // Will check the expected type against the actual type
  // Return tensors in the same order the keys have been provided
  Status ParseAll(vector<Tensor>* tensors, const avro_value_t& in_value);
private:
  std::unique_ptr<ValueParserTree> parser_tree_;
};

}
}

