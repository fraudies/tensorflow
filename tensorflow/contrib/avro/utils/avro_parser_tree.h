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
#ifndef TENSORFLOW_DATA_AVRO_PARSER_TREE_H_
#define TENSORFLOW_DATA_AVRO_PARSER_TREE_H_

#include <vector>
#include "tensorflow/contrib/avro/utils/avro_parser.h"
#include "tensorflow/contrib/avro/utils/prefix_tree.h"

namespace tensorflow {
namespace data {

typedef std::pair<string, DataType> KeyWithType;

// This will be a vector
template <typename T>
class UniqueVector {
public:
  bool Prepend(const T& value) {
    auto info = unique.insert(value);
    if (info.second) {
      order.insert(order.begin(), value);
    }
    return info.second;
  }
  bool Append(const T& value) {
    auto info = unique.insert(value);
    if (info.second) {
      order.push_back(value);
    }
    return info.second;
  }
  const std::vector<T>& GetOrdered() const {
    return order;
  }
  const std::set<T>& GetUnique() const {
    return unique;
  }
private:
  std::set<T> unique;
  std::vector<T> order;
};

class AvroParserTree {
public:
  // creates all the correct parser nodes with
  static Status Build(AvroParserTree* parser_tree, const string& avro_namespace,
    const std::vector<KeyWithType>& keys_and_types);

  Status ParseValues(std::map<string, ValueStoreUniquePtr>* key_to_value,
    const std::vector<AvroValueSharedPtr>& values);

  inline string GetAvroNamespace() const { return avro_namespace_; }

  // exposed for testing
  inline AvroParserSharedPtr getRoot() const { return root_; }
  string ToString() const { return (*root_).ToString(); };
private:
  static const char kSeparator = '.';
  static const string kArrayAllElements;
  static const string kDefaultNamespace;

  Status Build(AvroParser* parent, const std::vector<PrefixTreeNodeSharedPtr>& children);

  // Resolve and set namespace
  string ResolveAndSetNamespace(const string& avro_namespace);

  static std::vector<KeyWithType> OrderAndResolveKeyTypes(
    const std::vector<KeyWithType>& keys_and_types);

  static Status ValidateUniqueKeys(const std::vector<KeyWithType>& keys_and_types);
  static Status AddBeginMarks(std::map<string, ValueStoreUniquePtr>* key_to_value);
  static Status AddFinishMarks(std::map<string, ValueStoreUniquePtr>* key_to_value);

  static string ResolveFilterName(const string& user_name, const string& side_name,
    const string& filter_name);

  static std::vector<string> GetPartsWithoutAvroNamespace(const string& user_name,
    const string& avro_namespace);

  static string RemoveDefaultAvroNamespace(const string& name);
  static string RemoveDotBeforeBracket(const string& name);

  Status CreateAvroParser(AvroParserUniquePtr& value_parser, const string& infix,
    const string& avro_namespace) const;

  Status CreateValueParser(AvroParserUniquePtr& value_parser,
    const string& user_name, DataType data_type) const;

  static bool ContainsFilter(string* lhs_name, string* rhs_name, const string& name);
  static bool IsFilter(string* lhs, string* rhs, const string& key);
  static bool IsArrayAll(const string& infix);
  static bool IsArrayIndex(int* index, const string& infix);
  static bool IsMapKey(string* key, const string& infix);
  static bool IsAttribute(const string& infix);
  static bool IsStringConstant(string* constant, const string& infix);

  Status InitValueBuffers(std::map<string, ValueStoreUniquePtr>* key_to_value);

  static bool IsDefaultNamespace(const string& avro_namespace) { return avro_namespace == kDefaultNamespace; };

  string avro_namespace_;

  AvroParserSharedPtr root_;

  // used to preserve the order in the parse value method, InitValueBuffers before each parse call
  std::vector<std::pair<string, DataType> > keys_and_types_;
  // This map is a helper for fast access of the data type that corresponds to the key
  std::map<string, DataType> key_to_type_;
};

}
}

#endif // TENSORFLOW_DATA_AVRO_PARSER_TREE_H_