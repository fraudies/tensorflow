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

class AvroParserTree {
public:
  // creates all the correct parser nodes with
  static Status Build(AvroParserTree* parser_tree,
    const std::vector<std::pair<string, DataType>>& keys_and_types);

  Status ParseValues(std::map<string, ValueStoreUniquePtr>* key_to_value,
    const std::vector<AvroValueSharedPtr>& values);

  // default constructor
  AvroParserTree(const string& avro_namespace = kDefaultNamespace);

  inline string GetAvroNamespace() const { return avro_namespace_; }

  // exposed for testing
  inline AvroParserSharedPtr getRoot() const { return root_; }
  string ToString() const { return (*root_).ToString(); };
private:
  static const char kSeparator = '.';
  static const string kArrayAllElements;
  static const string kDefaultNamespace;

  Status Build(AvroParser* parent, const std::vector<PrefixTreeNodeSharedPtr>& children);
  Status BuildKeyWithInternalName(const std::vector<std::pair<string, DataType>>& keys_and_types);

  static Status GetUniqueKeys(std::unordered_set<string>* keys,
    const std::vector<std::pair<string, DataType>>& keys_and_types);
  static Status AddBeginMarks(std::map<string, ValueStoreUniquePtr>* key_to_value);
  static Status AddFinishMarks(std::map<string, ValueStoreUniquePtr>* key_to_value);

  Status CreateAvroParser(AvroParserUniquePtr& value_parser, const string& infix) const;

  Status CreateValueParser(AvroParserUniquePtr& value_parser,
    const string& name, DataType data_type) const;

  Status ConvertToUserName(string* user_name, const string& internal_name) const;

  static bool ContainsFilter(string* lhs, string* rhs, const string& key);
  static bool IsFilter(string* lhs, string* rhs, const string& key);
  static bool IsArrayAll(const string& infix);
  static bool IsArrayIndex(int* index, const string& infix);
  static bool IsMapKey(string* key, const string& infix);
  static bool IsAttribute(const string& infix);
  static bool IsStringConstant(string* constant, const string& infix);

  Status InitValueBuffers(std::map<string, ValueStoreUniquePtr>* key_to_value);

  inline bool IsDefaultNamespace() const { return avro_namespace_ == kDefaultNamespace; };

  const string avro_namespace_;

  AvroParserSharedPtr root_;

  // used to preserve the order in the parse value method, InitValueBuffers before each parse call
  std::vector<std::pair<string, DataType> > keys_and_types_;
  // This map is a helper for fast access of the data type that corresponds to the key
  std::map<string, DataType> key_to_type_;
};

}
}

#endif // TENSORFLOW_DATA_AVRO_PARSER_TREE_H_