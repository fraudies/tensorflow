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

// RE2 https://github.com/google/re2/blob/master/re2/re2.h
#include <algorithm>
#include "re2/re2.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/contrib/avro/utils/avro_parser_tree.h"

namespace tensorflow {
namespace data {

// TODO(fraudies): Replace by regex once tensorflow is compiled with GCC > 4.8
void SplitOnDelimiterButNotInsideSquareBrackets(std::vector<string>* results, char delimiter, const string& str) {
  // check that delimiter is not [ or ]
  string lastMatch = "";
  int nBrackets = 0;
  for (char c : str) {
    nBrackets += (c == '[');
    nBrackets -= (c == ']');
    if (nBrackets == 0 && c == delimiter) {
      (*results).push_back(lastMatch);
      lastMatch = "";
    } else {
      lastMatch += c;
    }
  }
  (*results).push_back(lastMatch);
}

AvroParserTree::AvroParserTree(const string& avro_namespace) : avro_namespace_(avro_namespace) { }

Status AvroParserTree::ParseValues(std::map<string, ValueStoreUniquePtr>* key_to_value,
  const std::vector<AvroValueSharedPtr>& values) {

  // new assignment of all buffers
  TF_RETURN_IF_ERROR(InitValueBuffers(key_to_value));

  // add being marks to all buffers
  TF_RETURN_IF_ERROR(AddBeginMarks(key_to_value));

  for (auto const& value : values) {
    TF_RETURN_IF_ERROR((*root_).Parse(key_to_value, *value));
  }

  // add end marks to all buffers
  TF_RETURN_IF_ERROR(AddFinishMarks(key_to_value));

  return Status::OK();
}

Status AvroParserTree::Build(AvroParserTree* parser_tree,
    const std::vector<std::pair<string, DataType>>& keys_and_types) {

  // TODO(fraudies): Clean this up by extending the scope of the build keys function
  TF_RETURN_IF_ERROR((*parser_tree).BuildKeyWithInternalName(keys_and_types));

  // Get all the keys and check for uniqueness
  std::unordered_set<string> keys;
  TF_RETURN_IF_ERROR(GetUniqueKeys(&keys, (*parser_tree).keys_and_types_));
  LOG(INFO) << "Created set of unique keys";

  // Ensure that we have keys for lhs/rhs of all filters first
  // Note, some keys in all_keys may not be used for the output but by filter expressions
  std::vector<string> all_keys; // holds all keys in proper order
  string lhs;
  string rhs;
  string index;
  string name;

  // defined a type
  for (const string& key : keys) {

    // If filter
    if (ContainsFilter(&lhs, &rhs, key)) {

      // Handle lhs
      size_t pos = key.find(lhs);
      const string prefix = key.substr(0, pos-1);
      const string lhs_key = prefix + kArrayAllElements + kSeparator + lhs;

      all_keys.push_back(lhs_key);
      keys.erase(lhs_key);

      // Handle rhs
      if (IsStringConstant(nullptr, rhs)) {
        // do nothing for constants
      } else if (str_util::StartsWith(rhs, "@")) {
        // If anchored at top resolve full name
        const string rhs_key = (*parser_tree).GetAvroNamespace() + kSeparator + rhs.substr(1);
        all_keys.insert(all_keys.begin(), rhs_key);
        keys.erase(rhs_key);
      } else {
        // Otherwise
        const string rhs_key = prefix + kArrayAllElements + kSeparator + rhs;
        all_keys.push_back(rhs_key);
        keys.erase(rhs_key);
      }
    }

    all_keys.push_back(key);
  }

  // Parse keys into prefixes and handle namespace
  std::vector< std::vector<string> > prefixes;
  for (const string& key : all_keys) {
    std::vector<string> key_prefixes;
    SplitOnDelimiterButNotInsideSquareBrackets(&key_prefixes, kSeparator, key);
    // erase first prefix, because this is the namespace and covered in the root
    key_prefixes.erase(key_prefixes.begin());
    prefixes.push_back(key_prefixes);
  }

  OrderedPrefixTree prefix_tree((*parser_tree).GetAvroNamespace());
  OrderedPrefixTree::Build(&prefix_tree, prefixes);
  LOG(INFO) << "Built prefix tree";
  LOG(INFO) << prefix_tree.ToString();

  // Map all keys to types from inputs
  for (const auto& key_and_type : (*parser_tree).keys_and_types_) {
    (*parser_tree).key_to_type_[key_and_type.first] = key_and_type.second;
  }
  for (const string& key : all_keys) {
    auto key_and_type = (*parser_tree).key_to_type_.find(key);
    // Not present yet then it must have come from one of the filters which is a string type
    if (key_and_type == (*parser_tree).key_to_type_.end()) {
      (*parser_tree).key_to_type_[key] = DT_STRING;
    }
  }

  // Use the expected type to decide which value parser node to add
  (*parser_tree).root_ = std::make_shared<NamespaceParser>(prefix_tree.GetRootPrefix());

  // Use the prefix tree to build the parser tree
  TF_RETURN_IF_ERROR((*parser_tree).Build(
    (*parser_tree).root_.get(),
    (*prefix_tree.GetRoot()).GetChildren()));
  LOG(INFO) << "Built parser tree";
  LOG(INFO) << (*parser_tree).ToString();

  return Status::OK();
}

Status AvroParserTree::Build(AvroParser* parent, const std::vector<PrefixTreeNodeSharedPtr>& children) {
  for (PrefixTreeNodeSharedPtr child : children) {
    AvroParserUniquePtr avro_parser(nullptr);

    // Create a parser and add it to the parent
    TF_RETURN_IF_ERROR(CreateAvroParser(avro_parser, (*child).GetPrefix()));

    // Build a parser for the terminal node and attach it
    if ((*child).IsTerminal()) {

      AvroParserUniquePtr avro_value_parser(nullptr);
      const string name((*child).GetName(kSeparator));
      auto key_and_type = key_to_type_.find(name);
      if (key_and_type == key_to_type_.end()) {
        return Status(errors::NotFound("Unable to find key '", name, "'!"));
      }
      LOG(INFO) << "Create value parser for " << name;

      TF_RETURN_IF_ERROR(CreateValueParser(avro_value_parser, name, (*key_and_type).second));
      (*avro_parser).AddChild(std::move(avro_value_parser));

    // Build a parser for all children of this non-terminal node
    } else {
      LOG(INFO) << "Create parser for " << (*child).GetName('.');
      TF_RETURN_IF_ERROR(Build(avro_parser.get(), (*child).GetChildren()));
    }

    (*parent).AddChild(std::move(avro_parser));
  }
  return Status::OK();
}

Status AvroParserTree::GetUniqueKeys(std::unordered_set<string>* keys,
  const std::vector<std::pair<string, DataType>>& keys_and_types) {

  for (const auto& key_and_type : keys_and_types) {
    const string& key = key_and_type.first;
    auto inserted = (*keys).insert(key);
    if (!inserted.second) {
      return Status(errors::InvalidArgument("Found duplicate key ", key));
    }
  }
  return Status::OK();
}

Status AvroParserTree::BuildKeyWithInternalName(const std::vector<std::pair<string, DataType>>& keys_and_types) {
  for (const auto& key_and_type : keys_and_types) {
    string new_key(key_and_type.first);
    RE2::GlobalReplace(&new_key, RE2("\\["), ".[");
    // TODO: Check for nested filters and throw error, e.g. disallow [name[first=last]age=friends.age]
    // convert all array into array marker []
    //RE2::GlobalReplace(&new_key, RE2("\\[\\*\\]"), ".[]");
    // convert the index parser
    //RE2::GlobalReplace(&new_key, RE2("\\[(\\d+)\\]"), ".[].[\\1]");
    // add additional array expression for filters
    //RE2::GlobalReplace(&new_key, RE2("\\[([A-Za-z_].*)=(.*)\\]"), ".[].[\\1=\\2]");
    // add namespace
    keys_and_types_.push_back(std::make_pair(
      GetAvroNamespace() + kSeparator + new_key,
      key_and_type.second));
  }
  return Status::OK();
}

Status AvroParserTree::CreateAvroParser(AvroParserUniquePtr& avro_parser,
  const string& infix) const {

  if (IsArrayAll(infix)) {
    avro_parser.reset(new ArrayAllParser());
    return Status::OK();
  }

  int index;
  if (IsArrayIndex(&index, infix)) {
    avro_parser.reset(new ArrayIndexParser(index));
    return Status::OK();
  }

  string lhs;
  string rhs;
  string constant;
  // TODO(fraudies): Check that lhs and rhs are valid using the prefix tree and current node
  // TODO(fraudies): For lhs, rhs use fully qualified names and remove the default namespace
  if (IsFilter(&lhs, &rhs, infix)) {

    if (IsStringConstant(&constant, rhs)) {
      avro_parser.reset(new ArrayFilterParser(lhs, constant, kRhsIsConstant));
    } else {
      avro_parser.reset(new ArrayFilterParser(lhs, rhs, kRhsIsValue));
    }

    return Status::OK();
  }

  string key;
  if (IsMapKey(&key, infix)) {
    avro_parser.reset(new MapKeyParser(key));
    return Status::OK();
  }

  // TODO(fraudies): Check that the name appears in the prefix tree
  if (IsAttribute(infix)) {
    avro_parser.reset(new AttributeParser(infix));
    return Status::OK();
  }

  return Status(errors::InvalidArgument("Unable to match ", infix, " to valid internal avro parser"));
}

Status AvroParserTree::CreateValueParser(AvroParserUniquePtr& value_parser,
  const string& name, DataType data_type) const {

  string user_defined_name;
  TF_RETURN_IF_ERROR(ConvertToUserName(&user_defined_name, name));

  switch (data_type) {
    case DT_BOOL:
      value_parser.reset(new BoolValueParser(user_defined_name));
      break;
    case DT_INT32:
      value_parser.reset(new IntValueParser(user_defined_name));
      break;
    case DT_INT64:
      value_parser.reset(new LongValueParser(user_defined_name));
      break;
    case DT_FLOAT:
      value_parser.reset(new FloatValueParser(user_defined_name));
      break;
    case DT_DOUBLE:
      value_parser.reset(new DoubleValueParser(user_defined_name));
      break;
    case DT_STRING:
      value_parser.reset(new StringValueParser(user_defined_name));
      break;
    default:
      return Status(errors::Unimplemented("Unable to build avro value parser for name '",
                user_defined_name, "', because data type '", DataTypeString(data_type),
                "' is not supported!"));
  }
  return Status::OK();
}

Status AvroParserTree::ConvertToUserName(string* user_defined_name, const string& internal_name) const {
  // Remove the default namespace, if we added it
  if (IsDefaultNamespace()) {
    size_t pos = internal_name.find(kDefaultNamespace);
    if (pos != 0) {
      return Status(errors::InvalidArgument("Unable to find the default namespace '",
        kDefaultNamespace, "' at the start of '", internal_name));
    }
    *user_defined_name = internal_name.substr(pos+kDefaultNamespace.size()+1, string::npos);
  }
  // Remove the . before the [ if we there are any
  RE2::GlobalReplace(user_defined_name, RE2(".\\["), "[");

  return Status::OK();
}

inline bool AvroParserTree::ContainsFilter(string* lhs, string* rhs, const string& key) {
  return RE2::PartialMatch(key, "\\[([A-Za-z_].*)=(.*)\\]", lhs, rhs);
}

inline bool AvroParserTree::IsFilter(string* lhs, string* rhs, const string& key) {
  return RE2::FullMatch(key, "\\[([A-Za-z_].*)=(.*)\\]", lhs, rhs);
}

inline bool AvroParserTree::IsArrayAll(const string& infix) {
  return string("[*]").compare(infix) == 0;
}

inline bool AvroParserTree::IsArrayIndex(int* index, const string& infix) {
  return RE2::FullMatch(infix, "\\[(\\d+)\\]", index);
}

inline bool AvroParserTree::IsMapKey(string* key, const string& infix) {
  return RE2::FullMatch(infix, "\\['(\\S+)'\\]", key);
}

inline bool AvroParserTree::IsAttribute(const string& infix) {
  return RE2::FullMatch(infix, "[A-Za-z_][\\w]*");
}

inline bool AvroParserTree::IsStringConstant(string* constant, const string& infix) {
  return RE2::FullMatch(infix, "'(\\S+)'", constant);
}

Status AvroParserTree::AddBeginMarks(std::map<string, ValueStoreUniquePtr>* key_to_value) {
  for (auto const& key_value : *key_to_value) {
    (*key_value.second).BeginMark();
  }
  return Status::OK();
}

Status AvroParserTree::AddFinishMarks(std::map<string, ValueStoreUniquePtr>* key_to_value) {
  for (auto const& key_value : *key_to_value) {
    (*key_value.second).FinishMark();
  }
  return Status::OK();
}

Status AvroParserTree::InitValueBuffers(std::map<string, ValueStoreUniquePtr>* key_to_value) {
  // Remove all existing entries, because we'll create new ones
  (*key_to_value).clear();

  // For all keys and their data types add a buffer
  for (const auto& key_and_type : keys_and_types_) {
    const string& key = key_and_type.first;
    string user_defined_name;
    TF_RETURN_IF_ERROR(ConvertToUserName(&user_defined_name, key));

    DataType data_type = key_and_type.second;
    switch (data_type) {
      // Fill in the ValueBuffer
      case DT_BOOL:
        (*key_to_value).insert(
          std::make_pair(user_defined_name, std::unique_ptr<BoolValueBuffer>(new BoolValueBuffer())));
        break;
      case DT_INT32:
        (*key_to_value).insert(
          std::make_pair(user_defined_name, std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));
        break;
      case DT_INT64:
        (*key_to_value).insert(
          std::make_pair(user_defined_name, std::unique_ptr<LongValueBuffer>(new LongValueBuffer())));
        break;
      case DT_FLOAT:
        (*key_to_value).insert(
          std::make_pair(user_defined_name, std::unique_ptr<FloatValueBuffer>(new FloatValueBuffer())));
        break;
      case DT_DOUBLE:
        (*key_to_value).insert(
          std::make_pair(user_defined_name, std::unique_ptr<DoubleValueBuffer>(new DoubleValueBuffer())));
        break;
      case DT_STRING:
        (*key_to_value).insert(
          std::make_pair(user_defined_name, std::unique_ptr<StringValueBuffer>(new StringValueBuffer())));
        break;
      default:
        return Status(errors::Unimplemented("Unable to build value buffer for key '", user_defined_name,
          "', because data type '", DataTypeString(data_type), "' is not supported!"));
    }
  }
  return Status::OK();
}

const string AvroParserTree::kArrayAllElements = "[*]";
const string AvroParserTree::kDefaultNamespace = "default";

}
}
