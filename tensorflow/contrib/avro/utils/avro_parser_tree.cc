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

// TODO(fraudies): Change log level from INFO to DEBUG for most items
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

Status AvroParserTree::Build(AvroParserTree* parser_tree, const string& avro_namespace,
    const std::vector<KeyWithType>& keys_and_types) {

  // Check unique keys
  LOG(INFO) << "Validate keys";
  TF_RETURN_IF_ERROR(ValidateUniqueKeys(keys_and_types));

  // TODO: Validate filters etc, no nesting, no name conflict in shorthand
  // TODO: Check for nested filters and throw error, e.g. disallow [name[first=last]age=friends.age]
  // TODO(fraudies): Add to validation that lhs/rhs being a constant won't be possible
  // lhs != rhs
  const string& resolved_avro_namespace = (*parser_tree).ResolveAndSetNamespace(avro_namespace);

  // Convert to internal names and order names to handle filters properly through parse order
  LOG(INFO) << "Order identifiers";
  std::vector<KeyWithType> ordered_keys_and_types = OrderAndResolveKeyTypes(keys_and_types);

  LOG(INFO) << "\nListing ordered identifiers";
  for (const KeyWithType& key_and_type : ordered_keys_and_types) {
    LOG(INFO) << "Name: '" << key_and_type.first << "' and type '" << DataTypeString(key_and_type.second) << "'";
  }

  // Parse keys into prefixes and build map from key to type
  std::vector< std::vector<string> > prefixes;
  for (const KeyWithType& key_and_type : ordered_keys_and_types) {

    // Built prefixes
    std::vector<string> key_prefixes = GetPartsWithoutAvroNamespace(key_and_type.first,
      resolved_avro_namespace);
    prefixes.push_back(key_prefixes);

    // Built map from key to type
    (*parser_tree).key_to_type_[key_and_type.first] = key_and_type.second;
  }

  LOG(INFO) << "Build prefix tree";
  OrderedPrefixTree prefix_tree(resolved_avro_namespace);
  OrderedPrefixTree::Build(&prefix_tree, prefixes);

  LOG(INFO) << "Prefix tree\n" << prefix_tree.ToString();

  (*parser_tree).keys_and_types_ = ordered_keys_and_types;

  // Use the expected type to decide which value parser node to add
  (*parser_tree).root_ = std::make_shared<NamespaceParser>(prefix_tree.GetRootPrefix());

  // Use the prefix tree to build the parser tree
  TF_RETURN_IF_ERROR((*parser_tree).Build(
    (*parser_tree).root_.get(),
    (*prefix_tree.GetRoot()).GetChildren()));

  LOG(INFO) << "Parser tree \n" << (*parser_tree).ToString();

  return Status::OK();
}

string AvroParserTree::ResolveAndSetNamespace(const string& avro_namespace) {
  if (avro_namespace.size() > 0) {
    avro_namespace_ = avro_namespace;
  } else {
    avro_namespace_ = kDefaultNamespace;
  }
  return avro_namespace_;
}

Status AvroParserTree::Build(AvroParser* parent, const std::vector<PrefixTreeNodeSharedPtr>& children) {
  for (PrefixTreeNodeSharedPtr child : children) {

    LOG(INFO) << "Creating parser for prefix " << (*child).GetPrefix();

    AvroParserUniquePtr avro_parser(nullptr);

    // Create a parser and add it to the parent
    const string& user_name(RemoveDotBeforeBracket(
      RemoveDefaultAvroNamespace((*child).GetName(kSeparator))));

    TF_RETURN_IF_ERROR(CreateAvroParser(avro_parser, (*child).GetPrefix(), user_name));

    // Build a parser for the terminal node and attach it
    if ((*child).IsTerminal()) {

      AvroParserUniquePtr avro_value_parser(nullptr);

      auto key_and_type = key_to_type_.find(user_name);
      if (key_and_type == key_to_type_.end()) {
        return errors::NotFound("Unable to find key '", user_name, "'!");
      }
      LOG(INFO) << "Create value parser for " << user_name;

      TF_RETURN_IF_ERROR(CreateValueParser(avro_value_parser, user_name, (*key_and_type).second));
      (*avro_parser).AddChild(std::move(avro_value_parser));

    // Build a parser for all children of this non-terminal node
    } else {
      LOG(INFO) << "Create parser for " << (*child).GetName(kSeparator);
      TF_RETURN_IF_ERROR(Build(avro_parser.get(), (*child).GetChildren()));
    }

    (*parent).AddChild(std::move(avro_parser));
  }
  return Status::OK();
}

Status AvroParserTree::ValidateUniqueKeys(
  const std::vector<KeyWithType>& keys_and_types) {

  std::unordered_set<string> unique_keys;

  for (const auto& key_and_type : keys_and_types) {
    const string& key = key_and_type.first;
    auto inserted = unique_keys.insert(key);
    if (!inserted.second) {
      return errors::InvalidArgument("Found duplicate key ", key);
    }
  }

  return Status::OK();
}

struct KeyWithTypeHash {
  std::size_t operator()(const KeyWithType& key_type) const {
    size_t const h1 ( std::hash<string>{}(key_type.first) );
    size_t const h2 ( std::hash<int>{}(key_type.second) );
    return h1 ^ (h2 << 1); // see boost hash_combine
  }
};

// Ensure that we have keys for lhs/rhs of all filters first
// Note, some keys in ordered may not be used for the output but by filter expressions
// Note, if we have duplicate keys in filters we require uniqueness and we need order

// TODO: Need to enhance the ordering constraint, the below algorithm won't do
// e.g.
// "friends[@name.first=name.first].name.initial" will result in
// 1. friends[*].name.first                         \  These are grouped by the pre-fix tree
// 2. friends[@name.first=name.first].name.initial  /
// 3. name.first
// but the proper order would be
// 1. name.first
// 2. friends[*].name.first
// 3. friends[@name.first=name.first].name.initial
std::vector<KeyWithType> AvroParserTree::OrderAndResolveKeyTypes(
  const std::vector<KeyWithType>& keys_and_types) {

  std::unordered_set<KeyWithType, KeyWithTypeHash> key_types(
    keys_and_types.begin(), keys_and_types.end());
  UniqueVector<KeyWithType> ordered;

  string lhs_name;
  string rhs_name;

  for (const KeyWithType& key_type : key_types) {

    const string& user_name = key_type.first;
    DataType data_type = key_type.second;

    if (ContainsFilter(&lhs_name, &rhs_name, user_name)) {

      const string& filter_name = lhs_name + "=" + rhs_name;

      LOG(INFO) << "Found filter with lhs '" << lhs_name << "' and rhs '" << rhs_name < "'";

      if (!IsStringConstant(nullptr, lhs_name)) {
        const string& lhs_resolved_name = ResolveFilterName(user_name, lhs_name, filter_name);

        LOG(INFO) << "  Resolved lhs " << lhs_resolved_name;

        const KeyWithType& lhs_resolved = std::make_pair(lhs_resolved_name, DT_STRING);

        ordered.Prepend(lhs_resolved);
        key_types.erase(lhs_resolved);
      }

      if (!IsStringConstant(nullptr, rhs_name)) {
        const string& rhs_resolved_name = ResolveFilterName(user_name, rhs_name, filter_name);

        LOG(INFO) << "  Resolved rhs " << rhs_resolved_name;

        const KeyWithType& rhs_resolved = std::make_pair(rhs_resolved_name, DT_STRING);

        ordered.Prepend(rhs_resolved);
        key_types.erase(rhs_resolved);
      }
    }
    ordered.Append(key_type);
  }

  return ordered.GetOrdered();
}

Status AvroParserTree::CreateAvroParser(AvroParserUniquePtr& avro_parser,
  const string& infix, const string& user_name) const {

  if (IsArrayAll(infix)) {
    avro_parser.reset(new ArrayAllParser());
    return Status::OK();
  }

  int index;
  if (IsArrayIndex(&index, infix)) {
    avro_parser.reset(new ArrayIndexParser(index));
    return Status::OK();
  }

  string lhs_name;
  string rhs_name;
  string lhs_resolved_name;
  string rhs_resolved_name;

  if (IsFilter(&lhs_name, &rhs_name, infix)) {

    LOG(INFO) << "Infix " << infix << " lhs " << lhs_name << " and rhs " << rhs_name;

    bool lhs_is_constant = IsStringConstant(&lhs_resolved_name, lhs_name);
    bool rhs_is_constant = IsStringConstant(&rhs_resolved_name, rhs_name);
    const string& filter_name = lhs_name + "=" + rhs_name;

    // If the lhs is not a constant, then find the resolved name from the user name
    if (!lhs_is_constant) {
      lhs_resolved_name = ResolveFilterName(user_name, lhs_name, filter_name);
    }

    // If the rhs is not a constant, then find the resolved name form the user name
    if (!rhs_is_constant) {
      rhs_resolved_name = ResolveFilterName(user_name, rhs_name, filter_name);
    }

    ArrayFilterType array_filter_type = ArrayFilterParser::ToArrayFilterType(lhs_is_constant, rhs_is_constant);

    avro_parser.reset(new ArrayFilterParser(lhs_resolved_name, rhs_resolved_name, array_filter_type));

    return Status::OK();
  }

  string key;
  if (IsMapKey(&key, infix)) {
    avro_parser.reset(new MapKeyParser(key));
    return Status::OK();
  }

  // TODO(fraudies): Check that the name appears in the prefix tree
  if (IsAttribute(infix)) {
    avro_parser.reset(new RecordParser(infix));
    return Status::OK();
  }

  if (IsBranch(infix)) {
    avro_parser.reset(new UnionParser(user_name));
    return Status::OK();
  }

  return Status(errors::InvalidArgument("Unable to match ", infix, " to valid internal avro parser"));
}

// Need to use user user_name here to place begin/finish marks properly
Status AvroParserTree::CreateValueParser(AvroParserUniquePtr& value_parser,
  const string& user_name, DataType data_type) const {

  switch (data_type) {
    case DT_BOOL:
      value_parser.reset(new BoolValueParser(user_name));
      break;
    case DT_INT32:
      value_parser.reset(new IntValueParser(user_name));
      break;
    case DT_INT64:
      value_parser.reset(new LongValueParser(user_name));
      break;
    case DT_FLOAT:
      value_parser.reset(new FloatValueParser(user_name));
      break;
    case DT_DOUBLE:
      value_parser.reset(new DoubleValueParser(user_name));
      break;
    case DT_STRING:
      value_parser.reset(new StringOrBytesValueParser(user_name));
      break;
    default:
      return errors::Unimplemented("Unable to build avro value parser for name '",
                user_name, "', because data type '", DataTypeString(data_type),
                "' is not supported!");
  }
  return Status::OK();
}

bool AvroParserTree::ContainsFilter(string* lhs_name, string* rhs_name, const string& name) {
  return RE2::PartialMatch(name, "\\[(['@A-Za-z_]['\\.\\w]*)=(['@A-Za-z_]['\\.\\w]*)\\]", lhs_name, rhs_name);
}

inline bool AvroParserTree::IsFilter(string* lhs_name, string* rhs_name, const string& name) {
  return RE2::FullMatch(name, "\\[(['@A-Za-z_]['\\.\\w]*)=(['@A-Za-z_]['\\.\\w]*)\\]", lhs_name, rhs_name);
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

inline bool AvroParserTree::IsBranch(const string& infix) {
  return RE2::FullMatch(infix, ":int|:long|:float|:double|:string|:boolean");
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


  // For all keys -- that hold the user defined name -- and their data types add a buffer
  for (const auto& key_and_type : keys_and_types_) {

    const string& key = key_and_type.first;
    DataType data_type = key_and_type.second;

    switch (data_type) {
      // Fill in the ValueBuffer
      case DT_BOOL:
        (*key_to_value).insert(
          std::make_pair(key, std::unique_ptr<BoolValueBuffer>(new BoolValueBuffer())));
        break;
      case DT_INT32:
        (*key_to_value).insert(
          std::make_pair(key, std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));
        break;
      case DT_INT64:
        (*key_to_value).insert(
          std::make_pair(key, std::unique_ptr<LongValueBuffer>(new LongValueBuffer())));
        break;
      case DT_FLOAT:
        (*key_to_value).insert(
          std::make_pair(key, std::unique_ptr<FloatValueBuffer>(new FloatValueBuffer())));
        break;
      case DT_DOUBLE:
        (*key_to_value).insert(
          std::make_pair(key, std::unique_ptr<DoubleValueBuffer>(new DoubleValueBuffer())));
        break;
      case DT_STRING:
        (*key_to_value).insert(
          std::make_pair(key, std::unique_ptr<StringValueBuffer>(new StringValueBuffer())));
        break;
      default:
        return Status(errors::Unimplemented("Unable to build value buffer for key '", key,
          "', because data type '", DataTypeString(data_type), "' is not supported!"));
    }
  }
  return Status::OK();
}

// TODO(fraudies): Replace by regex once tensorflow is compiled with GCC > 4.8
std::vector<string> SplitOnDelimiterButNotInsideSquareBrackets(const string& str, char delimiter) {
  std::vector<string> results;
  // check that delimiter is not [ or ]
  string lastMatch = "";
  int nBrackets = 0;
  for (char c : str) {
    nBrackets += (c == '[');
    nBrackets -= (c == ']');
    if (nBrackets == 0 && c == delimiter) {
      results.push_back(lastMatch);
      lastMatch = "";
    } else {
      lastMatch += c;
    }
  }
  results.push_back(lastMatch);
  return results;
}

string AvroParserTree::ResolveFilterName(const string& user_name, const string& filter_side_name,
  const string& filter_name) {

  if (str_util::StartsWith(filter_side_name, "@")) {
    LOG(INFO) << "Resolve absolute filter name: " << filter_side_name;

    return filter_side_name.substr(1);
  } else {
    size_t pos = user_name.find(filter_name);
    return user_name.substr(0, pos-1) + kArrayAllElements + kSeparator + filter_side_name;
  }
}

std::vector<string> AvroParserTree::GetPartsWithoutAvroNamespace(const string& user_name,
  const string& avro_namespace) {

  string name = user_name;
  if (str_util::StartsWith(name, avro_namespace)) {
    name = name.substr(avro_namespace.size() + 1, string::npos); // +1 to remove separator
  }
  RE2::GlobalReplace(&name, RE2("\\["), ".["); // [ -> .[
  RE2::GlobalReplace(&name, RE2(":"), ".:"); // : -> .:
  std::vector<string> parts = SplitOnDelimiterButNotInsideSquareBrackets(name, kSeparator);
  return parts;
}

// Will remove the default avro namespace if it exists; otherwise the value won't change
string AvroParserTree::RemoveDefaultAvroNamespace(const string& name) {
  if (str_util::StartsWith(name, kDefaultNamespace)) {
    return name.substr(kDefaultNamespace.size() + 1, string::npos); // +1 to remove separator
  } else {
    return name;
  }
}

string AvroParserTree::RemoveDotBeforeBracket(const string& name) {
  string removed(name);
  RE2::GlobalReplace(&removed, RE2("\\.\\["), "[");
  RE2::GlobalReplace(&removed, RE2("\\.:"), ":");
  return removed;
}

const string AvroParserTree::kArrayAllElements = "[*]";
const string AvroParserTree::kDefaultNamespace = "default";

}
}
