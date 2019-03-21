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
#include "re2/re2.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/contrib/avro/utils/avro_parser_tree.h"

namespace tensorflow {
namespace data {

// RE2 https://github.com/google/re2/blob/master/re2/re2.h
// str_util https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/strings/str_util.h
bool Re2Match(std::vector<string>* results, const string& pattern, const string& str) {
    RE2::Options opt;
    opt.set_log_errors(false);
    opt.set_case_sensitive(false);
    opt.set_utf8(false);
    RE2 regex("(" + pattern + ")", opt);
    if (!regex.ok()) { return false; }

    std::vector<RE2::Arg> args;
    std::vector<RE2::Arg*> args_pointers;
    size_t n_args = regex.NumberOfCapturingGroups();
    args.resize(n_args);
    args_pointers.resize(n_args);
    (*results).resize(n_args);
    // Wire up the pointers
    for (size_t i_arg = 0; i_arg < n_args; ++i_arg) {
        args[i_arg] = &(*results)[i_arg];
        args_pointers[i_arg] = &args[i_arg];
    }
    return RE2::FullMatchN(str, regex, args_pointers.data(), n_args);
}

AvroParserTree::AvroParserTree() : root_(nullptr) { }
AvroParserTree::~AvroParserTree() { }

Status AvroParserTree::ParseValue(std::vector<ValueStoreUniquePtr>* values, const AvroValueSharedPtr& value) {
  // will also be used to get the data type for the node
  std::map<string, ValueStoreUniquePtr> key_to_value;
  TF_RETURN_IF_ERROR(InitValueBuffers(&key_to_value));
  // Note, avro_value_t* will be valid as long as value is
  std::queue<std::pair<AvroParserSharedPtr, AvroValueSharedPtr> > parser_for_value;
  parser_for_value.push(std::make_pair(root_, value));
  // TODO(fraudies): Have threads working on the queue as performance optimization
  // Note, only parallelize all expression parsing between two filters in the queue
  while (!parser_for_value.empty()) {
    auto current = parser_for_value.front();
    AvroParserSharedPtr p = current.first;
    AvroValueSharedPtr v = current.second;
    parser_for_value.pop();
    if ((*p).IsTerminal()) {
      TF_RETURN_IF_ERROR((*p).ParseValue(&key_to_value, *v));
    //} else if ((*p).UsesParsedValues()) {  // This is the boundary for threading
    //  TF_RETURN_IF_ERROR((*p).ResolveValues(&parser_for_value, *v, *values));
    } else {
      TF_RETURN_IF_ERROR((*p).ResolveValues(&parser_for_value, *v, key_to_value));
    }
  }

  for (const auto& key_type : keys_and_types_) {
    (*values).push_back(std::move(key_to_value[key_type.first]));
  }
}

Status AvroParserTree::Build(AvroParserTree* parser_tree,
  const std::vector<std::pair<string, DataType> >& keys_and_types) {

  // Get all the keys and check for uniqueness
  std::unordered_set<string> keys;
  TF_RETURN_IF_ERROR(GetUniqueKeys(&keys, keys_and_types));

  // Ensure that we have keys for lhs/rhs of all filters first
  // Note, some keys in all_keys may not be used for the output but by filter expressions
  std::vector<string> all_keys; // holds all keys in proper order
  string lhs;
  string rhs;
  for (const string& key : keys) {
    if (IsFilter(&lhs, &rhs, key)) {
      // Ensure to add the lhs
      size_t pos = key.find(lhs);
      string prefix = key.substr(0, pos-1);
      all_keys.push_back(prefix + lhs);
      keys.erase(prefix + lhs);

      // If the rhs is a constant we are done
      if (IsStringConstant(nullptr, rhs)) {
        continue;
      }

      // Check if the rhs is anchored to the top
      if (str_util::StartsWith(rhs, "@")) {
        all_keys.push_back(rhs.substr(1));
        keys.erase(rhs.substr(1));
      } else {
        all_keys.push_back(prefix + rhs);
        keys.erase(prefix + rhs);
      }
    }
  }

  // Add all the remaining keys to the all_keys vector
  for (const string& key : keys) {
    all_keys.push_back(key);
  }

  // Parse keys into prefixes
  std::vector< std::vector<string> > prefixes;
  for (const string& key : all_keys) {
    std::vector<string> key_prefixes;
    // https://stackoverflow.com/questions/35418597/split-string-on-the-dot-characters-that-are-not-inside-of-brackets
    // Pattern match for [] or . but no . inside []
    if (Re2Match(&key_prefixes, "(^\\[.*?\\]|[^.]*)", key)) {
      return Status(errors::InvalidArgument("Unable to parse key ", key));
    }
  }
  OrderedPrefixTree prefix_tree;
  OrderedPrefixTree::Build(&prefix_tree, prefixes);

  // Use the expected type to decide which value parser node to add
  (*parser_tree).root_ = std::make_shared<NamespaceParser>(prefix_tree.GetRootPrefix());

  // Use the prefix tree to build the parser tree
  TF_RETURN_IF_ERROR((*parser_tree).Build(
    (*parser_tree).root_.get(),
    (*prefix_tree.GetRoot()).GetChildren()));

  // Set keys and types using the keys
  (*parser_tree).keys_and_types_ = keys_and_types;
  // Set key to type for fast type access based on key
  for (const auto& key_and_type : keys_and_types) {
    (*parser_tree).key_to_type_[key_and_type.first] = key_and_type.second;
  }
}

Status AvroParserTree::Build(AvroParser* parent, const std::vector<PrefixTreeNodeSharedPtr>& children) {
  for (PrefixTreeNodeSharedPtr child : children) {
    AvroParserUniquePtr avro_parser(nullptr);
    TF_RETURN_IF_ERROR(CreateAvroParser(avro_parser, (*child).GetPrefix()));
    (*parent).AddChild(std::move(avro_parser));
    if ((*child).IsTerminal()) {
      const string name((*child).GetName('.'));
      auto key_and_type = key_to_type_.find(name);
      if (key_and_type == key_to_type_.end()) {
        return Status(errors::NotFound("Could not find ", name, " as key"));
      }
      AvroParserUniquePtr child_parser(nullptr);
      TF_RETURN_IF_ERROR(CreateValueParser(child_parser, name, (*key_and_type).second));
    } else {
      TF_RETURN_IF_ERROR(Build(avro_parser.get(), (*child).GetChildren()));
    }
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

Status AvroParserTree::CreateAvroParser(AvroParserUniquePtr& avro_parser,
  const string& infix) {

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
  const string& name, DataType data_type) {

  switch (data_type) {
    case DT_BOOL:
      value_parser.reset(new BoolValueParser(name));
      break;
    case DT_INT32:
      value_parser.reset(new IntValueParser(name));
      break;
    case DT_INT64:
      return Status(errors::Unimplemented("Long value parser not supported yet"));
    case DT_FLOAT:
      return Status(errors::Unimplemented("Float value parser not supported yet"));
    case DT_DOUBLE:
      return Status(errors::Unimplemented("Double value parser not supported yet"));
    case DT_STRING:
      value_parser.reset(new StringValueParser(name));
    default:
      return Status(errors::Unimplemented("Type ", data_type, " not supported!"));
  }
  return Status::OK();
}

inline bool AvroParserTree::IsFilter(string* lhs, string* rhs, const string& key) {
  return RE2::FullMatch(key, "(^\\[([A-Za-z_][\\w]*)=(\\[\\S+\\])\\]$)", lhs, rhs);
}

inline bool AvroParserTree::IsArrayAll(const string& infix) {
  return string("[*]").compare(infix) == 0;
}

inline bool AvroParserTree::IsArrayIndex(int* index, const string& infix) {
  return RE2::FullMatch(infix, "[(\\d+)]", index);
}

inline bool AvroParserTree::IsMapKey(string* key, const string& infix) {
  return RE2::FullMatch(infix, "['(\\S+)']", key);
}

inline bool AvroParserTree::IsAttribute(const string& infix) {
  return RE2::FullMatch(infix, "([A-Za-z_][\\w]*)");
}

inline bool AvroParserTree::IsStringConstant(string* constant, const string& infix) {
  return RE2::FullMatch(infix, "'(\\S+)'", constant);
}

Status AvroParserTree::InitValueBuffers(std::map<string, ValueStoreUniquePtr>* key_to_value) {
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
        return Status(errors::Unimplemented("Long value buffer not supported yet"));
      case DT_FLOAT:
        return Status(errors::Unimplemented("Float value buffer not supported yet"));
      case DT_DOUBLE:
        return Status(errors::Unimplemented("Double value buffer not supported yet"));
      case DT_STRING:
        (*key_to_value).insert(
          std::make_pair(key, std::unique_ptr<StringValueBuffer>(new StringValueBuffer())));
      default:
        return Status(errors::Unimplemented("Type ", data_type, " not supported!"));
    }
  }
  return Status::OK();
}

}
}
