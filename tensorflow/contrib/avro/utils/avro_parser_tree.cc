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
#include "tensorflow/contrib/avro/utils/avro_parser_tree.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace data {

// RE2 https://github.com/google/re2/blob/master/re2/re2.h
/// str_util https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/strings/str_util.h

ParserTree::ParserTree() : root_(nullptr) {}
ParserTree::~ParserTree() { }

bool Re2Match(vector<string>* results, const string& pattern, const string& str) {
    RE2::Options opt;
    opt.set_log_errors(false);
    opt.set_case_sensitive(false);
    opt.set_utf8(false);
    RE2 regex("(" + pattern + ")", opt);
    if (!regex.ok()) { return false; }

    vector<RE2::Arg> args;
    vector<RE2::Arg*> args_pointers;
    size_t n_args = regex.NumberOfCapturingGroups();
    args.resize(n_args);
    args_pointers.resize(n_args);
    (*results).resize(n_args);
    // Wire up the pointers
    for (size_t i_arg = 0; i_arg < args_count; ++i_arg) {
        args[i] = &(*results)[i_arg];
        args_pointers[i] = &args[i_arg];
    }
    return RE2::FullMatchN(StringPiece(str), regex, args_pointers.data(), n_args);
}


Status ParserTree::Build(ParserTree* parser_tree, const vector<pair<string, DataType>>& keys_and_types) {
  // Get all the keys and check for uniqueness
  unordered_set<string> keys;
  TF_RETURN_IF_ERROR(GetUniqueKeys(&keys, keys_and_types));

  // Ensure that we have keys for lhs/rhs of all filters first
  // Note, some keys in all_keys may not be used for the output but by filters
  vector<string> all_keys; // this includes auxiliary keys and proper ordering
  string lhs;
  string rhs;
  for (auto key = keys.begin(); key != keys.end(); ++key) {
    if (IsFilter(&lhs, &rhs, *key)) {
      // Ensure to add the lhs
      size_t pos = key.find(lhs);
      string prefix = key.substr(0, pos-1);
      all_keys.push_back(prefix + lhs);
      keys.erase(prefix + lhs);

      // If the rhs is a constant we are done
      if (IsStringConstant(nullptr, rhs)) {
        continue;
      }

      // Check if the rhs is anchored to top
      if (str_util::StartsWith(rhs, "@")) {
        all_keys.push_back(rhs.substr(1));
        keys.erase(rhs.substr(1));
      } else {
        all_keys.push_back(prefix + rhs)
        keys.erase(prefix + rhs);
      }
    }
  }

  // Add all the remaining keys to the all_keys vector
  for (auto key = keys.begin(); key != keys.end(): ++key) {
    all_keys.push_back(*key);
  }

  // Parse keys into prefixes
  vector<vector<string>> prefixes;
  for (string& key : all_keys) {
    vector<string> key_prefixes;
    // https://stackoverflow.com/questions/35418597/split-string-on-the-dot-characters-that-are-not-inside-of-brackets
    // Split each along . or [ but not inside []
    if (Re2Match(&key_prefixes, key, R"([^.]*\[[^]]*\]|[^.]*)")) {
      return Status(error::InvalidArgument("Unable to parse key ", key));
    }
  }

  // Ensure the following
  PrefixTree prefix_tree;
  TF_RETURN_IF_ERROR(PrefixTree::Build(&prefix_tree, prefixes));
  TF_RETURN_IF_ERROR(InitValues(keys, expected_types));
  // Use the expected type to decide which value parser node to add
  root_.reset(new ValueParser(prefix_tree.GetRootPrefix()));
  vector<std::shared_ptr<TreeNode>> children;
  prefix_tree.GetChildren(&children);
  // Use the prefix tree to build the parser tree
  TF_RETURN_IF_ERROR(Build(root_.get(), children));
}

Status ParserTree::Build(ValueParser* father, const vector<std::shared_ptr<TreeNode>>& children) {
  for (std::shared_ptr<TreeNode> child : children) {
    std::unique_ptr<ValueParser> value_parser(nullptr);
    TF_RETURN_IF_ERROR(CreateValueParser(value_parser, (*child).GetPrefix()));
    (*father).children_.push_back(std::move(value_parser));
    if ((*child).IsTerminal()) {
      const string name;
      (*child).GetName(&name, '.');
      auto key_and_type = values_.find(name);
      if (key_and_type == values_.end()) {
        return Status(errors::NotFound("Could not find ", name, " as key"));
      }
      DataType data_type = key_and_type->second;
      std::unique_ptr<ValueParser> child_value_parser(nullptr);
      TF_RETURN_IF_ERROR(CreateValueParser(child_value_parser, data_type));
      value_parser.children_.push_back(std::move(child_value_parser));
    } else {
      vector<std::shared_ptr<TreeNode>> child_children;
      (*child).GetChildren(child_children);
      TF_RETURN_IF_ERROR(Build(value_parser, child_children));
    }
  }
  return Status::OK();
}

Status ParserTree::GetUniqueKeys(unordered_set<string>* keys,
  const vector<pair<string, DataType>>& keys_and_types) {

  for (auto& key_and_type : keys_and_types) {
    const string& key = key_and_type.first;
    auto inserted = (*keys).insert(key);
    if (!inserted.second) {
      return Status(error::InvalidArgument("Found duplicate key ", key));
    }
  }
  return Status::OK();
}

Status ParserTree::CreateValueParser(std::unique_ptr<ValueParser>& value_parser, const string& infix) {
  if (IsArrayAll(infix)) {
    value_parser.reset(std::make_unique<ArrayAllParser>());
    return Status::OK();
  }

  int index;
  if (IsArrayIndex(&index, infix)) {
    value_parser.reset(std::make_unique<ArrayIndexParser(index));
    return Status::OK();
  }

  string lhs;
  string rhs;
  // TODO(fraudies): Check that lhs and rhs are valid using the prefix tree and current note
  if (IsArrayFilter(&lhs, &rhs, infix)) {
    value_parser.reset(std::make_unique<ArrayFilterParser>(lhs, rhs));
    return Status::OK();
  }

  if (IsMapAll(infix)) {
    value_parser.reset(std::make_unique<MapAllParser>());
    return Status::OK();
  }

  string key;
  if (IsMapKey(&key, infix)) {
    value_parser.reset(std::make_unique<MapKeyParser>(key));
    return Status::OK();
  }

  string name;
  // TODO(fraudies): Check that the name appears in the prefix tree
  if (IsAttribute(&name, infix)) {
    value_parser.reset(std::make_unique<AttributeParser>(name));
    return Status::OK();
  }

  return Status(error::InvalidArgument("Unable to match ", infix, " to valid avro type"));
}

Status ParserTree::CreateValueParser(std::unique_ptr<ValueParser>& value_parser, DataType data_type) {
  switch (data_type) {
    case DT_BOOL:
      value_parser.reset(std::make_unique<BoolValueParser>());
      break;
    case DT_INT32:
      value_parser.reset(std::make_unique<IntValueParser>());
      break;
    case DT_INT64:
      return Status(errors::Unimplemented("Long value parser not supported yet"));
    case DT_FLOAT:
      return Status(errors::Unimplemented("Float value parser not supported yet"));
    case DT_DOUBLE:
      return Status(errors::Unimplemented("Double value parser not supported yet"));
    case DT_STRING:
      return Status(errors::Unimplemented("String value parser not supported yet"));
    default:
      return Status(errors::Unimplemented("Type ", data_type, " not supported!"));
  }
  return Status::OK();
}

inline bool ParserTree::IsArrayAll(const string& infix) {
  return "[*]".compare(infix) == 0;
}

inline bool ParserTree::IsArrayIndex(int* index, const string& infix) {
  return RE2::FullMatch(infix, R"[(\d+)]", index);
}

inline bool ParserTree::IsArrayFilter(string* lhs, string* rhs, const string& infix) {
  return RE2::FullMatch(infix, R"(([A-Za-z_][\w]*)=(\[\S+\]))", lhs, rhs);
}

inline bool ParserTree::IsMapAll(const string& infix) {
  return "[*]".compare(infix) == 0;
}

inline bool ParserTree::IsMapKey(string* key, const string& infix) {
  return RE2::FullMatch(infix, R"['(\S+)']", key);
}

inline bool ParserTree::IsAttribute(const string& infix) {
  return RE2::FullMatch(infix, R"([A-Za-z_][\w]*)");
}

inline bool ParserTree::IsStringConstant(string* constant, const string& infix) {
  return RE2::FullMatch(infix, R"'(\S+)'", constant);
}

Status ParserTree::InitValues(const vector<pair<string, DataType>& keys_and_types) {
  for (auto& key_and_type : keys_and_types) {
    const string& key = key_and_type.first;
    DataType data_type = key_and_type.second;
    switch (data_type) {
      // Fill in the correctly typed ValueBuffer
      case DT_BOOL:
        values_.insert(pair<string, std::unique_ptr<ValueBuffer>>(
          key, std::make_unique<BoolValueBuffer>());
        break;
      case DT_INT32:
        values_.insert(pair<string, std::unique_ptr<ValueBuffer>>(
          key, std::make_unique<IntValueBuffer>()));
        break;
      case DT_INT64:
        return Status(errors::Unimplemented("Long value buffer not supported yet"));
      case DT_FLOAT:
        return Status(errors::Unimplemented("Float value buffer not supported yet"));
      case DT_DOUBLE:
        return Status(errors::Unimplemented("Double value buffer not supported yet"));
      case DT_STRING:
        return Status(errors::Unimplemented("String value buffer not supported yet"));
      default:
        return Status(errors::Unimplemented("Type ", data_type, " not supported!"));
    }
  }
  return Status::OK();
}

}
}
