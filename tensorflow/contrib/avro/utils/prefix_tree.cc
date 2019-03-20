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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/avro/utils/prefix_tree.h"

namespace tensorflow {
namespace data {

PrefixTreeNode::PrefixTreeNode(const std::string& prefix, PrefixTreeNode* father)
  : prefix_(prefix), father_(father) { }

PrefixTreeNode::~PrefixTreeNode() { }

std::vector<PrefixTreeNodeSharedPtr> PrefixTreeNode::GetChildren() const {
  return children_;
}

std::string PrefixTreeNode::GetPrefix() const {
  return prefix_;
}

std::string PrefixTreeNode::GetName(char separator) const {
  std::string name;
  name += prefix_;
  PrefixTreeNode* father = father_;
  while (father != nullptr) {
    name = father->prefix_ + separator + name;
    father = father->father_;
  }
  return name;
}

bool PrefixTreeNode::IsTerminal() const {
  return children_.size() == 0;
}

bool PrefixTreeNode::HasPrefix() const {
  return prefix_.size() > 0;
}

// TODO(fraudies): Could be optimized using a set instead of a std::vector--but note that we need
// the std::vector to preserve order
PrefixTreeNodeSharedPtr PrefixTreeNode::Find(const std::string& child_prefix) const {
  LOG(INFO) << "Find " << child_prefix;
  for (auto child : children_) {
    std::string prefix((*child).GetPrefix());
    LOG(INFO) << "Checking child: " << prefix;
    if (prefix.compare(child_prefix) == 0) {
      return child;
    }
  }
  return nullptr;
}

PrefixTreeNodeSharedPtr PrefixTreeNode::FindOrAddChild(const std::string& child_prefix) {
  // If we could not find it make it, add it, and assign it; otherwise we assigned it
  PrefixTreeNodeSharedPtr child(Find(child_prefix));
  if (child == nullptr) {
    // Note, the child we found will be the father
    children_.push_back(std::make_shared<PrefixTreeNode>(child_prefix, this));
    return children_.back();
  } else {
    return child;
  }
}

// -------------------------------------------------------------------------------------------------
// Ordered prefix tree
// -------------------------------------------------------------------------------------------------
OrderedPrefixTree::OrderedPrefixTree(const std::string& root_name)
  : root_(new PrefixTreeNode(root_name)) { }

std::string OrderedPrefixTree::GetRootPrefix() const {
  return (*root_).GetPrefix();
}

PrefixTreeNodeSharedPtr OrderedPrefixTree::GetRoot() const {
  return root_;
}

void OrderedPrefixTree::Insert(const std::vector<std::string>& prefixes) {
  PrefixTreeNodeSharedPtr node = root_;
  for (auto prefix = prefixes.begin(); prefix != prefixes.end(); ++prefix) {
    node = (*node).FindOrAddChild(*prefix);
  }
}

void OrderedPrefixTree::Build(OrderedPrefixTree* tree,
  const std::vector<std::vector<std::string>>& prefixes_list) {
  for (auto prefixes = prefixes_list.begin(); prefixes != prefixes_list.end(); ++prefixes) {
    (*tree).Insert(*prefixes);
  }
}

PrefixTreeNodeSharedPtr OrderedPrefixTree::FindNearest(std::vector<std::string>* remaining,
  const std::vector<std::string>& prefixes) const {
  // copy vector elements
  *remaining = prefixes;
  // If the root has a prefix
  auto prefix = (*remaining).begin();
  if ((*root_).HasPrefix()) {
    // If we dont have any prefixes then we can't match the root
    if (prefix == (*remaining).end()) {
      return nullptr;
    } else {
      std::string root_prefix(GetRootPrefix());
      // If the root prefix does not match we can't find the node
      if (root_prefix.compare(*prefix) != 0) {
        return nullptr;
      }
      // We matched
      prefix = (*remaining).erase(prefix);
    }
  }

  PrefixTreeNodeSharedPtr node(root_);
  PrefixTreeNodeSharedPtr next_node;
  // We don't have a root with prefix, then start with the children
  while (prefix != (*remaining).end() && (next_node = (*node).Find(*prefix))) {
    prefix = (*remaining).erase(prefix);
    node = next_node;
  }

  return node;
}

PrefixTreeNodeSharedPtr OrderedPrefixTree::Find(const std::vector<std::string>& prefixes) const {
  std::vector<std::string> remaining;
  PrefixTreeNodeSharedPtr node = FindNearest(&remaining, prefixes);
  return remaining.size() == 0 ? node : nullptr;
}

}
}