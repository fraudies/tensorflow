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

TreeNode::TreeNode(const std::string& prefix, TreeNode* father)
  : prefix_(prefix), father_(father) { }

TreeNode::~TreeNode() { }

void TreeNode::GetChildren(std::vector<std::shared_ptr<TreeNode>>* children) const {
  *children = children_;
}

void TreeNode::GetPrefix(std::string* prefix) const {
  *prefix = prefix_;
}

void TreeNode::GetName(std::string* name, char separator) const {
  *name += prefix_;
  TreeNode* father = father_;
  while (father != nullptr) {
    *name = father->prefix_ + separator + *name;
    father = father->father_;
  }
}

bool TreeNode::IsTerminal() const {
  return children_.size() == 0;
}

bool TreeNode::HasPrefix() const {
  return prefix_.size() > 0;
}

// TODO(fraudies): Could be optimized using a set instead of a std::vector--but note that we need
// the std::vector to preserve order
bool TreeNode::Find(std::shared_ptr<TreeNode>& child, const std::string& child_prefix) const {
  LOG(INFO) << "Find " << child_prefix;
  std::string prefix;
  for (auto child_ : children_) {
    (*child_).GetPrefix(&prefix);
    LOG(INFO) << "Checking child: " << prefix;
    if (prefix.compare(child_prefix) == 0) {
      child = child_;
      return true;
    }
  }
  return false;
}

void TreeNode::FindOrAddChild(std::shared_ptr<TreeNode>& child, const std::string& child_prefix) {
  // If we could not find it make it, add it, and assign it; otherwise we assigned it
  if (!Find(child, child_prefix)) {
    // Note, the child we found will be the father
    children_.push_back(std::make_shared<TreeNode>(child_prefix, this));
    child = children_.back();
  }
}

// -------------------------------------------------------------------------------------------------
// Ordered prefix tree
// -------------------------------------------------------------------------------------------------
OrderedPrefixTree::OrderedPrefixTree(const std::string& root_name)
  : root_(new TreeNode(root_name)) { }

OrderedPrefixTree::~OrderedPrefixTree() { }

void OrderedPrefixTree::GetRootPrefix(std::string* root_prefix) const {
  (*root_).GetPrefix(root_prefix);
}

void OrderedPrefixTree::Insert(const std::vector<std::string>& prefixes) {
  std::shared_ptr<TreeNode> node = root_;
  for (auto prefix = prefixes.begin(); prefix != prefixes.end(); ++prefix) {
    (*node).FindOrAddChild(node, *prefix);
  }
}

void OrderedPrefixTree::Build(OrderedPrefixTree* tree,
  const std::vector<std::vector<std::string>>& prefixes_list) {
  for (auto prefixes = prefixes_list.begin(); prefixes != prefixes_list.end(); ++prefixes) {
    (*tree).Insert(*prefixes);
  }
}

bool OrderedPrefixTree::Find(std::shared_ptr<TreeNode>& node,
  std::vector<std::string>* remaining,
  const std::vector<std::string>& prefixes) const {

  // If the root has a prefix
  auto prefix = prefixes.begin();
  if ((*root_).HasPrefix()) {
    // If we dont have any prefixes then we can't match the root
    if (prefix == prefixes.end()) {
      return false;
    } else {
      std::string root_prefix;
      GetRootPrefix(&root_prefix);
      // If the root prefix does not match we can't find the node
      if (root_prefix.compare(*prefix) != 0) {
        return false;
      }
      // We matched
      prefix++;
    }
  }

  node = root_;
  // We don't have a root with prefix, then start with the children
  while (prefix != prefixes.end() && (*node).Find(node, *prefix)) {
    prefix++;
  }

  // If we exhausted all prefixes we found the node
  if (prefix == prefixes.end()) {
    return true;
  }

  // We still have further prefixes
  if (remaining == nullptr) {
    return false;
  }
  do {
    (*remaining).push_back(*prefix);
  } while (++prefix != prefixes.end());

  return false;
}

bool OrderedPrefixTree::Find(std::shared_ptr<TreeNode>& node,
  const std::vector<std::string>& prefixes) const {
  return Find(node, nullptr, prefixes);
}

}
}