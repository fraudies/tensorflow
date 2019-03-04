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
#ifndef TENSORFLOW_DATA_PREFIX_TREE_H_
#define TENSORFLOW_DATA_PREFIX_TREE_H_

#include <vector>
#include <string>
#include <memory>

namespace tensorflow {
namespace data {

class TreeNode {
public:
  TreeNode(const std::string& prefix = "", TreeNode* father = nullptr);
  virtual ~TreeNode();
  void GetChildren(std::vector<std::shared_ptr<TreeNode>>* children) const;
  void GetPrefix(std::string* prefix) const;
  // returns the full name using the separator
  void GetName(std::string* name, char separator) const;
  bool IsTerminal() const;
  // We define the prefix to exist if it is != "", which might be the case for the root
  bool HasPrefix() const;
  bool Find(std::shared_ptr<TreeNode>& child, const std::string& child_prefix) const; // true if found, otherwise false and child is not altered
  void FindOrAddChild(std::shared_ptr<TreeNode>& child, const std::string& child_prefix); // Child is ALWAYS assigned
private:
  std::string prefix_;
  TreeNode* father_; // Used to construct the full name
  std::vector<std::shared_ptr<TreeNode>> children_; // I use raw pointers because these are encapsulated and not exposed
};

// An ordered prefix tree maintains the order of it's children
// Note, that we leverage this property in the parser
class OrderedPrefixTree {
public:
  OrderedPrefixTree(const std::string& root_name = "");
  virtual ~OrderedPrefixTree();
  void GetRootPrefix(std::string* root_prefix) const;
  void Insert(const std::vector<std::string>& prefixes); // will try to insert if it exists won't change the tree
  // Assumes tree != nullptr
  static void Build(OrderedPrefixTree* tree, const std::vector<std::vector<std::string>>& prefixes_list);
  // Will return the node as far as the prefixes could be matched
  // Will only return true for a full match
  // If remaining is nullptr this method won't fill it
  bool Find(std::shared_ptr<TreeNode>& node, std::vector<std::string>* remaining,
    const std::vector<std::string>& prefixes) const;
  bool Find(std::shared_ptr<TreeNode>& node, const std::vector<std::string>& prefixes) const;
private:
  std::shared_ptr<TreeNode> root_;
};

}
}

#endif // TENSORFLOW_DATA_PREFIX_TREE_H_