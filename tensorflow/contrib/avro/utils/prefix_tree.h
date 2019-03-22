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

class PrefixTreeNode; // forward declare for pointer definition

using PrefixTreeNodeSharedPtr = std::shared_ptr<PrefixTreeNode>;


class PrefixTreeNode {
public:
  PrefixTreeNode(const std::string& prefix = "", PrefixTreeNode* father = nullptr);
  virtual ~PrefixTreeNode();
  // TODO(fraudies): For better performance convert this into an iterator
  std::vector<PrefixTreeNodeSharedPtr> GetChildren() const;
  std::string GetPrefix() const;
  // returns the full name using the separator
  std::string GetName(char separator) const;
  bool IsTerminal() const;
  // We define the prefix to exist if it is != "", which might be the case for the root
  bool HasPrefix() const;
  PrefixTreeNodeSharedPtr Find(const std::string& child_prefix) const; // true if found, otherwise false and child is not altered
  PrefixTreeNodeSharedPtr FindOrAddChild(const std::string& child_prefix); // Child is ALWAYS assigned
  string ToString(int level) const;
private:
  std::string prefix_;
  PrefixTreeNode* father_; // Used to construct the full name
  std::vector<PrefixTreeNodeSharedPtr> children_; // I use raw pointers because these are encapsulated and not exposed
};

// An ordered prefix tree maintains the order of it's children
// Note, that we leverage this property in the parser
class OrderedPrefixTree {
public:
  OrderedPrefixTree(const std::string& root_name = "");
  std::string GetRootPrefix() const;
  PrefixTreeNodeSharedPtr GetRoot() const;
  void Insert(const std::vector<std::string>& prefixes); // will try to insert if it exists won't change the tree

  // Assumes tree != nullptr
  static void Build(OrderedPrefixTree* tree, const std::vector<std::vector<std::string>>& prefixes_list);

  // Will return the node as far as the prefixes could be matched and put the unmatched part in remaining
  PrefixTreeNodeSharedPtr FindNearest(std::vector<std::string>* remaining, const std::vector<std::string>& prefixes) const;

  // Returns tree node if found otherwise nullptr
  PrefixTreeNodeSharedPtr Find(const std::vector<std::string>& prefixes) const;

  // Return a the ordered prefix tree in a human readable format
  string ToString() const;
private:
  PrefixTreeNodeSharedPtr root_;
};

}
}

#endif // TENSORFLOW_DATA_PREFIX_TREE_H_