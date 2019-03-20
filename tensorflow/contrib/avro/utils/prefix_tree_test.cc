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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/contrib/avro/utils/prefix_tree.h"

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// Tests for a tree node
// ------------------------------------------------------------
TEST(PrefixTreeNodeTest, IsTerminal) {
  PrefixTreeNode node("father");
  EXPECT_TRUE(node.IsTerminal());
  std::shared_ptr<PrefixTreeNode> child;
  node.FindOrAddChild(child, "child");
  EXPECT_TRUE(!node.IsTerminal());
}

TEST(PrefixTreeNodeTest, HasPrefix) {
  PrefixTreeNode wout;
  EXPECT_TRUE(!wout.HasPrefix());
  PrefixTreeNode with("name");
  EXPECT_TRUE(with.HasPrefix());
}

TEST(PrefixTreeNodeTest, GetPrefix) {
  PrefixTreeNode node("name");
  string prefix;
  node.GetPrefix(&prefix);
  EXPECT_EQ(prefix, "name");
}

// Tests: Find, FindOrAdd, GetPrefix
TEST(PrefixTreeNodeTest, SingleChild) {
  PrefixTreeNode node("father");
  std::shared_ptr<PrefixTreeNode> child;
  // Expect the child does not exist
  EXPECT_TRUE(!node.Find(child, "child"));
  // Insert the child
  node.FindOrAddChild(child, "child");
  // Child must be present now
  EXPECT_TRUE(node.Find(child, "child"));
  string prefix;
  (*child).GetPrefix(&prefix);
  EXPECT_EQ(prefix, "child");
  // Check the name
  string name;
  (*child).GetName(&name, '.');
  EXPECT_EQ(name, "father.child");
}

TEST(PrefixTreeNodeTest, GetChildren) {
  PrefixTreeNode node("father");
  std::shared_ptr<PrefixTreeNode> child;
  node.FindOrAddChild(child, "child1");
  node.FindOrAddChild(child, "child2");
  node.FindOrAddChild(child, "child3");
  std::vector<std::shared_ptr<PrefixTreeNode>> children;
  std::vector< std::string > names{"child1", "child2", "child3"};
  node.GetChildren(&children);
  int n_child = 3;
  EXPECT_EQ(children.size(), n_child);
  string prefix;
  for (int i_child = 0; i_child < n_child; ++i_child) {
    (*children[i_child]).GetPrefix(&prefix);
    EXPECT_EQ(prefix, names[i_child]);
  }
}


// ------------------------------------------------------------
// Tests for an ordered prefix tree
// ------------------------------------------------------------
TEST(OrderedPrefixTree, GetRootPrefix) {
  string root_prefix;

  OrderedPrefixTree wout;
  wout.GetRootPrefix(&root_prefix);
  EXPECT_EQ(root_prefix, "");

  OrderedPrefixTree with("namespace");
  with.GetRootPrefix(&root_prefix);
  EXPECT_EQ(root_prefix, "namespace");
}

TEST(OrderedPrefixTree, BuildEmpty) {
  std::vector< std::vector<std::string> > prefixes_list;
  OrderedPrefixTree tree;
  OrderedPrefixTree::Build(&tree, prefixes_list);
}

TEST(OrderedPrefixTree, BuildSmall) {
  std::vector< std::vector<std::string> > prefixes_list{{"com"}};
  std::shared_ptr<PrefixTreeNode> node;
  std::vector< std::string > present{"com"};
  std::vector< std::string > absent{"nothing"};
  string prefix;
  OrderedPrefixTree tree;
  OrderedPrefixTree::Build(&tree, prefixes_list);

  // Check for present prefixes
  EXPECT_TRUE(tree.Find(node, present));
  (*node).GetPrefix(&prefix);
  EXPECT_EQ(prefix, "com");

  // Check for absent prefixes
  EXPECT_TRUE(!tree.Find(node, absent));
}

TEST(OrderedPrefixTree, BuildLarge) {
  std::vector< std::vector<std::string> > prefixes_list{{"com", "google", "search"},
    {"com", "linkedin", "jobs"}, {"com", "linkedin", "members"}};
  std::shared_ptr<PrefixTreeNode> node;
  std::vector< std::string > present_with_remaining{"com", "google", "search", "cloud"};
  std::vector< std::string > present_partial_match{"com", "google"};
  std::vector< std::string > present_full_match{"com", "linkedin", "members"};
  std::vector< std::string > absent{"com", "linkedin", "members", "us"};
  OrderedPrefixTree tree;
  OrderedPrefixTree::Build(&tree, prefixes_list);

  // Check for present prefixes
  EXPECT_TRUE(tree.Find(node, present_partial_match));
  EXPECT_TRUE(tree.Find(node, present_full_match));

  // Check for absent prefixes
  EXPECT_TRUE(!tree.Find(node, absent));

  // Check that the partial match returns the right remaining
  std::vector< std::string > remaining;
  // A partial match returns false and the remaining part matches cloud
  EXPECT_TRUE(!tree.Find(node, &remaining, present_with_remaining));
  EXPECT_EQ(remaining.size(), 1);
  EXPECT_EQ(remaining.front(), "cloud");
}

}
}