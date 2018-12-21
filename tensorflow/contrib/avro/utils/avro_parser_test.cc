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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/strings/str_util.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that. Instead these tests only cover the wrapping in the mem readers
namespace tensorflow {
namespace data {

TEST(RegexTest, SplitExpressions) {
  std::vector<string> expressions = {"name.first", "friends[2].name.first",
    "friends[*].name.first", "friends[*].address[*].street",
    "friends[*].job[*].coworker[*].name.first", "car['nickname'].color",
    "friends[gender='unknown'].name.first",
    "friends[name.first=name.last].name.initial"};
}

}
}