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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/framework/tensor_testutil.h" // tensor equals
#include "tensorflow/contrib/avro/utils/value_buffer.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that.
namespace tensorflow {
namespace data {


// ------------------------------------------------------------
// Test shape builder
// ------------------------------------------------------------
TEST(ShapeBuilderTest, ShapeBuilderEmpty) {
  ShapeBuilder builder;
  size_t dim;
  builder.GetNumberOfDimensions(&dim);
  EXPECT_TRUE(dim == 0);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 0);

  EXPECT_TRUE(builder.HasAllElements(shape));
}

TEST(ShapeBuilderTest, ShapeBuilderSingleDimension) {
  ShapeBuilder builder;
  builder.BeginMark(); builder.Increment(); builder.Increment(); builder.FinishMark();
  builder.BeginMark(); builder.Increment(); builder.FinishMark();

  size_t dim;
  builder.GetNumberOfDimensions(&dim);
  EXPECT_EQ(dim, 1);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 1);
  EXPECT_EQ(shape.dim_size(0), 2);

  EXPECT_TRUE(!builder.HasAllElements(shape));
}

TEST(ShapeBuilderTest, ShapeBuilderTwoDimensions) {
  ShapeBuilder builder;
  builder.BeginMark();
    builder.BeginMark(); builder.Increment(); builder.FinishMark();
    builder.BeginMark(); builder.FinishMark();
  builder.FinishMark();

  size_t dim;
  builder.GetNumberOfDimensions(&dim);
  EXPECT_EQ(dim, 2);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 2);
  EXPECT_EQ(shape.dim_size(0), 2);
  EXPECT_EQ(shape.dim_size(1), 1);

  EXPECT_TRUE(!builder.HasAllElements(shape));
}

TEST(ShapeBuilderTest, ShapeBuilderManyDimensions) {
  ShapeBuilder builder;
  builder.BeginMark();
    builder.BeginMark();
      builder.BeginMark();
        builder.Increment();
      builder.FinishMark();
      builder.BeginMark();
        builder.Increment();
      builder.FinishMark();
    builder.FinishMark();
    builder.BeginMark();
      builder.BeginMark();
        builder.Increment();
      builder.FinishMark();
    builder.FinishMark();
    builder.BeginMark();
      builder.BeginMark();
      builder.FinishMark();
    builder.FinishMark();
  builder.FinishMark();

  size_t dim;
  builder.GetNumberOfDimensions(&dim);
  EXPECT_EQ(dim, 3);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 3);
  EXPECT_EQ(shape.dim_size(0), 3);
  EXPECT_EQ(shape.dim_size(1), 2);
  EXPECT_EQ(shape.dim_size(2), 1);

  EXPECT_TRUE(!builder.HasAllElements(shape));
}

// ------------------------------------------------------------
// Test Value Buffer -- Simple corner cases
// ------------------------------------------------------------
TEST(ValueBufferTest, BufferCreateAndDestroy) {
  IntValueBuffer buffer;
}


TEST(ValueBufferTest, DenseTensorForEmptyBuffer) {
  IntValueBuffer buffer;
  const TensorShape shape({0});
  Tensor tensor(DT_INT32, shape);
  Tensor defaults;
  TF_EXPECT_OK(buffer.MakeDense(&tensor, shape, defaults));
}

// ------------------------------------------------------------
// Test Value buffer -- Dense
// ------------------------------------------------------------
TEST(ValueBufferTest, Dense1DWithTensorDefault) {
  // Define the default tensor
  const TensorShape shape({3});
  Tensor defaults(DT_INT32, shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 11;
  defaults_flat(1) = 12;
  defaults_flat(2) = 13;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
  buffer.Add(0);
  buffer.Add(1);
  buffer.FinishMark();

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, shape);
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  // Create expected tensor and compare against the is tensor
  Tensor tensor_expected(DT_INT32, shape);
  auto tensor_expected_flat = tensor_expected.flat<int>();
  tensor_expected_flat(0) = 0;
  tensor_expected_flat(1) = 1;
  tensor_expected_flat(2) = 13;
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

TEST(ValueBufferTest, Dense1DWithScalarDefault) {
  // Define the default tensor
  const TensorShape defaults_shape({1});
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 100;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
  buffer.Add(1);
  buffer.FinishMark();

  // Make the tensor from buffer
  const TensorShape shape({3});
  Tensor tensor_actual(DT_INT32, shape);
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  // Create expected tensor and compare against the is tensor
  Tensor tensor_expected(DT_INT32, shape);
  auto tensor_expected_flat = tensor_expected.flat<int>();
  tensor_expected_flat(0) = 1;
  tensor_expected_flat(1) = 100;
  tensor_expected_flat(2) = 100;
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test when missing a complete inner nested element
// Test when missing elements in the innermost dimension
TEST(ValueBufferTest, Dense3DWithTensorDefault) {
  // Define the default tensor
  const TensorShape shape({2, 3, 4});
  Tensor defaults(DT_INT32, shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < shape.num_elements(); ++i_value) {
    defaults_flat(i_value) = i_value;
  }

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.Add(1); buffer.Add(2); buffer.Add(3); buffer.Add(4);
      buffer.FinishMark();
      buffer.BeginMark();
        buffer.Add(2); buffer.Add(3); buffer.Add(4); // misses
      buffer.FinishMark();
      // misses complete entries for 3rd component
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.Add(5); buffer.Add(6); buffer.Add(7); // misses
      buffer.FinishMark();
      buffer.BeginMark();
        buffer.Add(8); buffer.Add(9); buffer.Add(10); buffer.Add(11);
      buffer.FinishMark();
      buffer.BeginMark();
        buffer.Add(12); // misses 3
      buffer.FinishMark();
    buffer.FinishMark();
  buffer.FinishMark();

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, shape);
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  LOG(INFO) << "Tensor defaults: " << defaults.SummarizeValue(24);

  // Create expected tensor and compare against the is tensor
  // Reuse defaults for expected
  Tensor tensor_expected(defaults);
  auto tensor_expected_write = tensor_expected.tensor<int, 3>();
  // first block
  tensor_expected_write(0, 0, 0) = 1;
  tensor_expected_write(0, 0, 1) = 2;
  tensor_expected_write(0, 0, 2) = 3;
  tensor_expected_write(0, 0, 3) = 4;
  // second block
  tensor_expected_write(0, 1, 0) = 2;
  tensor_expected_write(0, 1, 1) = 3;
  tensor_expected_write(0, 1, 2) = 4;
  // third block
  tensor_expected_write(1, 0, 0) = 5;
  tensor_expected_write(1, 0, 1) = 6;
  tensor_expected_write(1, 0, 2) = 7;
  // fourth block
  tensor_expected_write(1, 1, 0) = 8;
  tensor_expected_write(1, 1, 1) = 9;
  tensor_expected_write(1, 1, 2) = 10;
  tensor_expected_write(1, 1, 3) = 11;
  // fifth block
  tensor_expected_write(1, 2, 0) = 12;

  LOG(INFO) << "Tensor is:       " << tensor_actual.SummarizeValue(24);
  LOG(INFO) << "Tensor expected: " << tensor_expected.SummarizeValue(24);

  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}


// In this test we never have the maximum number of elements for a dimension
// and we miss more then one element in the outer dimensions
TEST(ValueBufferTest, Dense4DWithTensorDefault) {
  // Define the default tensor
  const TensorShape shape({2, 3, 4, 5});
  Tensor defaults(DT_INT32, shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < shape.num_elements(); ++i_value) {
    defaults_flat(i_value) = i_value;
  }

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.BeginMark();
          buffer.Add(1); buffer.Add(2); buffer.Add(3); buffer.Add(4);
        buffer.FinishMark();
        buffer.BeginMark();
          buffer.Add(2); buffer.Add(3); buffer.Add(4);
        buffer.FinishMark();
      buffer.FinishMark();
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.BeginMark();
          buffer.Add(5); buffer.Add(6); buffer.Add(7);
        buffer.FinishMark();
        buffer.BeginMark();
          buffer.Add(8); buffer.Add(9); buffer.Add(10); buffer.Add(11);
        buffer.FinishMark();
        buffer.BeginMark();
          buffer.Add(12);
        buffer.FinishMark();
      buffer.FinishMark();
    buffer.FinishMark();
  buffer.FinishMark();

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, shape);
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  LOG(INFO) << "Tensor defaults: " << defaults.SummarizeValue(24);

  // Create expected tensor and compare against the is tensor
  // Reuse defaults for expected
  Tensor tensor_expected(defaults);
  auto tensor_expected_write = tensor_expected.tensor<int, 4>();
  // first block
  tensor_expected_write(0, 0, 0, 0) = 1;
  tensor_expected_write(0, 0, 0, 1) = 2;
  tensor_expected_write(0, 0, 0, 2) = 3;
  tensor_expected_write(0, 0, 0, 3) = 4;
  // second block
  tensor_expected_write(0, 0, 1, 0) = 2;
  tensor_expected_write(0, 0, 1, 1) = 3;
  tensor_expected_write(0, 0, 1, 2) = 4;
  // third block
  tensor_expected_write(1, 0, 0, 0) = 5;
  tensor_expected_write(1, 0, 0, 1) = 6;
  tensor_expected_write(1, 0, 0, 2) = 7;
  // fourth block
  tensor_expected_write(1, 0, 1, 0) = 8;
  tensor_expected_write(1, 0, 1, 1) = 9;
  tensor_expected_write(1, 0, 1, 2) = 10;
  tensor_expected_write(1, 0, 1, 3) = 11;
  // fifth block
  tensor_expected_write(1, 0, 2, 0) = 12;

  LOG(INFO) << "Tensor is:       " << tensor_actual.SummarizeValue(24);
  LOG(INFO) << "Tensor expected: " << tensor_expected.SummarizeValue(24);

  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test empty dimension
// Test dense 2D with scalar default
TEST(ValueBufferTest, Dense2DWithScalarDefault) {
  // Define the default tensor
  const TensorShape defaults_shape({1});
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 100;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.Add(1); buffer.Add(2); buffer.Add(3);
    buffer.FinishMark();
  buffer.FinishMark();

  // Make the tensor from buffer
  const TensorShape shape({2, 3});
  Tensor tensor_actual(DT_INT32, shape);
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  // Create expected tensor and compare against the is tensor
  Tensor tensor_expected(DT_INT32, shape);
  auto tensor_expected_write = tensor_expected.tensor<int, 2>();
  tensor_expected_write(0, 0) = 100;
  tensor_expected_write(0, 1) = 100;
  tensor_expected_write(0, 2) = 100;
  tensor_expected_write(1, 0) = 1;
  tensor_expected_write(1, 1) = 2;
  tensor_expected_write(1, 2) = 3;
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test with partial tensor shape, where the default provides
// the complete shape information
TEST(ValueBufferTest, ShapeFromDefault) {
  // Define the default tensor
  const TensorShape defaults_shape({2, 3});
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < defaults_shape.num_elements(); ++i_value) {
    defaults_flat(i_value) = i_value;
  }

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.Add(40); buffer.Add(50); buffer.Add(60);
    buffer.FinishMark();
  buffer.FinishMark();

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, defaults_shape);
  const PartialTensorShape shape({-1, -1});
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  // Create expected tensor and compare against the is tensor
  Tensor tensor_expected(defaults);
  auto tensor_expected_write = tensor_expected.tensor<int, 2>();
  tensor_expected_write(1, 0) = 40;
  tensor_expected_write(1, 1) = 50;
  tensor_expected_write(1, 2) = 60;
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}


// Test with partial tensor shape, where the value buffer provides
// the complete shape information
TEST(ValueBufferTest, ShapeFromBuffer) {
  // Define the default tensor
  const TensorShape defaults_shape({1});
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 100;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.Add(1); buffer.Add(2); buffer.Add(3);
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.Add(4); buffer.Add(5); buffer.Add(6);
    buffer.FinishMark();
  buffer.FinishMark();

  // Make the tensor from buffer
  const TensorShape full_shape({2, 3});
  Tensor tensor_actual(DT_INT32, full_shape);
  PartialTensorShape shape({-1, -1});
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  // to allocate the proper amount of memory
  Tensor tensor_expected(DT_INT32, full_shape);
  auto expected_flat = tensor_expected.flat<int>();
  for (int i_value = 0; i_value < full_shape.num_elements(); ++i_value) {
    expected_flat(i_value) = i_value + 1;
  }
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test with string content -- since we use move instead of copy
TEST(ValueBufferTest, StringContent) {
  const TensorShape defaults_shape({1});
  Tensor defaults(DT_STRING, defaults_shape);
  auto defaults_flat = defaults.flat<string>();
  defaults_flat(0) = "abc";

  // Initialize the buffer
  StringValueBuffer buffer;
  buffer.BeginMark();
    buffer.AddByRef("a"); buffer.AddByRef("b");
  buffer.FinishMark();

  const TensorShape shape({3});
  Tensor tensor_actual(DT_STRING, shape);
  TF_EXPECT_OK(buffer.MakeDense(&tensor_actual, shape, defaults));

  Tensor tensor_expected(DT_STRING, shape);
  auto expected_flat = tensor_expected.flat<string>();
  expected_flat(0) = "a";
  expected_flat(1) = "b";
  expected_flat(2) = "abc";
  test::ExpectTensorEqual<string>(tensor_actual, tensor_expected);
}


// Test variants
// dense, sparse
// 1D, 2D
// scalar default, tensor default
// complete and partial


}
}