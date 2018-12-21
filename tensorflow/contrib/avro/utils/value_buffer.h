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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace data {

// Up to 4 items are stored without allocating heap memory
template <typename T>
class ValueBuffer {
public:
  ValueBuffer();
  Status FinishDimension();
  Status Add(T value); // use for bool, int, long, float, double, specialized implementation for string
  Status AddByRef(const T& value);
  // Make sure to finish dimensions, also add the end, Make will not call finish dimension because
  // in certain cases 0 dimensions might be desired in the shape.
  Status Make(Tensor* tensor, const PartialTensorShape& shape, const Tensor& defaults) const;
private:
  gtl::InlinedVector<T, 4> values_; // For up to 4 values use inline
  std::vector<size_t> end_indices_;
  size_t n_elements_for_dimension;
};

// ========================================
// Implementation
template <typename T>
ValueBuffer<T>::ValueBuffer() : n_elements_for_dimension(0) { }

template <typename T>
Status ValueBuffer<T>::FinishDimension() {
  end_indices_.push_back(n_elements_for_dimension);
  n_elements_for_dimension = 0;
  return Status::OK();
}

template <typename T>
Status ValueBuffer<T>::Add(T value) {
  values_.push_back(value);
  n_elements_for_dimension++;
  return Status::OK();
}

template <typename T>
Status ValueBuffer<T>::AddByRef(const T& value) {
  values_.push_back(value);
  n_elements_for_dimension++;
  return Status::OK();
}

template <typename InputIterT, typename OutputIterT>
void CopyOrMoveBlock(const InputIterT b, const InputIterT e, OutputIterT t) {
  std::copy(b, e, t);
}

template <>
void CopyOrMoveBlock(const string* b, const string* e, string* t) {
  std::move(b, e, t);
}

template <typename T>
Status ValueBuffer<T>::Make(Tensor* tensor, const PartialTensorShape& shape,
  const Tensor& defaults) const {
  // Check that shape matches, with the dimensions
  // Check that the default type matches this type
  // Call the move and copy stuff to transfer the data from the buffer into the tensor
  return Status::OK();
}

}
}

