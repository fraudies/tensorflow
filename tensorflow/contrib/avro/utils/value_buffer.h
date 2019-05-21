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
#ifndef TENSORFLOW_DATA_VALUE_BUFFER_H_
#define TENSORFLOW_DATA_VALUE_BUFFER_H_

#include <stack>
#include <sstream>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace data {

static constexpr size_t kBeginMark = std::numeric_limits<size_t>::max() - 1;
static constexpr size_t kFinishMark = std::numeric_limits<size_t>::max();

// Forward declaration
class ValueStore;

// Pointer type
using ValueStoreUniquePtr = std::unique_ptr<ValueStore>;


// Non template base class
class ValueStore {
public:
  enum ResolverType {
    memoryShape,
    outputShape
  };

  virtual Status ResolveDenseShape(TensorShape* shape, const PartialTensorShape& partial_shape,
    const TensorShape& default_shape, ResolverType resolverType = ResolverType::memoryShape) const = 0;
  virtual Status GetSparseValueShape(TensorShape* shape) const = 0;
  virtual Status GetSparseIndexShape(TensorShape* shape) const = 0;

  virtual Status MakeDense(Tensor* tensor, const TensorShape& resolved_shape, const Tensor& defaults) const = 0;
  virtual Status MakeSparse(Tensor* values, Tensor* indices) const = 0;
  virtual bool ValuesMatchAtReverseIndex(const ValueStore& store, size_t reverse_index) const = 0;
  virtual bool ValueMatchesAtReverseIndex(const string& value, size_t reverse_index) const = 0;
  virtual bool IsEmpty() const = 0;
  virtual void BeginMark() = 0;
  virtual void FinishMark() = 0;
  virtual string ToString(size_t limit = 10) const = 0;
};

class ShapeBuilder {
public:
  ShapeBuilder();
  void BeginMark();
  void FinishMark();
  void Increment();
  size_t GetNumberOfDimensions() const;
  void GetDenseShape(TensorShape* shape) const;

  // indicates if the underlying buffer has all elements for the given shape
  bool HasAllElements(const TensorShape& shape) const;

  Status GetCopyInfo(std::vector<std::pair<size_t, size_t> >* copy_info, const TensorShape& shape) const;
  Status GetFillInfo(std::vector<std::pair<size_t, size_t> >* fill_info, const TensorShape& shape) const;
  Status GetIndices(Tensor* indices) const;
private:
  // Not recommended by google style guide but this is not part of the public API
  std::vector<size_t> CumulativeProductOfDimensionsWithOneAtEnd(const TensorShape& shape) const;
  std::vector<size_t> element_info_;
  size_t element_counter_; // intermediate counter used when creating a buffer
  bool has_begin;
};

// Up to 4 items are stored without allocating heap memory
template <typename T>
class ValueBuffer : public ValueStore {
public:
  void BeginMark() override;
  void FinishMark() override;
  // Make sure to finish dimensions, also add the end, Make will not call finish dimension because
  // in certain cases 0 dimensions might be desired in the shape
  void Add(T value); // use for bool, int, long, float, double, specialized implementation for string
  void AddByRef(const T& value);
  inline const T back() const { return values_.back(); }

  // Index must be starting from 1; which indexes the last element
  inline const T ReverseIndex(size_t index) const { return values_[values_.size() - index]; }

  // TODO(fraudies): May want to split this into two methods rather than using the flag
  Status ResolveDenseShape(TensorShape* shape, const PartialTensorShape& partial_shape,
    const TensorShape& default_shape,
    ResolverType resolverType = ValueStore::ResolverType::memoryShape) const override;
  Status GetSparseValueShape(TensorShape* shape) const override;
  Status GetSparseIndexShape(TensorShape* shape) const override;

  // May move the contents of this buffer into the tensor; hence not const
  // Assumes tensor has been initialized with the proper dimensions & type through allocate
  // TODO(fraudies): Do we want to check this in MakeDense?
  Status MakeDense(Tensor* tensor, const TensorShape& resolved_shape, const Tensor& defaults) const override;
  Status MakeSparse(Tensor* values, Tensor* indices) const override;
  virtual bool ValuesMatchAtReverseIndex(const ValueStore& store, size_t reverse_index) const override;
  virtual bool ValueMatchesAtReverseIndex(const string& value, size_t reverse_index) const override;
  inline bool IsEmpty() const {
    return GetNumberOfElements() == 0;
  }
  virtual string ToString(size_t limit) const override;
private:
  size_t GetNumberOfElements() const;

  // Assumes tensor has been initialized
  Status FillInFromBuffer(Tensor* tensor) const;
  // Assumes tensor has been initialized
  Status FillInFromDefault(Tensor* tensor, const Tensor& defaults) const;
  // Means this is not a tensor; it's just a scalar value
  inline static bool IsScalarValue(const PartialTensorShape& partial_shape) {
    return partial_shape.dims() < 1;
  }
  // Means this is a tensor which has a single dimension with value 1
  inline static bool IsOneElementTensor(const TensorShape& tensor_shape) {
    return tensor_shape.dims() == 1 && tensor_shape.dim_size(0) == 1;
  }
  /*
  inline static bool IsFullTensor(const TensorShape& tensor_shape) {
    return tensor_shape.dims() >= 1 && tensor_shape.dim_size(0) >= 1;
  }
  */
  gtl::InlinedVector<T, 4> values_; // For up to 4 values use inline
  ShapeBuilder shape_builder_;
};

// Template specializations for value buffer
typedef ValueBuffer<bool> BoolValueBuffer;
typedef ValueBuffer<int> IntValueBuffer;
typedef ValueBuffer<float> FloatValueBuffer;
typedef ValueBuffer<string> StringValueBuffer;

// helpers to copy or move data depending on the data type
template <typename InputIterT, typename OutputIterT>
inline void CopyOrMoveBlock(const InputIterT b, const InputIterT e, OutputIterT t) {
  std::copy(b, e, t);
}

template <>
inline void CopyOrMoveBlock(const string* b, const string* e, string* t) {
  std::move(b, e, t);
}


// Implementation of the value buffer

template <typename T>
void ValueBuffer<T>::BeginMark() {
  shape_builder_.BeginMark();
}

template <typename T>
void ValueBuffer<T>::FinishMark() {
  shape_builder_.FinishMark();
}

template <typename T>
void ValueBuffer<T>::Add(T value) {
  values_.push_back(value);
  shape_builder_.Increment();
}

template <typename T>
void ValueBuffer<T>::AddByRef(const T& value) {
  values_.push_back(value);
  shape_builder_.Increment();
}

template <typename T>
size_t ValueBuffer<T>::GetNumberOfElements() const {
  return values_.size();
}

// TODO(fraudies): Move validation of user defined shape and defaults into the avro dataset
// To resolve the proper shape for a dense tensor we honor:
// 1st the user provided partial shape
// 2nd the user provided defaults
// 3rd the value buffer's end indices
// In that order!
template<typename T>
Status ValueBuffer<T>::ResolveDenseShape(TensorShape* shape,
  const PartialTensorShape& partial_shape, const TensorShape& default_shape,
  ResolverType resolverType) const {

  bool defaultIsOneElementTensor = IsOneElementTensor(default_shape);
  bool partialIsScalarValue = IsScalarValue(partial_shape);

  LOG(INFO) << "Default shape is " << default_shape << " and is one element tensor " << (defaultIsOneElementTensor ? "true" : "false");

  LOG(INFO) << "Fully defined user shape " << (partial_shape.IsFullyDefined() ? "true" : "false");

  // Honor user defined shape if fully defined
  if (partial_shape.IsFullyDefined()) {
    // This is the case where the user provided [] as input and expects a scalar;
    // however internally we represent that as a one dimensional tensor with
    // dimension: 1
    PartialTensorShape tmp_shape;
    if (partialIsScalarValue && resolverType == memoryShape) {
      tmp_shape = TensorShape({1});
      LOG(INFO) << "Is scalar value with tmp shape " << tmp_shape;
    } else {
      tmp_shape = partial_shape;
    }
    if (!tmp_shape.AsTensorShape(shape)) {
      return errors::InvalidArgument("Expected ", tmp_shape, " to be fully defined"
        " and convertible into a dense shape.");
    }

  // If the default is not scalar
  } else if (!defaultIsOneElementTensor) {
    PartialTensorShape tmp_shape;
    // Honor any partially defined shape from user and supplement with that from default
    if (partial_shape.MergeWith(default_shape, &tmp_shape) == Status::OK()) {
      // Merged convert partial shape into shape
      if (!tmp_shape.AsTensorShape(shape)) {
        return errors::InvalidArgument("Expected ", tmp_shape, " to be fully defined"
          " and convertible into a dense shape.");
      }
    } else {
      // Could not merge, then use default
      *shape = default_shape;
    }
  // If the shape is not defined by the user nor the default, infer from provided data
  } else {
    TensorShape dense_shape;
    shape_builder_.GetDenseShape(&dense_shape);

    LOG(INFO) << "Dense shape from data " << dense_shape;

    PartialTensorShape tmp_shape;
    // Honor any partially defined shape from user and supplement with that from data
    if (partial_shape.MergeWith(dense_shape, &tmp_shape) == Status::OK()) {
      if (!tmp_shape.AsTensorShape(shape)) {
        return errors::InvalidArgument("Expected ", tmp_shape, " to be fully defined"
          " and convertible into a dense shape.");
      }
    } else {
      // Could not merge, then use dense shape
      *shape = dense_shape;
    }
  }

  return Status::OK();
}

template <typename T>
Status ValueBuffer<T>::GetSparseValueShape(TensorShape* shape) const {
  (*shape).AddDim(GetNumberOfElements());
  return Status::OK();
}

template <typename T>
Status ValueBuffer<T>::GetSparseIndexShape(TensorShape* shape) const {
  (*shape).AddDim(GetNumberOfElements());
  size_t n_dim = shape_builder_.GetNumberOfDimensions();
  // Only add this dimension if it is necessary, which is > 1
  if (n_dim > 1) {
    (*shape).AddDim(shape_builder_.GetNumberOfDimensions());
  }
  return Status::OK();
}

// Builds the tensor from the value buffer
template <typename T>
Status ValueBuffer<T>::MakeDense(Tensor* tensor, const TensorShape& resolved_shape,
  const Tensor& defaults) const {

  // Get the dense shape
  bool doFillFromDefault = !shape_builder_.HasAllElements(resolved_shape);

  // Check that shape matches, with the dimensions
  if (doFillFromDefault) {
    // fill in the default -- note might fill all values
    TF_RETURN_IF_ERROR(FillInFromDefault(tensor, defaults));
  }

  // Fill in the values into the tensor from the buffer
  TF_RETURN_IF_ERROR(FillInFromBuffer(tensor));

  return Status::OK();
}

// Builds the tensor for the values and the indices from the value buffer
// Assumes that values is pre-allocated with space for n_elements
// Assumes that indices is pre-allocated with space for n_elements x n_dim
template <typename T>
Status ValueBuffer<T>::MakeSparse(Tensor* values, Tensor* indices) const {

  // Copy values
  auto tensor_data = (*values).flat<T>().data();
  auto buffer_data = values_.begin();
  size_t n_elements(GetNumberOfElements());
  CopyOrMoveBlock(
    buffer_data,
    buffer_data + n_elements,
    tensor_data);

  // Create indices
  TF_RETURN_IF_ERROR(shape_builder_.GetIndices(indices));

  return Status::OK();
}


template <typename T>
Status ValueBuffer<T>::FillInFromBuffer(Tensor* tensor) const {
  TensorShape shape = (*tensor).shape();
  auto tensor_data = (*tensor).flat<T>().data();
  auto buffer_data = values_.begin();
  // These offsets are per fragment of data
  std::vector<std::pair<size_t, size_t> > copy_info;
  TF_RETURN_IF_ERROR(shape_builder_.GetCopyInfo(&copy_info, shape));
  size_t source_offset = 0;
  for (const auto& info : copy_info) {
    size_t target_offset = info.first;
    size_t length = info.second;
    CopyOrMoveBlock(
      buffer_data + source_offset,
      buffer_data + source_offset + length,
      tensor_data + target_offset);
    source_offset += length;
  }

  return Status::OK();
}

template <typename T>
Status ValueBuffer<T>::FillInFromDefault(Tensor* tensor, const Tensor& defaults) const {
  if (!defaults.IsInitialized()) {
    return errors::InvalidArgument("Need to provide a 'defaults' tensor with values");
  }

  TensorShape shape = (*tensor).shape();
  auto tensor_data = (*tensor).flat<T>().data();
  auto buffer_data = defaults.flat<T>().data();
  if (IsOneElementTensor(defaults.shape())) {
    // Fill tensor with default to create padding
    std::fill(tensor_data, tensor_data + shape.num_elements(), defaults.flat<T>()(0));
  } else {
    std::vector<std::pair<size_t, size_t> > fill_info;
    TF_RETURN_IF_ERROR(shape_builder_.GetFillInfo(&fill_info, shape));
    for (const auto& info : fill_info) {
      size_t offset = info.first;
      size_t length = info.second;
      CopyOrMoveBlock(
        buffer_data + offset,
        buffer_data + offset + length,
        tensor_data + offset);
    }
  }

  return Status::OK();
}

template <typename T>
bool ValueBuffer<T>::ValuesMatchAtReverseIndex(const ValueStore& store, size_t reverse_index) const {
  // If both stores are empty, then there is a match
  if (IsEmpty() && store.IsEmpty()) {
    return true;
  }
  // Note, buffer will be nullptr if the types don't match
  const ValueBuffer* buffer = dynamic_cast<const ValueBuffer*>(&store);
  return buffer != nullptr
    && ReverseIndex(reverse_index) == (*buffer).ReverseIndex(reverse_index);
}

template <typename T>
inline bool ValueBuffer<T>::ValueMatchesAtReverseIndex(const string& value, size_t reverse_index) const {
  // TODO(fraudies): Check if we want to match other types through parsing of string
  return false;
}

template <>
inline bool ValueBuffer<string>::ValueMatchesAtReverseIndex(const string& value, size_t reverse_index) const {
  // If there is no value to match there is no match
  if (IsEmpty()) {
    return false;
  }
  return ReverseIndex(reverse_index) == value;
}

template <typename T>
string ValueBuffer<T>::ToString(size_t limit) const {
  std::stringstream ss;
  size_t n_print = std::min(values_.size(),  limit);
  for (size_t i_print = 0; i_print < n_print; ++i_print) {
    ss << values_[i_print] << ", ";
  }
  if (values_.size() > limit) {
    ss << " ...";
  }
  return ss.str();
}

}
}

#endif // TENSORFLOW_DATA_VALUE_BUFFER_H_