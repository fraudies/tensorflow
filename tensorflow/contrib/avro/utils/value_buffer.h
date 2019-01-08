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
#include <stack>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace data {

static constexpr size_t kBeginMark = std::numeric_limits<size_t>::max() - 1;
static constexpr size_t kFinishMark = std::numeric_limits<size_t>::max();

// Non template base class
class ValueStore {
public:
  virtual Status MakeDense(Tensor* tensor, const PartialTensorShape& shape, const Tensor& defaults) const = 0;
  virtual Status MakeSparse(Tensor* values, Tensor* indices) const = 0;
protected:
};

class ShapeBuilder {
public:
  ShapeBuilder();
  void BeginMark();
  void FinishMark();
  inline void Increment() { element_counter_++; }
  void GetNumberOfDimensions(size_t* num) const;
  void GetDenseShape(TensorShape* shape) const;
  bool HasAllElements(const TensorShape& shape) const; // indicates if the underlying buffer has all elements
  Status GetCopyInfo(std::vector<std::pair<size_t, size_t> >* copy_info, const TensorShape& shape) const;
  Status GetFillInfo(std::vector<std::pair<size_t, size_t> >* fill_info, const TensorShape& shape) const;
private:
  // Uses c++11 move semantics and is not part of the public API, so seems OK to me
  std::vector<size_t> CumulativeProductOfDimensionsWithOneAtEnd(const TensorShape& shape) const;
  std::vector<size_t> element_info_;
  size_t element_counter_; // intermediate counter used when crating a buffer
  bool has_begin;
};

// Up to 4 items are stored without allocating heap memory
template <typename T>
class ValueBuffer : public ValueStore, public ShapeBuilder {
public:
  ValueBuffer();
  // Make sure to finish dimensions, also add the end, Make will not call finish dimension because
  // in certain cases 0 dimensions might be desired in the shape
  void Add(T value); // use for bool, int, long, float, double, specialized implementation for string
  void AddByRef(const T& value);
  // May move the contents of this buffer into the tensor; hence not const
  // Assumes tensor has been initialized with the proper dimensions & type through allocate in OpOutputList
  // TODO: Do we want to check this in MakeDense?
  Status MakeDense(Tensor* tensor, const PartialTensorShape& partial_shape, const Tensor& defaults) const override;
  Status MakeSparse(Tensor* values, Tensor* indices) const override;
private:
  //void InitTensor(Tensor* tensor, const TensorShape& shape) const;
  Status ResolveDenseShape(TensorShape* shape, const PartialTensorShape& partial_shape, const Tensor& defaults) const;
  // Assumes tensor has been initialized
  Status FillInFromBuffer(Tensor* tensor) const;
  // Assumes tensor has been initialized
  Status FillInFromDefault(Tensor* tensor, const Tensor& defaults) const;
  inline bool IsScalar(const Tensor& tensor) const {
    return tensor.dims() == 1 && tensor.dim_size(0) == 1;
  }
  gtl::InlinedVector<T, 4> values_; // For up to 4 values use inline
};

ShapeBuilder::ShapeBuilder() : element_counter_(0), has_begin(false) {}

// ========================================
// Implementation
void ShapeBuilder::BeginMark() {
  element_info_.push_back(kBeginMark);
  has_begin = true;
}

void ShapeBuilder::FinishMark() {
  // Only put the element count
  if (has_begin) {
    element_info_.push_back(element_counter_);
    element_counter_ = 0;
  }
  element_info_.push_back(kFinishMark);
  has_begin = false;
}

// Assumes that the value buffer has correct markers
void ShapeBuilder::GetNumberOfDimensions(size_t* num) const {
  *num = 0;
  for (size_t info : element_info_) {
    if (info != kBeginMark) {
      break;
    }
    (*num)++;
  }
}

// Assumes that the value buffer has correct markers
void ShapeBuilder::GetDenseShape(TensorShape* shape) const {
  size_t n_dim;
  GetNumberOfDimensions(&n_dim);
  std::vector<size_t> dimensions(n_dim, 0);
  std::vector<size_t> counts(n_dim+1, 0); // +1 to simplify logic of setting to 0
  size_t i_dim = -1; // -1 to make sure indices start from 0
  for (size_t info : element_info_) {
    if (info == kBeginMark) {
      i_dim++;
      counts[i_dim]++;
    } else if (info == kFinishMark) {
      dimensions[i_dim - 1] = std::max(dimensions[i_dim - 1], counts[i_dim]);
      counts[i_dim + 1] = 0;
      i_dim--;
    } else {
      dimensions[i_dim] = std::max(dimensions[i_dim], info);
    }
  }
  // Construct the shape
  *shape = TensorShape();
  for (size_t dimension : dimensions) {
    (*shape).AddDim(dimension);
  }
}

bool ShapeBuilder::HasAllElements(const TensorShape& shape) const {
  size_t n_dim;
  GetNumberOfDimensions(&n_dim);
  std::vector<size_t> counts(n_dim+1, 0); // +1 to simplify logic of setting to 0
  size_t i_dim = -1; // -1 to make sure indices start from 0
  for (size_t info : element_info_) {
    if (info == kBeginMark) {
      i_dim++;
      counts[i_dim]++;
    } else if (info == kFinishMark) {
      if (counts[i_dim] != shape.dim_size(i_dim)) {
        return false;
      }
      counts[i_dim + 1] = 0;
      i_dim--;
    } else {
      if (info != shape.dim_size(i_dim)) {
        return false;
      }
    }
  }
  return true;
}

Status ShapeBuilder::GetCopyInfo(std::vector<std::pair<size_t, size_t> >* copy_info,
  const TensorShape& shape) const {
  size_t offset = 0;
  size_t i_dim = 0; // -1 to make sure indices start from 0
  size_t n_dim = shape.dims();
  std::vector<size_t> counts(n_dim+1, 0);
  std::vector<size_t> dims = CumulativeProductOfDimensionsWithOneAtEnd(shape);

  for (size_t info : element_info_) {
    // Open a group
    if (info == kBeginMark) {
      counts[i_dim]++;
      i_dim++;

    // Close a group
    } else if (info == kFinishMark) {
      size_t count = counts[i_dim];
      size_t length = shape.dim_size(i_dim-1);

      // Handle the inner most element count
      if (i_dim == n_dim) {

        // If the user data contained more elements than expected
        if (count > length) {
          return errors::InvalidArgument("Per shape ", shape, " for dimension ", i_dim-1,
            " expected at most ", length, " elements but received ", count, " elements");
        }

        if (count > 0) {
          //LOG(INFO) << "Copy (" << offset << ", " << count << ")";
          (*copy_info).push_back(std::make_pair(offset, count));
        }
        offset += length;

      // Handle where the caller did not provide all finish marks
      } else if (count < length) {
        offset += dims[i_dim] * (length - count);
      }

      counts[i_dim] = 0;
      i_dim--;
    } else {
      counts[i_dim] = info;
    }
  }

  return Status::OK();
}

Status ShapeBuilder::GetFillInfo(std::vector<std::pair<size_t, size_t> >* fill_info,
  const TensorShape& shape) const {
  size_t offset = 0;
  size_t i_dim = 0;
  size_t n_dim = shape.dims();

  std::vector<size_t> counts(n_dim+1, 0);
  std::vector<size_t> dims = CumulativeProductOfDimensionsWithOneAtEnd(shape);

  for (size_t i_dim = n_dim; i_dim > 0; --i_dim) {
    dims[i_dim-1] = shape.dim_size(i_dim-1)*dims[i_dim];
  }

  for (size_t info : element_info_) {
    // Open a group
    if (info == kBeginMark) {
      counts[i_dim]++;
      i_dim++;

    // Close a group
    } else if (info == kFinishMark) {

      size_t count = counts[i_dim];
      size_t length = shape.dim_size(i_dim-1);
      bool smaller = count < length;

      // Handle the inner most element count
      if (i_dim == n_dim) {
        // If the user data contained more elements than expected
        if (count > length) {
          return errors::InvalidArgument("Per shape ", shape, " for dimension ", i_dim-1,
            " expected at most ", length, " elements but received ", count, " elements");
        }

        if (smaller) {
          //LOG(INFO) << "Fill (" << offset+count << ", " << length-count << ")";
          (*fill_info).push_back(std::make_pair(offset+count, length-count));
        }
        offset += length;

      // Handle where the caller did not provide all finish marks
      } else if (smaller) {
        size_t delta = dims[i_dim] * (length - count);
        //LOG(INFO) << "Fill (" << offset << ", " << delta << ")";
        (*fill_info).push_back(std::make_pair(offset, delta));
        offset += delta;
      }
      counts[i_dim] = 0;
      i_dim--;
    } else {
      counts[i_dim] = info;
    }
  }

  return Status::OK();
}

std::vector<size_t> ShapeBuilder::CumulativeProductOfDimensionsWithOneAtEnd(
  const TensorShape& shape) const {

  size_t n_dim = shape.dims();
  std::vector<size_t> dims(n_dim+1, 1);
  for (size_t i_dim = n_dim; i_dim > 0; --i_dim) {
    dims[i_dim-1] = shape.dim_size(i_dim-1)*dims[i_dim];
  }

  return dims;
}

template <typename T>
ValueBuffer<T>::ValueBuffer() : ShapeBuilder() { }

template <typename T>
void ValueBuffer<T>::Add(T value) {
  values_.push_back(value);
  Increment();
}

template <typename T>
void ValueBuffer<T>::AddByRef(const T& value) {
  values_.push_back(value);
  Increment();
}

template <typename InputIterT, typename OutputIterT>
void CopyOrMoveBlock(const InputIterT b, const InputIterT e, OutputIterT t) {
  std::copy(b, e, t);
}

template <>
void CopyOrMoveBlock(const string* b, const string* e, string* t) {
  std::move(b, e, t);
}

template<typename T>
Status ValueBuffer<T>::ResolveDenseShape(TensorShape* shape,
  const PartialTensorShape& partial_shape, const Tensor& defaults) const {

  bool isScalarDefault = IsScalar(defaults);

  // Honor user defined shape if fully defined
  if (partial_shape.IsFullyDefined()) {
    if (!partial_shape.AsTensorShape(shape)) {
      return errors::InvalidArgument("Expected ", partial_shape, " to be fully defined"
        " and convertible into a dense shape.");
    }
  // If the default is not scalar
  } else if (!isScalarDefault) {
    PartialTensorShape tmp_shape;
    TensorShape default_shape(defaults.shape());
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
    GetDenseShape(&dense_shape);
    PartialTensorShape tmp_shape;
    // Honor any partially defined shape from user and supplement with that from data
    if (partial_shape.MergeWith(dense_shape, &tmp_shape) == Status::OK()) {
      if (tmp_shape.AsTensorShape(shape)) {
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

// Honors
// shape first
// defaults second
// value buffer's end indices 3rd
//
// 'tensor' The output tensor
// 'shape' The tensor shape the
template <typename T>
Status ValueBuffer<T>::MakeDense(Tensor* tensor, const PartialTensorShape& partial_shape,
  const Tensor& defaults) const {

  TensorShape dense_shape;
  TF_RETURN_IF_ERROR(ResolveDenseShape(&dense_shape, partial_shape, defaults));

  LOG(INFO) << "Dense shape: " << dense_shape.DebugString();

  // Get the dense shape
  bool doFillFromDefault = !HasAllElements(dense_shape);

  // Check that shape matches, with the dimensions
  if (doFillFromDefault) {
    // fill in the default -- note might fill all values
    TF_RETURN_IF_ERROR(FillInFromDefault(tensor, defaults));
  }

  // Fill in the values into the tensor from the buffer
  TF_RETURN_IF_ERROR(FillInFromBuffer(tensor));

  // Check that the default type matches this type
  // Call the move and copy stuff to transfer the data from the buffer into the tensor
  return Status::OK();
}

template <typename T>
Status ValueBuffer<T>::FillInFromBuffer(Tensor* tensor) const {
  TensorShape shape = (*tensor).shape();
  auto tensor_data = (*tensor).flat<T>().data();
  auto buffer_data = values_.begin();
  // These offsets are per fragment of data
  std::vector<std::pair<size_t, size_t> > copy_info;
  TF_RETURN_IF_ERROR(GetCopyInfo(&copy_info, shape));
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
  if (IsScalar(defaults)) {
    // Fill tensor with default to create padding
    std::fill(tensor_data, tensor_data + shape.num_elements(), defaults.flat<T>()(0));
  } else {
    std::vector<std::pair<size_t, size_t> > fill_info;
    TF_RETURN_IF_ERROR(GetFillInfo(&fill_info, shape));
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
Status ValueBuffer<T>::MakeSparse(Tensor* values, Tensor* indices) const {
  // Build the tensor for the values and the indices from the value buffer
  return Status::OK();
}

// Template specializations for value buffer
typedef ValueBuffer<bool> BoolValueBuffer;
typedef ValueBuffer<int> IntValueBuffer;
typedef ValueBuffer<float> FloatValueBuffer;
typedef ValueBuffer<string> StringValueBuffer;

}
}

