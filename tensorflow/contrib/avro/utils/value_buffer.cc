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
#include "tensorflow/contrib/avro/utils/value_buffer.h"

namespace tensorflow {
namespace data {

// Implementation of the shape builder
ShapeBuilder::ShapeBuilder() : element_counter_(0), has_begin(false) {}

void ShapeBuilder::BeginMark() {
  element_info_.push_back(kBeginMark);
  has_begin = true;
}

void ShapeBuilder::FinishMark() {
  // Only put the element count if there was a beginning, necessary for nested dimensions
  if (has_begin) {
    element_info_.push_back(element_counter_);
    element_counter_ = 0;
  }
  element_info_.push_back(kFinishMark);
  has_begin = false;
}

void ShapeBuilder::Increment() {
  element_counter_++;
}

// Assumes that the value buffer has correct markers
size_t ShapeBuilder::GetNumberOfDimensions() const {
  size_t num = 0;
  for (size_t info : element_info_) {
    if (info != kBeginMark) {
      break;
    }
    num++;
  }
  return num;
}

// Assumes that the value buffer has correct markers
void ShapeBuilder::GetDenseShape(TensorShape* shape) const {
  size_t n_dim(GetNumberOfDimensions());
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
  size_t n_dim(GetNumberOfDimensions());
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
  std::vector<size_t> dims(CumulativeProductOfDimensionsWithOneAtEnd(shape));

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

        // If the user data has more elements than expected
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
  std::vector<size_t> dims(CumulativeProductOfDimensionsWithOneAtEnd(shape));

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
        // If the user data has more elements than expected
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

Status ShapeBuilder::GetIndices(Tensor* indices) const {
  size_t i_dim = 0;
  size_t offset = 0;
  size_t n_dim(GetNumberOfDimensions());
  std::vector<int64> counts(n_dim+1, -1); // initialize -1, create +1 number of dimensions
  auto counts_data = counts.begin();
  auto indices_data = (*indices).flat<int64>().data();

  for (size_t info : element_info_) {
    // Open a group
    if (info == kBeginMark) {
      counts[i_dim]++;
      i_dim++;

    // Close a group
    } else if (info == kFinishMark) {
      counts[i_dim] = -1;
      i_dim--;
    } else {

      // For each of the values create an index vector, while updating the inner-most index
      for (size_t i_value = 0; i_value < info; ++i_value) {
        counts[i_dim] = i_value;
        CopyOrMoveBlock(counts_data+1, counts_data+n_dim+1, indices_data + offset);
        offset += n_dim;
      }
    }
  }

  return Status::OK();
}

string ShapeBuilder::ToString() const {
  std::stringstream ss;
  for (size_t info : element_info_) {
    if (info == kBeginMark) {
      ss << "Begin, ";
    } else if (info == kFinishMark) {
      ss << "Finish, ";
    } else {
      ss << info << ", ";
    }
  }
  return ss.str();
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

}  // namespace data
}  // namespace tensorflow