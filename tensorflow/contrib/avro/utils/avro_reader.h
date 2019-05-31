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
#ifndef TENSORFLOW_DATA_AVRO_READER_H_
#define TENSORFLOW_DATA_AVRO_READER_H_

#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

#include "tensorflow/contrib/avro/utils/avro_parser_tree.h"
#include "tensorflow/contrib/avro/utils/avro_mem_reader.h"

namespace tensorflow {
namespace data {

struct AvroParseConfig {
  struct Dense {
    string feature_name;
    DataType dtype;
    PartialTensorShape shape;
    Tensor default_value;
  };

  struct Sparse {
    string feature_name;
    DataType dtype;
  };

  int64 batch_size;
  bool drop_remainder;
  std::vector<Dense> dense;
  std::vector<Sparse> sparse;
};

struct AvroResult {
  std::vector<Tensor> sparse_indices;
  std::vector<Tensor> sparse_values;
  std::vector<Tensor> sparse_shapes;
  std::vector<Tensor> dense_values;
};


class AvroReader {
public:
  AvroReader(const std::unique_ptr<RandomAccessFile>& file, const uint64 file_size,
             const string& filename, const string& reader_schema, const AvroParseConfig& config)
    : file_(std::move(file)),
      file_size_(file_size),
      filename_(filename),
      reader_schema_(reader_schema),
      config_(config),
      allocator_(tensorflow::cpu_allocator()) { }

  // Call for startup of work after construction.
  //
  // Loads data into memory and sets up the avro memory reader, and the parser tree.
  //
  Status OnWorkStartup();

  Status Read(AvroResult* result);

private:

  // Assumes tensor has been allocated appropriate space -- not checked
  Status ShapeToTensor(Tensor* tensor, const TensorShape& shape);

  // Checks that there are no duplicate keys in the sparse feature names and dense feature names
  std::vector<std::pair<string, DataType>> CreateKeysAndTypesFromConfig();

  const std::unique_ptr<RandomAccessFile>& file_;
  const uint64 file_size_;
  const string filename_;
  const string reader_schema_;
  const AvroParseConfig config_;

  AvroMemReader avro_mem_reader_;
  AvroParserTree avro_parser_tree_;
  std::unique_ptr<char[]> data_;
  std::map<string, ValueStoreUniquePtr> key_to_value_;
  // caching allocator here to avoid lock contention in `tensorflow::cpu_allocator()`
  Allocator* allocator_;
};

}
}

#endif  // TENSORFLOW_DATA_AVRO_READER_H_