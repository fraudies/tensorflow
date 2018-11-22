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
#ifndef TENSORFLOW_DATA_AVRO_MEM_READER_H_
#define TENSORFLOW_DATA_AVRO_MEM_READER_H_

#include <avro.h>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// Ideally this mem reader would not need a mapping to a file API
// However, the low level avro c API does not provide the full API
// of reading a schema from a avro reader and only an avro file
// reader
// Also schema resolution is only provided
class AvroMemReader {
  public:
    // Supply a filename if this memory is backed by a file
    static Status Create(AvroMemReader* reader, const std::shared_ptr<void>& mem_data,
      const uint64 mem_size, const string& filename="generic.avro");
    virtual Status ReadNext(avro_value_t* value);
    AvroMemReader(const tensorflow::data::AvroMemReader& mem_reader);
  protected:
    using FilePtr = std::shared_ptr<FILE>;

    using AvroFileReaderPtr = std::shared_ptr<avro_file_reader_t>;

    using AvroSchemaPtr = std::unique_ptr<struct avro_obj_t,
                                           void(*)(avro_schema_t)>;

    using AvroInterfacePtr = std::unique_ptr<avro_value_iface_t,
                                                  void(*)(avro_value_iface_t*)>;

    using AvroValuePtr = std::shared_ptr<avro_value_t>;

    explicit AvroMemReader(FilePtr file, AvroFileReaderPtr reader, AvroValuePtr value);

    FilePtr file_;
    AvroFileReaderPtr file_reader_;
    AvroValuePtr writer_value_;
};

// Will only create a resolved reader IF
// the reader schema is not empty AND
// the reader schema is different from the writer schema
// OTHERWISE
// this will return a AvroMemReader
/*
class AvroResolvedMemReader : public AvroMemReader {
  public:
    static Status Create(AvroMemReader* reader, const std::shared_ptr<void>& mem_data,
      const uint64 mem_size, const string& reader_schema_str, const string& filename="generic.avro");
    virtual Status ReadNext(avro_value_t* value);
  private:
    static Status Resolve(bool* resolve, const std::shared_ptr<void>& mem_data,
      const uint64 mem_size, const string& reader_schema_str, const string& filename);

    explicit AvroResolvedMemReader();

    AvroValueUPtr reader_value_;
};
*/

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_MEM_READER_H_