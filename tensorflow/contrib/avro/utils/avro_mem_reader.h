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

void AvroFileReaderDestructor(avro_file_reader_t* reader) {
  CHECK_GE(avro_file_reader_close(*reader), 0);
}

void AvroSchemaDestructor(avro_schema_t schema) {
  CHECK_GE(avro_schema_decref(schema), 0);
}

void AvroInterfaceDestructor(avro_value_iface_t* iface)  {
  avro_value_iface_decref(iface);
}

void AvroValueDestructor(avro_value_t* value) {
    avro_value_decref(value);
}

// Avro mem reader assumes that the memory block contains
// - header information for avro files
// - schema information about the data
// This reader uses the file reader on a memory mapped file
// This reader does not support schema resolution
class AvroMemReader {
  public:
    using AvroFileReaderPtr = std::unique_ptr<avro_file_reader_t, void(*)(avro_file_reader_t*)>;

    using AvroSchemaPtr = std::unique_ptr<struct avro_obj_t,
                                           void(*)(avro_schema_t)>;

    using AvroInterfacePtr = std::unique_ptr<avro_value_iface_t,
                                                  void(*)(avro_value_iface_t*)>;

    using AvroValuePtr = std::shared_ptr<avro_value_t>;

    AvroMemReader();
    virtual ~AvroMemReader();
    // Supply a filename if this memory is backed by a file
    static Status Create(AvroMemReader* reader, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& filename="generic.avro");
    // Note, value is only valid as long as no ReadNext is called since the internal method
    // re-uses the same memory for the next read
    virtual Status ReadNext(AvroValuePtr& value);
  protected:
    AvroFileReaderPtr file_reader_; // will close the file
    AvroValuePtr writer_value_;
};


// Will only create a resolved reader IF
// the reader schema is not empty AND
// the reader schema is different from the writer schema
// OTHERWISE
// this will return a AvroMemReader
class AvroResolvedMemReader : public AvroMemReader {
  public:
    AvroResolvedMemReader();
    virtual ~AvroResolvedMemReader();

    static Status Create(AvroResolvedMemReader* reader, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& reader_schema_str,
      const string& filename="generic.avro");

    virtual Status ReadNext(AvroValuePtr& value);

    static Status DoResolve(bool* resolve, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& reader_schema_str, const string& filename);
  private:
    AvroValuePtr reader_value_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_MEM_READER_H_