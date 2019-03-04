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
#include "tensorflow/contrib/avro/utils/avro_value.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// Avro mem reader assumes that the memory block contains
// - header information for avro files
// - schema information about the data
// This reader uses the file reader on a memory mapped file
// This reader can support schema resolution
class AvroMemReader {
  public:
    AvroMemReader();
    virtual ~AvroMemReader();
    // Supply a filename if this memory is backed by a file
    static Status Create(AvroMemReader* reader, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& filename);
    // Note, value is only valid as long as no ReadNext is called since the internal method
    // re-uses the same memory for the next read
    virtual Status ReadNext(AvroValuePtr& value);

    static void AvroFileReaderDestructor(avro_file_reader_t* reader) {
      CHECK_GE(avro_file_reader_close(*reader), 0);
      free(reader);
    }
  protected:
    mutex mu_;
    AvroFileReaderPtr file_reader_ GUARDED_BY(mu_); // will close the file
    AvroInterfacePtr writer_iface_ GUARDED_BY(mu_);
};

class AvroResolvedMemReader : public AvroMemReader {
  public:
    AvroResolvedMemReader();
    virtual ~AvroResolvedMemReader();
    static Status Create(AvroResolvedMemReader* reader, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& reader_schema_str,
      const string& filename);
    static Status DoResolve(bool* resolve, const std::unique_ptr<char[]>& mem_data,
      const uint64 mem_size, const string& reader_schema_str, const string& filename);
    virtual Status ReadNext(AvroValuePtr& value);
  protected:
    AvroInterfacePtr reader_iface_ GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_MEM_READER_H_