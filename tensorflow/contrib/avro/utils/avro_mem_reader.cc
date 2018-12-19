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
#include "tensorflow/contrib/avro/utils/avro_mem_reader.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace data {


AvroMemReader::AvroMemReader() :
  file_reader_(nullptr, AvroFileReaderDestructor),
  writer_value_(new avro_value_t, AvroValueDestructor),
  reader_value_(new avro_value_t, AvroValueDestructor) { }

AvroMemReader::~AvroMemReader() { }

Status AvroMemReader::Create(AvroMemReader* reader, const std::unique_ptr<char[]>& mem_data,
  const uint64 mem_size, const string& filename) {

  // Clear any previous error messages
  avro_set_error("");

  // Open a memory mapped file
  FILE* file(fmemopen(static_cast<void*>(mem_data.get()), mem_size, "rb"));
  if (file == nullptr) {
    return Status(errors::InvalidArgument("Unable to open file ", filename, " for memory."));
  }

  // Create an avro file reader with that file
  avro_file_reader_t* file_reader = new avro_file_reader_t; // use tmp not to clean up a partially created reader
  if (avro_file_reader_fp(file, filename.c_str(), 1, file_reader) != 0) {
    return Status(errors::InvalidArgument("Unable to open file ", filename,
                                          " in avro reader. ", avro_strerror()));
  }
  reader->file_reader_.reset(file_reader);

  // Get the writer schema
  AvroSchemaPtr writer_schema(avro_file_reader_get_writer_schema(*reader->file_reader_),
             AvroSchemaDestructor);
  if (writer_schema.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to retrieve schema from file ", filename));
  }

  // Get the writer interface for that schema
  AvroInterfacePtr writer_iface(avro_generic_class_from_schema(writer_schema.get()),
    AvroInterfaceDestructor);
  if (writer_iface.get() == nullptr) {
    // TODO(fraudies): Get a string representation of the schema, use avro_schema_to_json
    return Status(errors::ResourceExhausted("Unable to create interface for schema"));
  }

  // Initialize the value for that schema
  if (avro_generic_value_new(writer_iface.get(), reader->writer_value_.get()) != 0) {
    return Status(errors::InvalidArgument(
        "Unable to create instance for generic class."));
  }
  avro_value_copy_ref(reader->reader_value_.get(), reader->writer_value_.get());

  return Status::OK();
}

Status AvroMemReader::ReadNext(AvroValuePtr& value) {
  avro_set_error("");
  avro_value_reset(writer_value_.get());
  int ret = avro_file_reader_read_value(*file_reader_, writer_value_.get());
  // TODO(fraudies): Issue:
  // When reading from a memory mapped file we get this error
  // `Error reading file: Incorrect sync bytes`
  // Instead of EOF
  // Need to check why this is happening when opening the file with fmemopen and not with
  // fopen
  /*
  if (ret == EOF) {
    return errors::OutOfRange("eof");
  }
  if (ret != 0) {
    return errors::InvalidArgument("Unable to read value due to: ", avro_strerror());
  }
  */
  if (ret != 0) {
    return errors::OutOfRange("eof");
  }
  value = reader_value_;
  return Status::OK();
}

// An example of resolved reading can be found in this test case test_avro_984.c
// We follow that here
Status AvroMemReader::Create(AvroMemReader* reader, const std::unique_ptr<char[]>& mem_data,
  const uint64 mem_size, const string& reader_schema_str, const string& filename) {

  // Create a reader schema for the user passed string
  avro_schema_t reader_schema_tmp;
  if (avro_schema_from_json_length(reader_schema_str.data(),
                                   reader_schema_str.length(),
                                   &reader_schema_tmp) != 0) {
    return Status(errors::InvalidArgument(
        "The provided json schema is invalid. ", avro_strerror()));
  }
  AvroSchemaPtr reader_schema(reader_schema_tmp, AvroSchemaDestructor);

  // Create reader class
  AvroInterfacePtr reader_iface(
    avro_generic_class_from_schema(reader_schema.get()),
    AvroInterfaceDestructor
  );
  if (reader_iface.get() == nullptr) {
    // TODO(fraudies): Print the schemas in the error message
    return Status(errors::ResourceExhausted("Unable to create interface for schema"));
  }

  // Initialize value for reader class
  if (avro_generic_value_new(reader_iface.get(), reader->reader_value_.get()) != 0) {
    return Status(errors::InvalidArgument(
        "Unable to create instance for generic reader class."));
  }

  // Open a memory mapped file
  FILE* file = fmemopen(static_cast<void*>(mem_data.get()), mem_size, "rb");
  if (file == nullptr) {
    return Status(errors::InvalidArgument("Unable to open file ", filename, " for memory."));
  }

  // Closes the file handle
  avro_file_reader_t* file_reader = new avro_file_reader_t; // use tmp not to clean up a partially created reader
  if (avro_file_reader_fp(file, filename.c_str(), 1, file_reader) != 0) {
    return Status(errors::InvalidArgument("Unable to open file ", filename,
                                          " in avro reader. ", avro_strerror()));
  }
  reader->file_reader_.reset(file_reader);

  // Get the writer schema
  AvroSchemaPtr writer_schema(avro_file_reader_get_writer_schema(*reader->file_reader_),
             AvroSchemaDestructor);
  if (writer_schema.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to retrieve schema from file ", filename));
  }

  // Get the writer interface and initialize the value for that interface
  AvroInterfacePtr writer_iface(avro_resolved_writer_new(writer_schema.get(), reader_schema.get()),
    AvroInterfaceDestructor);
  if (writer_iface.get() == nullptr) {
    // TODO(fraudies): Get a string representation of the schema, use avro_schema_to_json
    return Status(errors::InvalidArgument("Schemas are incompatible. ",
                                          avro_strerror()));
  }

  // Create instance for resolved writer class
  if (avro_resolved_writer_new_value(writer_iface.get(), reader->writer_value_.get()) != 0) {
    return Status(
        errors::InvalidArgument("Unable to create resolved writer value."));
  }
  avro_resolved_writer_set_dest(reader->writer_value_.get(), reader->reader_value_.get());

  return Status::OK();
}

Status AvroMemReader::DoResolve(bool* resolve, const std::unique_ptr<char[]>& mem_data,
  const uint64 mem_size, const string& reader_schema_str, const string& filename) {

  // No schema supplied => no schema resolution is necessary
  if (reader_schema_str.length() <= 0) {
    *resolve = false;
    return Status::OK();
  }

  // Open the file to get the writer schema
  FILE* file(fmemopen(static_cast<void*>(mem_data.get()), mem_size, "r"));
  if (file == nullptr) {
    return Status(errors::InvalidArgument("Unable to open file ", filename, " for memory."));
  }

  // Open the avro file reader
  avro_file_reader_t file_reader_tmp;
  if (avro_file_reader_fp(file, filename.c_str(), 1, &file_reader_tmp) != 0) {
    return Status(errors::InvalidArgument("Unable to open file ", filename,
                                          " in avro reader. ", avro_strerror()));
  }
  AvroFileReaderPtr file_reader(&file_reader_tmp, AvroFileReaderDestructor);

  // Get the writer schema
  AvroSchemaPtr writer_schema(avro_file_reader_get_writer_schema(*file_reader),
             AvroSchemaDestructor);
  if (writer_schema.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to retrieve schema from file ", filename));
  }

  // Create a reader schema for the user passed string
  avro_schema_t reader_schema_tmp;
  if (avro_schema_from_json_length(reader_schema_str.data(),
                                   reader_schema_str.length(),
                                   &reader_schema_tmp) != 0) {
    return Status(errors::InvalidArgument(
        "The provided json schema is invalid. ", avro_strerror()));
  }
  AvroSchemaPtr reader_schema(reader_schema_tmp, AvroSchemaDestructor);

  // Do resolve only if the schemas are different
  *resolve = !avro_schema_equal(writer_schema.get(), reader_schema.get());

  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow