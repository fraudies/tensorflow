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

void AvroFileReaderDestructor(avro_file_reader_t* reader) {
  // I don't think we need the CHECK_NOT NULL
  CHECK_GE(avro_file_reader_close(*reader), 0);
}

void AvroSchemaDestructor(avro_schema_t schema) {
  // Confusingly, it appears that the avro_file_reader_t creates its
  // own reference to this schema, so the schema is not really
  // "uniquely" owned...
  CHECK_GE(avro_schema_decref(schema), 0);
};

void AvroInterfaceDestructor(avro_value_iface_t* iface)  {
  avro_value_iface_decref(iface);
}

void AvroValueDestructor(avro_value_t* value) {
    // This is unnecessary clunky because avro's free assumes that
    // the iface ptr is initialized which is only the case once used
    if (value->iface != nullptr) {
      avro_value_decref(value);
    } else {
      // free the container
      free(value);
    }
}

void FileDestructor(FILE* file) {
  if (file != nullptr) {
    fclose(file);
  }
}

AvroMemReader::AvroMemReader(FilePtr file, AvroFileReaderPtr reader, AvroValuePtr value) :
  file_(file),
  file_reader_(reader),
  writer_value_(value) { }

// Transfers ownership
AvroMemReader::AvroMemReader(const AvroMemReader& mem_reader) :
  file_(mem_reader.file_),
  file_reader_(mem_reader.file_reader_),
  writer_value_(mem_reader.writer_value_) { }

Status AvroMemReader::Create(AvroMemReader* reader, const std::shared_ptr<void>& mem_data,
  const uint64 mem_size, const string& filename) {

  // Clear any previous error messages
  avro_set_error("");

  // Open a memory mapped file
  FilePtr file(fmemopen(mem_data.get(), mem_size, "r"), FileDestructor);
  if (file.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to open file ", filename, " for memory."));
  }

  // Create an avro file reader with that file
  AvroFileReaderPtr file_reader(new avro_file_reader_t, AvroFileReaderDestructor);
  if (avro_file_reader_fp(file.get(), filename.c_str(), 1, file_reader.get()) != 0) {
    return Status(errors::InvalidArgument("Unable to open file ", filename,
                                          " in avro reader. ", avro_strerror()));
  }

  // Get the writer schema
  AvroSchemaPtr writer_schema(avro_file_reader_get_writer_schema(*file_reader),
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
  AvroValuePtr writer_value(new avro_value_t, AvroValueDestructor);
  writer_value.get()->iface = nullptr; // To work with the custom deleter
  if (avro_generic_value_new(writer_iface.get(), writer_value.get()) != 0) {
    return Status(errors::InvalidArgument(
        "Unable to create instance for generic class."));
  }

  //*reader = AvroMemReader(file, reader, writer_value);

  return Status::OK();
}

Status AvroMemReader::ReadNext(avro_value_t* value) {
  // Reset the value for the next read
  avro_value_reset(writer_value_.get());
  bool at_end =
    avro_file_reader_read_value(*file_reader_, writer_value_.get()) != 0;
  // Transfer ownership
  avro_value_move_ref(value, writer_value_.get());
  return at_end ? errors::OutOfRange("eof") : Status::OK();
}

/*
AvroResolvedMemReader::AvroResolvedMemReader() :
  AvroMemReader(),
  reader_value_(new avro_value_t, AvroValueDestructor) {
  reader_value_.get()->iface = nullptr;
}

// An example of resolved reading can be found in this test case test_avro_984.c
// We follow that here
Status AvroResolvedMemReader::Create(AvroMemReader* reader, const std::shared_ptr<void>& mem_data,
  const uint64 mem_size, const string& reader_schema_str, const string& filename) {

  bool resolve;
  TF_RETURN_IF_ERROR(Resolve(&resolve, mem_data, mem_size, reader_schema_str, filename));

  if (!resolve) {
    return AvroMemReader::Create(reader, mem_data, mem_size, filename);
  }

  *reader = AvroResolvedMemReader();

  // Open a memory mapped file
  FILE* file = fmemopen(mem_data.get(), mem_size, "r");
  if (file == nullptr) {
    return Status(errors::InvalidArgument("Unable to open file ", filename, " for memory."));
  }
  reader->file_.reset(file);

  // Create a avro file reader with that file
  avro_file_reader_t file_reader;
  if (avro_file_reader_fp(reader->file_.get(), filename.c_str(), 1, &file_reader) != 0) {
    return Status(errors::InvalidArgument("Unable to open file ", filename,
                                          " in avro reader. ", avro_strerror()));
  }
  reader->file_reader_.reset(file_reader);

  // Get the writer schema
  AvroSchemaUPtr writer_schema(avro_file_reader_get_writer_schema(reader->file_reader_.get()),
             AvroSchemaDestructor);
  if (writer_schema.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to retrieve schema from file ", filename));
  }

  // Get the writer interface and initialize the value for that interface
  AvroValueInterfacePtr writer_iface(avro_generic_class_from_schema(writer_schema.get()),
    AvroValueInterfaceDestructor);
  if (writer_iface.get() == nullptr) {
    // TODO(fraudies): Get a string representation of the schema, use avro_schema_to_json
    return Status(errors::ResourceExhausted("Unable to create interface for schema"));
  }

  // Create writer value
  if (avro_generic_value_new(writer_iface.get(), reader->writer_value_.get()) != 0) {
    return Status(errors::InvalidArgument(
        "Unable to create instance for generic class."));
  }

  // Create a reader schema for the user passed string
  AvroSchemaUPtr reader_schema(new avro_schema_t, AvroSchemaDestructor);
  // Create value to read into using the provided schema
  if (avro_schema_from_json_length(reader_schema_str.data(),
                                   reader_schema_str.length(),
                                   reader_schema.get()) != 0) {
    return Status(errors::InvalidArgument(
        "The provided json schema is invalid. ", avro_strerror()));
  }

  // Create resolved writer class
  AvroValueInterfacePtr reader_iface(avro_resolved_writer_new(writer_schema.get(),
    reader_schema.get()));
  if (writer_iface_.get() == nullptr) {
    // TODO(fraudies): Print the schemas in the error message
    return Status(errors::InvalidArgument("Schemas are incompatible. ",
                                          avro_strerror()));
  }

  // Create instance for resolved writer class
  if (avro_resolved_writer_new_value(writer_iface_.get(), reader->writer_value_.get()) != 0) {
    return Status(
        errors::InvalidArgument("Unable to create resolved writer."));
  }

  avro_resolved_writer_set_dest(reader->writer_value_.get(), reader->reader_value_.get());

  return Status::OK();
}

Status AvroResolvedMemReader::ReadNext(avro_value_t* value) {
  // Reset the value for the next read
  avro_value_reset(reader_value_.get());
  bool at_end =
    avro_file_reader_read_value(file_reader_.get(), reader_value_.get()) != 0;
  // Transfer ownership
  avro_value_move_ref(value, reader_value_.get());
  return at_end ? errors::OutOfRange("eof") : Status::OK();
}

Status AvroResolvedMemReader::Resolve(bool* resolve, const std::shared_ptr<void>& mem_data,
  const uint64 mem_size, const string& reader_schema_str, const string& filename) {

  // No schema supplied => no schema resolution is necessary
  if (reader_schema.length() <= 0) {
    *resolve = false;
    return Status::OK();
  }

  // Open the file to get the writer schema
  FileUPtr file(fmemopen(mem_data.get(), mem_size, "r"), FileDestructor);
  if (file.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to open file ", filename, " for memory."));
  }

  // Open the avro file reader
  AvroFileReaderUPtr file_reader(new avro_file_reader_t, AvroFileReaderDestructor);
  if (avro_file_reader_fp(reader->file_.get(), filename.c_str(), 1, file_reader.get()) != 0) {
    return Status(errors::InvalidArgument("Unable to open file ", filename,
                                          " in avro reader. ", avro_strerror()));
  }

  // Get the writer schema
  AvroSchemaUPtr writer_schema(avro_file_reader_get_writer_schema(file_reader.get()),
             AvroSchemaDestructor);
  if (writer_schema.get() == nullptr) {
    return Status(errors::InvalidArgument("Unable to retrieve schema from file ", filename));
  }

  // Create a reader schema for the user passed string
  AvroSchemaUPtr reader_schema(new avro_schema_t, AvroSchemaDestructor);
  if (avro_schema_from_json_length(reader_schema_str.data(),
                                   reader_schema_str.length(),
                                   reader_schema.get()) != 0) {
    return Status(errors::InvalidArgument(
        "The provided json schema is invalid. ", avro_strerror()));
  }

  // Do resolve only if the schemas are different
  *resolve = !avro_schema_equal(writer_schema.get(), reader_schema.get());

  return Status::OK();
}
*/

}  // namespace data
}  // namespace tensorflow