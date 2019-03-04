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

#include <thread>
#include "tensorflow/contrib/avro/utils/avro_mem_reader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that. Instead these tests only cover the wrapping in the mem readers
namespace tensorflow {
namespace data {

const string schema =
          "{"
          "  \"type\": \"record\","
          "  \"name\": \"person\","
          "  \"fields\": ["
          "    { \"name\": \"name\", \"type\": \"string\" },"
          "    { \"name\": \"age\", \"type\": \"int\" }"
          "  ]"
          "}";

const string resolved =
          "{"
          "  \"type\": \"record\","
          "  \"name\": \"person\","
          "  \"fields\": ["
          "    { \"name\": \"name\", \"type\": \"string\" }"
          "  ]"
          "}";

const int N_RECORD = 10;

class AvroField {
public:
  void GetName(string* name) const { *name = name_; }
  void GetType(avro_type_t* type) const { *type = type_; };
  virtual void GetValue(void* value) const = 0;
  virtual ~AvroField() { }
protected:
  AvroField(const string& name, avro_type_t type) : name_(name), type_(type) { }
private:
  string name_;
  avro_type_t type_;
};

class StringField : public AvroField {
public:
  StringField(const string& name, avro_type_t type, const string& value) :
    AvroField(name, type), value_(value) {}
  void GetValue(void* value) const {
    *(static_cast<string*>(value)) = value_;
  }
private:
  string value_;
};

class IntField : public AvroField {
public:
  IntField(const string& name, avro_type_t type, int value) :
    AvroField(name, type), value_(value) {}
  void GetValue(void* value) const {
    *(static_cast<int*>(value)) = value_;
  }
private:
  int value_;
};

class AvroUtils {
public:
  // Writes a simple valid avro file to the given path
  // Could improve by factoring out the schema, values, writing of values
  static Status WriteAvroFile(const string& filename, const std::vector<AvroField*>& fields, int n_record) {
    avro_set_error(""); // don't carry over any errors

    // Parse schema
    std::unique_ptr<avro_schema_t, std::function<void(avro_schema_t*)>> writer_schema(
      new avro_schema_t,
      [](avro_schema_t* ptr) { CHECK_GE(avro_schema_decref(*ptr), 0); }
    );
    if (avro_schema_from_json_length(schema.data(), schema.length(), writer_schema.get()) != 0) {
      return Status(errors::InvalidArgument("Unable to retrieve schema from file ", filename));
    }

    // Open file and write schema (file will be flushed and closed once out of scope)
    std::unique_ptr<avro_file_writer_t, std::function<void(avro_file_writer_t*)>> file_writer(
      new avro_file_writer_t,
      [](avro_file_writer_t* ptr) { CHECK_GE(avro_file_writer_close(*ptr), 0); }
    );
    if (avro_file_writer_create(filename.c_str(), *writer_schema, file_writer.get()) != 0) {
      return Status(errors::InvalidArgument("Unable to open file ", filename));
    }

    // Add a person
    // Create the iface
    std::unique_ptr<avro_value_iface_t, std::function<void(avro_value_iface_t*)>> iface(
      avro_generic_class_from_schema(*writer_schema),
      [](avro_value_iface_t* ptr) { avro_value_iface_decref(ptr); }
    );
    if (iface.get() == nullptr) {
      return Status(errors::InvalidArgument("Unable to create iface for schema ", schema));
    }

    // Create the generic value
    std::unique_ptr<avro_value_t, std::function<void(avro_value_t*)>> value(
      new avro_value_t,
      [](avro_value_t* ptr) { avro_value_decref(ptr); }
    );
    if (avro_generic_value_new(iface.get(), value.get())) {
      return Status(errors::InvalidArgument("Unable to create value for iface with schema ", schema));
    }

    // Fill fields in value
    for (auto field : fields) {
      TF_RETURN_IF_ERROR(SetFieldInValue(value, *field));
    }

    for (int i_record = 0; i_record < n_record; ++i_record) {
      if (avro_file_writer_append_value(*file_writer, value.get())) {
        return Status(errors::InvalidArgument("Unable to write value"));
      }
    }

    avro_file_writer_sync(*file_writer); // When no records are written

    return Status::OK();
  }

  static Status CheckValue(const avro_value_t& value, const std::vector<AvroField*>& fields) {
    for (auto field : fields) {
      TF_RETURN_IF_ERROR(CheckFieldInValue(value, *field));
    }
    return Status::OK();
  }

  static Status CheckResolvedValue(const avro_value_t& value, const AvroField& field) {
    TF_RETURN_IF_ERROR(CheckFieldInValue(value, field));
    return Status::OK();
  }

  static Status ReadFileIntoMem(std::unique_ptr<char[]>& mem_data, uint64* mem_size,
    const string& filename) {
    // Unfortunately, this somewhat violates the google c++ style guide... because smart pointers
    // need to be returned through &  -- the alternative could have been to use char** mem_data and
    // manage the memory without using smart pointers

    FILE* fp = fopen(filename.c_str(), "r");
    if (fp == nullptr) {
      return Status(errors::InvalidArgument("Unable to open file ", filename));
    }

    // Find the size of the file in By
    fseek(fp, 0, SEEK_END);
    *mem_size = ftell(fp);

    // Create memory for the entire file
    mem_data.reset(new (std::nothrow) char[*mem_size]);
    if (mem_data.get() == nullptr) {
      return Status(errors::InvalidArgument("Unable to allocate ", *mem_size, " B of memory"));
    }

    // Read the entire file into memory
    rewind(fp);
    if (fgets(mem_data.get(), *mem_size, fp) == nullptr) {
      return Status(errors::InvalidArgument("Unable read ", *mem_size, " bytes from file ", filename));
    }
    fclose(fp);

    return Status::OK();
  }

  static void LogValue(const avro_value_t& value) {
    char *json;
    if (avro_value_to_json(&value, 1, &json)) {
      LOG(ERROR) << "Error when converting value to JSON: " << avro_strerror();
    } else {
      LOG(INFO) << json;
      free(json);
    }
  }
private:
  // Fills direct attribute values of the avro value
  // Does not support nesting setting of nested fields
  // Fills only string and int types
  static Status SetFieldInValue(
    const std::unique_ptr<avro_value_t, std::function<void(avro_value_t*)>>& value,
    const AvroField& field) {

    string name;
    avro_type_t type;
    field.GetName(&name);
    field.GetType(&type);

    avro_value_t field_ref;
    if (avro_value_get_by_name(value.get(), name.c_str(), &field_ref, NULL)) {
      return Status(errors::InvalidArgument("Unable to get field ", name));
    }

    switch (type) {
      case (AVRO_STRING): {
        string field_value;
        field.GetValue(static_cast<void*>(&field_value));
        if (avro_value_set_string(&field_ref, field_value.c_str())) {
          return Status(errors::InvalidArgument("Unable to set '", field_value, "' to field ", name));
        }
        LOG(INFO) << "Set value " << field_value;
        break;
      }
      case (AVRO_INT32): {
        int field_value;
        field.GetValue(static_cast<void*>(&field_value));
        if (avro_value_set_int(&field_ref, field_value)) {
          return Status(errors::InvalidArgument("Unable to set '", field_value, "' to field ", name));
        }
        LOG(INFO) << "Set value " << field_value;
        break;
      }
      default: {
        return Status(errors::InvalidArgument("Type ", type, " is not supported"));
      }
    }
    return Status::OK();
  }

  static Status CheckFieldInValue(const avro_value_t& value, const AvroField& field) {

    string name;
    avro_type_t type;
    field.GetName(&name);
    field.GetType(&type);

    avro_value_t field_ref;
    if (avro_value_get_by_name(&value, name.c_str(), &field_ref, NULL)) {
      return Status(errors::InvalidArgument("Unable to get field ", name));
    }

    switch (type) {
      case (AVRO_STRING): {
        // Get the expected value
        string value_expected;
        field.GetValue(static_cast<void*>(&value_expected));

        // Get the actual value
        const char* ptr_actual = nullptr;
        size_t len_actual = 0;
        if (avro_value_get_string(&field_ref, &ptr_actual, &len_actual)) {
          return Status(errors::InvalidArgument("Unable to get string value ", name));
        }
        string value_actual(ptr_actual, len_actual);

        // Compare these values
        if (strcmp(value_actual.c_str(), value_expected.c_str()) != 0) {
          return Status(errors::InvalidArgument("Actual ", value_actual, " expected ", value_expected));
        }
        break;
      }
      case (AVRO_INT32): {
        // Get the expected value
        int value_expected;
        field.GetValue(static_cast<void*>(&value_expected));

        // Get the actual value
        int value_actual;
        if (avro_value_get_int(&field_ref, &value_actual)) {
          return Status(errors::InvalidArgument("Unable to get int value for ", name));
        }

        // Compare these values
        if (value_actual != value_expected) {
          return Status(errors::InvalidArgument("Actual ", value_actual, " expected ", value_expected));
        }
        break;
      }
      default: {
        return Status(errors::InvalidArgument("Type ", type, " is not supported"));
      }
    }
    return Status::OK();
  }
};

class AvroMemReaderTest : public ::testing::Test {
  protected:
    static void SetUpTestCase() {
      fields_.push_back(new StringField("name", AVRO_STRING, "Karl Steinbach"));
      fields_.push_back(new IntField("age", AVRO_INT32, 32));

      filename_ = io::GetTempFilename("avro");
      TF_EXPECT_OK(AvroUtils::WriteAvroFile(filename_, fields_, N_RECORD));
      LOG(INFO) << "Created tmp file: " << filename_;

      TF_EXPECT_OK(AvroUtils::ReadFileIntoMem(mem_data_, &mem_size_, filename_));
      LOG(INFO) << "Read file into memory with " << mem_size_ << " By";
    }
    static void TearDownTestCase() {
      remove(filename_.c_str());
      for (auto field : fields_) {
        delete field;
      }
    }

    static std::vector<AvroField*> fields_;
    static std::unique_ptr<char[]> mem_data_;
    static uint64 mem_size_;
    static string filename_;
};

std::unique_ptr<char[]> AvroMemReaderTest::mem_data_ = nullptr;
uint64 AvroMemReaderTest::mem_size_ = 0;
string AvroMemReaderTest::filename_ = "";
std::vector<AvroField*> AvroMemReaderTest::fields_;


class AvroMemReaderEmptyFileTest : public ::testing::Test {
  protected:
    static void SetUpTestCase() {
      filename_ = io::GetTempFilename("avro");
      std::vector<AvroField*> empty_fields;
      TF_EXPECT_OK(AvroUtils::WriteAvroFile(filename_, empty_fields, 0));
      LOG(INFO) << "Created tmp file: " << filename_;

      TF_EXPECT_OK(AvroUtils::ReadFileIntoMem(mem_data_, &mem_size_, filename_));
      LOG(INFO) << "Read file into memory with " << mem_size_ << " By";
    }
    static void TearDownTestCase() {
      remove(filename_.c_str());
    }
    static std::unique_ptr<char[]> mem_data_;
    static uint64 mem_size_;
    static string filename_;
};

std::unique_ptr<char[]> AvroMemReaderEmptyFileTest::mem_data_ = nullptr;
uint64 AvroMemReaderEmptyFileTest::mem_size_ = 0;
string AvroMemReaderEmptyFileTest::filename_ = "";

TEST_F(AvroMemReaderTest, CreateAndDelete) {
  AvroMemReader* reader = new AvroMemReader();
  TF_EXPECT_OK(AvroMemReader::Create(reader, mem_data_, mem_size_, filename_));
  delete reader;
}

/*
TODO(fraudies): Empty file with a memory mapped file seems to be a problem since it returns
Cannot read file block count: Cannot read 1 bytes from file
If an actual file handle is used then this test passes
TEST_F(AvroMemReaderEmptyFileTest, ReadFromEmptyFile) {
  AvroMemReader* reader = new AvroMemReader();
  TF_EXPECT_OK(AvroMemReader::Create(reader, mem_data_, mem_size_, filename_));
  AvroMemReader::AvroValuePtr value(nullptr, AvroMemReader::AvroValueDestructor);
  EXPECT_EQ(reader->ReadNext(value), Status(errors::OutOfRange("eof")));
  delete reader;
}
*/

TEST_F(AvroMemReaderTest, Read) {
  AvroMemReader* reader = new AvroMemReader();
  TF_EXPECT_OK(AvroMemReader::Create(reader, mem_data_, mem_size_, filename_));
  AvroValuePtr value(nullptr, AvroValueDestructor);
  for (int i_record = 0; i_record < N_RECORD; ++i_record) {
    TF_EXPECT_OK(reader->ReadNext(value));
    AvroUtils::LogValue(*value);
    // Ensure the contents of the avro value match
    TF_EXPECT_OK(AvroUtils::CheckValue(*value, fields_));
  }
  EXPECT_EQ(reader->ReadNext(value), Status(errors::OutOfRange("eof")));
  delete reader;
}

TEST_F(AvroMemReaderTest, ReadMultiThreaded) {
  AvroMemReader* reader = new AvroMemReader();
  TF_EXPECT_OK(AvroMemReader::Create(reader, mem_data_, mem_size_, filename_));
  std::thread readers[N_RECORD];
  for (int i_reader = 0; i_reader < N_RECORD; i_reader++) {
    readers[i_reader] = std::thread([&reader] {
      AvroValuePtr value(nullptr, AvroValueDestructor);
      TF_EXPECT_OK(reader->ReadNext(value));
      AvroUtils::LogValue(*value);
      // Ensure the contents of the avro value match
      TF_EXPECT_OK(AvroUtils::CheckValue(*value, fields_));
    });
  }
  for (int i_reader = 0; i_reader < N_RECORD; i_reader++) {
    readers[i_reader].join();
  }
  delete reader;
}

TEST_F(AvroMemReaderTest, ReadResolved) {
  AvroResolvedMemReader* reader = new AvroResolvedMemReader();
  TF_EXPECT_OK(AvroResolvedMemReader::Create(reader, mem_data_, mem_size_, resolved, filename_));
  AvroValuePtr value(nullptr, AvroValueDestructor);
  for (int i_record = 0; i_record < N_RECORD; ++i_record) {
    TF_EXPECT_OK(reader->ReadNext(value));
    AvroUtils::LogValue(*value);
    // Ensure the contents of the avro value match
    TF_EXPECT_OK(AvroUtils::CheckResolvedValue(*value, *fields_[0]));
  }
  EXPECT_EQ(reader->ReadNext(value), Status(errors::OutOfRange("eof")));
  delete reader;
}

TEST_F(AvroMemReaderTest, ReadResolvedMultiThreaded) {
  AvroResolvedMemReader* reader = new AvroResolvedMemReader();
  TF_EXPECT_OK(AvroResolvedMemReader::Create(reader, mem_data_, mem_size_, resolved, filename_));
  std::thread readers[N_RECORD];
  for (int i_reader = 0; i_reader < N_RECORD; i_reader++) {
    readers[i_reader] = std::thread([&reader] {
      AvroValuePtr value(nullptr, AvroValueDestructor);
      TF_EXPECT_OK(reader->ReadNext(value));
      AvroUtils::LogValue(*value);
      // Ensure the contents of the avro value match
      TF_EXPECT_OK(AvroUtils::CheckResolvedValue(*value, *fields_[0]));
    });
  }
  for (int i_reader = 0; i_reader < N_RECORD; i_reader++) {
    readers[i_reader].join();
  }
  delete reader;
}

TEST_F(AvroMemReaderTest, DoNotResolve) {
  bool resolve;
  TF_EXPECT_OK(AvroResolvedMemReader::DoResolve(&resolve, mem_data_, mem_size_, schema, filename_));
  EXPECT_FALSE(resolve);
}

TEST_F(AvroMemReaderTest, DoResolve) {
  bool resolve;
  TF_EXPECT_OK(AvroResolvedMemReader::DoResolve(&resolve, mem_data_, mem_size_, resolved, filename_));
  EXPECT_TRUE(resolve);
}

}  // namespace data
}  // namespace tensorflow
