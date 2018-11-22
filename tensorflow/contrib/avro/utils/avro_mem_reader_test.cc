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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace data {

TEST(AvroMemReaderTest, Create) {
  //static const uint64 MEM_SIZE = 65*1024;
  static char SCHEMA_JSON[] =
    "{"
    "  \"type\": \"record\","
    "  \"name\": \"test\","
    "  \"fields\": ["
    "    { \"name\": \"b\", \"type\": \"boolean\" },"
    "    { \"name\": \"i\", \"type\": \"int\" },"
    "    { \"name\": \"s\", \"type\": \"string\" }"
    "  ]"
    "}";
  /*
  avro_schema_t schema;
  char mem_data[MEM_SIZE];

  // Parse schema string
  if (avro_schema_from_json(SCHEMA_JSON, 0, &schema, NULL) != 0) {
    fprintf(stderr, "failed to parse  schema (%s)\n", avro_strerror());
  }

  // Write schema to memory
  avro_writer_t writer = avro_writer_memory(mem_data, MEM_SIZE);
  if (avro_schema_to_json(SCHEMA_JSON, writer)) {
    fprintf(stderr, "failed to write schema (%s)\n", avro_strerror());
  }
  avro_write(writer, (void *)"", 1);  // zero terminate
  avro_writer_free(writer);
  avro_schema_decref(schema);
  */
  // This should be good enough to have a valid schema

  AvroMemReader* reader = nullptr;
  std::string filename = "DummyExample.avro";
  //std::shared_ptr<void> mem_data(static_cast<void*>(SCHEMA_JSON));

  //TF_EXPECT_OK(AvroMemReader::Create(reader, mem_data, sizeof(SCHEMA_JSON), filename));
}

}  // namespace data
}  // namespace tensorflow
