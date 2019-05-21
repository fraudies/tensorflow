# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import shutil
import tempfile

from avro.io import DatumWriter
from avro.datafile import DataFileWriter

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import OpError, OutOfRangeError
from tensorflow.python.ops import parsing_ops
from tensorflow.contrib.avro.python.avro_dataset import AvroDatasetV1
from tensorflow.contrib.avro.python.utils.avro_serialization import \
  AvroDeserializer, AvroParser, AvroSchemaReader, AvroFileToRecords


class AvroDatasetTest(test_util.TensorFlowTestCase):

  def __init__(self, *args, **kwargs):
    super(AvroDatasetTest, self).__init__(*args, **kwargs)

    # Set by setup
    self.output_dir = ''
    self.filename = ''

    self.full_schema = """{
              "doc": "Test schema for avro records.",
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "index", "type": "int"}
              ]}"""

    self.test_records = [
      {"index": 0},
      {"index": 1},
      {"index": 2},
      {"index": 3},
      {"index": 4}
    ]

  @staticmethod
  def _write_records_to_file(records,
      filename,
      writer_schema,
      codec='deflate'):
    """
    Writes the string data into an avro encoded file

    :param records: Records to write
    :param filename: Filename for the file to be written
    :param writer_schema: The schema used when writing the records
    :param codec: Compression codec used to write the avro file
    """
    schema = AvroParser(writer_schema).get_schema_object()
    with open(filename, 'wb') as out:
      writer = DataFileWriter(
          out, DatumWriter(), schema, codec=codec)
      for record in records:
        writer.append(record)
      writer.close()

  def setUp(self):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)

    # Write test records into temporary output directory
    self.output_dir = tempfile.mkdtemp()
    self.filename = os.path.join(self.output_dir, "test.avro")

    print("Created dummy data {}", self.filename)

    AvroDatasetTest._write_records_to_file(
        records=self.test_records,
        writer_schema=self.full_schema,
        filename=self.filename)

  #def tearDown(self):
  #  shutil.rmtree(self.output_dir)

  def test_reading_data(self):
    logging.info("Running test for reading data")
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    with self.test_session(config=config) as sess:
      # Note, currently there is no easy way of supporting [] w/out batching
      features = {
        'index': parsing_ops.FixedLenFeature([1], tf_types.int32, default_value=0)
      }

      dataset = AvroDatasetV1(filenames=[self.filename], features=features)
      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()

      sess.run(iterator.initializer)

      while True:
        try:
          record_actual = sess.run(next_element)
          logging.info(record_actual)

        except OutOfRangeError:
          logging.info("Done")
          break


if __name__ == "__main__":
  test.main()
