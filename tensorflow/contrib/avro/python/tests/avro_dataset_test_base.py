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

# Examples: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/data/experimental/kernel_tests/stats_dataset_test_base.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import six
import shutil
import tempfile

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import test_util, sparse_tensor
from tensorflow.python.framework.errors import OpError, OutOfRangeError
from tensorflow.contrib.avro.python.avro_dataset import AvroDatasetV1
from tensorflow.contrib.avro.python.utils.avro_serialization import \
  AvroRecordsToFile


class AvroDatasetTestBase(test_util.TensorFlowTestCase):

  def __init__(self, *args, **kwargs):
    super(AvroDatasetTestBase, self).__init__(*args, **kwargs)
    self.output_dir = ''
    self.filename = ''
    self.schema = ''
    self.actual_records = []
    self.features = {}
    self.expected_tensors = []

  def setUp(self):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)

    # Write test records into temporary output directory
    self.output_dir = tempfile.mkdtemp()
    self.filename = os.path.join(self.output_dir, "test.avro")
    writer = AvroRecordsToFile(filename=self.filename, writer_schema=self.schema)
    writer.write_records(self.actual_records)

  def tearDown(self):
    shutil.rmtree(self.output_dir)

  def _assert_same_tensors(self, expected, actual):
    logging.info("Expected {}".format(expected))
    logging.info("Actual {}".format(actual))

    assert len(expected) == len(actual), \
      "Expected length {} but got actual length {}".format(
          len(expected), len(actual))
    for expected_item, actual_item in zip(expected, actual):

      assert len(expected_item) == len(actual_item), \
        "Expected {} pairs but " "got {} pairs".format(
            len(expected_item), len(actual_item))

      for name, actual_tensor in six.iteritems(actual_item):
        assert name in expected_item, "Expected key {} be present in {}".format(
            name, actual.keys())
        expected_tensor = expected_item[name]

        def assert_same_array(expected_array, actual_array):
          if np.issubdtype(actual_array.dtype, np.number):
            self.assertAllClose(expected_array, actual_array)
          else:
            self.assertAllEqual(expected_array, actual_array)

        # Sparse tensor?
        if isinstance(actual_tensor, sparse_tensor.SparseTensorValue):
          self.assertAllEqual(expected_tensor.indices, actual_tensor.indices)
          assert_same_array(expected_tensor.values, actual_tensor.values)
        else:
          assert_same_array(expected_tensor, actual_tensor)

  def _read_all_tensors(self):
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tensors = []
    with self.test_session(config=config) as sess:

      dataset = AvroDatasetV1(filenames=[self.filename], features=self.features)
      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()
      sess.run(iterator.initializer)

      while True:
        try:
          e = sess.run(next_element)
          logging.info("Next element: {}".format(e))
          tensors.append(e)
        except OutOfRangeError:
          break
    return tensors

  def test(self):
    actual_tensors = self._read_all_tensors()
    self._assert_same_tensors(expected=self.expected_tensors,
                              actual=actual_tensors)