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
from tensorflow.contrib.avro.python.avro_dataset import make_avro_datasetV1
from tensorflow.contrib.avro.python.utils.avro_serialization import \
  AvroRecordsToFile


class AvroDatasetTestBase(test_util.TensorFlowTestCase):

  def __init__(self, batch_size, *args, **kwargs):
    super(AvroDatasetTestBase, self).__init__(*args, **kwargs)
    self._batch_size = batch_size
    self._output_dir = ''
    self._filename = ''
    self._schema = ''
    self._actual_records = []
    self._features = {}
    self._expected_tensors = []

  def setUp(self):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)

    # Write test records into temporary output directory
    self._output_dir = tempfile.mkdtemp()
    self._filename = os.path.join(self._output_dir, "test.avro")
    writer = AvroRecordsToFile(filename=self._filename,
                               writer_schema=self._schema)
    writer.write_records(self._actual_records)

  def tearDown(self):
    shutil.rmtree(self._output_dir)

  def _assert_same_tensors(self, expected_tensors, actual_tensors):

    assert len(expected_tensors) == len(actual_tensors), \
      "Expected {} pairs but " "got {} pairs".format(
          len(expected_tensors), len(actual_tensors))

    for name, actual_tensor in six.iteritems(actual_tensors):
      assert name in expected_tensors, "Expected key {} be present in {}"\
        .format(name, actual_tensors.keys())
      expected_tensor = expected_tensors[name]

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

  def _verify_output(self):
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with self.test_session(config=config) as sess:
      # Turn off any parallelism and random for testing to have
      # reproducibility
      dataset = make_avro_datasetV1(file_pattern=[self._filename],
                                    features=self._features,
                                    batch_size=self._batch_size,
                                    shuffle=False,
                                    num_epochs=1)

      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()
      sess.run(iterator.initializer)

      for expected_tensors in self._expected_tensors:

        self._assert_same_tensors(expected_tensors=expected_tensors,
                                  actual_tensors=sess.run(next_element))

      with self.assertRaises(OutOfRangeError):
        sess.run(next_element)

  def test(self):
    self._verify_output()
