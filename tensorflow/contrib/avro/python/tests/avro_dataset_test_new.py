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

import numpy as np
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat

from tensorflow.contrib.avro.python.tests import avro_dataset_test_base as \
  avro_test_base


class AvroDatasetTest(avro_test_base.AvroDatasetTestBase):

  def test_dummy(self):
    print("dummy")

  # def test_filter_with_variable_length(self):
  #   writer_schema = """
  #         {
  #            "type": "record",
  #            "name": "data_row",
  #            "fields": [
  #               {
  #                  "name": "guests",
  #                  "type": {
  #                     "type": "array",
  #                     "items": {
  #                        "type": "record",
  #                        "name": "person",
  #                        "fields": [
  #                           {
  #                              "name": "name",
  #                              "type": "string"
  #                           },
  #                           {
  #                              "name": "gender",
  #                              "type": "string"
  #                           }
  #                        ]
  #                     }
  #                  }
  #               }
  #            ]
  #         }
  #         """
  #   record_data = [
  #     {
  #       "guests": [
  #         {
  #           "name": "Hans",
  #           "gender": "male"
  #         },
  #         {
  #           "name": "Mary",
  #           "gender": "female"
  #         },
  #         {
  #           "name": "July",
  #           "gender": "female"
  #         }
  #       ]
  #     },
  #     {
  #       "guests": [
  #         {
  #           "name": "Joel",
  #           "gender": "male"
  #         }, {
  #           "name": "JoAn",
  #           "gender": "female"
  #         }, {
  #           "name": "Marc",
  #           "gender": "male"
  #         }
  #       ]
  #     }
  #   ]
  #   features = {
  #     "guests[gender='male'].name":
  #       parsing_ops.VarLenFeature(tf_types.string),
  #     "guests[gender='female'].name":
  #       parsing_ops.VarLenFeature(tf_types.string)
  #   }
  #   expected_tensors = {
  #     "guests[gender='male'].name":
  #       sparse_tensor.SparseTensorValue(
  #           np.asarray([[0, 0], [1, 0], [1, 1]]),
  #           np.asarray([compat.as_bytes("Hans"), compat.as_bytes("Joel"),
  #                       compat.as_bytes("Marc")]),
  #           np.asarray([2, 1])),
  #     "guests[gender='female'].name":
  #       sparse_tensor.SparseTensorValue(
  #           np.asarray([[0, 0], [0, 1], [1, 0]]),
  #           np.asarray([compat.as_bytes("Mary"), compat.as_bytes("July"),
  #                       compat.as_bytes("JoAn")]),
  #           np.asarray([2, 1]))
  #   }
  #   self._test_pass_dataset(writer_schema=writer_schema,
  #                           record_data=record_data,
  #                           expected_tensors=expected_tensors,
  #                           features=features,
  #                           batch_size=2, num_epochs=1)


if __name__ == "__main__":
  test.main()
