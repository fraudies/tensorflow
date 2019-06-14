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

  # Not supported for now, will actually provide another dimension for filter
  # that can't be properly coerced
  # def test_filter_of_sparse_feature(self):
  #   writer_schema = """
  #     {
  #        "type": "record",
  #        "name": "data_row",
  #        "fields": [
  #           {
  #              "name": "guests",
  #              "type": {
  #                 "type": "array",
  #                 "items": {
  #                    "type": "record",
  #                    "name": "person",
  #                    "fields": [
  #                       {
  #                          "name": "name",
  #                          "type": "string"
  #                       },
  #                       {
  #                          "name": "gender",
  #                          "type": "string"
  #                       },
  #                       {
  #                          "name": "address",
  #                          "type": {
  #                             "type": "array",
  #                             "items": {
  #                                "type": "record",
  #                                "name": "postal",
  #                                "fields": [
  #                                   {
  #                                      "name":"street",
  #                                      "type":"string"
  #                                   },
  #                                   {
  #                                      "name":"zip",
  #                                      "type":"long"
  #                                   },
  #                                   {
  #                                      "name":"street_no",
  #                                      "type":"int"
  #                                   }
  #                                ]
  #                             }
  #                          }
  #                       }
  #                    ]
  #                 }
  #              }
  #           }
  #        ]
  #     }
  #     """
  #   record_data = [{
  #     "guests": [{
  #       "name":
  #         "Hans",
  #       "gender":
  #         "male",
  #       "address": [{
  #         "street": "California St",
  #         "zip": 94040,
  #         "state": "CA",
  #         "street_no": 1
  #       }, {
  #         "street": "New York St",
  #         "zip": 32012,
  #         "state": "NY",
  #         "street_no": 2
  #       }]
  #     }, {
  #       "name":
  #         "Mary",
  #       "gender":
  #         "female",
  #       "address": [{
  #         "street": "Ellis St",
  #         "zip": 29040,
  #         "state": "MA",
  #         "street_no": 3
  #       }]
  #     }]
  #   }]
  #   features = {
  #     "guests[gender='female'].address":
  #       parsing_ops.SparseFeature(
  #           index_key="zip",
  #           value_key="street_no",
  #           dtype=tf_types.int32,
  #           size=94040)
  #   }
  #   expected_tensors = [
  #     {
  #       "guests[gender='female'].address":
  #         sparse_tensor.SparseTensorValue(
  #             np.asarray([[0, 29040]]), np.asarray([3]),
  #             np.asarray([1, 94040]))
  #     }
  #   ]
  #   self._test_pass_dataset(writer_schema=writer_schema,
  #                           record_data=record_data,
  #                           expected_tensors=expected_tensors,
  #                           features=features,
  #                           batch_size=2, num_epochs=1)


if __name__ == "__main__":
  log_root = logging.getLogger()
  log_root.setLevel(logging.INFO)
  test.main()
