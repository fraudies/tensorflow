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
from tensorflow.python.ops import parsing_ops
from tensorflow.python.framework import sparse_tensor

from tensorflow.contrib.avro.python.tests import avro_dataset_test_base as avro_test_base


class AvroDatasetPrimitiveTypeTest(avro_test_base.AvroDatasetTestBase):

  def __init__(self, *args, **kwargs):
    super(AvroDatasetPrimitiveTypeTest, self).__init__(*args, **kwargs)
    self.schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
    self.actual_records = [
      {"int_value": 0},
      {"int_value": 1},
      {"int_value": 2}
    ]
    self.features = {
      "int_value": parsing_ops.FixedLenFeature([], tf_types.int32, default_value=0)
    }
    self.expected_tensors = [
      {"int_value": np.asarray([0])},
      {"int_value": np.asarray([1])},
      {"int_value": np.asarray([2])}
    ]


class AvroDatasetFixedLengthListTest(avro_test_base.AvroDatasetTestBase):

  def __init__(self, *args, **kwargs):
    super(AvroDatasetFixedLengthListTest, self).__init__(*args, **kwargs)
    self.schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
    self.actual_records = [
      {"int_list": [0, 1, 2]},
      {"int_list": [3, 4, 5]},
      {"int_list": [6, 7, 8]}
    ]
    self.features = {
      "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32)
    }
    self.expected_tensors = [
      {"int_list[*]": np.asarray([0, 1, 2])},
      {"int_list[*]": np.asarray([3, 4, 5])},
      {"int_list[*]": np.asarray([6, 7, 8])}
    ]

# Not implemented yet
# class AvroDatasetSparseFeatureTest(avro_test_base.AvroDatasetTestBase):
#
#   def __init__(self, *args, **kwargs):
#     super(AvroDatasetSparseFeatureTest, self).__init__(*args, **kwargs)
#     self.schema = """{
#               "type": "record",
#               "name": "row",
#               "fields": [
#                 {
#                   "name": "sparse_type",
#                   "type": {
#                     "type": "array",
#                     "items": {
#                        "type": "record",
#                        "name": "sparse_triplet",
#                        "fields": [
#                           {
#                              "name":"index",
#                              "type":"long"
#                           },
#                           {
#                              "name":"value",
#                              "type":"float"
#                           }
#                        ]
#                     }
#                  }
#               }
#         ]}"""
#     self.actual_records = [
#       {"sparse_type": [{"index": 0, "value": 5.0}, {"index": 3, "value": 2.0}]},
#       {"sparse_type": [{"index": 2, "value": 7.0}]},
#     ]
#     self.features = {
#       "sparse_type": parsing_ops.SparseFeature(index_key="index",
#                                                value_key="value",
#                                                dtype=tf_types.float32,
#                                                size=4)
#     }
#     self.expected_tensors = [
#       {"sparse_type": sparse_tensor.SparseTensorValue(
#           np.asarray([0, 3]), np.asarray([5.0, 2.0]), np.asarray([2]))},
#       {"sparse_type": sparse_tensor.SparseTensorValue(
#           np.asarray([2]), np.asarray([7.0]), np.asarray([1]))}
#     ]


# Makes only sense with batching
# class AvroDatasetVariableLengthListTest(avro_test_base.AvroDatasetTestBase):
#
#   def __init__(self, *args, **kwargs):
#     super(AvroDatasetVariableLengthListTest, self).__init__(*args, **kwargs)
#     self.schema = """{
#               "type": "record",
#               "name": "row",
#               "fields": [
#                   {
#                      "name": "int_list",
#                      "type": {
#                         "type": "array",
#                         "items": "int"
#                      }
#                   }
#               ]}"""
#     self.actual_records = [
#       {"int_list": [1, 2]},
#       {"int_list": [3, 4, 5]},
#       {"int_list": [6]}
#     ]
#     self.features = {
#       'int_list[*]': parsing_ops.VarLenFeature(tf_types.int32)
#     }
#     self.expected_tensors = [
#       {"int_list[*]":
#          sparse_tensor.SparseTensorValue(
#              np.asarray([0, 1]),
#              np.asarray([1, 2]),
#              np.asarray([1, 2])
#          )
#       },
#       {"int_list[*]":
#         sparse_tensor.SparseTensorValue(
#             np.asarray([0, 1, 2]),
#             np.asarray([3, 4, 5]),
#             np.asarray([1, 3])
#         )
#       },
#       {"int_list[*]":
#         sparse_tensor.SparseTensorValue(
#             np.asarray([0]),
#             np.asarray([6]),
#             np.asarray([1, 1])
#         )
#       }
#     ]


if __name__ == "__main__":
  test.main()