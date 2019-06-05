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

from tensorflow.contrib.avro.python.tests import avro_dataset_test_base as \
  avro_test_base


class AvroDatasetTest(avro_test_base.AvroDatasetTestBase):

  def test_primitive_type(self):
    writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
    record_data = [
      {"int_value": 0},
      {"int_value": 1},
      {"int_value": 2}
    ]
    features = {
      "int_value": parsing_ops.FixedLenFeature([], tf_types.int32)
    }
    expected_tensors = [
      {"int_value": np.asarray([0, 1, 2])}
    ]
    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=3, num_epochs=1)

  def test_batching(self):

    writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
    record_data = [
      {"int_value": 0},
      {"int_value": 1},
      {"int_value": 2}
    ]
    features = {
      "int_value": parsing_ops.FixedLenFeature([], tf_types.int32)
    }
    expected_tensors = [
      {"int_value": np.asarray([0, 1])},
      {"int_value": np.asarray([2])}
    ]

    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=2, num_epochs=1)

  def test_fixed_length_list(self):
    writer_schema = """{
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
    record_data = [
      {"int_list": [0, 1, 2]},
      {"int_list": [3, 4, 5]},
      {"int_list": [6, 7, 8]}
    ]
    features = {
      "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32)
    }
    expected_tensors = [
      {"int_list[*]": np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]])}
    ]

    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=3, num_epochs=1)

  def test_fixed_length_with_default_vector(self):
    writer_schema = """{
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
    record_data = [
      {"int_list": [0, 1, 2]},
      {"int_list": [3]},
      {"int_list": [6, 7]}
    ]
    features = {
      "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32,
                                                 default_value=[0, 1, 2])
    }
    expected_tensors = [
      {"int_list[*]": np.asarray([
        [0, 1, 2],
        [3, 1, 2],
        [6, 7, 2]])
      }
    ]

    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=3, num_epochs=1)

  def test_fixed_length_with_default_scalar(self):
    writer_schema = """{
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
    record_data = [
      {"int_list": [0, 1, 2]},
      {"int_list": [3]},
      {"int_list": [6, 7]}
    ]
    features = {
      "int_list[*]": parsing_ops.FixedLenFeature([], tf_types.int32,
                                                 default_value=0)
    }
    expected_tensors = [
      {"int_list[*]": np.asarray([
        [0, 1, 2],
        [3, 0, 0],
        [6, 7, 0]])
      }
    ]

    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=3, num_epochs=1)

  def test_dense_2d(self):
    writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items":
                          {
                             "name" : "name",
                             "type" : "record",
                             "fields" : [
                                {
                                   "name": "nested_int_list",
                                   "type":
                                      {
                                          "type": "array",
                                          "items": "int"
                                      }
                                }
                             ]
                          }
                     }
                  }
              ]}"""
    record_data = [
      {"int_list": [
        {"nested_int_list": [1, 2, 3]},
        {"nested_int_list": [4, 5, 6]}
      ]},
      {"int_list": [
        {"nested_int_list": [7, 8, 9]},
        {"nested_int_list": [10, 11, 12]}
      ]}
    ]
    features = {
      "int_list[*].nested_int_list[*]":
        parsing_ops.FixedLenFeature([2, 3], tf_types.int32)
    }
    expected_tensors = [
      {"int_list[*].nested_int_list[*]":
         np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])}
    ]

    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=2, num_epochs=1)

  def test_dense_3d(self):
    writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items":
                          {
                             "name" : "wrapper1",
                             "type" : "record",
                             "fields" : [
                                {
                                   "name": "nested_int_list",
                                   "type": {
                                      "type": "array",
                                      "items":
                                        {
                                           "name" : "wrapper2",
                                           "type" : "record",
                                           "fields" : [
                                              {
                                                 "name": "nested_nested_int_list",
                                                 "type":
                                                    {
                                                        "type": "array",
                                                        "items": "int"
                                                    }
                                              }
                                           ]
                                        }
                                   }
                                }
                             ]
                          }
                     }
                  }
              ]}"""
    record_data = [
      {"int_list": [
        {"nested_int_list":
          [
            {"nested_nested_int_list": [1, 2, 3, 4]},
            {"nested_nested_int_list": [5, 6, 7, 8]}
          ]
        },
        {"nested_int_list":
          [
            {"nested_nested_int_list": [9, 10, 11, 12]},
            {"nested_nested_int_list": [13, 14, 15, 16]}
          ]
        },
        {"nested_int_list":
          [
            {"nested_nested_int_list": [17, 18, 19, 20]},
            {"nested_nested_int_list": [21, 22, 23, 24]}
          ]
        },
      ]}
    ]
    features = {
      "int_list[*].nested_int_list[*].nested_nested_int_list[*]":
        parsing_ops.FixedLenFeature([3, 2, 4], tf_types.int32)
    }
    expected_tensors = [
      {"int_list[*].nested_int_list[*].nested_nested_int_list[*]": np.asarray(
          [[
            [
              [1, 2, 3, 4],
              [5, 6, 7, 8]
            ],
            [
              [9, 10, 11, 12],
              [13, 14, 15, 16]
            ],
            [
              [17, 18, 19, 20],
              [21, 22, 23, 24]
            ]
          ]]
          )},
    ]

    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=1, num_epochs=1)

  def test_variable_length(self):
    writer_schema = """{
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
    record_data = [
      {"int_list": [1, 2]},
      {"int_list": [3, 4, 5]},
      {"int_list": [6]}
    ]
    features = {
      'int_list[*]': parsing_ops.VarLenFeature(tf_types.int32)
    }
    expected_tensors = [
      {"int_list[*]":
         sparse_tensor.SparseTensorValue(
             np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0]]),
             np.asarray([1, 2, 3, 4, 5, 6]),
             np.asarray([2, 6])
         )
      }
    ]

    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=3, num_epochs=1)

  def test_sparse_feature(self):
    writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                {
                  "name": "sparse_type",
                  "type": {
                    "type": "array",
                    "items": {
                       "type": "record",
                       "name": "sparse_triplet",
                       "fields": [
                          {
                             "name":"index",
                             "type":"long"
                          },
                          {
                             "name":"value",
                             "type":"float"
                          }
                       ]
                    }
                 }
              }
        ]}"""
    record_data = [
      {"sparse_type": [{"index": 0, "value": 5.0}, {"index": 3, "value": 2.0}]},
      {"sparse_type": [{"index": 2, "value": 7.0}]},
    ]
    features = {
      "sparse_type": parsing_ops.SparseFeature(index_key="index",
                                               value_key="value",
                                               dtype=tf_types.float32,
                                               size=4)
    }
    expected_tensors = [
      {"sparse_type": sparse_tensor.SparseTensorValue(
          np.asarray([[0, 0], [0, 3], [1, 2]]),
          np.asarray([5.0, 2.0, 7.0]),
          np.asarray([2, 3]))}
    ]
    self._test_dataset(writer_schema=writer_schema, record_data=record_data,
                       expected_tensors=expected_tensors, features=features,
                       batch_size=2, num_epochs=1)


if __name__ == "__main__":
  test.main()
