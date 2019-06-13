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
from tensorflow.python.util import compat
from tensorflow.python.framework import sparse_tensor

from tensorflow.contrib.avro.python.tests import avro_dataset_test_base as \
  avro_test_base


class AvroDatasetTest(avro_test_base.AvroDatasetTestBase):

  def test_primitive_types(self):
    writer_schema = """{
              "type": "record",
              "name": "dataTypes",
              "fields": [
                  {  
                     "name":"string_value",
                     "type":"string"
                  },
                  {  
                     "name":"bytes_value",
                     "type":"bytes"
                  },
                  {  
                     "name":"double_value",
                     "type":"double"
                  },
                  {  
                     "name":"float_value",
                     "type":"float"
                  },
                  {  
                     "name":"long_value",
                     "type":"long"
                  },
                  {  
                     "name":"int_value",
                     "type":"int"
                  },
                  {  
                     "name":"boolean_value",
                     "type":"boolean"
                  }
              ]}"""
    record_data = [
      {
        "string_value": "",
        "bytes_value": b"",
        "double_value": 0.0,
        "float_value": 0.0,
        "long_value": 0,
        "int_value": 0,
        "boolean_value": False,
      },
      {
        "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
        "bytes_value": b"SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
        "double_value": 1.7976931348623157e+308,
        "float_value": 3.40282306074e+38,
        "long_value": 9223372036854775807,
        "int_value": 2147483648 - 1,
        "boolean_value": True,
      },
      {
        "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
        "bytes_value": b"ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
        "double_value": -1.7976931348623157e+308,
        "float_value": -3.40282306074e+38,
        "long_value": -9223372036854775807 - 1,
        "int_value": -2147483648,
        "boolean_value": False,
      }
    ]
    features = {
      "string_value": parsing_ops.FixedLenFeature([], tf_types.string),
      "bytes_value": parsing_ops.FixedLenFeature([], tf_types.string),
      "double_value": parsing_ops.FixedLenFeature([], tf_types.float64),
      "float_value": parsing_ops.FixedLenFeature([], tf_types.float32),
      "long_value": parsing_ops.FixedLenFeature([], tf_types.int64),
      "int_value": parsing_ops.FixedLenFeature([], tf_types.int32),
      "boolean_value": parsing_ops.FixedLenFeature([], tf_types.bool)
    }
    expected_tensors = [
      {
        "string_value":
          np.asarray([
            "",
            "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
            "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
          ]),
        "bytes_value":
          np.asarray([
            b"",
            b"SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
            b"ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
          ]),
        "double_value":
          np.asarray([
            0.0,
            1.7976931348623157e+308,
            -1.7976931348623157e+308
          ]),
        "float_value":
          np.asarray([
            0.0,
            3.40282306074e+38,
            -3.40282306074e+38
          ]),
        "long_value":
          np.asarray([
            0,
            9223372036854775807,
            -9223372036854775807-1
          ]),
        "int_value":
          np.asarray([
            0,
            2147483648-1,
            -2147483648
          ]),
        "boolean_value":
          np.asarray([
            False,
            True,
            False
          ])
      }
    ]
    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
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
    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
                            batch_size=2, num_epochs=1)

  def test_nesting(self):

    writer_schema = """
        {
           "type": "record",
           "name": "nesting",
           "fields": [
              {
                 "name": "nested_record",
                 "type": {
                    "type": "record",
                    "name": "nested_values",
                    "fields": [
                       {
                          "name": "nested_int",
                          "type": "int"
                       },
                       {
                          "name": "nested_float_list",
                          "type": {
                             "type": "array",
                             "items": "float"
                          }
                       }
                    ]
                 }
              },
              {
                 "name": "list_of_records",
                 "type": {
                    "type": "array",
                    "items": {
                       "type": "record",
                       "name": "person",
                       "fields": [
                          {
                             "name": "first_name",
                             "type": "string"
                          },
                          {
                             "name": "age",
                             "type": "int"
                          }
                       ]
                    }
                 }
              },
              {
                 "name": "map_of_records",
                 "type": {
                    "type": "map",
                    "values": {
                       "type": "record",
                       "name": "secondPerson",
                       "fields": [
                          {
                             "name": "first_name",
                             "type": "string"
                          },
                          {
                             "name": "age",
                             "type": "int"
                          }
                       ]
                    }
                 }
              }
           ]
        }
        """
    record_data = [
      {
        "nested_record": {
          "nested_int": 0,
          "nested_float_list": [0.0, 10.0]
        },
        "list_of_records": [{
          "first_name": "Herbert",
          "age": 70
        }],
        "map_of_records": {
          "first": {
            "first_name": "Herbert",
            "age": 70
          },
          "second": {
            "first_name": "Julia",
            "age": 30
          }
        }
      },
      {
        "nested_record": {
          "nested_int": 5,
          "nested_float_list": [-2.0, 7.0]
        },
        "list_of_records": [{
          "first_name": "Doug",
          "age": 55
        }, {
          "first_name": "Jess",
          "age": 66
        }, {
          "first_name": "Julia",
          "age": 30
        }],
        "map_of_records": {
          "first": {
            "first_name": "Doug",
            "age": 55
          },
          "second": {
            "first_name": "Jess",
            "age": 66
          }
        }
      },
      {
        "nested_record": {
          "nested_int": 7,
          "nested_float_list": [3.0, 4.0]
        },
        "list_of_records": [{
          "first_name": "Karl",
          "age": 32
        }],
        "map_of_records": {
          "first": {
            "first_name": "Karl",
            "age": 32
          },
          "second": {
            "first_name": "Joan",
            "age": 21
          }
        }
      }
    ]
    features = {
      "nested_record.nested_int": parsing_ops.FixedLenFeature([], tf_types.int32),
      "nested_record.nested_float_list[*]": parsing_ops.FixedLenFeature([2], tf_types.float32),
      "list_of_records[0].first_name": parsing_ops.FixedLenFeature([1], tf_types.string),
      "map_of_records['second'].age": parsing_ops.FixedLenFeature([1], tf_types.int32)
    }
    expected_tensors = [
      {
        "nested_record.nested_int":
          np.asarray([0, 5, 7]),
        "nested_record.nested_float_list[*]":
          np.asarray([[0.0, 10.0], [-2.0, 7.0], [3.0, 4.0]]),
        "list_of_records[0].first_name":
          np.asarray([[compat.as_bytes("Herbert")],
                      [compat.as_bytes("Doug")],
                      [compat.as_bytes("Karl")]]),
        "map_of_records['second'].age":
          np.asarray([30, 66, 21])
      }
    ]
    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
                            batch_size=3, num_epochs=1)

  def test_parse_int_as_long_fail(self):
    writer_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "index",
                   "type": "int"
                }
             ]
          }
          """
    record_data = [{"index": 0}]
    features = {"index": parsing_ops.FixedLenFeature([], tf_types.int64)}
    self._test_fail_dataset(writer_schema, record_data, features, 1)

  def test_parse_int_as_sparse_type_fail(self):
    writer_schema = """
      {
         "type": "record",
         "name": "data_row",
         "fields": [
            {
               "name": "index",
               "type": "int"
            }
         ]
      }        
      """
    record_data = [{"index": 5}]
    features = {
      "index":
        parsing_ops.SparseFeature(
            index_key="index",
            value_key="value",
            dtype=tf_types.float32,
            size=10)
    }
    self._test_fail_dataset(writer_schema, record_data, features, 1)

  def test_parse_float_as_double_fail(self):
    writer_schema = """
      {
         "type": "record",
         "name": "data_row",
         "fields": [
            {
               "name": "weight",
               "type": "float"
            }
         ]
      }
      """
    record_data = [{"weight": 0.5}]
    features = {"weight": parsing_ops.FixedLenFeature([], tf_types.float64)}
    self._test_fail_dataset(writer_schema, record_data, features, 1)

  def test_fixed_length_without_proper_default_fail(self):
    writer_schema = """
      {
         "type": "record",
         "name": "data_row",
         "fields": [
            {
               "name": "int_list_type",
               "type": {
                  "type":"array",
                  "items":"int"
               }
            }
         ]
      }        
      """
    record_data = [
      {
        "int_list_type": [0, 1, 2]
      },
      {
        "int_list_type": [0, 1]
      }
    ]
    features = {
      "int_list_type": parsing_ops.FixedLenFeature([], tf_types.int32)
    }
    self._test_fail_dataset(writer_schema, record_data, features, 1)

  def test_wrong_spelling_of_feature_name_fail(self):
    writer_schema = """
      {
         "type": "record",
         "name": "data_row",
         "fields": [
           {"name": "int_type", "type": "int"}
         ]
      }"""
    record_data = [{"int_type": 0}]
    features = {
      "wrong_spelling": parsing_ops.FixedLenFeature([], tf_types.int32)
    }
    self._test_fail_dataset(writer_schema, record_data, features, 1)

  def test_wrong_index(self):
    writer_schema = """
      {
         "type": "record",
         "name": "data_row",
         "fields": [
            {
               "name": "list_of_records",
               "type": {
                  "type": "array",
                  "items": {
                     "type": "record",
                     "name": "person",
                     "fields": [
                        {
                           "name": "first_name",
                           "type": "string"
                        }
                     ]
                  }
               }
            }
         ]
      }
      """
    record_data = [{
      "list_of_records": [{
        "first_name": "My name"
      }]
    }]
    features = {
      "list_of_records[2].name":
        parsing_ops.FixedLenFeature([], tf_types.string)
    }
    self._test_fail_dataset(writer_schema, record_data, features, 1)

  def test_filter_with_variable_length(self):
    writer_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "guests",
                   "type": {
                      "type": "array",
                      "items": {
                         "type": "record",
                         "name": "person",
                         "fields": [
                            {
                               "name": "name",
                               "type": "string"
                            },
                            {
                               "name": "gender",
                               "type": "string"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
    record_data = [
      {
        "guests": [
          {
            "name": "Hans",
            "gender": "male"
          },
          {
            "name": "Mary",
            "gender": "female"
          },
          {
            "name": "July",
            "gender": "female"
          }
        ]
      },
      {
        "guests": [
          {
            "name": "Joel",
            "gender": "male"
          }, {
            "name": "JoAn",
            "gender": "female"
          }, {
            "name": "Marc",
            "gender": "male"
          }
        ]
      }
    ]
    features = {
      "guests[gender='male'].name":
        parsing_ops.VarLenFeature(tf_types.string),
      "guests[gender='female'].name":
        parsing_ops.VarLenFeature(tf_types.string)
    }
    expected_tensors = [
      {
        "guests[gender='male'].name":
          sparse_tensor.SparseTensorValue(
              np.asarray([[0, 0], [1, 0], [1, 1]]),
              np.asarray([compat.as_bytes("Hans"), compat.as_bytes("Joel"),
                          compat.as_bytes("Marc")]),
              np.asarray([2, 1])),
        "guests[gender='female'].name":
          sparse_tensor.SparseTensorValue(
              np.asarray([[0, 0], [0, 1], [1, 0]]),
              np.asarray([compat.as_bytes("Mary"), compat.as_bytes("July"),
                          compat.as_bytes("JoAn")]),
              np.asarray([2, 1]))
      }
    ]

    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
                            batch_size=2, num_epochs=1)

  def test_filter_with_empty_result(self):
    writer_schema = """
      {
         "type": "record",
         "name": "data_row",
         "fields": [
            {
               "name": "guests",
               "type": {
                  "type": "array",
                  "items": {
                     "type": "record",
                     "name": "person",
                     "fields": [
                        {
                           "name":"name",
                           "type":"string"
                        },
                        {
                           "name":"gender",
                           "type":"string"
                        }
                     ]
                  }
               }
            }
         ]
      }
      """
    record_data = [{
      "guests": [{
        "name": "Hans",
        "gender": "male"
      }]
    }, {
      "guests": [{
        "name": "Joel",
        "gender": "male"
      }]
    }]
    features = {
      "guests[gender='wrong_value'].name":
        parsing_ops.VarLenFeature(tf_types.string)
    }
    expected_tensors = [
      {
        "guests[gender='wrong_value'].name":
          sparse_tensor.SparseTensorValue(
              np.empty(shape=[0, 2], dtype=np.int64),
              np.empty(shape=[0], dtype=np.str), np.asarray([2, 0]))
      }
    ]
    self._test_pass_dataset(writer_schema=writer_schema,
                            record_data=record_data,
                            expected_tensors=expected_tensors,
                            features=features,
                            batch_size=2, num_epochs=1)


if __name__ == "__main__":
  test.main()
