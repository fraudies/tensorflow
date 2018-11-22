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

# From CSV dataset
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/data/experimental/ops/readers.py
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/data/experimental/ops/parsing_ops.py

import os

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader
from tensorflow.contrib.avro.ops.gen_avro_dataset import gen_avro_dataset

# Load the shared library
lib_name = os.path.join(resource_loader.get_data_files_path(),
                        '_avro_dataset.so')
reader_module = load_library.load_op_library(lib_name)


class AvroDataset(dataset_ops.DatasetSource):
  """A `DatasetSource` that reads and parses Avro records from files."""

  def __init__(self, filenames, features, reader_schema=None):
    """Creates a `AvroDataset` and batches for performance.
    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      features: Is a map of keys that describe a single entry or sparse vector
                in the avro record and map that entry to a tensor. The syntax
                is as follows:

                features = {'my_meta_data.size':
                            tf.FixedLenFeature([], tf.int64)}

                Select the 'size' field from a record metadata that is in the
                field 'my_meta_data'. In this example we assume that the size is
                encoded as a long in the Avro record for the metadata.


                features = {'my_map_data['source'].ip_addresses':
                            tf.VarLenFeature([], tf.string)}

                Select the 'ip_addresses' for the 'source' key in the map
                'my_map_data'. Notice we assume that IP addresses are encoded as
                strings in this example.


                features = {'my_friends[1].first_name':
                            tf.FixedLenFeature([], tf.string)}

                Select the 'first_name' for the second friend with index '1'.
                This assumes that all of your data has a second friend. In
                addition, we assume that all friends have only one first name.
                For this reason we chose a 'FixedLenFeature'.


                features = {'my_friends[*].first_name':
                            tf.VarLenFeature([], tf.string)}

                Selects all first_names in each row. For this example we use the
                wildcard '*' to indicate that we want to select all 'first_name'
                entries from the array.

                features = {'sparse_features':
                            tf.SparseFeature(index_key='index',
                                             value_key='value',
                                             dtype=tf.float32, size=10)}

                We assume that sparse features contains an array with records
                that contain an 'index' field that MUST BE LONG and an 'value'
                field with floats (single precision).

      reader_schema: (Optional.) A `tf.string` scalar for schema resolution.

      num_parallel_reads: (Optional.) A `int` that defines the number of threads
                          used during loading/parsing.
      fixed_batch_size: (Optional.) A `bool` that indicates if we can assume a
                          fixed batch size.
    """
    super(AvroDataset, self).__init__()

    self._filenames = convert.convert_to_tensor(
        filenames, dtypes.string, name="filenames")
    self._features = features
    self._reader_schema = convert.optional_param_to_tensor(
        "reader_schema", reader_schema, argument_default="",
        argument_dtype=dtypes.string)
    # Prepare dimensions
    self._features = parsing_ops._prepend_none_dimension(features)

    # Copied from _ParseExampleDataset from data/experimental/ops/parsing_ops.py
    (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
     dense_shapes) = parsing_ops._features_to_raw_params(
        self._features, [
          parsing_ops.VarLenFeature, parsing_ops.SparseFeature,
          parsing_ops.FixedLenFeature, parsing_ops.FixedLenSequenceFeature
        ])

    (_, dense_defaults_vec, sparse_keys, sparse_types, dense_keys, dense_shapes,
     dense_shape_as_shape) = parsing_ops._process_raw_parameters(
        None, dense_defaults, sparse_keys, sparse_types, dense_keys,
        dense_types, dense_shapes)

    self._sparse_keys = sparse_keys
    self._sparse_types = sparse_types
    self._dense_keys = dense_keys
    self._dense_defaults = dense_defaults_vec
    self._dense_shapes = dense_shapes
    self._dense_types = dense_types
    dense_output_shapes = [
      self._input_dataset.output_shapes.concatenate(shape)
      for shape in dense_shape_as_shape
    ]
    sparse_output_shapes = [
      self._input_dataset.output_shapes.concatenate([None])
      for _ in range(len(sparse_keys))
    ]

    self._output_shapes = dict(
        zip(self._dense_keys + self._sparse_keys,
            dense_output_shapes + sparse_output_shapes))
    self._output_types = dict(
        zip(self._dense_keys + self._sparse_keys,
            self._dense_types + self._sparse_types))
    self._output_classes = dict(
        zip(self._dense_keys + self._sparse_keys,
            [ops.Tensor for _ in range(len(self._dense_defaults))] +
            [sparse_tensor.SparseTensor for _ in range(len(self._sparse_keys))
             ]))

    self._output_shapes = tuple(
        tensor_shape.scalar() for _ in range(len(record_defaults)))
    self._output_types = tuple(d.dtype for d in self._record_defaults)
    self._output_classes = tuple(
        ops.Tensor for _ in range(len(record_defaults)))

  def _as_variant_tensor(self):
    out_dataset = gen_avro_dataset(
        filenames=self._filenames,
        dense_defaults=self._dense_defaults,
        reader_schema=self._reader_schema,
        sparse_keys=self._sparse_keys,
        dense_keys=self._dense_keys,
        sparse_types=self._sparse_types,
        dense_shapes=self._dense_shapes,
        **dataset_ops.flat_structure(self))

    if any([
      isinstance(feature, parsing_ops.SparseFeature)
      for _, feature in features.items()
    ]):
      # pylint: disable=protected-access
      # pylint: disable=g-long-lambda
      out_dataset = out_dataset.map(
          lambda x: parsing_ops._construct_sparse_tensors_for_sparse_features(
              self._features, x), num_parallel_calls=self._num_parallel_calls)
    return out_dataset

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_classes(self):
    return self._output_classes