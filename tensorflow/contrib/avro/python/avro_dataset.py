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

# Parse example dataset
# https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/data/experimental/ops/parsing_ops.py

# For tf export use examples see
# sql_dataset_test_base.py for use and readers.py for definition


import os

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.data.util import structure
from tensorflow.python.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.dataset_ops import DatasetSource, DatasetV1Adapter
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.platform import resource_loader
from tensorflow.contrib.avro.ops.gen_avro_dataset import avro_dataset
from tensorflow.python.util.tf_export import tf_export

# Load the shared library
lib_name = os.path.join(resource_loader.get_data_files_path(),
                        '_avro_dataset.so')
reader_module = load_library.load_op_library(lib_name)


# TODO(fraudies): fixme @tf_export("contrib.avro.AvroDataset", v1=[])
# Note: I've hidden the dataset because it does not apply the mapping for
# sparse tensors
# Note: that such mapping is not possible inside the dataset, rather it
# needs to happen through a map on the output of the dataset which is a map
# of keys to tensors
# This can be changed when eager mode is the default and only mode supported
class _AvroDataset(DatasetSource):
  """A `DatasetSource` that reads and parses Avro records from files."""

  def __init__(self, filenames, features, reader_schema="", batch_size=128,
      num_parallel_calls=2):

    super(_AvroDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtypes.string, name="filenames")
    self._features = _AvroDataset._build_keys_for_sparse_features(features)
    self._reader_schema = reader_schema
    self._batch_size = batch_size
    self._num_parallel_calls = num_parallel_calls

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

    dense_output_shapes = dense_shape_as_shape
    sparse_output_shapes = [
      [None]
      for _ in range(len(sparse_keys))
    ]

    output_shapes = dict(
        zip(self._dense_keys + self._sparse_keys,
            dense_output_shapes + sparse_output_shapes))
    output_types = dict(
        zip(self._dense_keys + self._sparse_keys,
            self._dense_types + self._sparse_types))
    output_classes = dict(
        zip(self._dense_keys + self._sparse_keys,
            [ops.Tensor for _ in range(len(self._dense_defaults))] +
            [sparse_tensor.SparseTensor for _ in range(len(self._sparse_keys))
             ]))
    self._structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)

  def _as_variant_tensor(self):
    outputs = dataset_ops.flat_structure(self)

    print("File names: {}".format(self._filenames))
    print("Reader schema: {}".format(self._reader_schema))
    print("dense defaults: {}".format(self._dense_defaults))
    print("sparse keys: {}".format(self._sparse_keys))
    print("dense keys: {}".format(self._dense_keys))
    print("sparse types: {}".format(self._sparse_types))
    print("dense shapes: {}".format(self._dense_shapes))
    print("output shapes: {}".format(outputs['output_shapes']))
    print("output types: {}".format(outputs['output_types']))

    out_dataset = avro_dataset(
        filenames=self._filenames,
        batch_size=self._batch_size,
        reader_schema=self._reader_schema,
        dense_defaults=self._dense_defaults,
        sparse_keys=self._sparse_keys,
        dense_keys=self._dense_keys,
        sparse_types=self._sparse_types,
        dense_shapes=self._dense_shapes,
        **dataset_ops.flat_structure(self))

    print("output dataset: {}".format(out_dataset))

    return out_dataset


  @property
  def _element_structure(self):
    return self._structure

  @staticmethod
  def _build_keys_for_sparse_features(features):
    """
    Builds the fully qualified names for keys of sparse features.

    :param features:  A map of features with keys to TensorFlow features.

    :return: A map of features where for the sparse feature the 'index_key' and the 'value_key' have been expanded
             properly for the parser in the native code.
    """
    if features:
      # NOTE: We iterate over sorted keys to keep things deterministic.
      for key in sorted(features.keys()):
        feature = features[key]
        if isinstance(feature, parsing_ops.SparseFeature):
          features[key] = parsing_ops.SparseFeature(
              index_key=key + '[*].' + feature.index_key,
              value_key=key + '[*].' + feature.value_key,
              dtype=feature.dtype,
              size=feature.size,
              already_sorted=feature.already_sorted)
    return features


@tf_export("contrib.avro.make_avro_dataset", v1=[])
def make_avro_dataset_v2(
    file_pattern,
    features,
    batch_size,
    reader_schema="",
    num_parallel_calls=2,
    label_key=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=optimization.AUTOTUNE,
    num_parallel_reads=1,
    sloppy=False
):
  """Makes an avro dataset.

  Reads from avro files and parses the contents into tensors.

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

    num_parallel_calls: Number of parallel calls

  """
  filenames = readers._get_file_names(file_pattern, False)
  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if shuffle:
    dataset = dataset.shuffle(len(filenames), shuffle_seed)

  if label_key is not None and label_key not in features:
    raise ValueError("`label_key` provided must be in `features`.")

  def filename_to_dataset(filename):
    # Batches
    return _AvroDataset(
        filenames=filename,
        features=features,
        reader_schema=reader_schema,
        batch_size=batch_size,
        num_parallel_calls=num_parallel_calls
    )

  # Read files sequentially (if num_parallel_reads=1) or in parallel
  dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          filename_to_dataset, cycle_length=num_parallel_reads, sloppy=sloppy))

  dataset = readers._maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  if any(
      isinstance(feature, parsing_ops.SparseFeature)
      for _, feature in features.items()
  ):
    # pylint: disable=protected-access
    # pylint: disable=g-long-lambda
    dataset = dataset.map(
        lambda x: parsing_ops._construct_sparse_tensors_for_sparse_features(
            features, x), num_parallel_calls=num_parallel_calls)

  if label_key:
    if label_key not in features:
      raise ValueError(
          "The `label_key` provided (%r) must be one of the `features` keys." %
          label_key)
    dataset = dataset.map(lambda x: (x, x.pop(label_key)))

  dataset = dataset.prefetch(prefetch_buffer_size)

  return dataset

@tf_export(v1=["contrib.avro.make_avro_dataset"])
def make_avro_dataset_v1(
    file_pattern,
    features,
    batch_size,
    reader_schema="",
    num_parallel_calls=2,
    label_key=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=optimization.AUTOTUNE,
    num_parallel_reads=1,
    sloppy=False
):  # pylint: disable=missing-docstring
  return dataset_ops.DatasetV1Adapter(make_avro_dataset_v2(
      file_pattern, features, batch_size, reader_schema, num_parallel_calls,
      label_key, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed,
      prefetch_buffer_size, num_parallel_reads, sloppy))
make_avro_dataset_v1.__doc__ = make_avro_dataset_v2.__doc__