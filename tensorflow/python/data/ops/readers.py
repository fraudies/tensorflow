# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Python wrappers for reader Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util.tf_export import tf_export


# TODO(b/64974358): Increase default buffer size to 256 MB.
_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB


@tf_export("data.TextLineDataset")
class TextLineDataset(dataset_ops.Dataset):
  """A `Dataset` comprising lines from one or more text files."""

  def __init__(self, filenames, compression_type=None, buffer_size=None):
    """Creates a `TextLineDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
    """
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)
    variant_tensor = gen_dataset_ops.text_line_dataset(
        self._filenames, self._compression_type, self._buffer_size)
    super(TextLineDatasetV2, self).__init__(variant_tensor)

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string


class _TFRecordDataset(dataset_ops.Dataset):
  """A `Dataset` comprising records from one or more TFRecord files."""

  def __init__(self, filenames, compression_type=None, buffer_size=None):
    """Creates a `TFRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. 0 means no buffering.
    """
    # Force the type to string even if filenames is an empty list.
    self._filenames = ops.convert_to_tensor(
        filenames, dtypes.string, name="filenames")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size",
        buffer_size,
        argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)
    variant_tensor = gen_dataset_ops.tf_record_dataset(
        self._filenames, self._compression_type, self._buffer_size)
    super(_TFRecordDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([])

  @property
  def output_types(self):
    return dtypes.string


class ParallelInterleaveDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func, cycle_length, block_length,
               sloppy, buffer_output_elements, prefetch_input_elements):
    """See `tf.data.experimental.parallel_interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = dataset_ops.StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure,
                      dataset_ops.DatasetStructure):
      raise TypeError("`map_func` must return a `Dataset` object.")
    self._structure = self._map_func.output_structure._element_structure  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._sloppy = ops.convert_to_tensor(
        sloppy, dtype=dtypes.bool, name="sloppy")
    self._buffer_output_elements = convert.optional_param_to_tensor(
        "buffer_output_elements",
        buffer_output_elements,
        argument_default=2 * block_length)
    self._prefetch_input_elements = convert.optional_param_to_tensor(
        "prefetch_input_elements",
        prefetch_input_elements,
        argument_default=2 * cycle_length)
    variant_tensor = ged_ops.experimental_parallel_interleave_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        self._cycle_length,
        self._block_length,
        self._sloppy,
        self._buffer_output_elements,
        self._prefetch_input_elements,
        f=self._map_func,
        **dataset_ops.flat_structure(self))
    super(ParallelInterleaveDataset, self).__init__(input_dataset,
                                                    variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def _element_structure(self):
    return self._structure

  def _transformation_name(self):
    return "tf.data.experimental.parallel_interleave()"


@tf_export("data.TFRecordDataset")
class TFRecordDataset(dataset_ops.Dataset):
  """A `Dataset` comprising records from one or more TFRecord files."""

  def __init__(self, filenames, compression_type=None, buffer_size=None,
               num_parallel_reads=None):
    """Creates a `TFRecordDataset` to read for one or more TFRecord files.

    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. If your input pipeline is I/O bottlenecked,
        consider setting this parameter to a value 1-100 MBs. If `None`, a
        sensible default for both local and remote file systems is used.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.

    Raises:
      TypeError: If any argument does not have the expected type.
      ValueError: If any argument does not have the expected shape.
    """
    if isinstance(filenames, dataset_ops.DatasetV2):
      if dataset_ops.get_legacy_output_types(filenames) != dtypes.string:
        raise TypeError(
            "`filenames` must be a `tf.data.Dataset` of `tf.string` elements.")
      if not dataset_ops.get_legacy_output_shapes(filenames).is_compatible_with(
          tensor_shape.scalar()):
        raise ValueError(
            "`filenames` must be a `tf.data.Dataset` of scalar `tf.string` "
            "elements.")
    else:
      filenames = ops.convert_to_tensor(filenames, dtype=dtypes.string)
      filenames = array_ops.reshape(filenames, [-1], name="flat_filenames")
      filenames = dataset_ops.Dataset.from_tensor_slices(filenames)

    self._filenames = filenames
    self._compression_type = compression_type
    self._buffer_size = buffer_size
    self._num_parallel_reads = num_parallel_reads

    def read_one_file(filename):
      return _TFRecordDataset(filename, compression_type, buffer_size)

    if num_parallel_reads is None:
      self._impl = filenames.flat_map(read_one_file)
    else:
      self._impl = ParallelInterleaveDataset(
          filenames, read_one_file, cycle_length=num_parallel_reads,
          block_length=1, sloppy=False, buffer_output_elements=None,
          prefetch_input_elements=None)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(TFRecordDatasetV2, self).__init__(variant_tensor)

  def _clone(self,
             filenames=None,
             compression_type=None,
             buffer_size=None,
             num_parallel_reads=None):
    return TFRecordDataset(filenames or self._filenames,
                           compression_type or self._compression_type,
                           buffer_size or self._buffer_size,
                           num_parallel_reads or self._num_parallel_reads)

  def _inputs(self):
    return self._impl._inputs()  # pylint: disable=protected-access

  @property
  def output_classes(self):
    return self._impl.output_classes

  @property
  def output_shapes(self):
    return self._impl.output_shapes

  @property
  def output_types(self):
    return self._impl.output_types


@tf_export("data.FixedLengthRecordDataset")
class FixedLengthRecordDataset(dataset_ops.Dataset):
  """A `Dataset` of fixed-length records from one or more binary files."""

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None):
    """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in
        each record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
    """
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._record_bytes = ops.convert_to_tensor(
        record_bytes, dtype=dtypes.int64, name="record_bytes")

    self._header_bytes = convert.optional_param_to_tensor(
        "header_bytes", header_bytes)
    self._footer_bytes = convert.optional_param_to_tensor(
        "footer_bytes", footer_bytes)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    if (self._compression_type is not None or
        compat.forward_compatible(2018, 11, 30)):
      variant_tensor = gen_dataset_ops.fixed_length_record_dataset_v2(
          self._filenames, self._header_bytes, self._record_bytes,
          self._footer_bytes, self._buffer_size, self._compression_type)
    else:
      variant_tensor = gen_dataset_ops.fixed_length_record_dataset(
          self._filenames, self._header_bytes, self._record_bytes,
          self._footer_bytes, self._buffer_size)
    super(FixedLengthRecordDatasetV2, self).__init__(variant_tensor)

  @property
  def _element_structure(self):
    return structure.TensorStructure(dtypes.string, [])

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string
