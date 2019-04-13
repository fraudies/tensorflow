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
import os

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.data.util import structure
from tensorflow.python.framework import load_library
from tensorflow.python.data.ops.dataset_ops import DatasetSource, DatasetV1Adapter
from tensorflow.python.data.util import convert
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader
from tensorflow.contrib.avro.ops.gen_avro_record_dataset import avro_record_dataset

# Load the shared library
lib_name = os.path.join(resource_loader.get_data_files_path(),
                        '_avro_record_dataset.so')
reader_module = load_library.load_op_library(lib_name)

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB


class AvroRecordDataset(DatasetSource):
    """A `Dataset` comprising records from one or more Avro files."""

    def __init__(self, filenames, reader_schema=None):
        """Creates a `AvroRecordDataset`.
        Args:
          filenames: A `tf.string` tensor containing one or more filenames.
          reader_schema: (Optional.) A `tf.string` scalar for schema resolution.
        """
        super(AvroRecordDataset, self).__init__()

        # Force the type to string even if filenames is an empty list.
        self._filenames = ops.convert_to_tensor(
            filenames, dtypes.string, name="filenames")
        self._reader_schema = convert.optional_param_to_tensor(
            "reader_schema", reader_schema, argument_default="",
            argument_dtype=dtypes.string)
        #outputKey = 'output'
        #self._structure = structure.convert_legacy_structure(
        #    dict(outputKey=dtypes.string),
        #    dict(outputKey=tensor_shape.scalar()),
        #    dict(outputKey=ops.Tensor))
        #self._structure = structure.TensorStructure(dtypes.string, tensor_shape.scalar())
#structure.NestedStructure(
#    tuple(structure.TensorStructure(d.dtype, [])
#          for d in self._record_defaults))
#        self._structure = structure.NestedStructure([structure.TensorStructure(dtypes.string, [])])
        self._structure = structure.convert_legacy_structure(dtypes.string, tensor_shape.scalar(), ops.Tensor)

    def _as_variant_tensor(self):
        return avro_record_dataset(self._filenames, self._reader_schema)

    def _element_structure(self):
        return self._structure

    def make_initializable_iterator(self):
        return DatasetV1Adapter(self).make_initializable_iterator()
