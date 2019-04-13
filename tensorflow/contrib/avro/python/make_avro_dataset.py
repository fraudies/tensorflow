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


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental import readers

@tf_export("contrib.avro.make_avro_dataset")
def make_avro_dataset(
    file_pattern,
    batch_size,
    features,
    label_key=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=optimization.AUTOTUNE,
    num_parallel_reads=1,
    sloppy=False
):

  filenames = readers._get_file_names(file_pattern, False)
  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if shuffle:
    dataset = dataset.shuffle(len(filenames), shuffle_seed)

  if label_name is not None and label_name not in features:
    raise ValueError("`label_name` provided must be in `features`.")

  def filename_to_dataset(filename):
    # Batches
    return AvroDataset(
        filenames=filenames,
        features=features,
        reader_schema=reader_schema
    )

  # Read files sequentially (if num_parallel_reads=1) or in parallel
  dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          filename_to_dataset, cycle_length=num_parallel_reads, sloppy=sloppy))

  dataset = readers._maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  # TODO(fraudies): Batch in c++ to increase parallelism for parsing
  dataset = dataset.batch(batch_size=batch_size,
                          drop_remainder=num_epochs is None)

  if label_key:
    if label_key not in features:
      raise ValueError(
          "The `label_key` provided (%r) must be one of the `features` keys." %
          label_key)
    dataset = dataset.map(lambda x: (x, x.pop(label_key)))

  dataset = dataset.prefetch(prefetch_buffer_size)

  return dataset