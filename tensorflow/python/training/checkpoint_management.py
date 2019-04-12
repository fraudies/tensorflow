# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=invalid-name
"""Save and restore variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import re
import time

from google.protobuf import text_format

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


def _GetCheckpointFilename(save_dir, latest_filename):
  """Returns a filename for storing the CheckpointState.

  Args:
    save_dir: The directory for saving and restoring checkpoints.
    latest_filename: Name of the file in 'save_dir' that is used
      to store the CheckpointState.

  Returns:
    The path of the file that contains the CheckpointState proto.
  """
  if latest_filename is None:
    latest_filename = "checkpoint"
  return os.path.join(save_dir, latest_filename)


@tf_export("train.generate_checkpoint_state_proto")
def generate_checkpoint_state_proto(save_dir,
                                    model_checkpoint_path,
                                    all_model_checkpoint_paths=None,
                                    all_model_checkpoint_timestamps=None,
                                    last_preserved_timestamp=None):
  """Generates a checkpoint state proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    all_model_checkpoint_timestamps: A list of floats, indicating the number of
      seconds since the Epoch when each checkpoint was generated.
    last_preserved_timestamp: A float, indicating the number of seconds since
      the Epoch when the last preserved checkpoint was written, e.g. due to a
      `keep_checkpoint_every_n_hours` parameter (see
      `tf.contrib.checkpoint.CheckpointManager` for an implementation).
  Returns:
    CheckpointState proto with model_checkpoint_path and
    all_model_checkpoint_paths updated to either absolute paths or
    relative paths to the current save_dir.

  Raises:
    ValueError: If `all_model_checkpoint_timestamps` was provided but its length
      does not match `all_model_checkpoint_paths`.
  """
  if all_model_checkpoint_paths is None:
    all_model_checkpoint_paths = []

  if (not all_model_checkpoint_paths or
      all_model_checkpoint_paths[-1] != model_checkpoint_path):
    logging.info("%s is not in all_model_checkpoint_paths. Manually adding it.",
                 model_checkpoint_path)
    all_model_checkpoint_paths.append(model_checkpoint_path)

  if (all_model_checkpoint_timestamps
      and (len(all_model_checkpoint_timestamps)
           != len(all_model_checkpoint_paths))):
    raise ValueError(
        ("Checkpoint timestamps, if provided, must match checkpoint paths (got "
         "paths %s and timestamps %s)")
        % (all_model_checkpoint_paths, all_model_checkpoint_timestamps))

  # Relative paths need to be rewritten to be relative to the "save_dir"
  # if model_checkpoint_path already contains "save_dir".
  if not os.path.isabs(save_dir):
    if not os.path.isabs(model_checkpoint_path):
      model_checkpoint_path = os.path.relpath(model_checkpoint_path, save_dir)
    for i in range(len(all_model_checkpoint_paths)):
      p = all_model_checkpoint_paths[i]
      if not os.path.isabs(p):
        all_model_checkpoint_paths[i] = os.path.relpath(p, save_dir)

  coord_checkpoint_proto = CheckpointState(
      model_checkpoint_path=model_checkpoint_path,
      all_model_checkpoint_paths=all_model_checkpoint_paths,
      all_model_checkpoint_timestamps=all_model_checkpoint_timestamps,
      last_preserved_timestamp=last_preserved_timestamp)

  return coord_checkpoint_proto


@deprecation.deprecated(
    date=None,
    instructions=("Use `tf.train.CheckpointManager` to manage checkpoints "
                  "rather than manually editing the Checkpoint proto."))
@tf_export(v1=["train.update_checkpoint_state"])
def update_checkpoint_state(save_dir,
                            model_checkpoint_path,
                            all_model_checkpoint_paths=None,
                            latest_filename=None,
                            all_model_checkpoint_timestamps=None,
                            last_preserved_timestamp=None):
  """Updates the content of the 'checkpoint' file.

  This updates the checkpoint file containing a CheckpointState
  proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.
    all_model_checkpoint_timestamps: Optional list of timestamps (floats,
      seconds since the Epoch) indicating when the checkpoints in
      `all_model_checkpoint_paths` were created.
    last_preserved_timestamp: A float, indicating the number of seconds since
      the Epoch when the last preserved checkpoint was written, e.g. due to a
      `keep_checkpoint_every_n_hours` parameter (see
      `tf.contrib.checkpoint.CheckpointManager` for an implementation).
  Raises:
    RuntimeError: If any of the model checkpoint paths conflict with the file
      containing CheckpointSate.
  """
  update_checkpoint_state_internal(
      save_dir=save_dir,
      model_checkpoint_path=model_checkpoint_path,
      all_model_checkpoint_paths=all_model_checkpoint_paths,
      latest_filename=latest_filename,
      save_relative_paths=False,
      all_model_checkpoint_timestamps=all_model_checkpoint_timestamps,
      last_preserved_timestamp=last_preserved_timestamp)


def update_checkpoint_state_internal(save_dir,
                                     model_checkpoint_path,
                                     all_model_checkpoint_paths=None,
                                     latest_filename=None,
                                     save_relative_paths=False,
                                     all_model_checkpoint_timestamps=None,
                                     last_preserved_timestamp=None):
  """Updates the content of the 'checkpoint' file.

  This updates the checkpoint file containing a CheckpointState
  proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.
    save_relative_paths: If `True`, will write relative paths to the checkpoint
      state file.
    all_model_checkpoint_timestamps: Optional list of timestamps (floats,
      seconds since the Epoch) indicating when the checkpoints in
      `all_model_checkpoint_paths` were created.
    last_preserved_timestamp: A float, indicating the number of seconds since
      the Epoch when the last preserved checkpoint was written, e.g. due to a
      `keep_checkpoint_every_n_hours` parameter (see
      `tf.contrib.checkpoint.CheckpointManager` for an implementation).

  Raises:
    RuntimeError: If any of the model checkpoint paths conflict with the file
      containing CheckpointSate.
  """
  # Writes the "checkpoint" file for the coordinator for later restoration.
  coord_checkpoint_filename = _GetCheckpointFilename(save_dir, latest_filename)
  if save_relative_paths:
    if os.path.isabs(model_checkpoint_path):
      rel_model_checkpoint_path = os.path.relpath(
          model_checkpoint_path, save_dir)
    else:
      rel_model_checkpoint_path = model_checkpoint_path
    rel_all_model_checkpoint_paths = []
    for p in all_model_checkpoint_paths:
      if os.path.isabs(p):
        rel_all_model_checkpoint_paths.append(os.path.relpath(p, save_dir))
      else:
        rel_all_model_checkpoint_paths.append(p)
    ckpt = generate_checkpoint_state_proto(
        save_dir,
        rel_model_checkpoint_path,
        all_model_checkpoint_paths=rel_all_model_checkpoint_paths,
        all_model_checkpoint_timestamps=all_model_checkpoint_timestamps,
        last_preserved_timestamp=last_preserved_timestamp)
  else:
    ckpt = generate_checkpoint_state_proto(
        save_dir,
        model_checkpoint_path,
        all_model_checkpoint_paths=all_model_checkpoint_paths,
        all_model_checkpoint_timestamps=all_model_checkpoint_timestamps,
        last_preserved_timestamp=last_preserved_timestamp)

  if coord_checkpoint_filename == ckpt.model_checkpoint_path:
    raise RuntimeError("Save path '%s' conflicts with path used for "
                       "checkpoint state.  Please use a different save path." %
                       model_checkpoint_path)

  # Preventing potential read/write race condition by *atomically* writing to a
  # file.
  file_io.atomic_write_string_to_file(coord_checkpoint_filename,
                                      text_format.MessageToString(ckpt))


@tf_export("train.get_checkpoint_state")
def get_checkpoint_state(checkpoint_dir, latest_filename=None):
  """Returns CheckpointState proto from the "checkpoint" file.

  If the "checkpoint" file contains a valid CheckpointState
  proto, returns it.

  Args:
    checkpoint_dir: The directory of checkpoints.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.

  Returns:
    A CheckpointState if the state was available, None
    otherwise.

  Raises:
    ValueError: if the checkpoint read doesn't have model_checkpoint_path set.
  """
  ckpt = None
  coord_checkpoint_filename = _GetCheckpointFilename(checkpoint_dir,
                                                     latest_filename)
  f = None
  try:
    # Check that the file exists before opening it to avoid
    # many lines of errors from colossus in the logs.
    if file_io.file_exists(coord_checkpoint_filename):
      file_content = file_io.read_file_to_string(
          coord_checkpoint_filename)
      ckpt = CheckpointState()
      text_format.Merge(file_content, ckpt)
      if not ckpt.model_checkpoint_path:
        raise ValueError("Invalid checkpoint state loaded from "
                         + checkpoint_dir)
      # For relative model_checkpoint_path and all_model_checkpoint_paths,
      # prepend checkpoint_dir.
      if not os.path.isabs(ckpt.model_checkpoint_path):
        ckpt.model_checkpoint_path = os.path.join(checkpoint_dir,
                                                  ckpt.model_checkpoint_path)
      for i in range(len(ckpt.all_model_checkpoint_paths)):
        p = ckpt.all_model_checkpoint_paths[i]
        if not os.path.isabs(p):
          ckpt.all_model_checkpoint_paths[i] = os.path.join(checkpoint_dir, p)
  except errors.OpError as e:
    # It's ok if the file cannot be read
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  except text_format.ParseError as e:
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  finally:
    if f:
      f.close()
  return ckpt


def _prefix_to_checkpoint_path(prefix, format_version):
  """Returns the pathname of a checkpoint file, given the checkpoint prefix.

  For V1 checkpoint, simply returns the prefix itself (the data file).  For V2,
  returns the pathname to the index file.

  Args:
    prefix: a string, the prefix of a checkpoint.
    format_version: the checkpoint format version that corresponds to the
      prefix.
  Returns:
    The pathname of a checkpoint file, taking into account the checkpoint
      format version.
  """
  if format_version == saver_pb2.SaverDef.V2:
    return prefix + ".index"  # The index file identifies a checkpoint.
  return prefix  # Just the data file.


@tf_export("train.latest_checkpoint")
def latest_checkpoint(checkpoint_dir, latest_filename=None):
  """Finds the filename of latest saved checkpoint file.

  Args:
    checkpoint_dir: Directory where the variables were saved.
    latest_filename: Optional name for the protocol buffer file that
      contains the list of most recent checkpoint filenames.
      See the corresponding argument to `Saver.save()`.

  Returns:
    The full path to the latest checkpoint or `None` if no checkpoint was found.
  """
  # Pick the latest checkpoint based on checkpoint state.
  ckpt = get_checkpoint_state(checkpoint_dir, latest_filename)
  if ckpt and ckpt.model_checkpoint_path:
    # Look for either a V2 path or a V1 path, with priority for V2.
    v2_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V2)
    v1_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(
        v1_path):
      return ckpt.model_checkpoint_path
    else:
      logging.error("Couldn't match files for checkpoint %s",
                    ckpt.model_checkpoint_path)
  return None


@tf_export("train.checkpoint_exists")
def checkpoint_exists(checkpoint_prefix):
  """Checks whether a V1 or V2 checkpoint exists with the specified prefix.

  This is the recommended way to check if a checkpoint exists, since it takes
  into account the naming difference between V1 and V2 formats.

  Args:
    checkpoint_prefix: the prefix of a V1 or V2 checkpoint, with V2 taking
      priority.  Typically the result of `Saver.save()` or that of
      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
      V1/V2.
  Returns:
    A bool, true iff a checkpoint referred to by `checkpoint_prefix` exists.
  """
  pathname = _prefix_to_checkpoint_path(checkpoint_prefix,
                                        saver_pb2.SaverDef.V2)
  if file_io.get_matching_files(pathname):
    return True
  elif file_io.get_matching_files(checkpoint_prefix):
    return True
  else:
    return False


@tf_export("train.get_checkpoint_mtimes")
def get_checkpoint_mtimes(checkpoint_prefixes):
  """Returns the mtimes (modification timestamps) of the checkpoints.

  Globs for the checkpoints pointed to by `checkpoint_prefixes`.  If the files
  exist, collect their mtime.  Both V2 and V1 checkpoints are considered, in
  that priority.

  This is the recommended way to get the mtimes, since it takes into account
  the naming difference between V1 and V2 formats.

  Args:
    checkpoint_prefixes: a list of checkpoint paths, typically the results of
      `Saver.save()` or those of `tf.train.latest_checkpoint()`, regardless of
      sharded/non-sharded or V1/V2.
  Returns:
    A list of mtimes (in microseconds) of the found checkpoints.
  """
  mtimes = []

  def match_maybe_append(pathname):
    fnames = file_io.get_matching_files(pathname)
    if fnames:
      mtimes.append(file_io.stat(fnames[0]).mtime_nsec / 1e9)
      return True
    return False

  for checkpoint_prefix in checkpoint_prefixes:
    # Tries V2's metadata file first.
    pathname = _prefix_to_checkpoint_path(checkpoint_prefix,
                                          saver_pb2.SaverDef.V2)
    if match_maybe_append(pathname):
      continue
    # Otherwise, tries V1, where the prefix is the complete pathname.
    match_maybe_append(checkpoint_prefix)

  return mtimes


@tf_export("train.remove_checkpoint")
def remove_checkpoint(checkpoint_prefix,
                      checkpoint_format_version=saver_pb2.SaverDef.V2,
                      meta_graph_suffix="meta"):
  """Removes a checkpoint given by `checkpoint_prefix`.

  Args:
    checkpoint_prefix: The prefix of a V1 or V2 checkpoint. Typically the result
      of `Saver.save()` or that of `tf.train.latest_checkpoint()`, regardless of
      sharded/non-sharded or V1/V2.
    checkpoint_format_version: `SaverDef.CheckpointFormatVersion`, defaults to
      `SaverDef.V2`.
    meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
  """
  _delete_file_if_exists(
      meta_graph_filename(checkpoint_prefix, meta_graph_suffix))
  if checkpoint_format_version == saver_pb2.SaverDef.V2:
    # V2 has a metadata file and some data files.
    _delete_file_if_exists(checkpoint_prefix + ".index")
    _delete_file_if_exists(checkpoint_prefix + ".data-?????-of-?????")
  else:
    # V1, Legacy.  Exact match on the data file.
    _delete_file_if_exists(checkpoint_prefix)


def _delete_file_if_exists(filespec):
  """Deletes files matching `filespec`."""
  for pathname in file_io.get_matching_files(filespec):
    file_io.delete_file(pathname)


def meta_graph_filename(checkpoint_filename, meta_graph_suffix="meta"):
  """Returns the meta graph filename.

  Args:
    checkpoint_filename: Name of the checkpoint file.
    meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.

  Returns:
    MetaGraph file name.
  """
  # If the checkpoint_filename is sharded, the checkpoint_filename could
  # be of format model.ckpt-step#-?????-of-shard#. For example,
  # model.ckpt-123456-?????-of-00005, or model.ckpt-123456-00001-of-00002.
  basename = re.sub(r"-[\d\?]+-of-\d+$", "", checkpoint_filename)
  suffixed_filename = ".".join([basename, meta_graph_suffix])
  return suffixed_filename


# TODO(allenl): Allow tf.keras.Model instances in the constructor directly?
class CheckpointManager(object):
  """Deletes old checkpoints.

  Example usage:
  ```python
  import tensorflow as tf
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.contrib.checkpoint.CheckpointManager(
      checkpoint, directory="/tmp/model", max_to_keep=5)
  status = checkpoint.restore(manager.latest_checkpoint)
  while True:
    # train
    manager.save()
  ```

  `CheckpointManager` preserves its own state across instantiations (see the
  `__init__` documentation for details). Only one should be active in a
  particular directory at a time.
  """

  def __init__(self, checkpoint, directory,
               max_to_keep, keep_checkpoint_every_n_hours=None):
    """Configure a `CheckpointManager` for use in `directory`.

    If a `CheckpointManager` was previously used in `directory`, its
    state will be restored. This includes the list of managed checkpoints and
    the timestamp bookkeeping necessary to support
    `keep_checkpoint_every_n_hours`. The behavior of the new `CheckpointManager`
    will be the same as the previous `CheckpointManager`, including cleaning up
    existing checkpoints if appropriate.

    Checkpoints are only considered for deletion just after a new checkpoint has
    been added. At that point, `max_to_keep` checkpoints will remain in an
    "active set". Once a checkpoint is preserved by
    `keep_checkpoint_every_n_hours` it will not be deleted by this
    `CheckpointManager` or any future `CheckpointManager` instantiated in
    `directory` (regardless of the new setting of
    `keep_checkpoint_every_n_hours`). The `max_to_keep` checkpoints in the
    active set may be deleted by this `CheckpointManager` or a future
    `CheckpointManager` instantiated in `directory` (subject to its
    `max_to_keep` and `keep_checkpoint_every_n_hours` settings).

    Args:
      checkpoint: The `tf.train.Checkpoint` instance to save and manage
        checkpoints for.
      directory: The path to a directory in which to write checkpoints. A
        special file named "checkpoint" is also written to this directory (in a
        human-readable text format) which contains the state of the
        `CheckpointManager`.
      max_to_keep: An integer, the number of checkpoints to keep. Unless
        preserved by `keep_checkpoint_every_n_hours`, checkpoints will be
        deleted from the active set, oldest first, until only `max_to_keep`
        checkpoints remain. If `None`, no checkpoints are deleted and everything
        stays in the active set. Note that `max_to_keep=None` will keep all
        checkpoint paths in memory and in the checkpoint state protocol buffer
        on disk.
      keep_checkpoint_every_n_hours: Upon removal from the active set, a
        checkpoint will be preserved if it has been at least
        `keep_checkpoint_every_n_hours` since the last preserved checkpoint. The
        default setting of `None` does not preserve any checkpoints in this way.

    Raises:
      ValueError: If `max_to_keep` is not a positive integer.
    """
    self._checkpoint = checkpoint
    self._save_counter_assign = None
    if max_to_keep is not None and max_to_keep <= 0:
      raise ValueError(
          ("Expected a positive integer or `None` for `max_to_max_to_keep`, "
           "got %d.")
          % (max_to_keep,))
    self._max_to_keep = max_to_keep
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self._directory = directory
    self._checkpoint_prefix = os.path.join(directory, "ckpt")
    recovered_state = get_checkpoint_state(directory)
    current_clock = time.time()
    self._maybe_delete = collections.OrderedDict()
    if recovered_state is None:
      self._latest_checkpoint = None
      # Set the clock back slightly to avoid race conditions when quckly
      # re-creating a CheckpointManager.
      self._last_preserved_timestamp = current_clock - 1.
    else:
      self._latest_checkpoint = recovered_state.model_checkpoint_path
      self._last_preserved_timestamp = recovered_state.last_preserved_timestamp
      if current_clock < self._last_preserved_timestamp:
        # Time seems to have reversed itself. In addition to this warning, we'll
        # min() saved checkpoint timestamps with the current time to ensure that
        # old checkpoints don't get deleted accidentally.
        logging.warning(
            ("time.time() returned a value %f seconds behind the last "
             "preserved checkpoint timestamp.")
            % (self._last_preserved_timestamp - current_clock,))
        self._last_preserved_timestamp = current_clock
      all_timestamps = recovered_state.all_model_checkpoint_timestamps
      all_paths = recovered_state.all_model_checkpoint_paths
      del recovered_state  # Uses modified values from now on
      if not all_timestamps:
        all_timestamps = [self._last_preserved_timestamp] * len(all_paths)

      for filename, timestamp in zip(all_paths, all_timestamps):
        timestamp = min(timestamp, current_clock)
        if timestamp > self._last_preserved_timestamp:
          self._maybe_delete[filename] = timestamp

  @property
  def latest_checkpoint(self):
    """The prefix of the most recent checkpoint in `directory`.

    Equivalent to `tf.train.latest_checkpoint(directory)` where `directory` is
    the constructor argument to `CheckpointManager`.

    Suitable for passing to `tf.train.Checkpoint.restore` to resume training.

    Returns:
      The checkpoint prefix. If there are no checkpoints, returns `None`.
    """
    return self._latest_checkpoint

  @property
  def checkpoints(self):
    """A list of managed checkpoints.

    Note that checkpoints saved due to `keep_checkpoint_every_n_hours` will not
    show up in this list (to avoid ever-growing filename lists).

    Returns:
      A list of filenames, sorted from oldest to newest.
    """
    return list(self._maybe_delete.keys())

  def _sweep(self):
    """Deletes or preserves managed checkpoints."""
    if not self._max_to_keep:
      # Does not update self._last_preserved_timestamp, since everything is kept
      # in the active set.
      return
    while len(self._maybe_delete) > self._max_to_keep:
      filename, timestamp = self._maybe_delete.popitem(last=False)
      # Even if we're keeping this checkpoint due to
      # keep_checkpoint_every_n_hours, we won't reference it to avoid
      # infinitely-growing CheckpointState protos.
      if (self._keep_checkpoint_every_n_hours
          and (timestamp - self._keep_checkpoint_every_n_hours * 3600.
               >= self._last_preserved_timestamp)):
        self._last_preserved_timestamp = timestamp
        continue
      _delete_file_if_exists(filename + ".index")
      _delete_file_if_exists(filename + ".data-?????-of-?????")

  def _record_state(self):
    """Saves the `CheckpointManager`'s state in `directory`."""
    filenames, timestamps = zip(*self._maybe_delete.items())
    update_checkpoint_state_internal(
        self._directory,
        model_checkpoint_path=self.latest_checkpoint,
        all_model_checkpoint_paths=filenames,
        all_model_checkpoint_timestamps=timestamps,
        last_preserved_timestamp=self._last_preserved_timestamp,
        save_relative_paths=True)

  @property
  def _prefix(self):
    """A common prefix for all checkpoints saved with this manager.

    For example, if `directory` (a constructor argument) were `"/tmp/tf-model"`,
    `prefix` would be `"/tmp/tf-model/ckpt"` and checkpoints would generally be
    numbered `"/tmp/tf-model/ckpt-1"`, `"/tmp/tf-model/ckpt-2"`, and so on. Each
    checkpoint has several associated files
    (e.g. `"/tmp/tf-model/ckpt-2.index"`).

    Returns:
      A string prefix.
    """
    return self._checkpoint_prefix

  def save(self, session=None, checkpoint_number=None):
    """Creates a new checkpoint and manages it.

    Args:
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.
      checkpoint_number: An optional integer, or an integer-dtype `Variable` or
        `Tensor`, used to number the checkpoint. If `None` (default),
        checkpoints are numbered using `checkpoint.save_counter`. Even if
        `checkpoint_number` is provided, `save_counter` is still incremented. A
        user-provided `checkpoint_number` is not incremented even if it is a
        `Variable`.

    Returns:
      The path to the new checkpoint. It is also recorded in the `checkpoints`
      and `latest_checkpoint` properies.
    """
    # Save counter logic duplicated from tf.train.Checkpoint, soon to diverge
    # slightly with a custom numbering option.
    if context.executing_eagerly():
      save_counter = self._checkpoint.save_counter
      save_counter.assign_add(1)
    else:
      if session is None:
        session = ops.get_default_session()

      def _initializing_creator(next_creator, **kwargs):
        """Initialize the save counter if it has been newly created."""
        v = next_creator(**kwargs)
        session.run(v.initializer)
        return v

      with variable_scope.variable_creator_scope(_initializing_creator):
        save_counter = self._checkpoint.save_counter
      if self._save_counter_assign is None:
        self._save_counter_assign = save_counter.assign_add(1, read_value=False)
      session.run(self._save_counter_assign)
    if checkpoint_number is None:
      checkpoint_number = save_counter
    if not isinstance(checkpoint_number, compat.integral_types):
      checkpoint_number = training_util.global_step(
          sess=session, global_step_tensor=checkpoint_number)
    prefix = "%s-%d" % (self._prefix, checkpoint_number)
    save_path = self._checkpoint.write(prefix)
    timestamp = time.time()
    # If this is an overwritten checkpoint we were previously tracking, delete
    # and reinsert it to make sure it goes to the end of the queue.
    if save_path in self._maybe_delete:
      del self._maybe_delete[save_path]
    self._maybe_delete[save_path] = timestamp
    self._latest_checkpoint = save_path
    self._sweep()
    self._record_state()
    return save_path
