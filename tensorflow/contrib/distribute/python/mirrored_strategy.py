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
"""Class MirroredStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_strategy


# TODO(josh11b): Replace asserts in this file with if ...: raise ...


@contextlib.contextmanager
def _enter_graph(g):
  if context.executing_eagerly():
    with g.as_default(), context.eager_mode():
      yield
  else:
    with g.as_default():
      yield


def _cpu_device(device):
  cpu_device = tf_device.DeviceSpec.from_string(device)
  cpu_device.merge_from(tf_device.DeviceSpec(device_type="CPU", device_index=0))
  return cpu_device.to_string()


class _RequestedStop(Exception):
  pass


# _call_for_each_tower and _reduce_non_distributed_value are not members of
# MirroredStrategy so that they are generally not allowed to use anything
# specific to MirroredStrategy and thus can be shared with other distribution
# strategies.


# TODO(yuefengz): maybe create a common class for those who need to call this
# _call_for_each_tower.
def _call_for_each_tower(distribution, fn, *args, **kwargs):
  """Run `fn` in separate threads, once per tower/worker device.

  Args:
    distribution: the DistributionStrategy object.
    fn: function to run (will be run once per device, each in its own thread).
    *args: positional arguments for `fn`
    **kwargs: keyword arguments for `fn`.
        `"run_concurrently"`: Boolean indicating whether executions of `fn`
           can be run concurrently (under eager execution only), defaults to
           `True`.

  Returns:
    Merged return value of `fn` across all towers.

  Raises:
    RuntimeError: If fn() calls get_tower_context().merge_call() a different
        number of times from the available devices.
  """
  run_concurrently = kwargs.pop("run_concurrently", True)
  if not context.executing_eagerly():
    # Lots of TF library code isn't thread-safe in graph mode, and
    # there is little to be gained by turning on multithreading when
    # constructing a graph.
    run_concurrently = False
    # Needed for per-thread device, etc. contexts in graph mode.
    ops.get_default_graph().switch_to_thread_local()
  elif run_concurrently is None:
    run_concurrently = True

  coord = coordinator.Coordinator(clean_stop_exception_types=(_RequestedStop,))

  shared_variable_store = {}

  # TODO(isaprykin): Create these threads once instead of during every run()
  # call.
  threads = []
  for index, d in enumerate(distribution.worker_devices):
    variable_creator_fn = shared_variable_creator.make_fn(
        shared_variable_store, index)
    t = MirroredStrategy._MirroredTowerThread(  # pylint: disable=protected-access
        distribution, coord, d, variable_creator_fn, fn,
        *values.select_device(d, args), **values.select_device(d, kwargs))
    threads.append(t)

  for t in threads:
    t.start()

  # When `fn` starts `should_run` event is set on _MirroredTowerThread
  # (`MTT`) threads. The execution waits until
  # `MTT.has_paused` is set, which indicates that either `fn` is
  # complete or a `get_tower_context().merge_call()` is called.  If `fn` is
  # complete, then `MTT.done` is set to True.  Otherwise, arguments
  # of `get_tower_context().merge_call` from all paused threads are grouped
  # and the `merge_fn` is performed.  Results of the
  # `get_tower_context().merge_call` are then set to `MTT.merge_result`.
  # Each such `get_tower_context().merge_call` call returns the
  # `MTT.merge_result` for that thread when `MTT.should_run` event
  # is reset again. Execution of `fn` resumes.

  try:
    with coord.stop_on_exception():
      all_done = False
      while not all_done and not coord.should_stop():
        done = []
        if run_concurrently:
          for t in threads:
            t.should_run.set()
          for t in threads:
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        else:
          for t in threads:
            t.should_run.set()
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        if coord.should_stop():
          return None
        all_done = all(done)
        if not all_done:
          if any(done):
            raise RuntimeError("Some towers made a different number of "
                               "tower_context().merge_call() calls.")
          # get_tower_context().merge_call() case
          merge_args = values.regroup({t.device: t.merge_args for t in threads})
          merge_kwargs = values.regroup(
              {t.device: t.merge_kwargs for t in threads})
          # We capture the name_scope of the MTT when we call merge_fn
          # to ensure that if we have opened a name scope in the MTT,
          # it will be respected when executing the merge function. We only
          # capture the name_scope from the first MTT and assume it is
          # the same for all other MTTs.
          mtt_captured_name_scope = threads[0].captured_name_scope
          with ops.name_scope(mtt_captured_name_scope):
            merge_result = threads[0].merge_fn(distribution, *merge_args,
                                               **merge_kwargs)
          for t in threads:
            t.merge_result = values.select_device(t.device, merge_result)
  finally:
    for t in threads:
      t.should_run.set()
    coord.join(threads)

  return values.regroup({t.device: t.main_result for t in threads})


def _reduce_non_distributed_value(distribution, aggregation, value,
                                  destinations):
  """Reduce a non-DistributedValue `value` to `destinations`."""
  if isinstance(value, values.DistributedValues):
    raise ValueError("You are passing a `DistributedValue` to "
                     "`_reduce_non_distributed_value`, which is not allowed.")

  # If the same value is present on all towers then the PerDevice value will
  # be a single value. We also handle the case when `value` is a single value
  # and equal to 0.
  if value == 0:
    return 0
  # If the aggregation type is MEAN or ONLY_FIRST_TOWER, then this
  # essentially means that the same value should be on all destinations.
  if aggregation in (
      variable_scope.VariableAggregation.MEAN,
      variable_scope.VariableAggregation.ONLY_FIRST_TOWER):
    return value

  cross_tower_ops_lib.validate_destinations(destinations)
  # We do not support an aggregation type of SUM if the value is the same across
  # all towers. We call this as part of assign functions for MirroredVariables
  # and summing up identical values across towers is not clearly defined.
  if (len(distribution.worker_devices) != 1 or
      not cross_tower_ops_lib.check_destinations(destinations)):
    raise ValueError("A non-DistributedValues value %s cannot be reduced with "
                     "the given aggregation %s." % (value, aggregation))
  # TODO(anjalisridhar): Moves these methods to a device utility file?
  devices = cross_tower_ops_lib.get_devices_from(destinations)
  if len(devices) == 1:
    with ops.device(devices[0]):
      return array_ops.identity(value)
  else:
    value_updates = {}
    for d in devices:
      with ops.device(d):
        value_updates[d] = array_ops.identity(value)
    return values.Mirrored(value_updates)


def _create_mirrored_variable(devices, real_mirrored_creator, *args, **kwargs):  # pylint: disable=g-missing-docstring
  # Figure out what collections this variable should be added to.
  # We'll add the MirroredVariable to those collections instead.
  collections = kwargs.pop("collections", None)
  if collections is None:
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []

  # Get synchronization value
  synchronization = kwargs.get("synchronization",
                               variable_scope.VariableSynchronization.ON_WRITE)
  if synchronization == variable_scope.VariableSynchronization.NONE:
    raise ValueError("`NONE` variable synchronization mode is not "
                     "supported with `Mirrored` distribution strategy. Please"
                     " change the `synchronization` for variable: " +
                     kwargs["name"])
  elif synchronization == variable_scope.VariableSynchronization.ON_READ:
    # Variables that are to be synced on read are tower local.
    is_tower_local = True
    kwargs["trainable"] = False
  elif (synchronization == variable_scope.VariableSynchronization.ON_WRITE or
        synchronization == variable_scope.VariableSynchronization.AUTO):
    # `AUTO` synchronization for `MirroredStrategy` is `ON_WRITE`.
    is_tower_local = False
  else:
    raise ValueError("Invalid variable synchronization mode: " +
                     synchronization + " for variable: " + kwargs["name"])

  # Get aggregation value
  aggregation = kwargs.pop("aggregation",
                           variable_scope.VariableAggregation.NONE)
  if aggregation not in (
      variable_scope.VariableAggregation.NONE,
      variable_scope.VariableAggregation.SUM,
      variable_scope.VariableAggregation.MEAN,
      variable_scope.VariableAggregation.ONLY_FIRST_TOWER
  ):
    raise ValueError("Invalid variable aggregation mode: " + aggregation +
                     " for variable: " + kwargs["name"])

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with tape.stop_recording():
    index = real_mirrored_creator(devices, *args, **kwargs)

    if is_tower_local:
      result = values.TowerLocalVariable(index, index[devices[0]], aggregation)
    else:
      result = values.MirroredVariable(index, index[devices[0]], aggregation)

  # Add the wrapped variable to the requested collections.
  # The handling of eager mode and the global step matches
  # ResourceVariable._init_from_args().
  if not context.executing_eagerly():
    g = ops.get_default_graph()
    # If "trainable" is True, next_creator() will add the member variables
    # to the TRAINABLE_VARIABLES collection, so we manually remove
    # them and replace with the MirroredVariable. We can't set
    # "trainable" to False for next_creator() since that causes functions
    # like implicit_gradients to skip those variables.
    if kwargs.get("trainable", True):
      collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
      l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
      for v in index.values():
        if v in l:
          l.remove(v)
    g.add_to_collections(collections, result)
  elif ops.GraphKeys.GLOBAL_STEP in collections:
    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, result)

  return result


class MirroredStrategy(distribute_lib.StrategyV1):
  """Mirrors vars to distribute across multiple devices and machines.

  This strategy uses one tower per device and sync replication for its multi-GPU
  version.

  When `cluster_spec` is given by the `configure` method., it turns into the
  mulit-worker version that works on multiple workers with in-graph replication.
  Note: `configure` will be called by higher-level APIs if running in
  distributed environment.

  There are several important concepts for distributed TensorFlow, e.g.
  `client`, `job`, `task`, `cluster`, `in-graph replication` and
  `synchronous training` and they have already been defined in the
  [TensorFlow's documentation](https://www.tensorflow.org/deploy/distributed).
  The distribution strategy inherits these concepts as well and in addition to
  that we also clarify several more concepts:

  * **In-graph replication**: the `client` creates a single `tf.Graph` that
    specifies tasks for devices on all workers. The `client` then creates a
    client session which will talk to the `master` service of a `worker`. Then
    the `master` will partition the graph and distribute the work to all
    participating workers.
  * **Worker**: A `worker` is a TensorFlow `task` that usually maps to one
    physical machine. We will have multiple `worker`s with different `task`
    index. They all do similar things except for one worker checkpointing model
    variables, writing summaries, etc. in addition to its ordinary work.

  The multi-worker version of this class maps one tower to one device on a
  worker. It mirrors all model variables on all towers. For example, if you have
  two `worker`s and each `worker` has 4 GPUs, it will create 8 copies of the
  model variables on these 8 GPUs. Then like in MirroredStrategy, each tower
  performs their computation with their own copy of variables unless in
  cross-tower model where variable or tensor reduction happens.

  Args:
    devices: a list of device strings.
    num_gpus: number of GPUs. For local training, either specify `devices` or
      `num_gpus`. In distributed training, this must be specified as number of
      GPUs on each worker.
    num_gpus_per_worker: number of GPUs per worker. This is the same as
      `num_gpus` and only one of `num_gpus` and `num_gpus_per_worker` can be
      specified.
    cross_tower_ops: optional, a descedant of `CrossTowerOps`. If this is not
      set, the `configure` method will try to find the best one.
    prefetch_on_device: optional boolean to specify whether to prefetch input
      data to devices.
    auto_shard_dataset: whether to auto-shard the dataset when there are
      multiple workers.
  """

  def __init__(self,
               devices=None,
               num_gpus=None,
               num_gpus_per_worker=None,
               cross_tower_ops=None,
               prefetch_on_device=None,
               auto_shard_dataset=False):
    super(MirroredStrategy, self).__init__()

    self._cross_tower_ops = cross_tower_ops
    self._prefetch_on_device = prefetch_on_device
    self._auto_shard_dataset = auto_shard_dataset
    # Rememeber num GPUs which might be needed by `configure` method.
    if num_gpus is not None and num_gpus_per_worker is not None:
      raise ValueError(
          "You cannot specify both `num_gpus` and `num_gpus_per_worker`.")
    if num_gpus is None:
      num_gpus = num_gpus_per_worker
    extended = MirroredExtended(self, devices, num_gpus,
                                cross_device_ops or cross_tower_ops,
                                auto_shard_dataset)
    super(MirroredStrategy, self).__init__(extended)

  # Override to change the documentation to reflect the different handling of
  # global vs. local batch size between core and contrib.
  def make_dataset_iterator(self, dataset):  # pylint: disable=useless-super-delegation
    """Makes an iterator for input provided via `dataset`.

    NOTE: The batch size of the `dataset` argument is treated differently for
    this contrib version of `MirroredStrategy`.

    Data from the given dataset will be distributed evenly across all the
    compute replicas. We will assume that the input dataset is batched by the
    per-replica batch size.

    The user could also use `make_input_fn_iterator` if they want to
    customize which input is fed to which replica/worker etc.

    Args:
      dataset: `tf.data.Dataset` that will be distributed evenly across all
        replicas.

    Returns:
      An `tf.distribute.InputIterator` which returns inputs for each step of the
      computation.  User should call `initialize` on the returned iterator.
    """
    return super(MirroredStrategy, self).make_dataset_iterator(dataset)


    self._initialize_local(self._num_gpus, devices)

  def _initialize_local(self, num_gpus, devices):
    """Initializes the object for local training."""
    self._cluster_spec = None
    # Convert `num_gpus` into `devices`, shouldn't specify both.
    if devices is None:
      if num_gpus is None:
        num_gpus = context.num_gpus()
      if num_gpus == 0:
        devices = ["/device:CPU:0"]
      else:
        devices = ["/device:GPU:%d" % d for d in range(num_gpus)]
    elif num_gpus is not None:
      raise ValueError("Must only specify one of `devices` and `num_gpus`.")
    self._num_gpus = num_gpus
    # TODO(yuefengz): consider setting the default device.

  def _make_dataset_iterator(self, dataset):
    """Make iterator from dataset without splitting the batch.

    This implementation is different than the one in
    `tf.distribute.MirroredStrategy` for purposes of backward compatibility.
    We treat the incoming dataset's batch size as per replica batch size.

    Args:
      dataset: `tf.data.Dataset` for input.
    Returns:
      An `InputIterator` which returns inputs for each step of the computation.
    """
    return input_lib.DatasetIterator(dataset, self._input_workers)

  @property
  def _global_batch_size(self):
    """The contrib version of Mirrored strategy uses per-replica batch size."""
    return False

  @property
  def should_init(self):
    return True

  @property
  def should_checkpoint(self):
    return True

  @property
  def should_save_summary(self):
    return True

  def non_slot_devices(self, var_list):
    del var_list
    return list(self._devices)

  def _get_devices_from(self, colocate_with=None):
    if colocate_with is None:
      return self._devices
    else:
      return cross_tower_ops_lib.get_devices_from(colocate_with)

  class _MirroredTowerThread(threading.Thread):
    """A thread that runs() a function on a device."""

    def __init__(self, dist, coord, device, variable_creator_fn, fn, *args,
                 **kwargs):
      super(MirroredStrategy._MirroredTowerThread, self).__init__()  # pylint: disable=protected-access
      self.coord = coord
      self.distribution = dist
      self.device = device
      self.tower_id = dist.worker_devices.index(device)
      self.variable_creator_fn = variable_creator_fn
      # State needed to run and return the results of `fn`.
      self.main_fn = fn
      self.main_args = args
      self.main_kwargs = kwargs
      self.main_result = None
      self.done = False
      # State needed to run the next merge_call() (if any) requested via
      # TowerContext.
      self.merge_fn = None
      self.merge_args = None
      self.merge_kwargs = None
      self.merge_result = None
      self.captured_name_scope = None
      # We use a thread.Event for the main thread to signal when this
      # thread should start running (`should_run`), and another for
      # this thread to transfer control back to the main thread
      # (`has_paused`, either when it gets to a
      # `get_tower_context().merge_call` or when `fn` returns). In
      # either case the event starts cleared, is signaled by calling
      # set(). The receiving thread waits for the signal by calling
      # wait() and then immediately clearing the event using clear().
      self.should_run = threading.Event()
      self.has_paused = threading.Event()
      # These fields have to do with inheriting various contexts from the
      # parent thread:
      # pylint: disable=protected-access
      self.context_mode = context.context()._eager_context.mode
      if not context.context()._context_handle:
        context.context()._initialize_handle_and_devices()
      self.context_device_policy = (
          pywrap_tensorflow.TFE_ContextGetDevicePlacementPolicy(
              context.context()._context_handle))
      self.graph = ops.get_default_graph()
      self._variable_creator_stack = self.graph._variable_creator_stack[:]
      self._captured_var_scope = variable_scope.get_variable_scope()
      # Adding a "/" at end lets us re-enter this scope later.
      self._name_scope = self.graph.get_name_scope()
      if self._name_scope:
        self._name_scope += "/"
      if self.tower_id > 0:
        if not self._name_scope:
          self._name_scope = ""
        self._name_scope += "tower_%d/" % self.tower_id

    def run(self):
      # pylint: disable=protected-access
      self.graph._variable_creator_stack = self._variable_creator_stack
      self.should_run.wait()
      self.should_run.clear()
      try:
        if self.coord.should_stop():
          return
        with self.coord.stop_on_exception(), \
            context.context()._mode(self.context_mode), \
            context.context().device_policy(self.context_device_policy), \
            _enter_graph(self.graph), \
            MirroredTowerContext(self.distribution, self.tower_id), \
            ops.device(self.device), \
            ops.name_scope(self._name_scope), \
            variable_scope.variable_scope(
                self._captured_var_scope, reuse=self.tower_id > 0), \
            variable_scope.variable_creator_scope(self.variable_creator_fn):
          self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
          self.done = True
      finally:
        self.has_paused.set()


class MirroredTowerContext(distribute_lib.TowerContext):
  """TowerContext used in MirroredStrategy.call_for_each_tower().

  Opened in `_MirroredTowerThread`, to allow the user to invoke
  `MirroredStrategy`'s specific implementation of `merge_call()`,
  which works by delegating the function and its arguments to
  the main thread (the one that invoked
  `MirroredStrategy.call_for_each_tower()`).
  """

  def _merge_call(self, fn, *args, **kwargs):
    """Delegate to the main thread to actually perform merge_call()."""
    t = threading.current_thread()  # a _MirroredTowerThread
    t.merge_fn = fn
    t.merge_args = args
    t.merge_kwargs = kwargs
    t.captured_name_scope = t.graph.get_name_scope()
    # Adding a "/" at end lets us re-enter this scope later.
    if t.captured_name_scope:
      t.captured_name_scope += "/"
    t.has_paused.set()
    t.should_run.wait()
    t.should_run.clear()
    if t.coord.should_stop():
      raise _RequestedStop()
    return t.merge_result

  @property
  def device(self):
    distribute_lib.require_tower_context(self)
    return self._distribution_strategy.worker_devices[self._tower_id]
