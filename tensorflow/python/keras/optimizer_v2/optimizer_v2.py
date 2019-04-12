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

"""Version 2 of class Optimizer."""
# pylint: disable=g-bad-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


class _OptimizableVariable(object):
  """Interface for abstracting over variables in the optimizers."""

  @abc.abstractmethod
  def target(self):
    """Returns the optimization target for this variable."""
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def update_op(self, optimizer, g, *args):
    """Returns the update ops for updating the variable."""
    raise NotImplementedError("Calling an abstract method.")


class _RefVariableProcessor(_OptimizableVariable):
  """Processor for Variable."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v._ref()  # pylint: disable=protected-access

  def update_op(self, optimizer, g, *args):
    if isinstance(g, ops.Tensor):
      update_op = optimizer._apply_dense(g, self._v, *args)  # pylint: disable=protected-access
      if self._v.constraint is not None:
        with ops.control_dependencies([update_op]):
          return self._v.assign(self._v.constraint(self._v))
      else:
        return update_op
    else:
      assert isinstance(g, ops.IndexedSlices), ("Gradient ", g, " is neither a "
                                                "tensor nor IndexedSlices.")
      if self._v.constraint is not None:
        raise RuntimeError(
            "Cannot use a constraint function on a sparse variable.")
      # pylint: disable=protected-access
      return optimizer._apply_sparse_duplicate_indices(g, self._v, *args)


class _DenseReadResourceVariableProcessor(_OptimizableVariable):
  """Processor for dense ResourceVariables."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g, *args):
    # pylint: disable=protected-access
    update_op = optimizer._resource_apply_dense(g, self._v.op.inputs[0], *args)
    if self._v.constraint is not None:
      with ops.control_dependencies([update_op]):
        return self._v.assign(self._v.constraint(self._v))
    else:
      return update_op


class _DenseResourceVariableProcessor(_OptimizableVariable):
  """Processor for dense ResourceVariables."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g, *args):
    # pylint: disable=protected-access
    if isinstance(g, ops.IndexedSlices):
      if self._v.constraint is not None:
        raise RuntimeError(
            "Cannot use a constraint function on a sparse variable.")
      return optimizer._resource_apply_sparse_duplicate_indices(
          g.values, self._v, g.indices, *args)
    update_op = optimizer._resource_apply_dense(g, self._v, *args)
    if self._v.constraint is not None:
      with ops.control_dependencies([update_op]):
        return self._v.assign(self._v.constraint(self._v))
    else:
      return update_op


class _TensorProcessor(_OptimizableVariable):
  """Processor for ordinary Tensors.

  Even though a Tensor can't really be updated, sometimes it is useful to
  compute the gradients with respect to a Tensor using the optimizer. Updating
  the Tensor is, of course, unsupported.
  """

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

@six.add_metaclass(abc.ABCMeta)
@keras_export("keras.optimizers.Optimizer")
class OptimizerV2(trackable.Trackable):
  """Updated base class for optimizers.

  This class defines the API to add Ops to train a model.  You never use this
  class directly, but instead instantiate one of its subclasses such as
  `tf.keras.optimizers.SGD`, `tf.keras.optimizers.Adam`.

  ### Usage

  ```python
  # Create an optimizer with the desired parameters.
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  # `loss` is a callable that takes no argument and returns the value
  # to minimize.
  loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
  # In graph mode, returns op that minimizes the loss by updating the listed
  # variables.
  opt_op = opt.minimize(loss, var_list=[var1, var2])
  opt_op.run()
  # In eager mode, simply call minimize to update the list of variables.
  opt.minimize(loss, var_list=[var1, var2])
  ```

  ### Custom training loop with Keras models

  In Keras models, sometimes variables are created when the model is first
  called, instead of construction time. Examples include 1) sequential models
  without input shape pre-defined, or 2) subclassed models. Pass var_list as
  callable in these cases.

  Example:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
  model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid')
  loss_fn = lambda: tf.keras.losses.mse(model(input), output)
  var_list_fn = lambda: model.trainable_weights
  for input, output in data:
    opt.minimize(loss_fn, var_list_fn)
  ```

  ### Processing gradients before applying them.

  Calling `minimize()` takes care of both computing the gradients and
  applying them to the variables.  If you want to process the gradients
  before applying them you can instead use the optimizer in three steps:

  1.  Compute the gradients with `tf.GradientTape`.
  2.  Process the gradients as you wish.
  3.  Apply the processed gradients with `apply_gradients()`.

  Example:

  ```python
  # Create an optimizer.
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)

  # Compute the gradients for a list of variables.
  with tf.GradientTape() as tape:
    loss = <call_loss_function>
  vars = <list_of_variables>
  grads = tape.gradient(loss, vars)
  processed_grads = [process_gradient(g) for g in grads]
  grads_and_vars = zip(processed_grads, var_list)

  # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
  # need to the 'gradient' part, for example cap them, etc.
  capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

  # Ask the optimizer to apply the capped gradients.
  opt.apply_gradients(capped_grads_and_vars)
  ```

  ### Use with `tf.distribute.Strategy`.

  This optimizer class is `tf.distribute.Strategy` aware, which means it
  automatically sums gradients across all replicas. To average gradients,
  you divide your loss by the global batch size, which is done automatically
  if you use a member of `tf.keras.losses` or `tf.losses`. See the
  `reduction` argument of your loss which should be set to
  `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` for averaging or
  `tf.keras.losses.Reduction.SUM` for not.

  If you are not using these and you want to average gradients, you should use
  `tf.math.reduce_sum` to add up your per-example losses and then divide by the
  global batch size. Note that when using `tf.distribute.Strategy`, the first
  component of a tensor's shape is the *replica-local* batch size, which is off
  by a factor equal to the number of replicas being used to compute a single
  step. As a result, using `tf.math.reduce_mean` will give the wrong answer,
  resulting in gradients that can be many times too big.

  ### Variable Constraint

  All Keras optimizers respect variable constraints. If constraint function is
  passed to any variable, the constraint will be applied to the variable after
  the gradient has been applied to the variable.
  Important: If gradient is sparse tensor, variable constraint is not supported.

  ### Thread Compatibility

  The entire optimizer is currently thread compatible, not thread-safe. The user
  needs to perform synchronization if necessary.

  ### Slots

  Many optimizer subclasses, such as `Adam` and `Adagrad` allocate and manage
  additional variables associated with the variables to train.  These are called
  <i>Slots</i>.  Slots have names and you can ask the optimizer for the names of
  the slots that it uses.  Once you have a slot name you can ask the optimizer
  for the variable it created to hold the slot value.

  This can be useful if you want to log debug a training algorithm, report stats
  about the slots, etc.

  ### Non-slot variables

  Some optimizer subclasses, such as `AdamOptimizer` have variables that
  are not associated with the variables to train, just the step itself.

  ### Hyper parameters

  These are arguments passed to the optimizer subclass constructor
  (the `__init__` method), and then passed to `self._set_hyper()`.
  They can be either regular Python values (like 1.0), tensors, or
  callables. If they are callable, the callable will be called during
  `apply_gradients()` to get the value for the hyper parameter.

  Hyper parameters can be overwritten through user code:

  Example:

  ```python
  # Create an optimizer with the desired parameters.
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  # `loss` is a callable that takes no argument and returns the value
  # to minimize.
  loss = lambda: 3 * var1 + 2 * var2
  # In eager mode, simply call minimize to update the list of variables.
  opt.minimize(loss, var_list=[var1, var2])
  # update learning rate
  opt.learning_rate = 0.05
  opt.minimize(loss, var_list=[var1, var2])
  ```

  ### Write a customized optimizer.
  If you intend to create your own optimization algorithm, simply inherit from
  this class and override the following methods:

    - resource_apply_dense (update variable given gradient tensor is dense)
    - resource_apply_sparse (update variable given gradient tensor is sparse)
    - create_slots (if your optimizer algorithm requires additional variables)
    - get_config (serialization of the optimizer, include all hyper parameters)
  """

  # Values for gate_gradients.
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self, name):
    """Create a new Optimizer.

    This must be called by the constructors of subclasses.
    Note that Optimizer instances should not bind to a single graph,
    and so shouldn't keep Tensors as member variables. Generally
    you should be able to use the _set_hyper()/state.get_hyper()
    facility instead.

    Args:
      name: A non-empty string.  The name to use for accumulators created
        for the optimizer.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    Raises:
      ValueError: If name is malformed.
      RuntimeError: If _create_slots has been overridden instead of
          _create_vars.
    """
    allowed_kwargs = {"clipnorm", "clipvalue", "lr", "decay"}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError("Unexpected keyword argument "
                        "passed to optimizer: " + str(k))
      # checks that all keyword arguments are non-negative.
      if kwargs[k] < 0:
        raise ValueError("Expected {} >= 0, received: {}".format(k, kwargs[k]))

    self._use_locking = True
    self._name = name
    # Map from graph_key to state for that graph. We use the graph_key
    # since it works in both eager and graph mode, and gives the outer
    # graph inside functions.
    tower_context = distribution_strategy_context.get_tower_context()
    if tower_context is None:
      # In a cross-tower context for a DistributionStrategy, which means
      # only one Optimizer will be created, not one per tower.
      self._per_graph_state = {}
    else:
      # We use get_tower_context().merge_call() to get a single dict
      # shared across all model replicas when running with a
      # DistributionStrategy.
      self._per_graph_state = tower_context.merge_call(lambda _: {})

    # Hyper parameters, and whether they should be re-evaluated every step.
    self._hyper = {}
    # dict: {variable name : {slot name : variable}}
    self._slots = {}
    self._slot_names = []
    self._weights = []
    self._iterations = None

    # For implementing Trackable. Stores information about how to restore
    # slot variables which have not yet been created
    # (trackable._CheckpointPosition objects).
    #  {slot_name :
    #      {_var_key(variable_to_train): [checkpoint_position, ... ], ... },
    #   ... }
    self._deferred_slot_restorations = {}

    decay = kwargs.pop("decay", 0.0)
    if decay < 0.:
      raise ValueError("decay cannot be less than 0: {}".format(decay))
    self._initial_decay = decay
    if "clipnorm" in kwargs:
      self.clipnorm = kwargs.pop("clipnorm")
    if "clipvalue" in kwargs:
      self.clipvalue = kwargs.pop("clipvalue")

    self._hypers_created = False

  def minimize(self, loss, var_list, grad_loss=None, name=None):
    """Minimize `loss` by updating `var_list`.

    This method simply computes gradient using `tf.GradientTape` and calls
    `apply_gradients()`. If you want to process the gradient before applying
    then call `tf.GradientTape` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A callable taking no arguments which returns the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` since the variables are created at the first time `loss` is
        called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      stop_gradients: Optional. A Tensor or list of tensors not to differentiate
        through.
      scale_loss_by_num_towers: Optional boolean. If true, scale the loss
        down by the number of towers. By default, auto-detects whether this
        is needed.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.

    """
    grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss, stop_gradients=stop_gradients,
        scale_loss_by_num_towers=scale_loss_by_num_towers)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

    return self.apply_gradients(grads_and_vars, global_step=global_step,
                                name=name)

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None, stop_gradients=None,
                        scale_loss_by_num_towers=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A callable taking no arguments which returns the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` and the variables are created at the first time when `loss`
        is called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      stop_gradients: Optional. A Tensor or list of tensors not to differentiate
        through.
      scale_loss_by_num_towers: Optional boolean. If true, scale the loss
        down by the number of towers. By default, auto-detects whether this
        is needed.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
      RuntimeError: If called with eager execution enabled and `loss` is
        not callable.

    @compatibility(eager)
    When eager execution is enabled, `gate_gradients`, `aggregation_method`,
    and `colocate_gradients_with_ops` are ignored.
    @end_compatibility
    """
    # TODO(josh11b): Test that we handle weight decay in a reasonable way.
    with backprop.GradientTape() as tape:
      if not callable(var_list):
        tape.watch(var_list)
      loss_value = loss()
    if callable(var_list):
      var_list = var_list()
    var_list = nest.flatten(var_list)
    grads = tape.gradient(loss_value, var_list, grad_loss)

    if hasattr(self, "clipnorm"):
      grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
    if hasattr(self, "clipvalue"):
      grads = [
          clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
          for g in grads
      ]

    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes(
        [v for g, v in grads_and_vars
         if g is not None and v.dtype != dtypes.resource])
    return grads_and_vars

  def get_gradients(self, loss, params):
    """Returns gradients of `loss` with respect to `params`.

    Arguments:
      loss: Loss tensor.
      params: List of variables.

    Returns:
      List of gradient tensors.

    Raises:
      ValueError: In case any gradient cannot be computed (e.g. if gradient
        function not implemented).
    """
    with backend.get_graph().as_default():
      grads = gradients.gradients(loss, params)
    if None in grads:
      raise ValueError("An operation has `None` for gradient. "
                       "Please make sure that all of your ops have a "
                       "gradient defined (i.e. are differentiable). "
                       "Common ops without gradient: "
                       "K.argmax, K.round, K.eval.")
    if hasattr(self, "clipnorm"):
      grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
    if hasattr(self, "clipvalue"):
      grads = [
          clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
          for g in grads
      ]
    return grads

  def apply_gradients(self, grads_and_vars, name=None):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    """
    grads_and_vars = _filter_grads(grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]

    # Create iteration if necessary.
    with ops.init_scope():
      _ = self.iterations
      self._create_hypers()
      self._create_slots(var_list)

    self._prepare(var_list)

    return distribute_ctx.get_replica_context().merge_call(
        self._distributed_apply, args=(grads_and_vars,), kwargs={"name": name})

  def _distributed_apply(self, distribution, grads_and_vars, name):
    """`apply_gradients` using a `DistributionStrategy`."""
    reduced_grads = distribution.extended.batch_reduce_to(
        ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    var_list = [v for _, v in grads_and_vars]
    grads_and_vars = zip(reduced_grads, var_list)

    def apply_grad_to_update_var(var, grad):
      """Apply gradient to variable."""
      if isinstance(var, ops.Tensor):
        raise NotImplementedError("Trying to update a Tensor ", var)
      if isinstance(grad, ops.IndexedSlices):
        if var.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")
        return self._resource_apply_sparse_duplicate_indices(
            grad.values, var, grad.indices)
      update_op = self._resource_apply_dense(grad, var)
      if var.constraint is not None:
        with ops.control_dependencies([update_op]):
          return var.assign(var.constraint(var))
      else:
        return update_op

    update_ops = []
    with ops.name_scope(name, self._name) as name:
      per_graph_state = self._get_or_create_state(var_list=unwrapped_var_list)
      # Include the current value of any dynamic hyper parameters in `state`.
      non_slot_devices = distribution.non_slot_devices(var_list)
      state = per_graph_state._copy_with_dynamic_hyper(  # pylint: disable=protected-access
          self._hyper, distribution, non_slot_devices)

    # Create any slot and non-slot variables we need in `state`.
    with ops.init_scope():
      self._create_vars(var_list, state)

    with ops.name_scope(name):  # Re-enter name_scope created above
      # Give the child class a chance to do something before we start
      # applying gradients.
      self._prepare(state)

      def update(v, g):
        """Update variable `v` using gradient `g`."""
        assert v is not None

        # Convert the grad to Tensor or IndexedSlices if necessary, and
        # look up a processor for each variable's type.
        try:
          g = ops.convert_to_tensor_or_indexed_slices(g)
        except TypeError:
          raise TypeError(
              "Gradient must be convertible to a Tensor"
              " or IndexedSlices, or None: %s" % g)
        if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
          raise TypeError(
              "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
        processor = _get_processor(v)

        # We colocate all ops created in _apply_dense or _apply_sparse
        # on the same device as the variable.
        # TODO(apassos): figure out how to get the variable name here.
        scope_name = "" if eager_execution else v.op.name
        # device_policy is set because non-mirrored tensors will be read in
        # `update_op`.
        # TODO(josh11b): Make different state objects for each device to
        # avoid needing to set the device_policy.
        with ops.name_scope("update_" + scope_name), \
            context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
          return processor.update_op(self, g, state)

      # Use the processors to update the variables.
      update_ops = []
      for grad, var in grads_and_vars:
        scope_name = ("" if ops.executing_eagerly_outside_functions() else
                      "_" + var.op.name)
        with ops.name_scope("update" + scope_name):
          update_ops.extend(
              distribution.extended.update(
                  var, apply_grad_to_update_var, args=(grad,), group=False))

      any_symbolic = any(isinstance(i, ops.Operation) or
                         tf_utils.is_symbolic_tensor(i) for i in update_ops)
      if not context.executing_eagerly() or any_symbolic:
        # If the current context is graph mode or any of the update ops are
        # symbolic then the step update should be carried out under a graph
        # context. (eager updates execute immediately)
        with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
          with ops.control_dependencies(update_ops):
            return self._iterations.assign_add(1).op

      return self._iterations.assign_add(1)

          def update_global_step(global_step, name):
            return global_step.assign_add(1, read_value=False, name=name)

  def _set_hyper(self, name, value):
    """set hyper `name` to value. value can be callable, tensor, numeric."""
    if isinstance(value, trackable.Trackable):
      self._track_trackable(value, name, overwrite=True)
    if name not in self._hyper:
      self._hyper[name] = value
    else:
      prev_value = self._hyper[name]
      if (callable(prev_value)
          or isinstance(prev_value,
                        (ops.Tensor, int, float,
                         learning_rate_schedule.LearningRateSchedule))
          or isinstance(value, learning_rate_schedule.LearningRateSchedule)):
        self._hyper[name] = value
      else:
        backend.set_value(self._hyper[name], value)

  def _get_hyper(self, name, dtype=None):
    if not self._hypers_created:
      self._create_hypers()
    value = self._hyper[name]
    if isinstance(value, learning_rate_schedule.LearningRateSchedule):
      return value
    if callable(value):
      value = value()
    if dtype:
      return math_ops.cast(value, dtype)
    else:
      return value

  def __getattribute__(self, name):
    """Overridden to support hyperparameter access."""
    try:
      return super(OptimizerV2, self).__getattribute__(name)
    except AttributeError as e:
      # Needed to avoid infinite recursion with __setattr__.
      if name == "_hyper":
        raise e
      # Backwards compatibility with Keras optimizers.
      if name == "lr":
        name = "learning_rate"
      if name in self._hyper:
        return self._get_hyper(name)
      raise e

  def __setattr__(self, name, value):
    """Override setattr to support dynamic hyperparameter setting."""
    # Backwards compatibility with Keras optimizers.
    if name == "lr":
      name = "learning_rate"
    if hasattr(self, "_hyper") and name in self._hyper:
      self._set_hyper(name, value)
    else:
      super(OptimizerV2, self).__setattr__(name, value)

  def get_slot_names(self):
    """A list of names for this optimizer's slots."""
    return self._slot_names

  def add_slot(self, var, slot_name, initializer="zeros"):
    """Add a new slot variable for `var`."""
    if slot_name not in self._slot_names:
      self._slot_names.append(slot_name)
    var_key = _var_key(var)
    slot_dict = self._slots.setdefault(var_key, {})
    weight = slot_dict.get(slot_name, None)
    if weight is None:
      if isinstance(initializer, six.string_types) or callable(initializer):
        initializer = initializers.get(initializer)
        initial_value = functools.partial(
            initializer, shape=var.shape, dtype=var.dtype)
      else:
        initial_value = initializer
      strategy = distribute_ctx.get_strategy()
      with strategy.extended.colocate_vars_with(var):
        weight = tf_variables.Variable(
            name="%s/%s" % (var._shared_name, slot_name),  # pylint: disable=protected-access
            dtype=var.dtype,
            trainable=False,
            initial_value=initial_value)
      backend.track_variable(weight)
      slot_dict[slot_name] = weight
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=weight)
      self._weights.append(weight)
    return weight

  def get_slot(self, var, slot_name):
    var_key = _var_key(var)
    slot_dict = self._slots[var_key]
    return slot_dict[slot_name]

  def _prepare(self, var_list):
    pass

  def _create_hypers(self):
    if self._hypers_created:
      return
    # Iterate hyper values deterministically.
    for name, value in sorted(self._hyper.items()):
      if isinstance(value, ops.Tensor) or callable(value):
        continue
      else:
        self._hyper[name] = self.add_weight(
            name,
            shape=[],
            trainable=False,
            initializer=value,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
    self._hypers_created = True

  @property
  def iterations(self):
    """Variable. The number of training steps this Optimizer has run."""
    if self._iterations is None:
      self._iterations = self.add_weight(
          "iter",
          shape=[],
          dtype=dtypes.int64,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self._weights.append(self._iterations)
    return self._iterations

  @iterations.setter
  def iterations(self, variable):
    if self._iterations is not None:
      raise RuntimeError("Cannot set `iterations` to a new Variable after"
                         "the Optimizer weights have been created")
    self._iterations = variable
    self._weights.append(self._iterations)

  def _decayed_lr(self, var_dtype):
    """Get decayed learning rate as a Tensor with dtype=var_dtype."""
    lr_t = self._get_hyper("learning_rate", var_dtype)
    if isinstance(lr_t, learning_rate_schedule.LearningRateSchedule):
      local_step = math_ops.cast(self.iterations, var_dtype)
      lr_t = math_ops.cast(lr_t(local_step), var_dtype)
    if self._initial_decay > 0.:
      local_step = math_ops.cast(self.iterations, var_dtype)
      decay_t = self._get_hyper("decay", var_dtype)
      lr_t = lr_t / (1. + decay_t * local_step)
    return lr_t

  def get_slot(self, var, name):
    """Return a slot named `name` created for `var` by the Optimizer.

    Some `Optimizer` subclasses use additional variables.  For example
    `Momentum` and `Adagrad` use variables to accumulate updates.  This method
    gives access to these `Variable` objects if for some reason you need them.

    Use `get_slot_names()` to get the list of slot names created by the
    `Optimizer`.

    Args:
      var: A variable passed to `minimize()` or `apply_gradients()`.
      name: A string.

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    state = self._get_state_for_var(var)
    return state.get_slot(var, name) if state is not None else None

  def get_slot_names(self):
    """Return a list of the names of slots created by the `Optimizer`.

    See `get_slot()`.

    Returns:
      A list of strings.
    """
    state = self._get_per_graph_state()
    return state.get_slot_names() if state is not None else []

  def variables(self):
    """A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    """
    if "lr" in config:
      config["learning_rate"] = config.pop("lr")
    if "learning_rate" in config:
      if isinstance(config["learning_rate"], dict):
        config["learning_rate"] = learning_rate_schedule.deserialize(
            config["learning_rate"])
    return cls(**config)

  def _serialize_hyperparameter(self, hyperparameter_name):
    """Serialize a hyperparameter that can be a float, callable, or Tensor."""
    value = self._hyper[hyperparameter_name]
    if isinstance(value, learning_rate_schedule.LearningRateSchedule):
      return learning_rate_schedule.serialize(value)
    if callable(value):
      return value()
    if tensor_util.is_tensor(value):
      return backend.get_value(value)
    return value

    Args:
      var_list: A list of `Variable` objects.
      state: An object with these methods:
        `create_slot(var, val, slot_name, optional_op_name)`,
        `create_slot_with_initializer(`
            `var, initializer, shape, dtype, slot_name, optional_op_name)`,
        `zeros_slot(var, slot_name, optional_op_name)`,
        `create_non_slot_variable(initial_value, name, colocate_with)`,
        `get_hyper(name)`
    """
    # No slots needed by default
    pass

  def _prepare(self, state):
    """Code to execute before applying gradients.

    Note that most uses of _prepare() in Optimizer have been subsumed
    by explicit support for hyper parameters in OptimizerV2

    Args:
      state: An object with a `get_hyper(name)` method.

    Returns:
      Return value will be ignored.
    """
    pass

  def _apply_dense(self, grad, var, state):
    """Add ops to apply dense gradients to `var`.

    Args:
      grad: A `Tensor`.
      var: A `Variable` object.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation`.
    """
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle, state):
    """Add ops to apply dense gradients to the variable `handle`.

    Args:
      grad: a `Tensor` representing the gradient.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError()

  def _resource_apply_sparse_duplicate_indices(
      self, grad, handle, indices, state):
    """Add ops to apply sparse gradients to `handle`, with repeated indices.

    Optimizers which override this method must deal with repeated indices. See
    the docstring of `_apply_sparse_duplicate_indices` for details. By default
    the correct behavior, to sum non-unique indices and their associated
    gradients, is enforced by first pre-processing `grad` and `indices` and
    passing them on to `_resource_apply_sparse`. Optimizers which deal correctly
    with duplicate indices may instead override this method to avoid the
    overhead of summing.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.
      indices: a `Tensor` of integral type representing the indices for
       which the gradient is nonzero. Indices may be repeated.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    # pylint: disable=protected-access
    summed_grad, unique_indices = optimizer_v1._deduplicate_indexed_slices(
        values=grad, indices=indices)
    # pylint: enable=protected-access
    return self._resource_apply_sparse(
        summed_grad, handle, unique_indices, state)

  def _resource_apply_sparse(self, grad, handle, indices, state):
    """Add ops to apply sparse gradients to the variable `handle`.

    Similar to `_apply_sparse`, the `indices` argument to this method has been
    de-duplicated. Optimizers which deal correctly with non-unique indices may
    instead override `_resource_apply_sparse_duplicate_indices` to avoid this
    overhead.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.
      indices: a `Tensor` of integral type representing the indices for
       which the gradient is nonzero. Indices are unique.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError()

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_scatter_update(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_update(x.handle, i, v)]):
      return x.value()

  # ---------------
  # For implementing the trackable interface
  # ---------------

    Optimizers which override this method must deal with IndexedSlices objects
    such as the following:

      IndexedSlicesValue(values=[1, 1], indices=[0, 0], dense_shape=[1])

    The correct interpretation is:

      IndexedSlicesValue(values=[2], indices=[0], dense_shape=[1])

    Many optimizers deal incorrectly with repeated indices when updating based
    on sparse gradients (e.g. summing squares rather than squaring the sum, or
    applying momentum terms multiple times). Adding first is always the correct
    behavior, so this is enforced here by reconstructing the IndexedSlices to
    have only unique indices, then calling _apply_sparse.

    Optimizers which deal correctly with repeated indices may instead override
    this method to avoid the overhead of summing indices.

    Args:
      slot_variable_position: A `trackable._CheckpointPosition` object
        indicating the slot variable `Trackable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    variable_key = _var_key(variable)
    slot_dict = self._slots.get(variable_key, {})
    slot_variable = slot_dict.get(slot_name, None)
    if (slot_variable is None and context.executing_eagerly() and
        slot_variable_position.is_simple_variable()
        # Defer slot variable creation if there is an active variable creator
        # scope. Generally we'd like to eagerly create/restore slot variables
        # when possible, but this may mean that scopes intended to catch
        # `variable` also catch its eagerly created slot variable
        # unintentionally (specifically make_template would add a dependency on
        # a slot variable if not for this case). Deferring is mostly harmless
        # (aside from double initialization), and makes variable creator scopes
        # behave the same way they do when graph building.
        and not ops.get_default_graph()._variable_creator_stack):  # pylint: disable=protected-access
      initializer = trackable.CheckpointInitialValue(
          checkpoint_position=slot_variable_position)
      slot_variable = self.add_slot(
          var=variable,
          initializer=initializer,
          slot_name=slot_name)
      # Slot variables are not owned by any one object (because we don't want to
      # save the slot variable if the optimizer is saved without the non-slot
      # variable, or if the non-slot variable is saved without the optimizer;
      # it's a dependency hypergraph with edges of the form (optimizer, non-slot
      # variable, variable)). So we don't _track_ slot variables anywhere, and
      # instead special-case this dependency and otherwise pretend it's a normal
      # graph.
    if slot_variable is not None:
      # If we've either made this slot variable, or if we've pulled out an
      # existing slot variable, we should restore it.
      slot_variable_position.restore(slot_variable)
    else:
      # We didn't make the slot variable. Defer restoring until it gets created
      # normally. We keep a list rather than the one with the highest restore
      # UID in case slot variables have their own dependencies, in which case
      # those could differ between restores.
      self._deferred_slot_restorations.setdefault(
          slot_name, {}).setdefault(variable_key, []).append(
              slot_variable_position)

    Args:
      grad: `IndexedSlices`, with no repeated indices.
      var: A `Variable` object.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation`.
    """
    raise NotImplementedError()

  def _finish(self, state):
    """Do what is needed to finish the update.

    This is called inside a scope colocated with any non-slot variables.

    Args:
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      The operation to apply updates, or None if no updates.
    """
    return None

  # --------------
  # Utility methods for subclasses.
  # --------------
  def _get_per_graph_state(self):
    # pylint: disable=protected-access
    return self._per_graph_state.get(ops.get_default_graph()._graph_key, None)

def _var_key(var):
  """Key for representing a primary variable, for looking up slots.

    Args:
      slot_variable_position: A `checkpointable._CheckpointPosition` object
        indicating the slot variable `Checkpointable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    state = self._get_or_create_state(var_list=[variable])
    state._create_or_restore_slot_variable(  # pylint: disable=protected-access
        slot_variable_position=slot_variable_position,
        slot_name=slot_name,
        variable=variable,
        optional_op_name=self._name)

  def get_config(self):
    """Returns the config of the optimimizer.

    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer.
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.

  # pylint: disable=protected-access
  # Get the distributed variable if it exists.
  if getattr(var, "_distributed_container", None) is not None:
    var = var._distributed_container()
  if var._in_graph_mode:
    return var._shared_name
  return var._unique_id

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An optimizer instance.
    """
    return cls(**config)

  name = _var_key(var)
  return name + "/" + slot_name


class _RestoredOptimizer(OptimizerV2):
  """A non-functional Optimizer implementation for checkpoint compatibility.

  Holds slot variables and hyperparameters when an optimizer is restored from a
  SavedModel. These variables may be referenced in functions along with ops
  created by the original optimizer, but currently we do not support using the
  optimizer object iself (e.g. through `apply_gradients`).
  """
  # TODO(allenl): Make the restored optimizer functional by tracing its apply
  # methods.

  def __init__(self):
    super(_RestoredOptimizer, self).__init__("_RestoredOptimizer")
    self._hypers_created = True

  def get_config(self):
    # TODO(allenl): Save and restore the Optimizer's config
    raise NotImplementedError(
        "Restoring functional Optimzers from SavedModels is not currently "
        "supported. Please file a feature request if this limitation bothers "
        "you.")

revived_types.register_revived_type(
    "optimizer",
    lambda obj: isinstance(obj, OptimizerV2),
    versions=[revived_types.VersionedTypeRegistration(
        object_factory=lambda proto: _RestoredOptimizer(),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=_RestoredOptimizer._set_hyper  # pylint: disable=protected-access
    )])
