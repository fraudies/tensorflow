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

"""Adagrad optimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Adagrad')
class Adagrad(optimizer_v2.OptimizerV2):
  """Adagrad optimizer.

  It is recommended to leave the parameters of this optimizer at their default
  values.

  Initialization:
  $$accum_{g_0} := \text{initial_accumulator_value}$$

  Update step:
  $$t := t + 1$$
  $$accum_{g_t} := accum_{g_{t-1}} + g^2$$
  $$\theta_t := \theta_{t-1} - lr * g / (\sqrt{accum_{g_t}} + \epsilon)$$

  References:

  * [Paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  * [Introduction]
    (https://ppasupat.github.io/a9online/uploads/proximal_notes.pdf).
  """

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               name='Adagrad',
               **kwargs):
    """Construct a new Adagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      epsilon: A floating point value.
        Starting value for the accumulators, must be positive.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    Raises:
      ValueError: If the `initial_accumulator_value` or `epsilon` is invalid.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate` can be a callable that
    takes no arguments and returns the actual value to use. This can be useful
    for changing these values across different invocations of optimizer
    functions.
    @end_compatibility
    """
    if initial_accumulator_value < 0.0:
      raise ValueError('initial_accumulator_value must be non-negative: %s' %
                       initial_accumulator_value)
    if epsilon is None:
      epsilon = backend_config.epsilon()
    if epsilon < 1e-7:
      raise ValueError('epsilon must be larger than 1e-7: %s' % epsilon)
    super(Adagrad, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._initial_accumulator_value = initial_accumulator_value

  def _create_vars(self, var_list, state):
    for v in var_list:
      dtype = v.dtype.base_dtype
      if v.get_shape().is_fully_defined():
        init = init_ops.constant_initializer(self._initial_accumulator_value,
                                             dtype=dtype)
      else:
        def init(v=v, dtype=dtype):
          # Use a Tensor instead of initializer if variable does not have
          # static shape.
          init_constant = gen_array_ops.fill(array_ops.shape(v),
                                             self._initial_accumulator_value)
          return math_ops.cast(init_constant, dtype)
      state.create_slot_with_initializer(v, init, v.get_shape(), dtype,
                                         "accumulator")

  def _apply_dense(self, grad, var, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.apply_adagrad(
        var,
        acc,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.resource_apply_adagrad(
        var.handle,
        acc.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.sparse_apply_adagrad(
        var,
        acc,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.resource_sparse_apply_adagrad(
        var.handle,
        acc.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad,
        indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(Adagrad, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "initial_accumulator_value": self._initial_accumulator_value
    })
    return config
