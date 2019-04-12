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

"""Adadelta for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Adadelta')
class Adadelta(optimizer_v2.OptimizerV2):
  """Adadelta optimizer.

  It is recommended to leave the parameters of this optimizer at their default
  values.

  $$E[g^2]_0 := 0 \text{(Initialize gradient 2nd order moment vector)}$$
  $$E[\Delta x^2]_0 := 0 \text{(Initialize 2nd order variable update)}$$

  $$t := t + 1$$
  $$E[g^2]_t := \rho * E[g^2]_{t-1} + (1 - \rho) * g^2$$
  $$\Delta x_t = -RMS[\Delta x]_{t-1} * g_t / RMS[g]_t$$
  $$E[\Delta x^2]_t := \rho * E[\Delta x^2]_{t-1} + (1 - \rho) * \Delta x_t^2$$
  $$x_t := x_{t-1} + \Delta x_{t}

  Arguments:

      learning_rate: float hyperparameter >= 0. Learning rate. It is recommended
        to leave it at the default value.
      rho: float hyperparameter >= 0. The decay rate.
      epsilon: float hyperparameter >= 0. Fuzz factor. A constant epsilon used
        to better condition the grad update.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'Adadelta'.
  """

  def __init__(self,
               learning_rate=0.001,
               rho=0.95,
               epsilon=1e-7,
               name='Adadelta',
               **kwargs):
    """Construct a new Adadelta optimizer.

    Adadelta is a more robust extension of Adagrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past gradients. This way, Adadelta continues learning even when many updates
    have been done. Compared to Adagrad, in the original version of Adadelta you
    don't have to set an initial learning rate. In this version, initial
    learning rate can be set, as in most other Keras optimizers.

    Args:
      learning_rate: A `Tensor` or a floating point value. The learning rate.
        To match the exact form in the original paper use 1.0.
      rho: A `Tensor` or a floating point value. The decay rate.
      epsilon: A `Tensor` or a floating point value.  A constant epsilon used
               to better conditioning the grad update.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adadelta".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate`, `rho`, and `epsilon` can
    each be a callable that takes no arguments and returns the actual value to
    use. This can be useful for changing these values across different
    invocations of optimizer functions.
    @end_compatibility
    """
    if epsilon is None:
      epsilon = backend_config.epsilon()
    super(Adadelta, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('rho', rho)
    self._set_hyper('epsilon', epsilon)

  def _create_slots(self, var_list):
    # Separate for-loops to respect the ordering of slot variables from v1.
    for v in var_list:
      self.add_slot(v, 'accum_grad')
    for v in var_list:
      state.zeros_slot(v, "accum")
      state.zeros_slot(v, "accum_update")

  def _apply_dense(self, grad, var, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.apply_adadelta(
        var,
        accum,
        accum_update,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.resource_apply_adadelta(
        var.handle,
        accum.handle,
        accum_update.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.sparse_apply_adadelta(
        var,
        accum,
        accum_update,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.resource_sparse_apply_adadelta(
        var.handle,
        accum.handle,
        accum_update.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad,
        indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(Adadelta, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "rho": self._serialize_hyperparameter("rho"),
        "epsilon": self._serialize_hyperparameter("epsilon")
    })
    return config
