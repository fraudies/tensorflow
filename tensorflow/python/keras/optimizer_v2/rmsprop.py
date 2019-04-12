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
"""RMSprop optimizer for Tensorflow.

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export

A detailed description of rmsprop.

@keras_export("keras.optimizers.RMSprop")
class RMSprop(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the RMSprop algorithm.

mean_square = rho * mean_square{t-1} + (1-rho) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t / sqrt(mean_square)
delta = - mom

This implementation of RMSProp uses plain momentum, not Nesterov momentum.

  $$mean_square_t = rho * mean_square{t-1} + (1-rho) * gradient ** 2$$
  $$mom_t = momentum * mom_{t-1} + learning_rate * gradient / \sqrt{ /
      mean_square_t + \epsilon}$$
  $$variable_t := variable_{t-1} - mom_t$$

mean_grad = rho * mean_square{t-1} + (1-rho) * gradient
mean_square = rho * mean_square{t-1} + (1-rho) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t /
    sqrt(mean_square - mean_grad**2)
delta = - mom
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops

  $$mean_grad_t = rho * mean_grad_{t-1} + (1-rho) * gradient$$
  $$mean_square_t = rho * mean_square_{t-1} + (1-rho) * gradient ** 2$$
  $$mom_t = momentum * mom_{t-1} + learning_rate * gradient /
      sqrt(mean_square_t - mean_grad_t**2 + epsilon)$$
  $$variable_t := variable_{t-1} - mom_t$$


class RMSProp(optimizer_v2.OptimizerV2):
  """RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer at their default
  values (except the learning rate, which can be freely tuned).

  This optimizer is usually a good choice for recurrent neural networks.

  Some of the args below are hyperparameters, where a hyperparameter is
  defined as a scalar Tensor, a regular Python value, or a callable (which
  will be evaluated when `apply_gradients` is called) returning a scalar
  Tensor or a Python value.

  Note that in the dense implementation of this algorithm, variables and their
  corresponding accumulators (momentum, gradient moving average, square
  gradient moving average) will be updated even if the gradient is zero
  (i.e. accumulators will decay, momentum will be applied). The sparse
  implementation (used when the gradient is an `IndexedSlices` object,
  typically because of `tf.gather` or an embedding lookup in the forward pass)
  will not update variable slices or their accumulators unless those slices
  were used in the forward pass (nor is there an "eventual" correction to
  account for these omitted updates). This leads to more efficient updates for
  large embedding lookup tables (where most of the slices are not accessed in
  a particular graph execution), but differs from the published algorithm.

  Arguments:
      learning_rate: A float hyperparameter >= 0. The learning rate.
      rho: A float hyperparameter >= 0. Discounting factor for the
        history/coming gradient.
      momentum: A float hyperparameter >= 0.
      epsilon: A float hyperparameter >= 0 . Small value to initialize the
        average square gradient variable and avoid zero denominator.
      centered: If True, gradients are normalized by the estimated variance of
        the gradient; if False, by the uncentered second moment. Setting this to
        True may help with training, but is slightly more expensive in terms of
        computation and memory. Defaults to False.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSprop".  @compatibility(eager) When eager
        execution is enabled, `learning_rate`, `decay`, `momentum`, and
        `epsilon` can each be a callable that takes no arguments and returns the
        actual value to use. This can be useful for changing these values across
        different invocations of optimizer functions. @end_compatibility
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    if epsilon is None:
      epsilon = backend_config.epsilon()
    super(RMSprop, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("rho", rho)

  def __init__(self,
               learning_rate=0.001,
               rho=0.9,
               momentum=None,
               epsilon=1e-10,
               centered=False,
               name="RMSProp"):
    super(RMSProp, self).__init__(name)
    # Momentum default is `None` for consistency with SGD
    # but underlying implementation uses `momentum` hyperparameter here
    # regardless unlike SGD. Since extneral Keras RMSProp does not have
    # a `momentum` weight, for compatibility with external Keras h5 files,
    # when  `momentum` was set as `None` we should ignore the `momentum`
    # variable in `get_weights` and not require it in `set_weights`.
    if momentum is None:
      momentum = 0.0
    self._set_hyper("learning_rate", learning_rate)
    self._set_hyper("rho", rho)
    self._set_hyper("momentum", momentum)
    self._set_hyper("epsilon", epsilon)
    self.centered = centered

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, "rms")
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")
    if self.centered:
      for var in var_list:
        self.add_slot(var, "mg")

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    rms = self.get_slot(var, "rms")
    rho = self._get_hyper("rho", var_dtype)
    momentum = self._get_hyper("momentum", var_dtype)
    epsilon = self._get_hyper("epsilon", var_dtype)
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      if self.centered:
        mg = self.get_slot(var, "mg")
        return training_ops.resource_apply_centered_rms_prop(
            var.handle,
            mg.handle,
            rms.handle,
            mom.handle,
            lr_t,
            rho,
            momentum,
            epsilon,
            grad,
            use_locking=self._use_locking)
      else:
        return training_ops.resource_apply_rms_prop(
            var.handle,
            rms.handle,
            mom.handle,
            lr_t,
            rho,
            momentum,
            epsilon,
            grad,
            use_locking=self._use_locking)
    else:
      rms_t = rho * rms + (1. - rho) * math_ops.square(grad)
      rms_t = state_ops.assign(rms, rms_t, use_locking=self._use_locking)
      denom_t = rms_t
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_t = rho * mg + (1. - rho) * grad
        mg_t = state_ops.assign(mg, mg_t, use_locking=self._use_locking)
        denom_t = rms_t - math_ops.square(mg_t)
      var_t = var - lr_t * grad / (math_ops.sqrt(denom_t) + epsilon)
      return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices):
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    rms = self.get_slot(var, "rms")
    rho = self._get_hyper("rho", var_dtype)
    momentum = self._get_hyper("momentum", var_dtype)
    epsilon = self._get_hyper("epsilon", var_dtype)
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      if self.centered:
        mg = self.get_slot(var, "mg")
        return training_ops.resource_sparse_apply_centered_rms_prop(
            var.handle,
            mg.handle,
            rms.handle,
            mom.handle,
            lr_t,
            rho,
            momentum,
            epsilon,
            grad,
            indices,
            use_locking=self._use_locking)
      else:
        return training_ops.resource_sparse_apply_rms_prop(
            var.handle,
            rms.handle,
            mom.handle,
            lr_t,
            rho,
            momentum,
            epsilon,
            grad,
            indices,
            use_locking=self._use_locking)
    else:
      rms_scaled_g_values = (grad * grad) * (1. - rho)
      rms_t = state_ops.assign(rms, rms * rho, use_locking=self._use_locking)
      with ops.control_dependencies([rms_t]):
        rms_t = self._resource_scatter_add(rms, indices, rms_scaled_g_values)
        rms_slice = array_ops.gather(rms_t, indices)
      denom_slice = rms_slice
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_scaled_g_values = grad * (1. - rho)
        mg_t = state_ops.assign(mg, mg * rho, use_locking=self._use_locking)
        with ops.control_dependencies([mg_t]):
          mg_t = self._resource_scatter_add(mg, indices, mg_scaled_g_values)
          mg_slice = array_ops.gather(mg_t, indices)
          denom_slice = rms_slice - math_ops.square(mg_slice)
      var_update = self._resource_scatter_add(
          var, indices, -lr_t * grad / (math_ops.sqrt(denom_slice) + epsilon))
      if self.centered:
        return control_flow_ops.group(*[var_update, rms_t, mg_t])
      return control_flow_ops.group(*[var_update, rms_t])

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(RMSprop, self).set_weights(weights)

  def get_config(self):
    config = super(RMSProp, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "rho": self._serialize_hyperparameter("rho"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "epsilon": self._serialize_hyperparameter("epsilon"),
        "centered": self._centered
    })
    return config
