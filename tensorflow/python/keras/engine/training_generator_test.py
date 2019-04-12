# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for training routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import unittest

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import training_generator
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest


def custom_generator(mode=2):
  batch_size = 10
  num_samples = 50
  arr_data = np.random.random((num_samples, 2))
  arr_labels = np.random.random((num_samples, 4))
  arr_weights = np.random.random((num_samples,))
  i = 0
  while True:
    batch_index = i * batch_size % num_samples
    i += 1
    start = batch_index
    end = start + batch_size
    x = arr_data[start: end]
    y = arr_labels[start: end]
    w = arr_weights[start: end]
    if mode == 1:
      yield x
    elif mode == 2:
      yield x, y
    else:
      yield x, y, w


class ForkRobustTestCase(keras_parameterized.TestCase):
  _sleep_at_end = False

  def setUp(self):
    # When setting up a test simply make a best effort to start from a clean
    # state.
    self._starting_remnants = data_utils.terminate_keras_multiprocessing_pools(
        use_sigkill=False)

    self._sleep_at_end = False
    super(ForkRobustTestCase, self).setUp()

  def tearDown(self):
    # Give multiprocessing pools some time to finish on their own before
    # cleanup_all_keras_forkpools yanks the rug out from under them. This is
    # particularly important because calling .close() on a pool that is already
    # in the process of spinning down can cause an uncatchable segmentation
    # fault at which point the tearDown will hang.
    if self._sleep_at_end:
      time.sleep(1)

    # If a test finishes and leaves behind uncleanable artifacts then that is a
    # failure condition. However, if the state was not clean to begin with the
    # test should not fail on that account.
    new_remnants = set(data_utils.terminate_keras_multiprocessing_pools(
        use_sigkill=True)).difference(self._starting_remnants)

    if new_remnants:
      raise ValueError('Test left behind stubborn orphans:\n  {}'.format(
          '\n  '.join(new_remnants)))
    super(ForkRobustTestCase, self).tearDown()


class TestGeneratorMethods(ForkRobustTestCase):

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_fit_generator_method(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        metrics=['mae', metrics_module.CategoricalAccuracy()])

    self._sleep_at_end = True
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        workers=4,
                        use_multiprocessing=True)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        validation_data=custom_generator(),
                        validation_steps=10)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        workers=0)

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_evaluate_generator_method(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        run_eagerly=testing_utils.should_run_eagerly())

    self._sleep_at_end = True
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             workers=2,
                             verbose=1,
                             use_multiprocessing=True)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False,
                             workers=0)

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_predict_generator_method(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.run_eagerly = testing_utils.should_run_eagerly()

    self._sleep_at_end = True
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            workers=2,
                            use_multiprocessing=True)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            workers=0)
    # Test generator with just inputs (no targets)
    model.predict_generator(custom_generator(mode=1),
                            steps=5,
                            max_queue_size=10,
                            workers=4,
                            use_multiprocessing=True)
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=False)
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=False,
                            validation_data=custom_generator(),
                            validation_steps=10)
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            validation_data=custom_generator(),
                            validation_steps=1,
                            workers=0)
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                workers=2,
                                use_multiprocessing=True)
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                use_multiprocessing=False)
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                workers=0)
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 workers=2,
                                 verbose=1,
                                 use_multiprocessing=True)
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 use_multiprocessing=False)
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 use_multiprocessing=False,
                                 workers=0)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_generator_methods_with_sample_weights(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        run_eagerly=testing_utils.should_run_eagerly())

    model.fit_generator(custom_generator(mode=3),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)
    model.fit_generator(custom_generator(mode=3),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        validation_data=custom_generator(mode=3),
                        validation_steps=10)
    model.predict_generator(custom_generator(mode=3),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.evaluate_generator(custom_generator(mode=3),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_generator_methods_invalid_use_case(self):

    def invalid_generator():
      while 1:
        yield 0

    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(loss='mse', optimizer=rmsprop.RMSprop(1e-3),
                  run_eagerly=testing_utils.should_run_eagerly())

      model.fit_generator(custom_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False)
      model.fit_generator(custom_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False,
                          validation_data=custom_generator(),
                          validation_steps=10)
      model.predict_generator(custom_generator(),
                              steps=5,
                              max_queue_size=10,
                              use_multiprocessing=False)
      model.evaluate_generator(custom_generator(),
                               steps=5,
                               max_queue_size=10,
                               use_multiprocessing=False)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_generator_input_to_fit_eval_predict(self):
    val_data = np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    def ones_generator():
      while True:
        yield np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    model = testing_utils.get_small_mlp(
        num_hidden=10, num_classes=1, input_dim=10)

    model.compile(rmsprop.RMSprop(0.001), 'binary_crossentropy',
                  run_eagerly=testing_utils.should_run_eagerly())
    model.fit(
        ones_generator(),
        steps_per_epoch=2,
        validation_data=val_data,
        epochs=2)
    model.evaluate(ones_generator(), steps=2)
    model.predict(ones_generator(), steps=2)

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(1, input_shape=(2,)))
      model.compile(loss='mse', optimizer='sgd')

class TestGeneratorMethodsWithSequences(ForkRobustTestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_training_with_sequences(self):

    class DummySequence(keras.utils.Sequence):

      def __getitem__(self, idx):
        return np.zeros([10, 2]), np.ones([10])

      def __len__(self):
        return 10

    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(loss='mse', optimizer=rmsprop.RMSprop(1e-3))

    model.fit_generator(DummySequence(),
                        steps_per_epoch=10,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=True)
    model.fit_generator(DummySequence(),
                        steps_per_epoch=10,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=False)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_sequence_input_to_fit_eval_predict(self):
    val_data = np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    class CustomSequence(keras.utils.Sequence):

      def __getitem__(self, idx):
        return np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

      def __len__(self):
        return 2

    inputs = keras.layers.Input(shape=(10,))
    x = keras.layers.Dense(10, activation='relu')(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(rmsprop.RMSprop(0.001), 'binary_crossentropy')
    model.fit(CustomSequence(), validation_data=val_data, epochs=2)
    model.evaluate(CustomSequence())
    model.predict(CustomSequence())

    with self.assertRaisesRegexp(ValueError, '`y` argument is not supported'):
      model.fit(CustomSequence(), y=np.ones([10, 1]))

    with self.assertRaisesRegexp(ValueError,
                                 '`sample_weight` argument is not supported'):
      model.fit(CustomSequence(), sample_weight=np.ones([10, 1]))


if __name__ == '__main__':
  test.main()
