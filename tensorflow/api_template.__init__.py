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
"""Bring in all of the public TensorFlow interface into this module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import distutils as _distutils
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys

# pylint: disable=g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

try:
  # Add `estimator` attribute to allow access to estimator APIs via
  # "tf.estimator..."
  from tensorflow.python.estimator.api import estimator  # pylint: disable=g-import-not-at-top

  # Add `estimator` to the __path__ to allow "from tensorflow.estimator..."
  # style imports.
  from tensorflow.python.estimator import api as estimator_api  # pylint: disable=g-import-not-at-top
  __path__ += [_os.path.dirname(estimator_api.__file__)]
  del estimator_api
except (ImportError, AttributeError):
  print('tf.estimator package not installed.')

# API IMPORTS PLACEHOLDER

# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
# We're using bitwise, but there's nothing special about that.
_API_MODULE = bitwise  # pylint: disable=undefined-variable
_current_module = _sys.modules[__name__]
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# pylint: disable=g-bad-import-order
from tensorflow.python.tools import component_api_helper as _component_api_helper
_component_api_helper.package_hook(
    parent_package_str=__name__,
    child_package_str=('tensorboard.summary._tf.summary'),
    error_msg="Limited tf.summary API due to missing TensorBoard installation")
_component_api_helper.package_hook(
    parent_package_str=__name__,
    child_package_str=(
        'tensorflow_estimator.python.estimator.api._v2.estimator'))

if not hasattr(_current_module, 'estimator'):
  _component_api_helper.package_hook(
      parent_package_str=__name__,
      child_package_str=(
          'tensorflow_estimator.python.estimator.api.estimator'))
_component_api_helper.package_hook(
    parent_package_str=__name__,
    child_package_str=('tensorflow.python.keras.api._v2.keras'))

# Enable TF2 behaviors
from tensorflow.python.compat import v2_compat as _compat  # pylint: disable=g-import-not-at-top
_compat.enable_v2_behavior()


# Load all plugin libraries from site-packages/tensorflow-plugins if we are
# running under pip.
# TODO(gunan): Enable setting an environment variable to define arbitrary plugin
# directories.
# TODO(gunan): Find a better location for this code snippet.
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi

# Get sitepackages directories for the python installation.
_site_packages_dirs = []
_site_packages_dirs += [_site.USER_SITE]
_site_packages_dirs += [_p for _p in _sys.path if 'site-packages' in _p]
if 'getsitepackages' in dir(_site):
  _site_packages_dirs += _site.getsitepackages()

if 'sysconfig' in dir(_distutils):
  _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]

_site_packages_dirs = list(set(_site_packages_dirs))

# Find the location of this exact file.
_current_file_location = _inspect.getfile(_inspect.currentframe())

def _running_from_pip_package():
  return any(
      _current_file_location.startswith(dir_) for dir_ in _site_packages_dirs)

if _running_from_pip_package():
  for s in _site_packages_dirs:
    # TODO(gunan): Add sanity checks to loaded modules here.
    plugin_dir = _os.path.join(s, 'tensorflow-plugins')
    if _fi.file_exists(plugin_dir):
      _ll.load_library(plugin_dir)

# These symbols appear because we import the python package which
# in turn imports from tensorflow.core and tensorflow.python. They
# must come from this module. So python adds these symbols for the
# resolution to succeed.
# pylint: disable=undefined-variable
try:
  del python
  del core
except NameError:
  # Don't fail if these modules are not available.
  # For e.g. we are using this file for compat.v1 module as well and
  # 'python', 'core' directories are not under compat/v1.
  pass

# Add module aliases
if hasattr(_current_module, 'keras'):
  losses = keras.losses
  metrics = keras.metrics
  optimizers = keras.optimizers
  initializers = keras.initializers

# pylint: enable=undefined-variable
