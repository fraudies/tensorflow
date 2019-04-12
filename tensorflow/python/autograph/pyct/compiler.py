# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Converting AST to code and Python entities.

Adapted from Tangent.
"""

# TODO(mdan): Consolidate with parser and rename to parsing.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mdan): Use six for compatibility here.
import atexit
import imp
import os
import tempfile

import astor
import gast

from tensorflow.python.autograph.pyct import origin_info


def ast_to_source(node, indentation='  '):
  """Return the source code of given AST.

  Args:
    node: The code to compile, as an AST object.
    indentation: The string to use for indentation.

  Returns:
    code: The source code generated from the AST object
    source_mapping: A mapping between the user and AutoGraph generated code.
  """
  if not isinstance(node, (list, tuple)):
    node = (node,)
  generator = astor.code_gen.SourceGenerator(indentation, False,
                                             astor.string_repr.pretty_string)

  for n in node:
    if isinstance(n, gast.AST):
      n = gast.gast_to_ast(n)
    generator.visit(n)
    generator.result.append('\n')

  # In some versions of Python, literals may appear as actual values. This
  # ensures everything is string.
  code = ''.join(map(str, generator.result))

  # Strip leading blank lines.
  code_lines = code.split('\n')
  trimmed_code_lines = []
  for l in code_lines:
    if l.rstrip() or trimmed_code_lines:
      trimmed_code_lines.append(l)
  code = '\n'.join(trimmed_code_lines)

  return code


def source_to_entity(source, delete_on_exit):
  with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    module_name = os.path.basename(f.name[:-3])
    f.write(source)

  # TODO(mdan): Try flush() and delete=False instead.
  if delete_on_exit:
    atexit.register(lambda: os.remove(f.name))
  return imp.load_source(module_name, f.name), f.name


# TODO(mdan): Rename: ast_to_entity
def ast_to_object(nodes,
                  indentation='  ',
                  include_source_map=False,
                  delete_on_exit=True):
  """Return the Python objects represented by given AST.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    nodes: Union[ast.AST, Iterable[ast.AST]], the code to compile, as an AST
        object.
    indentation: Text, the string to use for indentation.
    include_source_map: bool, whether return a source map.
    delete_on_exit: bool, whether to delete the temporary file used for
        compilation on exit.

  Returns:
    Tuple[module, Text, Dict[LineLocation, OriginInfo]], containing:
    the module containing the unparsed nodes, the source code corresponding to
    nodes, and the source map. Is include_source_map is False, the source map
    will be None.
  """
  if not isinstance(nodes, (list, tuple)):
    nodes = (nodes,)

  source = ast_to_source(nodes, indentation=indentation)
  module, filename = source_to_entity(source, delete_on_exit)

  if include_source_map:
    source_map = origin_info.create_source_map(nodes, source, filename)
  else:
    source_map = None

  # TODO(mdan): Return a structured object.
  return module, source, source_map
