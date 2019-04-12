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
"""Assemble common TF Dockerfiles from many parts.

- Assembles Dockerfiles
- Builds images (and optionally runs image tests)
- Pushes images to Docker Hub (provided with credentials)

Logs are written to stderr; the list of successfully built images is
written to stdout.

Read README.md (in this directory) for instructions!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import errno
import os
import platform
import re
import shutil
import textwrap

from absl import app
from absl import flags
import cerberus
import yaml

FLAGS = flags.FLAGS

flags.DEFINE_string('hub_username', None,
                    'Dockerhub username, only used with --upload_to_hub')

flags.DEFINE_string(
    'hub_password', None,
    ('Dockerhub password, only used with --upload_to_hub. Use from an env param'
     ' so your password isn\'t in your history.'))

flags.DEFINE_integer('hub_timeout', 3600,
                     'Abort Hub upload if it takes longer than this.')

flags.DEFINE_string(
    'repository', 'tensorflow',
    'Tag local images as {repository}:tag (in addition to the '
    'hub_repository, if uploading to hub)')

flags.DEFINE_string(
    'hub_repository', None,
    'Push tags to this Docker Hub repository, e.g. tensorflow/tensorflow')

flags.DEFINE_boolean(
    'upload_to_hub',
    False,
    ('Push built images to Docker Hub (you must also provide --hub_username, '
     '--hub_password, and --hub_repository)'),
    short_name='u',
)

flags.DEFINE_boolean(
    'construct_dockerfiles', False, 'Do not build images', short_name='d')

flags.DEFINE_boolean(
    'keep_temp_dockerfiles',
    False,
    'Retain .temp.Dockerfiles created while building images.',
    short_name='k')

flags.DEFINE_boolean(
    'dry_run', False, 'Do not actually generate Dockerfiles', short_name='n')

flags.DEFINE_string(
    'spec_file',
    './spec.yml',
    'Path to a YAML specification file',
    short_name='s')

flags.DEFINE_string(
    'output_dir',
    './dockerfiles', ('Path to an output directory for Dockerfiles. '
                      'Will be created if it doesn\'t exist.'),
    short_name='o')

flags.DEFINE_string(
    'partial_dir',
    './partials',
    'Path to a directory containing foo.partial.Dockerfile partial files.',
    short_name='p')

flags.DEFINE_boolean(
    'quiet_dry_run',
    True,
    'Do not print contents of dry run Dockerfiles.',
    short_name='q')

flags.DEFINE_boolean(
    'nocache', False,
    'Disable the Docker build cache; identical to "docker build --no-cache"')

flags.DEFINE_string(
    'spec_file',
    './spec.yml',
    'Path to the YAML specification file',
    short_name='s')

# Schema to verify the contents of spec.yml with Cerberus.
# Must be converted to a dict from yaml to work.
# Note: can add python references with e.g.
# !!python/name:builtins.str
# !!python/name:__main__.funcname
SCHEMA_TEXT = """
header:
  type: string

partials:
  type: dict
  keyschema:
    type: string
  valueschema:
    type: dict
    schema:
      desc:
        type: string
      args:
        type: dict
        keyschema:
          type: string
        valueschema:
          anyof:
            - type: [ boolean, number, string ]
            - type: dict
              schema:
                 default:
                    type: [ boolean, number, string ]
                 desc:
                    type: string
                 options:
                    type: list
                    schema:
                       type: string

images:
  keyschema:
    type: string
  valueschema:
    type: dict
    schema:
      desc:
        type: string
      arg-defaults:
        type: list
        schema:
          anyof:
            - type: dict
              keyschema:
                type: string
                arg_in_use: true
              valueschema:
                type: string
            - type: string
              isimage: true
      create-dockerfile:
        type: boolean
      partials:
        type: list
        schema:
          anyof:
            - type: dict
              keyschema:
                type: string
                regex: image
              valueschema:
                type: string
                isimage: true
            - type: string
              ispartial: true
"""


class TfDockerValidator(cerberus.Validator):
  """Custom Cerberus validator for TF dockerfile spec.

  Note: Each _validate_foo function's docstring must end with a segment
  describing its own validation schema, e.g. "The rule's arguments are...". If
  you add a new validator, you can copy/paste that section.
  """

  def _validate_ispartial(self, ispartial, field, value):
    """Validate that a partial references an existing partial spec.

    Args:
      ispartial: Value of the rule, a bool
      field: The field being validated
      value: The field's value

    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if ispartial and value not in self.root_document.get('partials', dict()):
      self._error(field, '{} is not an existing partial.'.format(value))

  def _validate_isimage(self, isimage, field, value):
    """Validate that an image references an existing partial spec.

    Args:
      isimage: Value of the rule, a bool
      field: The field being validated
      value: The field's value

    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if isimage and value not in self.root_document.get('images', dict()):
      self._error(field, '{} is not an existing image.'.format(value))

  def _validate_arg_in_use(self, arg_in_use, field, value):
    """Validate that an arg references an existing partial spec's args.

    Args:
      arg_in_use: Value of the rule, a bool
      field: The field being validated
      value: The field's value

    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if arg_in_use:
      for partial in self.root_document.get('partials', dict()).values():
        if value in partial.get('args', tuple()):
          return

      self._error(field, '{} is not an arg used in any partial.'.format(value))


def build_partial_description(partial_spec):
  """Create the documentation lines for a specific partial.

  Generates something like this:

    # This is the partial's description, from spec.yml.
    # --build-arg ARG_NAME=argdefault
    #    this is one of the args.
    # --build-arg ANOTHER_ARG=(some|choices)
    #    another arg.

  Args:
    partial_spec: A dict representing one of the partials from spec.yml. Doesn't
      include the name of the partial; is a dict like { desc: ..., args: ... }.

  Returns:
    A commented string describing this partial.
  """

  # Start from linewrapped desc field
  lines = []
  wrapper = textwrap.TextWrapper(
      initial_indent='# ', subsequent_indent='# ', width=80)
  description = wrapper.fill(partial_spec.get('desc', '( no comments )'))
  lines.extend(['#', description])

  # Document each arg
  for arg, arg_data in partial_spec.get('args', dict()).items():
    # Wrap arg description with comment lines
    desc = arg_data.get('desc', '( no description )')
    desc = textwrap.fill(
        desc,
        initial_indent='#    ',
        subsequent_indent='#    ',
        width=80,
        drop_whitespace=False)

    # Document (each|option|like|this)
    if 'options' in arg_data:
      arg_options = ' ({})'.format('|'.join(arg_data['options']))
    else:
      arg_options = ''

    # Add usage sample
    arg_use = '# --build-arg {}={}{}'.format(arg,
                                             arg_data.get('default', '(unset)'),
                                             arg_options)
    lines.extend([arg_use, desc])

  return '\n'.join(lines)


def construct_contents(partial_specs, image_spec):
  """Assemble the dockerfile contents for an image spec.

  It assembles a concrete list of partial references into a single, large
  string.
  Also expands argument defaults, so that the resulting Dockerfile doesn't have
  to be configured with --build-arg=... every time. That is, any ARG directive
  will be updated with a new default value.

  Args:
    partial_specs: The dict from spec.yml["partials"].
    image_spec: One of the dict values from spec.yml["images"].

  Returns:
    A string containing a valid Dockerfile based on the partials listed in
    image_spec.
  """
  processed_partial_strings = []
  for partial_name in image_spec['partials']:
    # Apply image arg-defaults to existing arg defaults
    partial_spec = copy.deepcopy(partial_specs[partial_name])
    args = partial_spec.get('args', dict())
    for k_v in image_spec.get('arg-defaults', []):
      arg, value = list(k_v.items())[0]
      if arg in args:
        args[arg]['default'] = value

    # Read partial file contents
    filename = partial_spec.get('file', partial_name)
    partial_path = os.path.join(FLAGS.partial_dir,
                                '{}.partial.Dockerfile'.format(filename))
    with open(partial_path, 'r') as f_partial:
      partial_contents = f_partial.read()

    # Replace ARG FOO=BAR with ARG FOO=[new-default]
    for arg, arg_data in args.items():
      if 'default' in arg_data and arg_data['default']:
        default = '={}'.format(arg_data['default'])
      else:
        default = ''
      partial_contents = re.sub(r'ARG {}.*'.format(arg), 'ARG {}{}'.format(
          arg, default), partial_contents)

    # Store updated partial contents
    processed_partial_strings.append(partial_contents)

  # Join everything together
  return '\n'.join(processed_partial_strings)


def mkdir_p(path):
  """Create a directory and its parents, even if it already exists."""
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

def gather_tag_args(slices, cli_input_args, required_args):
  """Build a dictionary of all the CLI and slice-specified args for a tag."""
  args = {}

def construct_documentation(header, partial_specs, image_spec):
  """Assemble all of the documentation for a single dockerfile.

  Builds explanations of included partials and available build args.

  Args:
    header: The string from spec.yml["header"]; will be commented and wrapped.
    partial_specs: The dict from spec.yml["partials"].
    image_spec: The spec for the dockerfile being built.

  Returns:
    A string containing a commented header that documents the contents of the
    dockerfile.

  """
  # Comment and wrap header and image description
  commented_header = '\n'.join(
      [('# ' + l).rstrip() for l in header.splitlines()])
  commented_desc = '\n'.join(
      ['# ' + l for l in image_spec.get('desc', '').splitlines()])
  partial_descriptions = []

  # Build documentation for each partial in the image
  for partial in image_spec['partials']:
    # Copy partial data for default args unique to this image
    partial_spec = copy.deepcopy(partial_specs[partial])
    args = partial_spec.get('args', dict())

    # Overwrite any existing arg defaults
    for k_v in image_spec.get('arg-defaults', []):
      arg, value = list(k_v.items())[0]
      if arg in args:
        args[arg]['default'] = value

    # Build the description from new args
    partial_description = build_partial_description(partial_spec)
    partial_descriptions.append(partial_description)

  contents = [commented_header, '#', commented_desc] + partial_descriptions
  return '\n'.join(contents) + '\n'


def normalize_partial_args(partial_specs):
  """Normalize the shorthand form of a partial's args specification.

  Turns this:

    partial:
      args:
        SOME_ARG: arg_value

  Into this:

    partial:
       args:
         SOME_ARG:
            default: arg_value

  Args:
    partial_specs: The dict from spec.yml["partials"]. This dict is modified in
      place.

  Returns:
    The modified contents of partial_specs.

  """
  for _, partial in partial_specs.items():
    args = partial.get('args', dict())
    for arg, value in args.items():
      if not isinstance(value, dict):
        new_value = {'default': value}
        args[arg] = new_value

  return partial_specs


def flatten_args_references(image_specs):
  """Resolve all default-args in each image spec to a concrete dict.

  Turns this:

    example-image:
      arg-defaults:
        - MY_ARG: ARG_VALUE

    another-example:
      arg-defaults:
        - ANOTHER_ARG: ANOTHER_VALUE
        - example_image

  Into this:

    example-image:
      arg-defaults:
        - MY_ARG: ARG_VALUE

    another-example:
      arg-defaults:
        - ANOTHER_ARG: ANOTHER_VALUE
        - MY_ARG: ARG_VALUE

  Args:
    image_specs: A dict of image_spec dicts; should be the contents of the
      "images" key in the global spec.yaml. This dict is modified in place and
      then returned.

  Returns:
    The modified contents of image_specs.
  """
  partials = {}
  for path, _, files in os.walk(partial_path):
    for name in files:
      fullpath = os.path.join(path, name)
      if '.partial.Dockerfile' not in fullpath:
        eprint(('> Probably not a problem: skipping {}, which is not a '
                'partial.').format(fullpath))
        continue
      # partial_dir/foo/bar.partial.Dockerfile -> foo/bar
      simple_name = fullpath[len(partial_path) + 1:-len('.partial.dockerfile')]
      with open(fullpath, 'r') as f:
        partial_contents = f.read()
      partials[simple_name] = partial_contents
  return partials


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unexpected command line args found: {}'.format(argv))

  with open(FLAGS.spec_file, 'r') as spec_file:
    tf_spec = yaml.load(spec_file)

  # Abort if spec.yaml is invalid
  if FLAGS.validate:
    schema = yaml.load(SCHEMA_TEXT)
    v = TfDockerValidator(schema)
    if not v.validate(tf_spec):
      print('>> ERROR: {} is an invalid spec! The errors are:'.format(
          FLAGS.spec_file))
      print(yaml.dump(v.errors, indent=2))
      exit(1)
    if not FLAGS.hub_repository:
      eprint(
          '> Error: please set --hub_repository when uploading to Dockerhub.')
      exit(1)
    if not FLAGS.hub_password:
      eprint('> Error: please set --hub_password when uploading to Dockerhub.')
      exit(1)
    dock.login(
        username=FLAGS.hub_username,
        password=FLAGS.hub_password,
    )

  # Each tag has a name ('tag') and a definition consisting of the contents
  # of its Dockerfile, its build arg list, etc.
  failed_tags = []
  succeeded_tags = []
  for tag, tag_defs in all_tags.items():
    for tag_def in tag_defs:
      eprint('> Working on {}'.format(tag))

      if FLAGS.exclude_tags_matching and re.match(FLAGS.exclude_tags_matching,
                                                  tag):
        eprint('>> Excluded due to match against "{}".'.format(
            FLAGS.exclude_tags_matching))
        continue

      if FLAGS.only_tags_matching and not re.match(FLAGS.only_tags_matching,
                                                   tag):
        eprint('>> Excluded due to failure to match against "{}".'.format(
            FLAGS.only_tags_matching))
        continue

      # Write releases marked "is_dockerfiles" into the Dockerfile directory
      if FLAGS.construct_dockerfiles and tag_def['is_dockerfiles']:
        path = os.path.join(FLAGS.dockerfile_dir,
                            tag_def['dockerfile_subdirectory'],
                            tag + '.Dockerfile')
        eprint('>> Writing {}...'.format(path))
        if not FLAGS.dry_run:
          mkdir_p(os.path.dirname(path))
          with open(path, 'w') as f:
            f.write(tag_def['dockerfile_contents'])

      # Don't build any images for dockerfile-only releases
      if not FLAGS.build_images:
        continue

      # Only build images for host architecture
      proc_arch = platform.processor()
      is_x86 = proc_arch.startswith('x86')
      if (is_x86 and any([arch in tag for arch in ['ppc64le']]) or
          not is_x86 and proc_arch not in tag):
        continue

      # Generate a temporary Dockerfile to use to build, since docker-py
      # needs a filepath relative to the build context (i.e. the current
      # directory)
      dockerfile = os.path.join(FLAGS.dockerfile_dir, tag + '.temp.Dockerfile')
      if not FLAGS.dry_run:
        with open(dockerfile, 'w') as f:
          f.write(tag_def['dockerfile_contents'])
      eprint('>> (Temporary) writing {}...'.format(dockerfile))

      repo_tag = '{}:{}'.format(FLAGS.repository, tag)
      eprint('>> Building {} using build args:'.format(repo_tag))
      for arg, value in tag_def['cli_args'].items():
        eprint('>>> {}={}'.format(arg, value))

      # Note that we are NOT using cache_from, which appears to limit
      # available cache layers to those from explicitly specified layers. Many
      # of our layers are similar between local builds, so we want to use the
      # implied local build cache.
      tag_failed = False
      image, logs = None, []
      if not FLAGS.dry_run:
        try:
          image, logs = dock.images.build(
              timeout=FLAGS.hub_timeout,
              path='.',
              nocache=FLAGS.nocache,
              dockerfile=dockerfile,
              buildargs=tag_def['cli_args'],
              tag=repo_tag)

          # Print logs after finishing
          log_lines = [l.get('stream', '') for l in logs]
          eprint(''.join(log_lines))

          # Run tests if requested, and dump output
          # Could be improved by backgrounding, but would need better
          # multiprocessing support to track failures properly.
          if FLAGS.run_tests_path:
            if not tag_def['tests']:
              eprint('>>> No tests to run.')
            for test in tag_def['tests']:
              eprint('>> Testing {}...'.format(test))
              container, = dock.containers.run(
                  image,
                  '/tests/' + test,
                  working_dir='/',
                  log_config={'type': 'journald'},
                  detach=True,
                  stderr=True,
                  stdout=True,
                  volumes={
                      FLAGS.run_tests_path: {
                          'bind': '/tests',
                          'mode': 'ro'
                      }
                  },
                  runtime=tag_def['test_runtime']),
              ret = container.wait()
              code = ret['StatusCode']
              out = container.logs(stdout=True, stderr=False)
              err = container.logs(stdout=False, stderr=True)
              container.remove()
              if out:
                eprint('>>> Output stdout:')
                eprint(out.decode('utf-8'))
              else:
                eprint('>>> No test standard out.')
              if err:
                eprint('>>> Output stderr:')
                eprint(out.decode('utf-8'))
              else:
                eprint('>>> No test standard err.')
              if code != 0:
                eprint('>> {} failed tests with status: "{}"'.format(
                    repo_tag, code))
                failed_tags.append(tag)
                tag_failed = True
                if FLAGS.stop_on_failure:
                  eprint('>> ABORTING due to --stop_on_failure!')
                  exit(1)
              else:
                eprint('>> Tests look good!')

        except docker.errors.BuildError as e:
          eprint('>> {} failed to build with message: "{}"'.format(
              repo_tag, e.msg))
          eprint('>> Build logs follow:')
          log_lines = [l.get('stream', '') for l in e.build_log]
          eprint(''.join(log_lines))
          failed_tags.append(tag)
          tag_failed = True
          if FLAGS.stop_on_failure:
            eprint('>> ABORTING due to --stop_on_failure!')
            exit(1)

        # Clean temporary dockerfiles if they were created earlier
        if not FLAGS.keep_temp_dockerfiles:
          os.remove(dockerfile)

      # Upload new images to DockerHub as long as they built + passed tests
      if FLAGS.upload_to_hub:
        if not tag_def['upload_images']:
          continue
        if tag_failed:
          continue

        eprint('>> Uploading to {}:{}'.format(FLAGS.hub_repository, tag))
        if not FLAGS.dry_run:
          p = multiprocessing.Process(
              target=upload_in_background,
              args=(FLAGS.hub_repository, dock, image, tag))
          p.start()

      if not tag_failed:
        succeeded_tags.append(tag)

  if failed_tags:
    eprint(
        '> Some tags failed to build or failed testing, check scrollback for '
        'errors: {}'.format(','.join(failed_tags)))
    exit(1)

  eprint('> Writing built{} tags to standard out.'.format(
      ' and tested' if FLAGS.run_tests_path else ''))
  for tag in succeeded_tags:
    print('{}:{}'.format(FLAGS.repository, tag))


if __name__ == '__main__':
  app.run(main)
