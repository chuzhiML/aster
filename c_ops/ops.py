import os
import shutil
import uuid
from os.path import join, dirname, realpath, exists
import tensorflow as tf

OPLIB_NAME = 'aster'
OPLIB_SUFFIX = '.so'



def _load_oplib(lib_name):
  """
  Load TensorFlow operator library.
  """
  lib_path = join(dirname(realpath(__file__)), 'lib{0}{1}'.format(lib_name, OPLIB_SUFFIX))
  assert exists(lib_path), '{0} not found'.format(lib_path)

  # duplicate library with a random new name so that
  # a running program will not be interrupted when the lib file is updated
  lib_copy_path = '/tmp/lib{0}_{1}{2}'.format(lib_name, str(uuid.uuid4())[:8], OPLIB_SUFFIX)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib

_oplib = _load_oplib(OPLIB_NAME)

# map C++ operators to python objects
string_filtering = _oplib.string_filtering
string_reverse = _oplib.string_reverse
divide_curve = _oplib.divide_curve
