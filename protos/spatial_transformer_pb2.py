# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: aster/protos/spatial_transformer.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from aster.protos import convnet_pb2 as aster_dot_protos_dot_convnet__pb2
from aster.protos import hyperparams_pb2 as aster_dot_protos_dot_hyperparams__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='aster/protos/spatial_transformer.proto',
  package='aster.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n&aster/protos/spatial_transformer.proto\x12\x0c\x61ster.protos\x1a\x1a\x61ster/protos/convnet.proto\x1a\x1e\x61ster/protos/hyperparams.proto\"\x88\x03\n\x12SpatialTransformer\x12&\n\x07\x63onvnet\x18\x01 \x01(\x0b\x32\x15.aster.protos.Convnet\x12\x31\n\x0e\x66\x63_hyperparams\x18\x02 \x01(\x0b\x32\x19.aster.protos.Hyperparams\x12\x1a\n\x0elocalization_h\x18\x03 \x01(\x05:\x02\x36\x34\x12\x1b\n\x0elocalization_w\x18\x04 \x01(\x05:\x03\x31\x32\x38\x12\x14\n\x08output_h\x18\x05 \x01(\x05:\x02\x33\x32\x12\x15\n\x08output_w\x18\x06 \x01(\x05:\x03\x31\x30\x30\x12\x15\n\x08margin_x\x18\x07 \x01(\x02:\x03\x30.1\x12\x15\n\x08margin_y\x18\x08 \x01(\x02:\x03\x30.1\x12\x1e\n\x12num_control_points\x18\t \x01(\x05:\x02\x32\x30\x12#\n\x11init_bias_pattern\x18\n \x01(\t:\x08identity\x12\x18\n\nactivation\x18\x0b \x01(\t:\x04none\x12$\n\x15summarize_activations\x18\x0c \x01(\x08:\x05\x66\x61lse')
  ,
  dependencies=[aster_dot_protos_dot_convnet__pb2.DESCRIPTOR,aster_dot_protos_dot_hyperparams__pb2.DESCRIPTOR,])




_SPATIALTRANSFORMER = _descriptor.Descriptor(
  name='SpatialTransformer',
  full_name='aster.protos.SpatialTransformer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='convnet', full_name='aster.protos.SpatialTransformer.convnet', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fc_hyperparams', full_name='aster.protos.SpatialTransformer.fc_hyperparams', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='localization_h', full_name='aster.protos.SpatialTransformer.localization_h', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=64,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='localization_w', full_name='aster.protos.SpatialTransformer.localization_w', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=128,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_h', full_name='aster.protos.SpatialTransformer.output_h', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=32,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_w', full_name='aster.protos.SpatialTransformer.output_w', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='margin_x', full_name='aster.protos.SpatialTransformer.margin_x', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='margin_y', full_name='aster.protos.SpatialTransformer.margin_y', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_control_points', full_name='aster.protos.SpatialTransformer.num_control_points', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=20,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='init_bias_pattern', full_name='aster.protos.SpatialTransformer.init_bias_pattern', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("identity").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation', full_name='aster.protos.SpatialTransformer.activation', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("none").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='summarize_activations', full_name='aster.protos.SpatialTransformer.summarize_activations', index=11,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=509,
)

_SPATIALTRANSFORMER.fields_by_name['convnet'].message_type = aster_dot_protos_dot_convnet__pb2._CONVNET
_SPATIALTRANSFORMER.fields_by_name['fc_hyperparams'].message_type = aster_dot_protos_dot_hyperparams__pb2._HYPERPARAMS
DESCRIPTOR.message_types_by_name['SpatialTransformer'] = _SPATIALTRANSFORMER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SpatialTransformer = _reflection.GeneratedProtocolMessageType('SpatialTransformer', (_message.Message,), dict(
  DESCRIPTOR = _SPATIALTRANSFORMER,
  __module__ = 'aster.protos.spatial_transformer_pb2'
  # @@protoc_insertion_point(class_scope:aster.protos.SpatialTransformer)
  ))
_sym_db.RegisterMessage(SpatialTransformer)


# @@protoc_insertion_point(module_scope)
