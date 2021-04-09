# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlos/Grpc/OptimizerMonitoringService.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import MlosCommonMessageTypes_pb2 as MlosCommonMessageTypes__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlos/Grpc/OptimizerMonitoringService.proto',
  package='mlos.optimizer_monitoring_service',
  syntax='proto3',
  serialized_options=b'\252\002\037Mlos.OptimizerMonitoringService',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n*mlos/Grpc/OptimizerMonitoringService.proto\x12!mlos.optimizer_monitoring_service\x1a\x1cMlosCommonMessageTypes.proto\"\x8d\x01\n\x19OptimizerConvergenceState\x12\x43\n\x0fOptimizerHandle\x18\x01 \x01(\x0b\x32*.mlos_common_message_types.OptimizerHandle\x12+\n#SerializedOptimizerConvergenceState\x18\x02 \x01(\t\"M\n\rOptimizerList\x12<\n\nOptimizers\x18\x01 \x03(\x0b\x32(.mlos_common_message_types.OptimizerInfo\"\x8c\x01\n\x0ePredictRequest\x12\x43\n\x0fOptimizerHandle\x18\x01 \x01(\x0b\x32*.mlos_common_message_types.OptimizerHandle\x12\x35\n\x08\x46\x65\x61tures\x18\x02 \x01(\x0b\x32#.mlos_common_message_types.Features\"Y\n\x19SingleObjectivePrediction\x12\x15\n\rObjectiveName\x18\x01 \x01(\t\x12%\n\x1dPredictionDataFrameJsonString\x18\x02 \x01(\t\"m\n\x0fPredictResponse\x12Z\n\x14ObjectivePredictions\x18\x01 \x03(\x0b\x32<.mlos.optimizer_monitoring_service.SingleObjectivePrediction2\xff\x06\n\x1aOptimizerMonitoringService\x12l\n\x16ListExistingOptimizers\x12 .mlos_common_message_types.Empty\x1a\x30.mlos.optimizer_monitoring_service.OptimizerList\x12h\n\x10GetOptimizerInfo\x12*.mlos_common_message_types.OptimizerHandle\x1a(.mlos_common_message_types.OptimizerInfo\x12\x88\x01\n\x1cGetOptimizerConvergenceState\x12*.mlos_common_message_types.OptimizerHandle\x1a<.mlos.optimizer_monitoring_service.OptimizerConvergenceState\x12r\n\x1b\x43omputeGoodnessOfFitMetrics\x12*.mlos_common_message_types.OptimizerHandle\x1a\'.mlos_common_message_types.SimpleString\x12\x61\n\tIsTrained\x12*.mlos_common_message_types.OptimizerHandle\x1a(.mlos_common_message_types.SimpleBoolean\x12p\n\x07Predict\x12\x31.mlos.optimizer_monitoring_service.PredictRequest\x1a\x32.mlos.optimizer_monitoring_service.PredictResponse\x12i\n\x12GetAllObservations\x12*.mlos_common_message_types.OptimizerHandle\x1a\'.mlos_common_message_types.Observations\x12J\n\x04\x45\x63ho\x12 .mlos_common_message_types.Empty\x1a .mlos_common_message_types.EmptyB\"\xaa\x02\x1fMlos.OptimizerMonitoringServiceb\x06proto3'
  ,
  dependencies=[MlosCommonMessageTypes__pb2.DESCRIPTOR,])




_OPTIMIZERCONVERGENCESTATE = _descriptor.Descriptor(
  name='OptimizerConvergenceState',
  full_name='mlos.optimizer_monitoring_service.OptimizerConvergenceState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='OptimizerHandle', full_name='mlos.optimizer_monitoring_service.OptimizerConvergenceState.OptimizerHandle', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='SerializedOptimizerConvergenceState', full_name='mlos.optimizer_monitoring_service.OptimizerConvergenceState.SerializedOptimizerConvergenceState', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=112,
  serialized_end=253,
)


_OPTIMIZERLIST = _descriptor.Descriptor(
  name='OptimizerList',
  full_name='mlos.optimizer_monitoring_service.OptimizerList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Optimizers', full_name='mlos.optimizer_monitoring_service.OptimizerList.Optimizers', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=255,
  serialized_end=332,
)


_PREDICTREQUEST = _descriptor.Descriptor(
  name='PredictRequest',
  full_name='mlos.optimizer_monitoring_service.PredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='OptimizerHandle', full_name='mlos.optimizer_monitoring_service.PredictRequest.OptimizerHandle', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Features', full_name='mlos.optimizer_monitoring_service.PredictRequest.Features', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=335,
  serialized_end=475,
)


_SINGLEOBJECTIVEPREDICTION = _descriptor.Descriptor(
  name='SingleObjectivePrediction',
  full_name='mlos.optimizer_monitoring_service.SingleObjectivePrediction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ObjectiveName', full_name='mlos.optimizer_monitoring_service.SingleObjectivePrediction.ObjectiveName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='PredictionDataFrameJsonString', full_name='mlos.optimizer_monitoring_service.SingleObjectivePrediction.PredictionDataFrameJsonString', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=477,
  serialized_end=566,
)


_PREDICTRESPONSE = _descriptor.Descriptor(
  name='PredictResponse',
  full_name='mlos.optimizer_monitoring_service.PredictResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ObjectivePredictions', full_name='mlos.optimizer_monitoring_service.PredictResponse.ObjectivePredictions', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=568,
  serialized_end=677,
)

_OPTIMIZERCONVERGENCESTATE.fields_by_name['OptimizerHandle'].message_type = MlosCommonMessageTypes__pb2._OPTIMIZERHANDLE
_OPTIMIZERLIST.fields_by_name['Optimizers'].message_type = MlosCommonMessageTypes__pb2._OPTIMIZERINFO
_PREDICTREQUEST.fields_by_name['OptimizerHandle'].message_type = MlosCommonMessageTypes__pb2._OPTIMIZERHANDLE
_PREDICTREQUEST.fields_by_name['Features'].message_type = MlosCommonMessageTypes__pb2._FEATURES
_PREDICTRESPONSE.fields_by_name['ObjectivePredictions'].message_type = _SINGLEOBJECTIVEPREDICTION
DESCRIPTOR.message_types_by_name['OptimizerConvergenceState'] = _OPTIMIZERCONVERGENCESTATE
DESCRIPTOR.message_types_by_name['OptimizerList'] = _OPTIMIZERLIST
DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['SingleObjectivePrediction'] = _SINGLEOBJECTIVEPREDICTION
DESCRIPTOR.message_types_by_name['PredictResponse'] = _PREDICTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OptimizerConvergenceState = _reflection.GeneratedProtocolMessageType('OptimizerConvergenceState', (_message.Message,), {
  'DESCRIPTOR' : _OPTIMIZERCONVERGENCESTATE,
  '__module__' : 'mlos.Grpc.OptimizerMonitoringService_pb2'
  # @@protoc_insertion_point(class_scope:mlos.optimizer_monitoring_service.OptimizerConvergenceState)
  })
_sym_db.RegisterMessage(OptimizerConvergenceState)

OptimizerList = _reflection.GeneratedProtocolMessageType('OptimizerList', (_message.Message,), {
  'DESCRIPTOR' : _OPTIMIZERLIST,
  '__module__' : 'mlos.Grpc.OptimizerMonitoringService_pb2'
  # @@protoc_insertion_point(class_scope:mlos.optimizer_monitoring_service.OptimizerList)
  })
_sym_db.RegisterMessage(OptimizerList)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTREQUEST,
  '__module__' : 'mlos.Grpc.OptimizerMonitoringService_pb2'
  # @@protoc_insertion_point(class_scope:mlos.optimizer_monitoring_service.PredictRequest)
  })
_sym_db.RegisterMessage(PredictRequest)

SingleObjectivePrediction = _reflection.GeneratedProtocolMessageType('SingleObjectivePrediction', (_message.Message,), {
  'DESCRIPTOR' : _SINGLEOBJECTIVEPREDICTION,
  '__module__' : 'mlos.Grpc.OptimizerMonitoringService_pb2'
  # @@protoc_insertion_point(class_scope:mlos.optimizer_monitoring_service.SingleObjectivePrediction)
  })
_sym_db.RegisterMessage(SingleObjectivePrediction)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTRESPONSE,
  '__module__' : 'mlos.Grpc.OptimizerMonitoringService_pb2'
  # @@protoc_insertion_point(class_scope:mlos.optimizer_monitoring_service.PredictResponse)
  })
_sym_db.RegisterMessage(PredictResponse)


DESCRIPTOR._options = None

_OPTIMIZERMONITORINGSERVICE = _descriptor.ServiceDescriptor(
  name='OptimizerMonitoringService',
  full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=680,
  serialized_end=1575,
  methods=[
  _descriptor.MethodDescriptor(
    name='ListExistingOptimizers',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.ListExistingOptimizers',
    index=0,
    containing_service=None,
    input_type=MlosCommonMessageTypes__pb2._EMPTY,
    output_type=_OPTIMIZERLIST,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetOptimizerInfo',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.GetOptimizerInfo',
    index=1,
    containing_service=None,
    input_type=MlosCommonMessageTypes__pb2._OPTIMIZERHANDLE,
    output_type=MlosCommonMessageTypes__pb2._OPTIMIZERINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetOptimizerConvergenceState',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.GetOptimizerConvergenceState',
    index=2,
    containing_service=None,
    input_type=MlosCommonMessageTypes__pb2._OPTIMIZERHANDLE,
    output_type=_OPTIMIZERCONVERGENCESTATE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ComputeGoodnessOfFitMetrics',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.ComputeGoodnessOfFitMetrics',
    index=3,
    containing_service=None,
    input_type=MlosCommonMessageTypes__pb2._OPTIMIZERHANDLE,
    output_type=MlosCommonMessageTypes__pb2._SIMPLESTRING,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='IsTrained',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.IsTrained',
    index=4,
    containing_service=None,
    input_type=MlosCommonMessageTypes__pb2._OPTIMIZERHANDLE,
    output_type=MlosCommonMessageTypes__pb2._SIMPLEBOOLEAN,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.Predict',
    index=5,
    containing_service=None,
    input_type=_PREDICTREQUEST,
    output_type=_PREDICTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetAllObservations',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.GetAllObservations',
    index=6,
    containing_service=None,
    input_type=MlosCommonMessageTypes__pb2._OPTIMIZERHANDLE,
    output_type=MlosCommonMessageTypes__pb2._OBSERVATIONS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Echo',
    full_name='mlos.optimizer_monitoring_service.OptimizerMonitoringService.Echo',
    index=7,
    containing_service=None,
    input_type=MlosCommonMessageTypes__pb2._EMPTY,
    output_type=MlosCommonMessageTypes__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_OPTIMIZERMONITORINGSERVICE)

DESCRIPTOR.services_by_name['OptimizerMonitoringService'] = _OPTIMIZERMONITORINGSERVICE

# @@protoc_insertion_point(module_scope)
