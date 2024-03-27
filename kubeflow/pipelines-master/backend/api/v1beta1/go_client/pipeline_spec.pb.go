// Copyright 2018 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.33.0
// 	protoc        v3.17.3
// source: backend/api/v1beta1/pipeline_spec.proto

package go_client

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	structpb "google.golang.org/protobuf/types/known/structpb"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type PipelineSpec struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Optional input field. The ID of the pipeline user uploaded before.
	PipelineId string `protobuf:"bytes,1,opt,name=pipeline_id,json=pipelineId,proto3" json:"pipeline_id,omitempty"`
	// Optional output field. The name of the pipeline.
	// Not empty if the pipeline id is not empty.
	PipelineName string `protobuf:"bytes,5,opt,name=pipeline_name,json=pipelineName,proto3" json:"pipeline_name,omitempty"`
	// Optional input field. The marshalled raw argo JSON workflow.
	// This will be deprecated when pipeline_manifest is in use.
	WorkflowManifest string `protobuf:"bytes,2,opt,name=workflow_manifest,json=workflowManifest,proto3" json:"workflow_manifest,omitempty"`
	// Optional input field. The raw pipeline JSON spec.
	PipelineManifest string `protobuf:"bytes,3,opt,name=pipeline_manifest,json=pipelineManifest,proto3" json:"pipeline_manifest,omitempty"`
	// The parameter user provide to inject to the pipeline JSON.
	// If a default value of a parameter exist in the JSON,
	// the value user provided here will replace. V1 only
	Parameters []*Parameter `protobuf:"bytes,4,rep,name=parameters,proto3" json:"parameters,omitempty"`
	// Runtime config of the pipeline. V2 only
	RuntimeConfig *PipelineSpec_RuntimeConfig `protobuf:"bytes,6,opt,name=runtime_config,json=runtimeConfig,proto3" json:"runtime_config,omitempty"`
}

func (x *PipelineSpec) Reset() {
	*x = PipelineSpec{}
	if protoimpl.UnsafeEnabled {
		mi := &file_backend_api_v1beta1_pipeline_spec_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *PipelineSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PipelineSpec) ProtoMessage() {}

func (x *PipelineSpec) ProtoReflect() protoreflect.Message {
	mi := &file_backend_api_v1beta1_pipeline_spec_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PipelineSpec.ProtoReflect.Descriptor instead.
func (*PipelineSpec) Descriptor() ([]byte, []int) {
	return file_backend_api_v1beta1_pipeline_spec_proto_rawDescGZIP(), []int{0}
}

func (x *PipelineSpec) GetPipelineId() string {
	if x != nil {
		return x.PipelineId
	}
	return ""
}

func (x *PipelineSpec) GetPipelineName() string {
	if x != nil {
		return x.PipelineName
	}
	return ""
}

func (x *PipelineSpec) GetWorkflowManifest() string {
	if x != nil {
		return x.WorkflowManifest
	}
	return ""
}

func (x *PipelineSpec) GetPipelineManifest() string {
	if x != nil {
		return x.PipelineManifest
	}
	return ""
}

func (x *PipelineSpec) GetParameters() []*Parameter {
	if x != nil {
		return x.Parameters
	}
	return nil
}

func (x *PipelineSpec) GetRuntimeConfig() *PipelineSpec_RuntimeConfig {
	if x != nil {
		return x.RuntimeConfig
	}
	return nil
}

// The runtime config of a PipelineSpec.
type PipelineSpec_RuntimeConfig struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The runtime parameters of the PipelineSpec. The parameters will be
	// used to replace the placeholders
	// at runtime.
	Parameters map[string]*structpb.Value `protobuf:"bytes,1,rep,name=parameters,proto3" json:"parameters,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	// A path in a object store bucket which will be treated as the root
	// output directory of the pipeline. It is used by the system to
	// generate the paths of output artifacts. Ref:(https://www.kubeflow.org/docs/components/pipelines/pipeline-root/)
	PipelineRoot string `protobuf:"bytes,2,opt,name=pipeline_root,json=pipelineRoot,proto3" json:"pipeline_root,omitempty"`
}

func (x *PipelineSpec_RuntimeConfig) Reset() {
	*x = PipelineSpec_RuntimeConfig{}
	if protoimpl.UnsafeEnabled {
		mi := &file_backend_api_v1beta1_pipeline_spec_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *PipelineSpec_RuntimeConfig) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PipelineSpec_RuntimeConfig) ProtoMessage() {}

func (x *PipelineSpec_RuntimeConfig) ProtoReflect() protoreflect.Message {
	mi := &file_backend_api_v1beta1_pipeline_spec_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PipelineSpec_RuntimeConfig.ProtoReflect.Descriptor instead.
func (*PipelineSpec_RuntimeConfig) Descriptor() ([]byte, []int) {
	return file_backend_api_v1beta1_pipeline_spec_proto_rawDescGZIP(), []int{0, 0}
}

func (x *PipelineSpec_RuntimeConfig) GetParameters() map[string]*structpb.Value {
	if x != nil {
		return x.Parameters
	}
	return nil
}

func (x *PipelineSpec_RuntimeConfig) GetPipelineRoot() string {
	if x != nil {
		return x.PipelineRoot
	}
	return ""
}

var File_backend_api_v1beta1_pipeline_spec_proto protoreflect.FileDescriptor

var file_backend_api_v1beta1_pipeline_spec_proto_rawDesc = []byte{
	0x0a, 0x27, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x31,
	0x62, 0x65, 0x74, 0x61, 0x31, 0x2f, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x5f, 0x73,
	0x70, 0x65, 0x63, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x03, 0x61, 0x70, 0x69, 0x1a, 0x23,
	0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x31, 0x62, 0x65,
	0x74, 0x61, 0x31, 0x2f, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x62, 0x75, 0x66, 0x2f, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x22, 0x85, 0x04, 0x0a, 0x0c, 0x50, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x53, 0x70,
	0x65, 0x63, 0x12, 0x1f, 0x0a, 0x0b, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x5f, 0x69,
	0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e,
	0x65, 0x49, 0x64, 0x12, 0x23, 0x0a, 0x0d, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x5f,
	0x6e, 0x61, 0x6d, 0x65, 0x18, 0x05, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x70, 0x69, 0x70, 0x65,
	0x6c, 0x69, 0x6e, 0x65, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x2b, 0x0a, 0x11, 0x77, 0x6f, 0x72, 0x6b,
	0x66, 0x6c, 0x6f, 0x77, 0x5f, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x10, 0x77, 0x6f, 0x72, 0x6b, 0x66, 0x6c, 0x6f, 0x77, 0x4d, 0x61, 0x6e,
	0x69, 0x66, 0x65, 0x73, 0x74, 0x12, 0x2b, 0x0a, 0x11, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e,
	0x65, 0x5f, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x10, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x4d, 0x61, 0x6e, 0x69, 0x66, 0x65,
	0x73, 0x74, 0x12, 0x2e, 0x0a, 0x0a, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73,
	0x18, 0x04, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x0e, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x50, 0x61, 0x72,
	0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x52, 0x0a, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65,
	0x72, 0x73, 0x12, 0x46, 0x0a, 0x0e, 0x72, 0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x5f, 0x63, 0x6f,
	0x6e, 0x66, 0x69, 0x67, 0x18, 0x06, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1f, 0x2e, 0x61, 0x70, 0x69,
	0x2e, 0x50, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x53, 0x70, 0x65, 0x63, 0x2e, 0x52, 0x75,
	0x6e, 0x74, 0x69, 0x6d, 0x65, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x52, 0x0d, 0x72, 0x75, 0x6e,
	0x74, 0x69, 0x6d, 0x65, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x1a, 0xdc, 0x01, 0x0a, 0x0d, 0x52,
	0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x12, 0x4f, 0x0a, 0x0a,
	0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b,
	0x32, 0x2f, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x50, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x53,
	0x70, 0x65, 0x63, 0x2e, 0x52, 0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x43, 0x6f, 0x6e, 0x66, 0x69,
	0x67, 0x2e, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73, 0x45, 0x6e, 0x74, 0x72,
	0x79, 0x52, 0x0a, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73, 0x12, 0x23, 0x0a,
	0x0d, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x5f, 0x72, 0x6f, 0x6f, 0x74, 0x18, 0x02,
	0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x52, 0x6f,
	0x6f, 0x74, 0x1a, 0x55, 0x0a, 0x0f, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73,
	0x45, 0x6e, 0x74, 0x72, 0x79, 0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x09, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x12, 0x2c, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x16, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x52, 0x05,
	0x76, 0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x42, 0x3d, 0x5a, 0x3b, 0x67, 0x69, 0x74,
	0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x6b, 0x75, 0x62, 0x65, 0x66, 0x6c, 0x6f, 0x77,
	0x2f, 0x70, 0x69, 0x70, 0x65, 0x6c, 0x69, 0x6e, 0x65, 0x73, 0x2f, 0x62, 0x61, 0x63, 0x6b, 0x65,
	0x6e, 0x64, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x31, 0x62, 0x65, 0x74, 0x61, 0x31, 0x2f, 0x67,
	0x6f, 0x5f, 0x63, 0x6c, 0x69, 0x65, 0x6e, 0x74, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_backend_api_v1beta1_pipeline_spec_proto_rawDescOnce sync.Once
	file_backend_api_v1beta1_pipeline_spec_proto_rawDescData = file_backend_api_v1beta1_pipeline_spec_proto_rawDesc
)

func file_backend_api_v1beta1_pipeline_spec_proto_rawDescGZIP() []byte {
	file_backend_api_v1beta1_pipeline_spec_proto_rawDescOnce.Do(func() {
		file_backend_api_v1beta1_pipeline_spec_proto_rawDescData = protoimpl.X.CompressGZIP(file_backend_api_v1beta1_pipeline_spec_proto_rawDescData)
	})
	return file_backend_api_v1beta1_pipeline_spec_proto_rawDescData
}

var file_backend_api_v1beta1_pipeline_spec_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_backend_api_v1beta1_pipeline_spec_proto_goTypes = []interface{}{
	(*PipelineSpec)(nil),               // 0: api.PipelineSpec
	(*PipelineSpec_RuntimeConfig)(nil), // 1: api.PipelineSpec.RuntimeConfig
	nil,                                // 2: api.PipelineSpec.RuntimeConfig.ParametersEntry
	(*Parameter)(nil),                  // 3: api.Parameter
	(*structpb.Value)(nil),             // 4: google.protobuf.Value
}
var file_backend_api_v1beta1_pipeline_spec_proto_depIdxs = []int32{
	3, // 0: api.PipelineSpec.parameters:type_name -> api.Parameter
	1, // 1: api.PipelineSpec.runtime_config:type_name -> api.PipelineSpec.RuntimeConfig
	2, // 2: api.PipelineSpec.RuntimeConfig.parameters:type_name -> api.PipelineSpec.RuntimeConfig.ParametersEntry
	4, // 3: api.PipelineSpec.RuntimeConfig.ParametersEntry.value:type_name -> google.protobuf.Value
	4, // [4:4] is the sub-list for method output_type
	4, // [4:4] is the sub-list for method input_type
	4, // [4:4] is the sub-list for extension type_name
	4, // [4:4] is the sub-list for extension extendee
	0, // [0:4] is the sub-list for field type_name
}

func init() { file_backend_api_v1beta1_pipeline_spec_proto_init() }
func file_backend_api_v1beta1_pipeline_spec_proto_init() {
	if File_backend_api_v1beta1_pipeline_spec_proto != nil {
		return
	}
	file_backend_api_v1beta1_parameter_proto_init()
	if !protoimpl.UnsafeEnabled {
		file_backend_api_v1beta1_pipeline_spec_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*PipelineSpec); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_backend_api_v1beta1_pipeline_spec_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*PipelineSpec_RuntimeConfig); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_backend_api_v1beta1_pipeline_spec_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_backend_api_v1beta1_pipeline_spec_proto_goTypes,
		DependencyIndexes: file_backend_api_v1beta1_pipeline_spec_proto_depIdxs,
		MessageInfos:      file_backend_api_v1beta1_pipeline_spec_proto_msgTypes,
	}.Build()
	File_backend_api_v1beta1_pipeline_spec_proto = out.File
	file_backend_api_v1beta1_pipeline_spec_proto_rawDesc = nil
	file_backend_api_v1beta1_pipeline_spec_proto_goTypes = nil
	file_backend_api_v1beta1_pipeline_spec_proto_depIdxs = nil
}