#ifndef WGPU_H_
#define WGPU_H_

#include "webgpu/webgpu.h"

typedef enum WGPUNativeSType {
    // Start at 0003 since that's allocated range for wgpu-native
    WGPUSType_DeviceExtras = 0x00030001,
    WGPUSType_RequiredLimitsExtras = 0x00030002,
    WGPUSType_PipelineLayoutExtras = 0x00030003,
    WGPUSType_ShaderModuleGLSLDescriptor = 0x00030004,
    WGPUSType_SupportedLimitsExtras = 0x00030005,
    WGPUSType_InstanceExtras = 0x00030006,
    WGPUSType_BindGroupEntryExtras = 0x00030007,
    WGPUSType_BindGroupLayoutEntryExtras = 0x00030008,
    WGPUNativeSType_Force32 = 0x7FFFFFFF
} WGPUNativeSType;

typedef enum WGPUNativeFeature {
    WGPUNativeFeature_PushConstants = 0x00030001,
    WGPUNativeFeature_TextureAdapterSpecificFormatFeatures = 0x00030002,
    WGPUNativeFeature_MultiDrawIndirect = 0x00030003,
    WGPUNativeFeature_MultiDrawIndirectCount = 0x00030004,
    WGPUNativeFeature_VertexWritableStorage = 0x00030005,
    WGPUNativeFeature_TextureBindingArray = 0x00030006,
    WGPUNativeFeature_SampledTextureAndStorageBufferArrayNonUniformIndexing = 0x00030007,
    WGPUNativeFeature_Force32 = 0x7FFFFFFF
} WGPUNativeFeature;

typedef enum WGPULogLevel {
    WGPULogLevel_Off = 0x00000000,
    WGPULogLevel_Error = 0x00000001,
    WGPULogLevel_Warn = 0x00000002,
    WGPULogLevel_Info = 0x00000003,
    WGPULogLevel_Debug = 0x00000004,
    WGPULogLevel_Trace = 0x00000005,
    WGPULogLevel_Force32 = 0x7FFFFFFF
} WGPULogLevel;

typedef enum WGPUInstanceBackend {
    WGPUInstanceBackend_Vulkan = 1 << 1,
    WGPUInstanceBackend_GL = 1 << 5,
    WGPUInstanceBackend_Metal = 1 << 2,
    WGPUInstanceBackend_DX12 = 1 << 3,
    WGPUInstanceBackend_DX11 = 1 << 4,
    WGPUInstanceBackend_BrowserWebGPU = 1 << 6,
    WGPUInstanceBackend_Primary = WGPUInstanceBackend_Vulkan | WGPUInstanceBackend_Metal |
        WGPUInstanceBackend_DX12 |
        WGPUInstanceBackend_BrowserWebGPU,
    WGPUInstanceBackend_Secondary = WGPUInstanceBackend_GL | WGPUInstanceBackend_DX11,
    WGPUInstanceBackend_None = 0x00000000,
    WGPUInstanceBackend_Force32 = 0x7FFFFFFF
} WGPUInstanceBackend;
typedef WGPUFlags WGPUInstanceBackendFlags;

typedef enum WGPUDx12Compiler {
    WGPUDx12Compiler_Undefined = 0x00000000,
    WGPUDx12Compiler_Fxc = 0x00000001,
    WGPUDx12Compiler_Dxc = 0x00000002,
    WGPUDx12Compiler_Force32 = 0x7FFFFFFF
} WGPUDx12Compiler;

typedef enum WGPUGles3MinorVersion {
    WGPUGles3MinorVersion_Automatic = 0x00000000,
    WGPUGles3MinorVersion_Version0 = 0x00000001,
    WGPUGles3MinorVersion_Version1 = 0x00000002,
    WGPUGles3MinorVersion_Version2 = 0x00000003,
    WGPUGles3MinorVersion_Force32 = 0x7FFFFFFF
} WGPUGles3MinorVersion;

typedef struct WGPUInstanceExtras {
    WGPUChainedStruct chain;
    WGPUInstanceBackendFlags backends;
    WGPUDx12Compiler dx12ShaderCompiler;
    WGPUGles3MinorVersion gles3MinorVersion;
    const char * dxilPath;
    const char * dxcPath;
} WGPUInstanceExtras;

typedef struct WGPUDeviceExtras {
    WGPUChainedStruct chain;
    const char * tracePath;
} WGPUDeviceExtras;

typedef struct WGPURequiredLimitsExtras {
    WGPUChainedStruct chain;
    uint32_t maxPushConstantSize;
} WGPURequiredLimitsExtras;

typedef struct WGPUSupportedLimitsExtras {
    WGPUChainedStructOut chain;
    uint32_t maxPushConstantSize;
} WGPUSupportedLimitsExtras;

typedef struct WGPUPushConstantRange {
    WGPUShaderStageFlags stages;
    uint32_t start;
    uint32_t end;
} WGPUPushConstantRange;

typedef struct WGPUPipelineLayoutExtras {
    WGPUChainedStruct chain;
    uint32_t pushConstantRangeCount;
    WGPUPushConstantRange* pushConstantRanges;
} WGPUPipelineLayoutExtras;

typedef uint64_t WGPUSubmissionIndex;

typedef struct WGPUWrappedSubmissionIndex {
    WGPUQueue queue;
    WGPUSubmissionIndex submissionIndex;
} WGPUWrappedSubmissionIndex;

typedef struct WGPUShaderDefine {
    char const * name;
    char const * value;
} WGPUShaderDefine;

typedef struct WGPUShaderModuleGLSLDescriptor {
    WGPUChainedStruct chain;
    WGPUShaderStage stage;
    char const * code;
    uint32_t defineCount;
    WGPUShaderDefine * defines;
} WGPUShaderModuleGLSLDescriptor;

typedef struct WGPUStorageReport {
    size_t numOccupied;
    size_t numVacant;
    size_t numError;
    size_t elementSize;
} WGPUStorageReport;

typedef struct WGPUHubReport {
    WGPUStorageReport adapters;
    WGPUStorageReport devices;
    WGPUStorageReport pipelineLayouts;
    WGPUStorageReport shaderModules;
    WGPUStorageReport bindGroupLayouts;
    WGPUStorageReport bindGroups;
    WGPUStorageReport commandBuffers;
    WGPUStorageReport renderBundles;
    WGPUStorageReport renderPipelines;
    WGPUStorageReport computePipelines;
    WGPUStorageReport querySets;
    WGPUStorageReport buffers;
    WGPUStorageReport textures;
    WGPUStorageReport textureViews;
    WGPUStorageReport samplers;
} WGPUHubReport;

typedef struct WGPUGlobalReport {
    WGPUStorageReport surfaces;
    WGPUBackendType backendType;
    WGPUHubReport vulkan;
    WGPUHubReport metal;
    WGPUHubReport dx12;
    WGPUHubReport dx11;
    WGPUHubReport gl;
} WGPUGlobalReport;

typedef struct WGPUInstanceEnumerateAdapterOptions {
    WGPUChainedStruct const * nextInChain;
    WGPUInstanceBackendFlags backends;
} WGPUInstanceEnumerateAdapterOptions;

typedef struct WGPUBindGroupEntryExtras {
    WGPUChainedStruct chain;
    WGPUBuffer const * buffers;
    size_t bufferCount;
    WGPUSampler const * samplers;
    size_t samplerCount;
    WGPUTextureView const * textureViews;
    size_t textureViewCount;
} WGPUBindGroupEntryExtras;

typedef struct WGPUBindGroupLayoutEntryExtras {
    WGPUChainedStruct chain;
    uint32_t count;
} WGPUBindGroupLayoutEntryExtras;

typedef void (*WGPULogCallback)(WGPULogLevel level, char const * message, void * userdata);

#ifdef __cplusplus
extern "C" {
#endif

void wgpuGenerateReport(WGPUInstance instance, WGPUGlobalReport * report);
size_t wgpuInstanceEnumerateAdapters(WGPUInstance instance, WGPUInstanceEnumerateAdapterOptions const * options, WGPUAdapter * adapters);

WGPUSubmissionIndex wgpuQueueSubmitForIndex(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const * commands);

// Returns true if the queue is empty, or false if there are more queue submissions still in flight.
WGPUBool wgpuDevicePoll(WGPUDevice device, WGPUBool wait, WGPUWrappedSubmissionIndex const * wrappedSubmissionIndex);

void wgpuSetLogCallback(WGPULogCallback callback, void * userdata);

void wgpuSetLogLevel(WGPULogLevel level);

uint32_t wgpuGetVersion(void);

void wgpuRenderPassEncoderSetPushConstants(WGPURenderPassEncoder encoder, WGPUShaderStageFlags stages, uint32_t offset, uint32_t sizeBytes, void* const data);

void wgpuRenderPassEncoderMultiDrawIndirect(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint32_t count);
void wgpuRenderPassEncoderMultiDrawIndexedIndirect(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint32_t count);

void wgpuRenderPassEncoderMultiDrawIndirectCount(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, WGPUBuffer count_buffer, uint64_t count_buffer_offset, uint32_t max_count);
void wgpuRenderPassEncoderMultiDrawIndexedIndirectCount(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, WGPUBuffer count_buffer, uint64_t count_buffer_offset, uint32_t max_count);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
