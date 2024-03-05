#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.grpc.pb.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_rpc_init(const char * endpoint);
GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend);

class BackendImpl : public ggml::Backend::Service {
public:
    BackendImpl();
    ~BackendImpl();

    grpc::Status GetAlignment(grpc::ServerContext* context, const ggml::GetAlignmentRequest* request, ggml::GetAlignmentReply* reply) override;
    grpc::Status GetAllocSize(grpc::ServerContext* context, const ggml::GetAllocSizeRequest* request, ggml::GetAllocSizeReply* reply) override;
    grpc::Status AllocateBuffer(grpc::ServerContext* context, const ggml::AllocateBufferRequest* request, ggml::AllocateBufferReply* reply) override;
    grpc::Status BufferGetBase(grpc::ServerContext* context, const ggml::BufferGetBaseRequest* request, ggml::BufferGetBaseReply* reply) override;
    grpc::Status FreeBuffer(grpc::ServerContext* context, const ggml::FreeBufferRequest* request, ggml::FreeBufferReply* reply) override;
    grpc::Status BufferClear(grpc::ServerContext* context, const ggml::BufferClearRequest* request, ggml::BufferClearReply* reply) override;
    grpc::Status InitTensor(grpc::ServerContext* context, const ggml::InitTensorRequest* request, ggml::InitTensorReply* reply) override;
    grpc::Status SetTensor(grpc::ServerContext* context, const ggml::SetTensorRequest* request, ggml::SetTensorReply* reply) override;
    grpc::Status GetTensor(grpc::ServerContext* context, const ggml::GetTensorRequest* request, ggml::GetTensorReply* reply) override;
    grpc::Status CopyTensor(grpc::ServerContext* context, const ggml::CopyTensorRequest* request, ggml::CopyTensorReply* reply) override;
    grpc::Status GraphCompute(grpc::ServerContext* context, const ggml::GraphComputeRequest* request, ggml::GraphComputeReply* reply) override;

private:
    ggml_backend_t backend = NULL;
};

#ifdef  __cplusplus
}
#endif
