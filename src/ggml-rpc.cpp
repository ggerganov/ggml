#include "ggml-rpc.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-rpc.grpc.pb.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>

#define UNUSED GGML_UNUSED

#define GGML_DEBUG 1
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    return &guid;
}

struct ggml_backend_rpc_context {
    std::string endpoint;
    std::string name;
    std::shared_ptr<ggml::Backend::Stub> stub;
};

// rpc buffer

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<ggml::Backend::Stub> stub;
    uint64_t remote_ptr;
    std::string name;
};

GGML_CALL static const char * ggml_backend_rpc_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_PRINT_DEBUG("buffer get name\n");
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_PRINT_DEBUG("buffer free\n");
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ggml::FreeBufferRequest request;
    request.set_bufptr(ctx->remote_ptr);
    ggml::FreeBufferReply reply;
    grpc::ClientContext context;
    grpc::Status status = ctx->stub->FreeBuffer(&context, request, &reply);
    GGML_ASSERT(status.ok());
    delete ctx;
}

GGML_CALL static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_PRINT_DEBUG("buffer get base\n");
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ggml::BufferGetBaseRequest request;
    request.set_bufptr(ctx->remote_ptr);
    ggml::BufferGetBaseReply reply;
    grpc::ClientContext context;
    grpc::Status status = ctx->stub->BufferGetBase(&context, request, &reply);
    GGML_ASSERT(status.ok());
    return reinterpret_cast<void *>(reply.baseptr());
}

static void serialize_tensor(const ggml_tensor * tensor, ggml::Tensor * protobuf_tensor) {
    protobuf_tensor->set_id(reinterpret_cast<uint64_t>(tensor));
    protobuf_tensor->set_type(tensor->type);
    protobuf_tensor->set_backend(tensor->backend);
    if (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        protobuf_tensor->set_bufptr(ctx->remote_ptr);
    } else {
        protobuf_tensor->set_bufptr(0);
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        protobuf_tensor->add_ne(tensor->ne[i]);
    }
    protobuf_tensor->set_op(tensor->op);
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        protobuf_tensor->add_op_params(tensor->op_params[i]);
    }
    protobuf_tensor->set_flags(tensor->flags);
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        protobuf_tensor->add_src(reinterpret_cast<uint64_t>(tensor->src[i]));
    }
    protobuf_tensor->set_view_src(reinterpret_cast<uint64_t>(tensor->view_src));
    protobuf_tensor->set_view_offs(tensor->view_offs);
    protobuf_tensor->set_data(reinterpret_cast<uint64_t>(tensor->data));
    protobuf_tensor->set_name(tensor->name);
    protobuf_tensor->set_extra(reinterpret_cast<uint64_t>(tensor->extra));
}

static ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const ggml::Tensor & protobuf_tensor) {
    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) protobuf_tensor.type(),
        protobuf_tensor.ne(0), protobuf_tensor.ne(1), protobuf_tensor.ne(2), protobuf_tensor.ne(3));
    result->backend = (ggml_backend_type) protobuf_tensor.backend();
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(protobuf_tensor.bufptr());
    result->op = (ggml_op) protobuf_tensor.op();
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = protobuf_tensor.op_params(i);
    }
    result->flags = protobuf_tensor.flags();
    result->data = reinterpret_cast<void *>(protobuf_tensor.data());
    strncpy(result->name, protobuf_tensor.name().c_str(), GGML_MAX_NAME);
    result->extra = reinterpret_cast<void *>(protobuf_tensor.extra());
    return result;
}

GGML_CALL static void ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_PRINT_DEBUG("buffer init tensor\n");
    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ggml::InitTensorRequest request;
    ggml::Tensor * protobuf_tensor = request.mutable_tensor();
    serialize_tensor(tensor, protobuf_tensor);
    ggml::InitTensorReply reply;
    grpc::ClientContext context;
    grpc::Status status = ctx->stub->InitTensor(&context, request, &reply);
    GGML_ASSERT(status.ok());
    const ggml::Tensor & initialized_tensor = reply.tensor();
    tensor->backend = (ggml_backend_type) initialized_tensor.backend();
    tensor->extra = reinterpret_cast<void *>(initialized_tensor.extra());
}

GGML_CALL static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_PRINT_DEBUG("buffer set tensor\n");
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ggml::SetTensorRequest request;
    ggml::Tensor * protobuf_tensor = request.mutable_tensor();
    serialize_tensor(tensor, protobuf_tensor);
    request.set_offset(offset);
    std::string datastr((const char *)data, size);
    request.set_data(datastr);
    ggml::SetTensorReply reply;
    grpc::ClientContext context;
    grpc::Status status = ctx->stub->SetTensor(&context, request, &reply);
    GGML_ASSERT(status.ok());
}

GGML_CALL static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_PRINT_DEBUG("buffer get tensor\n");
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ggml::GetTensorRequest request;
    ggml::Tensor * protobuf_tensor = request.mutable_tensor();
    serialize_tensor(tensor, protobuf_tensor);
    request.set_offset(offset);
    request.set_size(size);
    ggml::GetTensorReply reply;
    grpc::ClientContext context;
    grpc::Status status = ctx->stub->GetTensor(&context, request, &reply);
    GGML_ASSERT(status.ok());
    memcpy(data, reply.data().c_str(), size);
}

GGML_CALL static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_PRINT_DEBUG("buffer cpy tensor\n");
    UNUSED(buffer);
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ggml::CopyTensorRequest request;
    ggml::Tensor * protobuf_src = request.mutable_src();
    serialize_tensor(src, protobuf_src);
    ggml::Tensor * protobuf_dst = request.mutable_dst();
    serialize_tensor(dst, protobuf_dst);
    ggml::CopyTensorReply reply;
    grpc::ClientContext context;
    grpc::Status status = ctx->stub->CopyTensor(&context, request, &reply);
    GGML_ASSERT(status.ok());
    return reply.result();
}

GGML_CALL static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_PRINT_DEBUG("buffer clear\n");
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ggml::BufferClearRequest request;
    request.set_bufptr(ctx->remote_ptr);
    request.set_value(value);
    ggml::BufferClearReply reply;
    grpc::ClientContext context;
    grpc::Status status = ctx->stub->BufferClear(&context, request, &reply);
    GGML_ASSERT(status.ok());
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .get_name        = */ ggml_backend_rpc_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

// rpc buffer type
struct ggml_backend_rpc_buffer_type_context {
    std::shared_ptr<ggml::Backend::Stub> stub;
    std::string name;
};

GGML_CALL static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_PRINT_DEBUG("buffer type name\n");
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_PRINT_DEBUG("alloc buffer, size: %zu\n", size);
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml::AllocateBufferRequest request;
    request.set_size(size);
    ggml::AllocateBufferReply reply;
    grpc::ClientContext context;
    grpc::Status status = buft_ctx->stub->AllocateBuffer(&context, request, &reply);
    GGML_ASSERT(status.ok());

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
        ggml_backend_rpc_buffer_interface,
        new ggml_backend_rpc_buffer_context{buft_ctx->stub, reply.bufptr(), "RPC Buffer"},
        reply.size());

    return buffer;
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_PRINT_DEBUG("get alignment\n");
    ggml::GetAlignmentRequest request;
    ggml::GetAlignmentReply reply;
    grpc::ClientContext context;
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    grpc::Status status = buft_ctx->stub->GetAlignment(&context, request, &reply);
    GGML_ASSERT(status.ok());
    return reply.alignment();
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_PRINT_DEBUG("get alloc size\n");
    ggml::GetAllocSizeRequest request;
    ggml::Tensor * protobuf_tensor = request.mutable_tensor();
    serialize_tensor(tensor, protobuf_tensor);
    ggml::GetAllocSizeReply reply;
    grpc::ClientContext context;
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    grpc::Status status = buft_ctx->stub->GetAllocSize(&context, request, &reply);
    GGML_ASSERT(status.ok());
    return reply.size();
}

GGML_CALL static bool ggml_backend_rpc_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    GGML_PRINT_DEBUG("buffer type supports backend\n");
    if (!ggml_backend_is_rpc(backend)) {
        return false;
    }

    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return buft_ctx->stub == rpc_ctx->stub;
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_rpc_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};


GGML_CALL static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    GGML_PRINT_DEBUG("backend name\n");
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_free(ggml_backend_t backend) {
    GGML_PRINT_DEBUG("backend free\n");
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_rpc_get_default_buffer_type(ggml_backend_t backend) {
    GGML_PRINT_DEBUG("get default buffer type\n");
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    static ggml_backend_buffer_type buft;
    buft.iface = ggml_backend_rpc_buffer_type_interface;
    buft.context = new ggml_backend_rpc_buffer_type_context{rpc_ctx->stub, "RPC"};
    return &buft;
}

GGML_CALL static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_PRINT_DEBUG("synchronize\n");
    UNUSED(backend);
    GGML_ASSERT(false && "not implemented");
}

static void add_node(ggml::GraphComputeRequest & request, ggml_tensor * node, std::unordered_set<ggml_tensor*> & visited) {
    if (node == nullptr) {
        return;
    }
    if (visited.find(node) != visited.end()) {
        return;
    }
    visited.insert(node);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_node(request, node->src[i], visited);
    }
    add_node(request, node->view_src, visited);

    ggml::Tensor * protobuf_tensor = request.add_tensors();
    GGML_PRINT_DEBUG("add node: %p\n", (void*)node);
    serialize_tensor(node, protobuf_tensor);
}

GGML_CALL static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_PRINT_DEBUG("graph compute\n");
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml::GraphComputeRequest request;
    std::unordered_set<ggml_tensor*> visited;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        add_node(request, node, visited);
        request.add_nodes(reinterpret_cast<uint64_t>(node));
    }
    ggml::GraphComputeReply reply;
    grpc::ClientContext context;
    grpc::Status status = rpc_ctx->stub->GraphCompute(&context, request, &reply);
    GGML_ASSERT(status.ok());
    return (enum ggml_status)reply.status();
}

GGML_CALL static bool ggml_backend_rpc_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    GGML_PRINT_DEBUG("supports op\n");
    UNUSED(backend);
    UNUSED(op);
    GGML_ASSERT(false && "not implemented");
    return true;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .get_default_buffer_type = */ ggml_backend_rpc_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .supports_op             = */ ggml_backend_rpc_supports_op,
};

GGML_CALL ggml_backend_t ggml_backend_rpc_init(const char * endpoint) {
    GGML_PRINT_DEBUG("backend init\n");
    auto channel = grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials());
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint = */ endpoint,
        /* .name   = */ "RPC",
        /* .stub   = */ ggml::Backend::NewStub(channel)
    };

    ggml_backend_t rpc_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_rpc_guid(),
        /* .interface = */ ggml_backend_rpc_interface,
        /* .context   = */ ctx
    };

    return rpc_backend;
}

GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

// Server-side implementation of the RPC backend

BackendImpl::BackendImpl() {
    // the RPC backend simply delegates to one of the existing backends
#ifdef GGML_USE_CUBLAS
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        backend = ggml_backend_cpu_init();
    }
}

BackendImpl::~BackendImpl() {
    ggml_backend_free(backend);
}

grpc::Status BackendImpl::GetAlignment(grpc::ServerContext* context, const ggml::GetAlignmentRequest* request, ggml::GetAlignmentReply* reply) {
    GGML_PRINT_DEBUG("GetAlignment\n");
    UNUSED(context);
    UNUSED(request);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    reply->set_alignment(ggml_backend_buft_get_alignment(buft));
    return grpc::Status::OK;
}

grpc::Status BackendImpl::GetAllocSize(grpc::ServerContext* context, const ggml::GetAllocSizeRequest* request, ggml::GetAllocSizeReply* reply) {
    GGML_PRINT_DEBUG("GetAllocSize\n");
    UNUSED(context);
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, request->tensor());
    size_t size = ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(backend), tensor);
    reply->set_size(size);
    ggml_free(ctx);
    return grpc::Status::OK;
}

grpc::Status BackendImpl::AllocateBuffer(grpc::ServerContext* context, const ggml::AllocateBufferRequest* request, ggml::AllocateBufferReply* reply) {
    GGML_PRINT_DEBUG("AllocateBuffer, size: %u\n", request->size());
    UNUSED(context);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request->size());
    reply->set_bufptr(reinterpret_cast<uint64_t>(buffer));
    reply->set_size(buffer->size);
    GGML_PRINT_DEBUG("bufptr: %p, size: %lu\n", (void*)buffer, buffer->size);
    return grpc::Status::OK;
}

grpc::Status BackendImpl::BufferGetBase(grpc::ServerContext* context, const ggml::BufferGetBaseRequest* request, ggml::BufferGetBaseReply* reply) {
    GGML_PRINT_DEBUG("BufferGetBase, bufptr: %p\n", (void*)request->bufptr());
    UNUSED(context);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request->bufptr());
    void * base = ggml_backend_buffer_get_base(buffer);
    GGML_PRINT_DEBUG("baseptr: %p\n", base);
    reply->set_baseptr(reinterpret_cast<uint64_t>(base));
    return grpc::Status::OK;
}

grpc::Status BackendImpl::FreeBuffer(grpc::ServerContext* context, const ggml::FreeBufferRequest* request, ggml::FreeBufferReply* reply) {
    GGML_PRINT_DEBUG("FreeBuffer, bufptr: %p\n", (void*)request->bufptr());
    UNUSED(context);
    UNUSED(reply);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request->bufptr());
    ggml_backend_buffer_free(buffer);
    return grpc::Status::OK;
}

grpc::Status BackendImpl::BufferClear(grpc::ServerContext* context, const ggml::BufferClearRequest* request, ggml::BufferClearReply* reply) {
    GGML_PRINT_DEBUG("BufferClear, bufptr: %p, value: %u\n", (void*)request->bufptr(), request->value());
    UNUSED(context);
    UNUSED(reply);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request->bufptr());
    ggml_backend_buffer_clear(buffer, request->value());
    return grpc::Status::OK;
}

grpc::Status BackendImpl::InitTensor(grpc::ServerContext* context, const ggml::InitTensorRequest* request, ggml::InitTensorReply* reply) {
    GGML_PRINT_DEBUG("InitTensor\n");
    UNUSED(context);
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, request->tensor());
    ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
    serialize_tensor(tensor, reply->mutable_tensor());
    ggml_free(ctx);
    return grpc::Status::OK;
}

grpc::Status BackendImpl::SetTensor(grpc::ServerContext* context, const ggml::SetTensorRequest* request, ggml::SetTensorReply* reply) {
    UNUSED(context);
    UNUSED(reply);
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * tensor = deserialize_tensor(ctx, request->tensor());
    GGML_PRINT_DEBUG("SetTensor, bufptr: %p, dataptr: %p, offset: %u, size: %lu\n", (void*)tensor->buffer,
        (void*)tensor->data, request->offset(), request->data().size());
    ggml_backend_tensor_set(tensor, request->data().c_str(), request->offset(), request->data().size());
    ggml_free(ctx);
    return grpc::Status::OK;
}

grpc::Status BackendImpl::GetTensor(grpc::ServerContext* context, const ggml::GetTensorRequest* request, ggml::GetTensorReply* reply) {
    UNUSED(context);
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * tensor = deserialize_tensor(ctx, request->tensor());
    GGML_PRINT_DEBUG("GetTensor, bufptr: %p, dataptr: %p, offset: %u, size: %u\n", (void*)tensor->buffer,
        (void*)tensor->data, request->offset(), request->size());
    std::string data(request->size(), 0);
    ggml_backend_tensor_get(tensor, &data[0], request->offset(), request->size());
    reply->set_data(data);
    ggml_free(ctx);
    return grpc::Status::OK;
}

grpc::Status BackendImpl::CopyTensor(grpc::ServerContext* context, const ggml::CopyTensorRequest* request, ggml::CopyTensorReply* reply) {
    UNUSED(context);
    struct ggml_init_params params {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * src = deserialize_tensor(ctx, request->src());
    struct ggml_tensor * dst = deserialize_tensor(ctx, request->dst());
    GGML_PRINT_DEBUG("CopyTensor, src: %p, dst: %p\n", (void*)src, (void*)dst);
    bool result = ggml_backend_buffer_copy_tensor(src, dst);
    reply->set_result(result);
    ggml_free(ctx);
    return grpc::Status::OK;
}

static struct ggml_tensor * create_node(uint64_t id,
                                        struct ggml_context * ctx,
                                        const ggml::GraphComputeRequest* request,
                                        std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (id == 0) {
        return nullptr;
    }
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    for (int i = 0; i < request->tensors_size(); i++) {
        if (request->tensors(i).id() == id) {
            GGML_PRINT_DEBUG("create node: %lx\n", id);
            const ggml::Tensor & protobuf_tensor = request->tensors(i);
            struct ggml_tensor * result = deserialize_tensor(ctx, protobuf_tensor);
            tensor_map[id] = result;
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                result->src[i] = create_node(protobuf_tensor.src(i), ctx, request, tensor_map);
            }
            result->view_src = create_node(protobuf_tensor.view_src(), ctx, request, tensor_map);
            result->view_offs = protobuf_tensor.view_offs();
            return result;
        }
    }
    GGML_ASSERT(false && "tensor not found");
    return nullptr;
}

grpc::Status BackendImpl::GraphCompute(grpc::ServerContext* context, const ggml::GraphComputeRequest* request, ggml::GraphComputeReply* reply) {
    GGML_PRINT_DEBUG("GraphCompute\n");
    UNUSED(context);
    int num_tensors = request->tensors_size();
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;

    int num_nodes = request->nodes_size();
    static size_t buf_size = ggml_tensor_overhead()*num_nodes + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, num_nodes, false);
    graph->n_nodes = num_nodes;
    for (int i = 0; i < num_nodes; i++) {
        graph->nodes[i] = create_node(request->nodes(i), ctx, request, tensor_map);
    }
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    reply->set_status(status);
    ggml_free(ctx);
    ggml_free(ctx0);
    return grpc::Status::OK;
}
