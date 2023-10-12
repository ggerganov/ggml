#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#include <stack>
#include <stdexcept>
#include <string>

namespace ggml {
    struct context {
        context() : ctx(nullptr) {}
        context(size_t mem_size, void * mem_buffer, bool no_alloc) {
            ggml_init_params params = {
                /*.mem_size   = */ mem_size,
                /*.mem_buffer = */ mem_buffer,
                /*.no_alloc   = */ no_alloc
            };
            ctx = ggml_init(params);
            if (ctx == nullptr) {
                throw std::runtime_error("failed to initialize ggml");
            }
        }
        context(const context & ctx) = delete;
        context(context && ctx) {
            this->ctx = ctx.ctx;
            ctx.ctx = nullptr;
        }
        ~context() {
            ggml_free(ctx);
        }
        context & operator=(const context & rhs) = delete;
        context & operator=(context && rhs) {
            if (this != &rhs) {
                this->ctx = rhs.ctx;
                rhs.ctx = nullptr;
            }
            return *this;
        }


        operator bool() const {
            return ctx != nullptr;
        }

        ggml_context * get() {
            GGML_ASSERT(ctx != nullptr && "context not initialized");
            return ctx;
        }

        private:
        ggml_context * ctx;
    };

    // the global context stack allows using tensors without explicitly passing the context
    // tensors must be created within a context_guard
    struct ctx_stack {
        std::stack<ggml_context *> stack;
    };

    inline ctx_stack & get_ctx_stack() {
        static ctx_stack s;
        return s;
    }

    inline ggml_context * ctx() {
        ggml_context * g_ctx = get_ctx_stack().stack.empty() ? nullptr : get_ctx_stack().stack.top();
        GGML_ASSERT(g_ctx != nullptr && "this function must be called within a context_guard");
        return g_ctx;
    }

    // TODO: nested context guards are not always properly handled
    struct context_guard {
        context_guard(context & ctx) : ctx(ctx.get()) {
            get_ctx_stack().stack.push(ctx.get());
        }
        context_guard(const context_guard & ctx) = delete;
        context_guard(context_guard && ctx) {
            this->ctx = ctx.ctx;
            ctx.ctx = nullptr;
        }

        context_guard & operator=(const context_guard & rhs) = delete;
        context_guard & operator=(context_guard && rhs) {
            this->ctx = rhs.ctx;
            rhs.ctx = nullptr;
            return *this;
        }

        ~context_guard() {
            if (ctx != nullptr) {
                release();
            }
        }

        void release() {
            GGML_ASSERT(ctx != nullptr && "this context_guard has already been released");
            GGML_ASSERT(get_ctx_stack().stack.top() == ctx && "only the top context_guard can be released");
            ctx = nullptr;
            get_ctx_stack().stack.pop();
        }


        ggml_context * ctx;
    };

    struct tensor {
        tensor() : val(nullptr) {}
        tensor(ggml_tensor * val) : val(val) {}
        tensor(const tensor & val) = delete; // reference copies can be performed by initializing from get()
        tensor(tensor && val) {
            this->val = val.val;
            val.val = nullptr;
        }
        tensor & operator=(const tensor & rhs) = delete;
        tensor & operator=(tensor && rhs) {
            if (this != &rhs) {
                this->val = rhs.val;
                rhs.val = nullptr;
            }
            return *this;
        }

        // new tensor
        tensor(ggml_type type) {
            val = ggml_new_tensor_1d(ctx(), type, 1);
        }
        tensor(ggml_type type, int64_t ne0) {
            val = ggml_new_tensor_1d(ctx(), type, ne0);
        }
        tensor(ggml_type type, int64_t ne0, int64_t ne1) {
            val = ggml_new_tensor_2d(ctx(), type, ne0, ne1);
        }
        tensor(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
            val = ggml_new_tensor_3d(ctx(), type, ne0, ne1, ne2);
        }
        tensor(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
            val = ggml_new_tensor_4d(ctx(), type, ne0, ne1, ne2, ne3);
        }

        // new float tensor
        tensor(int64_t ne0) : tensor(GGML_TYPE_F32, ne0) {}
        tensor(int64_t ne0, int64_t ne1) : tensor(GGML_TYPE_F32, ne0, ne1) {}
        tensor(int64_t ne0, int64_t ne1, int64_t ne2) : tensor(GGML_TYPE_F32, ne0, ne1, ne2) {}
        tensor(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) : tensor(GGML_TYPE_F32, ne0, ne1, ne2, ne3) {}

        // view
        tensor view(int64_t ne0, size_t offset = 0) {
            return ggml_view_1d(ctx(), get(), ne0, offset);
        }
        tensor view(int64_t ne0, int64_t ne1, size_t nb1, size_t offset = 0) {
            return ggml_view_2d(ctx(), get(), ne0, ne1, nb1, offset);
        }
        tensor view(int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset = 0) {
            return ggml_view_3d(ctx(), get(), ne0, ne1, ne2, nb1, nb2, offset);
        }
        tensor view(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset = 0) {
            return ggml_view_4d(ctx(), get(), ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
        }

        // reshape
        tensor reshape(int64_t ne0) {
            return ggml_reshape_1d(ctx(), get(), ne0);
        }
        tensor reshape(int64_t ne0, int64_t ne1) {
            return ggml_reshape_2d(ctx(), get(), ne0, ne1);
        }
        tensor reshape(int64_t ne0, int64_t ne1, int64_t ne2) {
            return ggml_reshape_3d(ctx(), get(), ne0, ne1, ne2);
        }
        tensor reshape(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
            return ggml_reshape_4d(ctx(), get(), ne0, ne1, ne2, ne3);
        }

        // permute
        tensor permute(int axis0, int axis1, int axis2, int axis3) {
            return ggml_permute(ctx(), get(), axis0, axis1, axis2, axis3);
        }

        // cont
        tensor cont() const {
            return ggml_cont(ctx(), val);
        }
        tensor cont(int64_t ne0) const {
            return ggml_cont_1d(ctx(), get(), ne0);
        }
        tensor cont(int64_t ne0, int64_t ne1) const {
            return ggml_cont_2d(ctx(), get(), ne0, ne1);
        }
        tensor cont(int64_t ne0, int64_t ne1, int64_t ne2) const {
            return ggml_cont_3d(ctx(), get(), ne0, ne1, ne2);
        }
        tensor cont(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) const {
            return ggml_cont_4d(ctx(), get(), ne0, ne1, ne2, ne3);
        }

        // copy
        tensor cpy(const tensor & a) {
            return ggml_cpy(ctx(), a.get(), get());
        }

        tensor dup_tensor() const {
            return ggml_dup_tensor(ctx(), get());
        }

        // operators
        tensor operator*(const tensor & rhs) const {
            if (rhs.ne(0) == 1 && rhs.ne(1) == 1 && rhs.ne(2) == 1 && rhs.ne(3) == 1) {
                return ggml_scale(ctx(), get(), rhs.get());
            }
            return ggml_mul(ctx(), get(), rhs.get());
        }

        tensor operator+(const tensor & rhs) const {
            return ggml_add(ctx(), get(), rhs.get());
        }

        tensor operator()(const tensor & rhs) const {
            return ggml_mul_mat(ctx(), get(), rhs.get());
        }

        operator bool() const {
            return val != nullptr;
        }

        // getters
        int64_t ne(int n) const {
            return get()->ne[n];
        }

        size_t nb(int n) const {
            return get()->nb[n];
        }

        ggml_type type() const {
            return get()->type;
        }

        size_t element_size() const {
            return ggml_element_size(get());
        }

        size_t nbytes() const {
            return ggml_nbytes(get());
        }

        int64_t nelements() const {
            return ggml_nelements(get());
        }

        void * data() {
            return ggml_get_data(get());
        }

        ggml_tensor * get() const {
            GGML_ASSERT(val != nullptr && "tensor not initialized");
            return val;
        }

        std::string get_name() const {
            return ggml_get_name(get());
        }

        // setters
        void set_name(const std::string & name) {
            ggml_set_name(get(), name.c_str());
        }

        ggml_tensor * val;

        // backend
        void backend_set(const void * data, size_t offset, size_t nbytes) {
            ggml_backend_tensor_set(get(), data, offset, nbytes);
        }

        void backend_get(void * data, size_t offset, size_t nbytes) {
            ggml_backend_tensor_get(get(), data, offset, nbytes);
        }

        void backend_copy(tensor & dst) {
            ggml_backend_tensor_copy(get(), dst.get());
        }
    };

    struct graph {
        graph() {
            gf = ggml_new_graph(ctx());
        }

        graph(const graph & g) = delete;

        graph(graph && g) {
            this->gf = g.gf;
            g.gf = nullptr;
        }

        graph & operator=(const graph & rhs) = delete;

        graph & operator=(graph && rhs) {
            if (this != &rhs) {
                this->gf = rhs.gf;
                rhs.gf = nullptr;
            }
            return *this;
        }

        void expand(const tensor & t) {
            ggml_build_forward_expand(gf, t.get());
        }

        tensor get_node(int i) {
            return get()->nodes[i];
        }

        size_t n_nodes() const {
            return get()->n_nodes;
        }

        ggml_cgraph * get() const {
            return gf;
        }

        ggml_cgraph * gf;
    };

    inline tensor get_rows(const tensor & a, const tensor & b) {
        return ggml_get_rows(ctx(), a.get(), b.get());
    }

    inline tensor norm(const tensor & t, float eps) {
        return ggml_norm(ctx(), t.get(), eps);
    }

    inline tensor diag_mask_inf(const tensor & t, int n_past) {
        return ggml_diag_mask_inf(ctx(), t.get(), n_past);
    }

    inline tensor soft_max(const tensor & t) {
        return ggml_soft_max(ctx(), t.get());
    }

    inline tensor gelu(const tensor & t) {
        return ggml_gelu(ctx(), t.get());
    }

    inline tensor mul_mat(const tensor & a, const tensor & b) {
        return ggml_mul_mat(ctx(), a.get(), b.get());
    }

    // backend
    struct backend_buffer {
        backend_buffer() : val(nullptr) {}
        backend_buffer(ggml_backend_buffer_t val) : val(val) {}
        backend_buffer(const backend_buffer & val) = delete;
        backend_buffer(backend_buffer && val) {
            this->val = val.val;
            val.val = nullptr;
        }
        ~backend_buffer() {
            free();
        }

        backend_buffer & operator=(const backend_buffer & rhs) = delete;
        backend_buffer & operator=(backend_buffer && rhs) {
            if (this != &rhs) {
                free();
                this->val = rhs.val;
                rhs.val = nullptr;
            }
            return *this;
        }

        operator bool() const {
            return val != nullptr;
        }

        void free() {
            ggml_backend_buffer_free(val);
            val = nullptr;
        }

        size_t get_alignment() const {
            return ggml_backend_buffer_get_alignment(get());
        }

        void * get_base() const {
            return ggml_backend_buffer_get_base(get());
        }

        size_t get_size() const {
            return ggml_backend_buffer_get_size(get());
        }

        size_t get_alloc_size(tensor & tensor) const {
            return ggml_backend_buffer_get_alloc_size(get(), tensor.get());
        }

        void init_tensor(tensor & tensor) {
            ggml_backend_buffer_init_tensor(get(), tensor.get());
        }

        void free_tensor(tensor & tensor) {
            ggml_backend_buffer_free_tensor(get(), tensor.get());
        }

        ggml_backend_buffer_t get() const {
            GGML_ASSERT(val != nullptr && "backend_buffer not initialized");
            return val;
        }

        ggml_backend_buffer_t val;
    };

    struct backend {
        backend() : val(nullptr) {}
        backend(ggml_backend_t val) : val(val) {}
        backend(const backend & val) = delete;
        backend(backend && val) {
            this->val = val.val;
            val.val = nullptr;
        }
        ~backend() {
            free();
        }

        backend & operator=(const backend & rhs) = delete;
        backend & operator=(backend && rhs) {
            if (this != &rhs) {
                free();
                this->val = rhs.val;
                rhs.val = nullptr;
            }
            return *this;
        }

        operator bool() const {
            return val != nullptr;
        }

        std::string name() const {
            return ggml_backend_name(get());
        }

        void free() {
            ggml_backend_free(val);
            val = nullptr;
        }

        size_t get_alignment() const {
            return ggml_backend_get_alignment(get());
        }

        backend_buffer alloc_buffer(size_t size) {
            return ggml_backend_alloc_buffer(get(), size);
        }

        void graph_compute(const graph & gf) {
            ggml_backend_graph_compute(get(), gf.get());
        }

        ggml_backend_t get() const {
            GGML_ASSERT(val != nullptr && "backend not initialized");
            return val;
        }

        ggml_backend_t val;
    };


    struct allocr {
        allocr() : val(nullptr) {}
        allocr(void * data, size_t size, size_t alignment) {
            val = ggml_allocr_new(data, size, alignment);
        }
        allocr(backend_buffer & buffer) {
            val = ggml_allocr_new_from_buffer(buffer.get());
        }
        allocr(ggml_allocr_t val) : val(val) {}
        allocr(const allocr & val) = delete;
        allocr(allocr && val) {
            this->val = val.val;
            val.val = nullptr;
        }
        ~allocr() {
            free();
        }

        static allocr new_measure(size_t alignment) {
            return ggml_allocr_new_measure(alignment);
        }

        allocr & operator=(const allocr & rhs) = delete;
        allocr & operator=(allocr && rhs) {
            if (this != &rhs) {
                free();
                this->val = rhs.val;
                rhs.val = nullptr;
            }
            return *this;
        }

        operator bool() const {
            return val != nullptr;
        }

        void free() {
            ggml_allocr_free(val);
            val = nullptr;
        }

        bool is_measure() const {
            return ggml_allocr_is_measure(get());
        }

        void reset() {
            ggml_allocr_reset(get());
        }

        void alloc(tensor & tensor) {
            ggml_allocr_alloc(get(), tensor.get());
        }

        size_t alloc_graph(graph & graph) {
            return ggml_allocr_alloc_graph(get(), graph.get());
        }

        size_t max_size() const {
            return ggml_allocr_max_size(get());
        }

        ggml_allocr_t get() const {
            GGML_ASSERT(val != nullptr && "allocr not initialized");
            return val;
        }


        ggml_allocr_t val;
    };
}
