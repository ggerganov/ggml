#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_dnnl_init();
GGML_API GGML_CALL bool ggml_backend_is_dnnl(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
