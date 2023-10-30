#pragma once

#include "ggml.h"

// GGML internal header

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


#define GGML_HASHTABLE_FULL ((size_t)-1)
#define GGML_HASHTABLE_ALREADY_EXISTS ((size_t)-2)

bool   ggml_hash_contains      (const struct ggml_hash_set hash_set, struct ggml_tensor * key);

// returns GGML_HASHTABLE_FULL if table is full, otherwise the current index of the key or where it should be inserted
size_t ggml_hash_find          (const struct ggml_hash_set hash_set, struct ggml_tensor * key);

// returns GGML_HAHSHTABLE_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
size_t ggml_hash_insert        (      struct ggml_hash_set hash_set, struct ggml_tensor * key);

// return index, asserts if table is full
size_t ggml_hash_find_or_insert(      struct ggml_hash_set hash_set, struct ggml_tensor * key);

#ifdef __cplusplus
}
#endif
