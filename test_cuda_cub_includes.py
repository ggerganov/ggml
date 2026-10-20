#!/usr/bin/env python3
"""Regression guard for ggml-org/ggml#1624.

The CCCL-version-gated CUB code paths in the ggml-cuda backend use libcu++
(`cuda::...`) iterator / execution-policy / stream APIs. `<cub/cub.cuh>` does
NOT pull those declarations in, so each gated block must directly `#include`
the header that declares every `cuda::` symbol it uses, or the file fails to
compile with CUDA Toolkit 13.4 (CCCL 3.2+).

No CUDA toolchain is available in this environment, so instead of a real
`nvcc` compile this test statically maps each `cuda::` API used inside a gated
block to the header that declares it and asserts the `#include` is present.
Reverting the fix (dropping an include) drops the matching entry and fails the
test with an explicit "missing header" message.
"""
import os
import re
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "src", "ggml-cuda")

# cuda:: API prefix -> header that must be included when the API is used.
API_HEADER = {
    "cuda::execution::": "<cuda/execution>",
    "cuda::std::execution::": "<cuda/execution>",
    "cuda::stream_ref": "<cuda/stream_ref>",
    "cuda::make_counting_iterator": "<cuda/iterator>",
    "cuda::make_strided_iterator": "<cuda/iterator>",
    "cuda::discard_iterator": "<cuda/iterator>",
}

# Gated blocks to check: (file, guard macro that enables the CUB code path).
CASES = [
    ("argsort.cu", "STRIDED_ITERATOR_AVAILABLE"),
    ("top-k.cu", "CUB_TOP_K_AVAILABLE"),
]


def check(filename, guard):
    path = os.path.join(SRC_DIR, filename)
    with open(path, encoding="utf-8") as f:
        text = f.read()

    includes = set(re.findall(r"#\s*include\s*(<[^>]+>)", text))
    used = {api: hdr for api, hdr in API_HEADER.items() if api in text}
    if not used:
        return [f"{filename}: no gated cuda:: API found (parser out of date?)"]

    problems = []
    for api, hdr in sorted(used.items()):
        status = "OK" if hdr in includes else "MISSING"
        print(f"  {filename} [{guard}] uses {api:<32} needs {hdr:<20} {status}")
        if hdr not in includes:
            problems.append(f"{filename}: uses {api} but never #include {hdr}")
    return problems


def main():
    problems = []
    for filename, guard in CASES:
        problems += check(filename, guard)
    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("\nPASS: every cuda:: API used in a gated CUB block has its header included")


if __name__ == "__main__":
    main()
