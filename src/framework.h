#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include "wgpu.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED(x) (void)x;

typedef struct frmwrk_buffer_init_descriptor {
  WGPU_NULLABLE char const *label;
  WGPUBufferUsageFlags usage;
  void *content;
  size_t content_size;
} frmwrk_buffer_init_descriptor;

void frmwrk_setup_logging(WGPULogLevel level);
WGPUShaderModule frmwrk_load_shader_module(WGPUDevice device, const char *name);
void frmwrk_print_global_report(WGPUGlobalReport report);
WGPUBuffer frmwrk_device_create_buffer_init(
    WGPUDevice device, const frmwrk_buffer_init_descriptor *descriptor);

#endif // FRAMEWORK_H
