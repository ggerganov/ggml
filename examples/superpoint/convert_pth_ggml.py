#!/usr/bin/env python3
'''
convert superpoint pth paramters  to gguf parameters

'''

import sys
import gguf
import numpy as np
from demo_superpoint import *
from benchmark import *



def list_params(net):
    params_list = []
    list_vars = net.state_dict()
    for name in list_vars.keys():
        data = list_vars[name].numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)
        params_list.append((name, data))
    return  params_list



def onnx_export(net):
      # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 1,480, 640, requires_grad=True)

    # Export the model
    torch.onnx.export(net,         # model being run
        dummy_input,       # model input (or a tuple for multiple inputs)
        "superpoint.onnx",       # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,    # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['modelInput'],   # the model's input names
        output_names = ['output1', 'output2'], # the model's output names
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    return

def isBias(name):
    return (name.strip().split(".")[1] == "bias")

def isConv(name):
    return (name.strip().split(".")[1] == "weight")

def save_conv2d_layer( name , conv, index, gguf_writer):
    layername = "l" + str(index) + "_weights"
    print(f"save {layername} with shape {conv.shape}")
    ## ggml doesn't support f32 convolution yet, use f16 instead

    flat_conv = conv.astype(np.float16).flatten()
    # print(type(conv))
    # exit(0)
    gguf_writer.add_tensor(layername, flat_conv, raw_shape= conv.shape)
    return

def save_bias_layer( name ,biases, index, gguf_writer):
    filters = biases.shape[0]
    layername = "l" + str(index) + "_biases"
    print(f"save {layername} with shape {biases.shape}")
    gguf_writer.add_tensor(layername, biases, raw_shape=(1, filters, 1, 1))
    return



if __name__ == '__main__':
    #hyper parameters provided by superpoint#
    weights_path = 'superpoint_v1.pth'
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7

    outfile = "superpoint.gguf"
    gguf_writer = gguf.GGUFWriter(outfile, 'superpoint')

    fe = SuperPointFrontend(weights_path= weights_path,
                        nms_dist= nms_dist,
                        conf_thresh= conf_thresh,
                        nn_thresh= nn_thresh,
                        cuda= False)
    conv_list = list_params(fe.net)
    conv_idx = 0
    bias_idx = 0
    for name, layer in conv_list:
        print(f"processing {name}")
        if(isConv(name)):
            # if(conv_idx==10):
            #     print(f"name: {name}: {layer.flatten()[:10]}")
            # if(conv_idx==11):
            #     print(f"name: {name}: {layer.flatten()[:10]}")
            #     # exit(0)


            save_conv2d_layer(name, layer, conv_idx, gguf_writer)
            conv_idx+=1
        elif(isBias(name)):
            save_bias_layer(name, layer, bias_idx, gguf_writer)
            bias_idx+=1

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("In total {} conv layers, {} bias ".format(conv_idx, bias_idx))
    print("{} converted to {}".format(weights_path, outfile))
