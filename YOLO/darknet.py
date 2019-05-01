from __future__ import division

import os

import numpy as np
import torch
import torch.nn as nn
from utils import predict_transform

# read the CfG path
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path,'cfg','yolov3.cfg')

def cfgToBlocks(myCfg):
    #Read the Cfg file in Lines format
    file = open(myCfg, 'r')
    lines = file.read().split('\n')
    file.close()

    # Remove empty files and left/Right Space for block consistency
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks

# NN.Modules Sub classes used for

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()  #submodels
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # 1) Convolutional Layers Type

        if (x["type"] == "convolutional"):

            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters,
                             filters,
                             kernel_size,
                             stride,
                             pad,
                             bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)     # with learning parameters
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True) # slope == 0.1
                module.add_module("leaky_{0}".format(index), activn)

        # 2) Upsample Layers Type
        elif (x["type"] == "upsample"):  # 2 times in Yolo v3

            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # 3) Route Layers Type
        elif (x["type"] == "route"):

            x["layers"] = x["layers"].split(',')

            head = int(x["layers"][0])
            try:
                tail = int(x["layers"][1])
            except:
                tail = 0

            if head > 0:                  # A head Case
                head = tail - index

            elif tail > 0:                # Back Case
                tail = -(index - tail)

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if tail < 0:
                filters = output_filters[index + head] + output_filters[index + tail]
            else:
                filters = output_filters[index + head]

        # 4) Residual Layers Type
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # 5) YOLO Layers Type
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


class Darknet(nn.Module):

    def __init__(self, myCfg):    # darknet instantiated using cfgToBlocks() && createModules()
        super(Darknet, self).__init__()
        self.blocks = cfgToBlocks(myCfg)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            # 1) Convolutional&&Upsample Layers Type
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            # 3) Route Layers Type
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            # 4) Residual Layers Type
            elif module_type == "shortcut":
                tail_3 = int(module["from"])
                x = outputs[i - 1] + outputs[i + tail_3]

            # 5) YOLO Layers Type
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)