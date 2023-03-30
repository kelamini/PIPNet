import os
import os.path as osp
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
# import cv2
# import numpy as np
# import pickle
import importlib
from math import floor
# from faceboxes_detector import *
# import time

import torch
# import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
# import torch.optim as optim
import torch.utils.data
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
# from functions import *
from mobilenetv3 import mobilenetv3_large

import onnx
import onnxsim


def convert():

    if not len(sys.argv) == 3:
        print('Format:')
        print('python lib/pytorch2onnx.py config_file, output')
        exit(0)
    experiment_name = sys.argv[1].split('/')[-1][:-3]
    data_name = sys.argv[1].split('/')[-2]
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
    save_dir = sys.argv[2]

    my_config = importlib.import_module(config_path, package='PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name

    weights = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)

    print("=====> load pytorch checkpoint...")
    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=cfg.pretrained)
        net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet101':
        resnet101 = models.resnet101(pretrained=cfg.pretrained)
        net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v2':
        mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
        net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v3':
        mbnet = mobilenetv3_large()
        if cfg.pretrained:
            mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
        net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    else:
        print('No such backbone!')
        exit(0)
 
    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)

    weight_file = os.path.join(weights, 'epoch%d.pth' % (cfg.num_epochs-1))
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    print("=====> convert pytorch model to onnx...")
    save_onnx_model = osp.join(save_dir, cfg.data_name+"_"+cfg.backbone+".onnx")
    dummy_input = Variable(torch.randn(1, 3, 256, 256))
    dummy_input = dummy_input.to(device)
    input_names = ["input_1"]
    output_names = ["output_1"]
    torch.onnx.export(net,
                    dummy_input,
                    save_onnx_model,
                    verbose=True,
                    input_names=input_names,
                    output_names=output_names)

    print("====> check onnx model...")
    model = onnx.load(save_onnx_model)
    onnx.checker.check_model(model)

    print("====> Simplifying...")
    model_opt, check = onnxsim.simplify(save_onnx_model)
    # print("model_opt", model_opt)
    onnx.save(model_opt, save_onnx_model.replace(".onnx", "_sim.onnx"))
    print("onnx model simplify Ok!")


if __name__ == "__main__":
    convert()
