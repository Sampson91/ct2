#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/27 上午9:44
# @Author : wangyangyang

import torch
from torchvision.models import resnet18
model =resnet18()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("resnet18.pt")
