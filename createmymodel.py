# Author: Wentao WEI
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
import pdb
import numpy as np
model_ft = models.mobilenet_v2(pretrained=True)
# print(model_ft)
# for param in model_ft.parameters():
#             param.requires_grad = False
# print(model_ft)
# pdb.set_trace()
# model_ft = models.mobilenet_v2(pretrained=True)
# model_ft.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(model_ft.last_channel, 3),
#         )
# model_ft.classifier[1] = nn.Linear(model_ft.last_channel,3)
model_ft.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model_ft.last_channel, 3),
        )
# for name, param in model_ft.named_parameters():
# 	print(name, '      ', param.size())
# pdb.set_trace()
# num_ftrs = model_ft.fc.in_features

# model_ft.fc = nn.Linear(num_ftrs, 3)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载权重参数
model_ft.load_state_dict(torch.load('./model_best.pth'))

example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model_ft, example)
traced_script_module.save("traced_resnet_model_my.pt")