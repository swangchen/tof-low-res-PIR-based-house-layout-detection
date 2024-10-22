import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from data_loader import load_concat_data
from modle import SimplifiedAlexNet
import argparse
import torch
import torch.nn as nn

import numpy as np


model_path = "simplified_alexnet_model_with_iou.pth"
input_data_path = "datas/20240703_162917_tof_rawdata.npz"
predict_data= load_concat_data(input_data_path)


# 自定义颜色映射
def visualize_prediction(predicted, actual):
    cmap = mcolors.ListedColormap(['white', 'yellow', 'blue', 'darkblue', 'purple'])
    bounds = [0, 1, 2, 3, 4, 5]  # 定义边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 5))

    # 预测标签可视化
    plt.subplot(1, 2, 1)
    plt.imshow(predicted, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0, 1, 2, 3, 4])
    plt.title("Predicted Label")
    plt.axis('off')

    # 实际标签可视化
    plt.subplot(1, 2, 2)
    plt.imshow(actual, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0, 1, 2, 3, 4])
    plt.title("Actual Label")
    plt.axis('off')

    plt.show()








