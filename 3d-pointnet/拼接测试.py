import numpy as np

# 假设这是您加载的标签字典，包含每个设备的 8x8 标签矩阵
labels_dict = {
    "L1_device1": np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 2, 2]]),

    "L1_device2": np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [2, 2, 0, 0, 0, 0, 0, 0]]),

    "L1_device3": np.array([[0, 0, 0, 0, 0, 0, 2, 2],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]]),

    "L1_device4": np.array([[2, 2, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]])
}


# 通过 (1, 2) 和 (3, 4) 的顺序来拼接标签矩阵
def concatenate_labels(labels_dict, version):
    device1 = labels_dict[f"{version}_device1"]
    device2 = labels_dict[f"{version}_device2"]
    device3 = labels_dict[f"{version}_device3"]
    device4 = labels_dict[f"{version}_device4"]

    # 拼接为 16x16 矩阵
    concatenated_label = np.block([
        [device1, device2],
        [device3, device4]
    ])
    return concatenated_label


# 使用示例
concatenated_label = concatenate_labels(labels_dict, "L1")
print("Concatenated Label Matrix:\n", concatenated_label)
