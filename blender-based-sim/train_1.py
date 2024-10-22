import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 读取点云数据
def load_point_cloud(file_path):
    point_cloud = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            x = float(parts[0].split(':')[1].strip())
            y = float(parts[1].split(':')[1].strip())
            z = float(parts[2].split(':')[1].strip())
            point_cloud.append([x, y, z])
    return np.array(point_cloud)

# 加载数据
point_cloud_data = load_point_cloud("point_cloud_zones.txt")
print(f"Loaded point cloud data shape: {point_cloud_data.shape}")

# 自定义数据集类
class PointCloudDataset(Dataset):
    def __init__(self, point_cloud_data, labels):
        self.data = torch.tensor(point_cloud_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 简化的 PointNet 模型
class SimplePointNet(nn.Module):
    def __init__(self, num_classes):
        super(SimplePointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 输入 3D 坐标 (x, y, z)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# 加载训练和测试数据
def load_data(point_cloud_data, labels, batch_size=32, test_split=0.2):
    num_samples = len(point_cloud_data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(num_samples * (1 - test_split))
    train_indices, test_indices = indices[:split], indices[split:]

    train_dataset = PointCloudDataset(point_cloud_data[train_indices], labels[train_indices])
    test_dataset = PointCloudDataset(point_cloud_data[test_indices], labels[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 训练函数
def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")

# 测试函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 准备数据和训练
if __name__ == "__main__":
    # 假设 point_cloud_data 是 3D 点云数据, labels 是物体类型的标签
    labels = np.random.randint(0, 2, len(point_cloud_data))  # 假设有 2 类物体，0 和 1

    train_loader, test_loader = load_data(point_cloud_data, labels, batch_size=32)

    # 创建 PointNet 模型，假设有 2 类物体
    model = SimplePointNet(num_classes=2)

    # 训练模型
    train_model(model, train_loader, epochs=10)

    # 测试模型
    test_model(model, test_loader)
