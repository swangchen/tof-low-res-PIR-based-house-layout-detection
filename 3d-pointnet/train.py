import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse
from modle import SimplifiedAlexNet, iou_loss
from data_loader import data_set, load_washed_dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# 加载数据集
def load_data(dataset_path='washed_labeled_data.pt', batch_size=32):
    dataset = load_washed_dataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset

# 训练模型
def train_model(train_loader, epochs=8, learning_rate=0.00009, model_path="simplified_alexnet_model.pth"):
    model = SimplifiedAlexNet(num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
            iou_loss_value = iou_loss(preds, targets)
            total_loss = ce_loss + iou_loss_value
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

# 评估模型
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))



# 自定义颜色映射
colors = ["white", "yellow", "lightblue", "blue", "purple"]
cmap = mcolors.ListedColormap(colors)


def plot_predicted_vs_actual(predicted_matrix, actual_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制预测标签
    im_pred = axes[0].imshow(predicted_matrix, cmap=cmap, vmin=0, vmax=4)
    axes[0].set_title("Predicted Label")
    axes[0].grid(True, color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_xticks(np.arange(-0.5, 16, 1))
    axes[0].set_yticks(np.arange(-0.5, 16, 1))
    axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 绘制实际标签
    im_actual = axes[1].imshow(actual_matrix, cmap=cmap, vmin=0, vmax=4)
    axes[1].set_title("Actual Label")
    axes[1].grid(True, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_xticks(np.arange(-0.5, 16, 1))
    axes[1].set_yticks(np.arange(-0.5, 16, 1))
    axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 在整个图中添加颜色条
    cbar = fig.colorbar(im_pred, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, ticks=[0, 1, 2, 3, 4])
    cbar.set_label("Class")

    plt.tight_layout()
    plt.show()


def predict_sample(model, test_dataset):
    # 选取一个样本进行预测
    sample_data, sample_label = test_dataset[0]
    sample_data = sample_data.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(sample_data)
        _, predicted = torch.max(output, 1)

    # 将预测结果和实际标签进行可视化
    print("Visualizing Predicted and Actual Labels:")
    plot_predicted_vs_actual(predicted.squeeze().numpy(), sample_label.numpy())


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate or predict using SimplifiedAlexNet.")
    parser.add_argument("action", choices=["train", "evaluate", "predict"], help="Action to perform")
    parser.add_argument("--dataset_path", default="washed_labeled_data.pt", help="Path to the dataset file")
    parser.add_argument("--model_path", default="simplified_alexnet_model.pth", help="Path to save/load the model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    args = parser.parse_args()

    train_loader, test_loader, test_dataset = load_data(args.dataset_path, args.batch_size)

    if args.action == "train":
        model = train_model(train_loader, args.epochs, model_path=args.model_path)
    else:
        model = SimplifiedAlexNet(num_classes=5)
        model.load_state_dict(torch.load(args.model_path))
        if args.action == "evaluate":
            evaluate_model(model, test_loader)
        elif args.action == "predict":
            predict_sample(model, test_dataset)

if __name__ == "__main__":
    main()
