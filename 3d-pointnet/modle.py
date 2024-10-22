
import torch.nn as nn


class SimplifiedAlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(SimplifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def iou_loss(preds, targets, num_classes=5):
    iou = 0.0
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).float().sum()  # 交集
        union = (pred_inds | target_inds).float().sum()  # 并集

        if union == 0:
            iou += 1.0  # 防止空集情况下的除0错误
        else:
            iou += intersection / union
    return 1 - (iou / num_classes)  # 返回1减去平均IoU作为损失
