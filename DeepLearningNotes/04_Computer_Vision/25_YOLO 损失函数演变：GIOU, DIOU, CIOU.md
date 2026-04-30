# 25_YOLO 损失函数演变：GIOU, DIOU, CIOU

## 核心概念

- **IoU（交并比）**：衡量预测框与真实框重叠程度的指标，$IoU = |A \cap B| / |A \cup B|$。取值范围[0,1]，具有尺度不变性。
- **IoU作为损失的缺陷**：当预测框与真实框不重叠时，IoU=0，梯度也为0，无法提供有效的优化方向。此外，IoU无法区分"靠得很近但未重叠"和"离得很远"两种情况。
- **GIoU（Generalized IoU）**：在IoU基础上引入最小外接矩形（convex hull），当两框不重叠时仍然提供梯度信号。$GIoU = IoU - \frac{|C \setminus (A \cup B)|}{|C|}$，其中 $C$ 是 $A$ 和 $B$ 的最小外接矩形。
- **DIoU（Distance IoU）**：考虑两个框中心点的归一化距离，直接最小化中心点距离，收敛速度比GIoU更快。$DIoU = IoU - \frac{\rho^2(b, b^{gt})}{c^2}$，其中 $\rho$ 是欧氏距离，$c$ 是最小外接矩形对角线长度。
- **CIoU（Complete IoU）**：在DIoU基础上加入长宽比一致性惩罚，同时考虑重叠面积、中心点距离和长宽比三个因素。$CIoU = IoU - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v$，是目前最完善的IoU损失变体。

## 数学推导

**IoU损失：**
$$
L_{IoU} = 1 - IoU = 1 - \frac{|A \cap B|}{|A \cup B|}
$$

**GIoU损失：**
$$
L_{GIoU} = 1 - IoU + \frac{|C \setminus (A \cup B)|}{|C|}
$$

其中 $C$ 是 $A$ 和 $B$ 的最小凸包（外接矩形）。当 $A$ 和 $B$ 完全重叠时，$L_{GIoU}=0$；当 $A$ 和 $B$ 相距无穷远时，$L_{GIoU} \to 2$。

**DIoU损失：**
$$
L_{DIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2}
$$

其中 $b, b^{gt}$ 分别代表预测框和真实框的中心点，$\rho$ 是欧氏距离，$c$ 是覆盖两个框的最小外接矩形的对角线长度。

**CIoU损失：**
$$
L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$
$$
v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2
$$
$$
\alpha = \frac{v}{(1 - IoU) + v}
$$

其中 $v$ 衡量长宽比的一致性，$\alpha$ 是平衡系数。

## 直观理解

IoU损失的演变反映了对"什么是一个好的边界框损失"认知的逐步深化：

- **IoU**：只关心"重叠了多少"——不重叠就没有梯度，像个"全有或全无"的判官
- **GIoU**：关心"重叠了多少，以及不重叠时你们离得有多远"——通过外接矩形给不重叠的框提供梯度，使框趋向重叠
- **DIoU**：更关心"你们的中心点是否对齐"——即使重叠面积相同，中心点对齐的框更好
- **CIoU**：进一步关心"你们的形状是否匹配"——在中心对齐的基础上，长宽比也要接近

整个过程就像调整两个矩形框使其完全重合——先推到一起（GIoU），再对齐中心（DIoU），最后调整形状（CIoU）。

## 代码示例

```python
import torch
import math

def ciou_loss(pred_boxes, target_boxes):
    """CIoU损失实现"""
    # pred_boxes, target_boxes: (N, 4) [x1, y1, x2, y2]
    x1_p, y1_p, x2_p, y2_p = pred_boxes.unbind(1)
    x1_t, y1_t, x2_t, y2_t = target_boxes.unbind(1)

    # 计算IoU
    inter_x1 = torch.max(x1_p, x1_t)
    inter_y1 = torch.max(y1_p, y1_t)
    inter_x2 = torch.min(x2_p, x2_t)
    inter_y2 = torch.min(y2_p, y2_t)
    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_t = (x2_t - x1_t) * (y2_t - y1_t)
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=1e-6)

    # 最小外接矩形
    c_x1 = torch.min(x1_p, x1_t)
    c_y1 = torch.min(y1_p, y1_t)
    c_x2 = torch.max(x2_p, x2_t)
    c_y2 = torch.max(y2_p, y2_t)
    c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-6

    # 中心点距离
    c_p = ((x1_p + x2_p) / 2, (y1_p + y2_p) / 2)
    c_t = ((x1_t + x2_t) / 2, (y1_t + y2_t) / 2)
    center_dist = (c_p[0] - c_t[0]) ** 2 + (c_p[1] - c_t[1]) ** 2

    # DIoU项
    diou_term = center_dist / c_diag

    # 长宽比一致性
    w_p, h_p = x2_p - x1_p, y2_p - y1_p
    w_t, h_t = x2_t - x1_t, y2_t - y1_t
    v = (4 / (math.pi ** 2)) * (torch.atan(w_t / h_t) - torch.atan(w_p / h_p)) ** 2
    alpha = v / ((1 - iou) + v + 1e-6)

    ciou = iou - diou_term - alpha * v
    return 1 - ciou

# 测试
pred = torch.tensor([[10, 10, 100, 100]], dtype=torch.float32)
target = torch.tensor([[20, 20, 120, 120]], dtype=torch.float32)
loss = ciou_loss(pred, target)
print(f"CIoU损失: {loss.item():.4f}")
```

## 深度学习关联

- **YOLO系列的标准配置**：GIoU被YOLOv4采用，DIoU和CIoU被YOLOv5及后续版本采用作为默认边界框回归损失，显著提升了YOLO系列的定位精度和收敛速度。
- **通用检测损失替代**：CIoU已超越YOLO系列，成为目标检测领域通用的边界框回归损失函数，被广泛应用于Faster R-CNN、RetinaNet、DETR等各类检测器中。
- **评估指标的演进**：IoU损失的改进也推动了评估指标的发展——从mAP@0.5到mAP@0.5:0.95，再到更精细的定位质量评估（如Localization Recall Precision），反映了对检测精度要求的不断提升。
