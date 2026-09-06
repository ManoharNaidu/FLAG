import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig
from sklearn.manifold import TSNE
from torch_geometric.utils import subgraph, index_to_mask, k_hop_subgraph, mask_to_index


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def load_model_config(model_name):
    config = AutoConfig.from_pretrained(model_name)
    rope_parameters = getattr(config, 'rope_parameters', None)
    if isinstance(rope_parameters, dict):
        original_max_position_embeddings = rope_parameters.get(
            'original_max_position_embeddings'
        )
        max_position_embeddings = getattr(config, 'max_position_embeddings', None)
        if original_max_position_embeddings and max_position_embeddings:
            rope_parameters = dict(rope_parameters)
            rope_parameters['factor'] = (
                max_position_embeddings / original_max_position_embeddings
            )
            rope_parameters.pop('original_max_position_embeddings', None)
            config.rope_parameters = rope_parameters
    return config


def generate_homo(data):
    labels = data.y
    edge_index = data.edge_index

    neighbors_labels = []
    index_test = mask_to_index(data.test_mask)
    for i in index_test:
        # 获取节点 i 的邻居索引
        neighbors = edge_index[1][edge_index[0] == i]
        # 将邻居标签存入列表
        neighbors_labels.append(labels[neighbors])

    homophilous_mask = torch.zeros(len(index_test), dtype=torch.bool)
    heterophilous_mask = torch.zeros(len(index_test), dtype=torch.bool)

    for i in range(len(index_test)):
        # 获取中心节点 i 的标签和邻居标签
        center_label = labels[index_test[i]]
        neighbor_labels = neighbors_labels[i]

        # 计算同标签邻居的数量
        num_same_label = (neighbor_labels == center_label).sum().item()
        num_neighbors = neighbor_labels.size(0)

        # 判断是同配还是异配
        if num_same_label > num_neighbors / 2:
            homophilous_mask[i] = True
        else:
            heterophilous_mask[i] = True
    return homophilous_mask, heterophilous_mask



def visualization(data, labels):
    label_colors = {0: 'blue', 1: 'red'}

    # 创建散点图
    plt.figure(figsize=(8, 6))
    for point, label in zip(data, labels):
        plt.scatter(point[0], point[1], color=label_colors[label])

    # 添加图例和显示
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('scatter_plot.png', format='png')  # 你可以选择不同的格式，如 'pdf', 'svg' 等
    plt.close()  # 关闭图表以释放内存

def t_sne(data, labels):
    data = np.array(data)

    # 使用 t-SNE 将高维数据降到 2D
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    # 标签对应的颜色
    label_colors = {0: 'blue', 1: 'red'}

    # 创建散点图
    plt.figure(figsize=(8, 6))
    for point, label in zip(data_2d, labels):
        plt.scatter(point[0], point[1], color=label_colors[label], alpha=0.6)

    # 添加轴标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')

    # 保存为文件
    plt.savefig('tsne.png', format='png')
    plt.close()  # 关闭图表以释放内存

def causal_loss(causal_output, target_labels):
    """
    Cross-entropy loss for the causal output compared with one-hot target labels.
    """
    return F.cross_entropy(causal_output, target_labels)

def non_causal_loss(non_causal_output, num_classes=2):
    """
    Residual loss, paper Eq. 8:  L_Res = D_KL( p_hat || u_hat )

    ``p_hat`` is the model's predicted distribution over classes and ``u_hat``
    is the uniform distribution.  Note that ``F.kl_div(input, target)`` computes
    ``KL(target || input)``, i.e. the *reverse* of what the paper asks for, so
    the divergence is computed explicitly here instead.

    Works for both a 1-D ``(C,)`` logit vector and a batched ``(N, C)`` one; the
    divergence is taken over the last dimension and averaged over the batch.
    """
    log_p = F.log_softmax(non_causal_output, dim=-1)
    p = log_p.exp()
    log_u = torch.log(torch.full_like(log_p, 1.0 / num_classes))
    kl = (p * (log_p - log_u)).sum(dim=-1)  # KL(p_hat || u_hat) per sample
    return kl.mean()

def orthogonal_loss(causal_embeddings, non_causal_embeddings):
    """
    Orthogonality loss, paper Eq. 9:  L_Orthog = || Z_D . Z_R ||^2_2

    Returns the SQUARED dot product of the L2-normalised representations, so the
    minimum (0) is attained exactly at orthogonality.  The raw signed cosine
    similarity would instead be minimised by anti-parallel vectors (-1), which
    is not what the paper specifies.

    Works for both 1-D ``(C,)`` and batched ``(N, C)`` inputs; the dot product is
    taken over the last dimension and the squared value averaged over the batch.
    """
    causal_norm = F.normalize(causal_embeddings, p=2, dim=-1)
    non_causal_norm = F.normalize(non_causal_embeddings, p=2, dim=-1)
    cosine_similarity = torch.sum(causal_norm * non_causal_norm, dim=-1)
    return (cosine_similarity ** 2).mean()


def ECELoss(logits, labels, n_bins=15):
    """
    Expected Calibration Error with ``n_bins`` equal-width confidence bins.

    Confidence is the max softmax probability, the prediction is the argmax.
    Returns the bin-count-weighted average of ``|accuracy - confidence|`` as a
    plain Python ``float`` (callers feed the result to ``statistics.mean``).
    """
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    logits = logits.detach().float()
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    labels = labels.detach().reshape(-1).long()

    if logits.size(0) == 0:
        return 0.0

    softmaxes = F.softmax(logits, dim=-1)
    confidences, predictions = torch.max(softmaxes, dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece.item())


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: 类别的权重因子，适用于不平衡数据集
        :param gamma: 调节因子，控制难易样本的惩罚力度
        :param reduction: 损失的聚合方式，'mean' 返回平均损失，'sum' 返回总损失，'none' 不进行聚合
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: 模型的输出 [batch_size, num_classes]
        # targets: 真实的类别标签 [batch_size]

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 计算普通交叉熵损失
        pt = torch.exp(-ce_loss)  # 计算 pt = exp(-CE)，代表预测的正确性概率

        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def remove_empty_lines(text):
    pattern = r"\n\s*\n"  
    return re.sub(pattern, "\n", text)