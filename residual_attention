import torch
import torch.nn as nn
# 主要当做分类头来用, 是否能用于模型中间的注意力还有待考证


class CSRA(nn.Module):  # one basic block
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T      # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)  # all m_i in the paper
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        # self.head(x) is x^T_j·m_i. ‘norm’ can be seen in 3.4 in the paper.
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)  # [B, C, HW]
        base_logit = torch.mean(score, dim=2)  # Equal to avg_pool in each channel. [B, C]

        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=2)[0]  # Equal to max_pool in each channel. [B, C]
            print(att_logit.shape)
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit


model = CSRA(128, 10, 99, 0.5)
inp = torch.ones([16, 128, 6, 6])
out = model(inp)
