import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

import resnet


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = head_projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = head_projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)

        # [2*B, D]
        out = torch.cat([x, y], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(x * y, dim=-1) / self.args.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss


class BYOL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.target_network, _ = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = head_projector(args, self.embedding)

        for param_q, param_k in zip(self.backbone.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward(self, x1, x2):
        pred_x1 = self.projector(self.backbone(x1))
        pred_x2 = self.projector(self.backbone(x2))

        with torch.no_grad():
            target_x1 = self.target_network(x1)
            target_x2 = self.target_network(x2)

        pred_x1 = torch.cat(FullGatherLayer.apply(pred_x1), dim=0)
        pred_x2 = torch.cat(FullGatherLayer.apply(pred_x2), dim=0)
        target_x1 = torch.cat(FullGatherLayer.apply(target_x1), dim=0)
        target_x2 = torch.cat(FullGatherLayer.apply(target_x2), dim=0)

        pred_x1 = F.normalize(pred_x1, dim=1)
        pred_x2 = F.normalize(pred_x2, dim=1)
        target_x1 = F.normalize(target_x1, dim=1)
        target_x2 = F.normalize(target_x2, dim=1)

        loss = 2 - 2 * (pred_x1 * target_x1).sum(dim=-1)
        loss += 2 - 2 * (pred_x2 * target_x2).sum(dim=-1)

        return loss.mean()


class MoCo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.target_network, _ = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector_q = head_projector(args, self.embedding)
        self.projector_k = head_projector(args, self.embedding)

        for param_q, param_k in zip(self.backbone.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for proj_param_q, proj_param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            proj_param_k.data.copy_(proj_param_q.data)
            proj_param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.args.queue_size, int(self.args.mlp.split("-")[-1])))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x1, x2):
        x_q = self.projector_q(self.backbone(x1))

        with torch.no_grad():
            # Momentum update
            for param_q, param_k in zip(self.backbone.parameters(), self.target_network.parameters()):
                param_k.data = param_k.data * self.args.momentum + param_q.data * (1. - self.args.momentum)

            x_k = self.projector_k(self.target_network(x2))

        x_q = torch.cat(FullGatherLayer.apply(x_q), dim=0)
        x_k = torch.cat(FullGatherLayer.apply(x_k), dim=0)

        loss = self.info_nce_loss(x_q, x_k)

        # Queue update
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr+self.args.batch_size, :] = x_k
        ptr = (ptr + self.args.batch_size) % self.args.queue_size
        self.queue_ptr[0] = ptr

        return loss

    def info_nce_loss(self, f_q, f_k):
        f_k = f_k.detach()
        f_mem = self.queue.clone().detach()

        f_q = F.normalize(f_q, dim=1)
        f_k = F.normalize(f_k, dim=1)
        f_mem = F.normalize(f_mem, dim=1)

        pos = torch.bmm(f_q.view(f_q.size(0), 1, -1),
                        f_k.view(f_q.size(0), -1, 1)).squeeze(-1)
        neg = torch.mm(f_q, f_mem.transpose(1, 0))

        logits = torch.cat((pos, neg), dim=1) / self.args.mem_temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)


class CaCo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.target_network, _ = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector_q = head_projector(args, self.embedding)
        self.projector_k = head_projector(args, self.embedding)

        for param_q, param_k in zip(self.backbone.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for proj_param_q, proj_param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            proj_param_k.data.copy_(proj_param_q.data)
            proj_param_k.requires_grad = False

        self.MemoryBank = CaCoPN(self.args.queue_size, int(self.args.mlp.split("-")[-1]))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x1, x2):
        x_q = self.projector_q(self.backbone(x1))
        x_q = F.normalize(x_q, dim=1)
        x_k = self.projector_q(self.backbone(x2))
        x_k = F.normalize(x_k, dim=1)

        x_q = torch.cat(FullGatherLayer.apply(x_q), dim=0)
        x_k = torch.cat(FullGatherLayer.apply(x_k), dim=0)

        with torch.no_grad():
            # Momentum update
            for param_q, param_k in zip(self.backbone.parameters(), self.target_network.parameters()):
                param_k.data = param_k.data * self.args.momentum + param_q.data * (1. - self.args.momentum)

            q = self.projector_k(self.target_network(x1))
            q = F.normalize(q, dim=1)
            q = q.detach()

            k = self.projector_k(self.target_network(x2))
            k = F.normalize(k, dim=1)
            k = k.detach()

            q = torch.cat(FullGatherLayer.apply(q), dim=0)
            k = torch.cat(FullGatherLayer.apply(k), dim=0)

        d_norm1, d1, logits1 = self.MemoryBank(x_q)
        d_norm2, d2, logits2 = self.MemoryBank(x_k)

        with torch.no_grad():
            logits_keep1 = logits1.clone()
            logits_keep2 = logits2.clone()
        logits1 /= self.args.temperature
        logits2 /= self.args.temperature

        with torch.no_grad():
            d_norm21, d21, check_logits1 = self.MemoryBank(k)
            check_logits1 = check_logits1.detach()
            filter_index1 = torch.argmax(check_logits1, dim=1)
            labels1 = filter_index1

            d_norm22, d22, check_logits2 = self.MemoryBank(q)
            check_logits2 = check_logits2.detach()
            filter_index2 = torch.argmax(check_logits2, dim=1)
            labels2 = filter_index2

        loss = self.criterion(logits1, labels1) + self.criterion(logits2, labels2)

        with torch.no_grad():
            # Update memory bank
            logits1 = logits_keep1 / self.args.mem_temperature
            logits2 = logits_keep2 / self.args.mem_temperature

            p_qd1 = F.softmax(logits1, dim=1)
            p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]),
                                                                             filter_index1]
            g1 = torch.einsum('cn,nk->ck', [x_q.T, p_qd1]) / logits1.shape[0] - torch.mul(
                torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)

            p_qd2 = F.softmax(logits2, dim=1)
            p_qd2[torch.arange(logits2.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]),
                                                                             filter_index2]
            g2 = torch.einsum('cn,nk->ck', [x_k.T, p_qd2]) / logits2.shape[0] - torch.mul(
                torch.mean(torch.mul(p_qd2, logits_keep2), dim=0), d_norm2)

            g = -torch.div(g1, torch.norm(d1, dim=0)) - torch.div(g2, torch.norm(d2, dim=0))
            g /= self.args.mem_temperature

            self.MemoryBank.update(self.args.momentum, self.args.mem_lr, g)

        return loss


class CaCoPN(nn.Module):
    def __init__(self, bank_size, dim):
        super(CaCoPN, self).__init__()
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))

    def forward(self, q):
        memory_bank = self.W
        memory_bank = F.normalize(memory_bank, dim=0)
        logit = torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit

    def update(self, m, lr, g):
        # g = g + wd * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v


def head_projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

