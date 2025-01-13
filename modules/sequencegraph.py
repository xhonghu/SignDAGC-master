from torch_geometric.nn import GCNConv, SAGEConv, EdgeConv
import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F


def ForEucDis(x, y):
    with torch.no_grad():
        b, c, t, n = x.shape
        x = x.permute(0, 2,3, 1)
        y = y.permute(0, 2,3, 1)
        x = x.reshape(b, t, n, c)
        y = y.reshape(b, t, n, c)
        return torch.cdist(x, y)


class TemporalGraph(nn.Module):
    def __init__(self, in_channels, k=4):
        super(TemporalGraph, self).__init__()
        self.k = k
        self.reduction_channel = in_channels
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.reduction_channel, kernel_size=(3,1,1), bias=False,padding=(1,0,0)),
            nn.BatchNorm3d(self.reduction_channel)
        )
        self.up_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.reduction_channel, kernel_size=(3,1,1), bias=False,padding=(1,0,0)),
            nn.BatchNorm3d(self.reduction_channel)
        )
        self.gconv = GCNConv(self.reduction_channel, self.reduction_channel)

    def forward(self, x, batch=4, span=1):
        tlen, c, h, w = x.shape
        x = rearrange(x.view(batch, tlen // batch, c, h, w), "b v c h w-> b c v h w")
        x = self.down_conv(x)
        x = rearrange(x, "b c v h w-> b c v (h w)")
        x1, x2 = x[:, :, :-span, :], x[:,:,span:,:]
        sim = -ForEucDis(x1, x2)
        b, t_1, hw, hw = sim.shape
        sim = F.normalize(sim.view(b,t_1, -1), dim=-1)
        _, topk_indices = torch.topk(sim, k=self.k)
        row_indices, col_indices = topk_indices // hw, topk_indices % hw
        finaledge = torch.zeros((b, t_1, self.k, 2), dtype=torch.int)
        for i in range(t_1):
            finaledge[:,i, :, 0] = row_indices[:, i,:]+ i* hw
            finaledge[:,  i, :, 1] = col_indices[:, i, :] + (i+span) * hw
        finaledge = finaledge.view(b, t_1*self.k, 2)
        finaledge_re = torch.stack((finaledge[:,:,1], finaledge[:,:,0]), dim=-1)
        finaledge = torch.cat((finaledge, finaledge_re), dim=1).permute(0,2,1).detach()
        x = rearrange(x, "b c v n-> b (v n) c")
        out = torch.zeros_like(x).to(x.device)
        for i in range(batch):
            out[i] = self.gconv(x[i], finaledge[i].to(x.device).long())
        x = out.permute(0,2,1).view(b, self.reduction_channel, tlen//b, h, w)
        x = self.up_conv(x).permute(0, 2, 1, 3, 4).contiguous().view(tlen, c, h, w)
        return x