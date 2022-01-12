import os

import torch
import torch.nn as nn
import numpy as np
from torch.functional import F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from data.dataqm93 import GraphReader
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


def collate(samples):
    graph_b, graph_c, label = map(list, zip(*samples))
    # graph_b, graph_c, label, lhomo, llumo, lgap, comp = map(list, zip(*samples))
    # graph_b, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graph_b)
    batched_graph_key = dgl.batch(graph_c)
    # print(batched_graph,labels)
    return batched_graph, batched_graph_key, torch.tensor(label)  #, torch.tensor(lhomo), torch.tensor(llumo), torch.tensor(lgap),torch.tensor(comp)
    # return batched_graph, torch.tensor(labels)


class DotProductPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score']


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2 + 4, 32, bias=True)
        self.bn = nn.BatchNorm1d(32)
        self.W2 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.W3 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.W4 = nn.Linear(16, 1)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        # score = self.W(torch.cat([h_u, h_v, edges.data['feature']], 1))
        # score = self.W2(score)
        edgesfeature = torch.cat([h_u, h_v, edges.data['feature']], 1)
        # print(len(h_u), len(h_v), len(edgesfeature))
        score = F.relu(self.bn(self.W(edgesfeature)))
        score = F.relu(self.bn1(self.W2(score)))
        score = F.relu(self.bn2(self.W3(score)))
        score = self.W4(score)
        # score = self.W2(score)
        return {'score': score}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)

            return g.edata['score']


class NKMLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2 + 12, 28, bias=True)
        self.bn = nn.BatchNorm1d(28)
        self.W2 = nn.Linear(28, 8)
        self.bn1 = nn.BatchNorm1d(8)
        self.W3 = nn.Linear(8, 1)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        # score = self.W(torch.cat([h_u, h_v, edges.data['feature']], 1))
        # score = self.W2(score)
        edgesfeature = torch.cat([h_u, h_v, edges.data['feature']], 1)
        # print(len(h_u), len(h_v), len(edgesfeature))
        score = F.relu(self.bn(self.W(edgesfeature)))
        score = F.relu(self.bn1(self.W2(score)))
        score = self.W3(score)
        # score = self.W2(score)
        return {'score': score}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)

            return g.edata['score']


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.eupdate1 = EdgeUpdate(hid_feats, 4, 32)
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')
        self.eupdate2 = EdgeUpdate(hid_feats, 8, 32)
        self.conv3 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        graph.edata['feature'] = torch.cat([graph.edata['feature'], self.eupdate1(graph, h)], 1)
        h = F.relu(self.conv2(graph, h))
        graph.edata['feature'] = torch.cat([graph.edata['feature'], self.eupdate2(graph, h)], 1)
        h = self.conv3(graph, h)
        return h


class Model(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hid_features, out_features)
        # self.prediction = DotProductPredictor()
        self.prediction = MLPPredictor(out_features, 1)

    def forward(self, g, x):
        # print("进入MOdel 的 forward")
        h = self.sage(g, x)
        return self.prediction(g, h)

class EdgeUpdate(nn.Module):
    def __init__(self, in_feats, edge_infeat, hid_feats):
        super().__init__()
        self.W = nn.Linear(in_feats * 2 + edge_infeat, hid_feats)
        self.W2 = nn.Linear(hid_feats, 4)
        # self.W3 = nn.Linear(10, 4)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        # print("h_v",h_v.size())
        edgesfeature = torch.cat([h_u, h_v, edges.data['feature']], 1)
        # print(edgesfeature.size())
        e_feats = F.relu(self.W(edgesfeature))
        e_feats = self.W2(e_feats)
        # e_feats = F.relu(self.W3(e_feats))
        return {'feature': e_feats}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['feature']

    # def forward(self, g, h):
    #     with g.local_scope():
    #         g.ndata['h'] = h
    #         g.apply_edges(self.apply_edges)
    #         return g.edata['feature']

class NKSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.bn = nn.BatchNorm1d(hid_feats)
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')
        self.bn1 = nn.BatchNorm1d(hid_feats)
        self.eupdate1 = EdgeUpdate(hid_feats, 4, 32)
        self.conv3 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')
        self.bn2 = nn.BatchNorm1d(hid_feats)
        self.conv4 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
        self.bn3 = nn.BatchNorm1d(out_feats)
        self.eupdate2 = EdgeUpdate(out_feats, 8, 32)


    def forward(self, graph, gk, inputs):
        h = self.conv1(gk, inputs)
        h = F.relu(self.bn(h))
        h = F.relu(self.bn1(self.conv2(gk, h)))
        graph.ndata['h'] = h
        graph.edata['feature'] = torch.cat([graph.edata['feature'], self.eupdate1(graph, h)], 1)
        h = F.relu(self.bn2(self.conv3(gk, h)))
        h = self.bn3(self.conv4(gk, h))
        # graph.ndata['h'] = h
        graph.ndata['h'] = h
        graph.edata['feature'] = torch.cat([graph.edata['feature'], self.eupdate2(graph, h)], 1)
        return h


class NKModel(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.sage = NKSAGE(in_features, hid_features, out_features)
        # self.prediction = DotProductPredictor()
        self.prediction = NKMLPPredictor(out_features, 1)

    def forward(self, g, gk, x):
        # print("进入MOdel 的 forward")
        h = self.sage(g, gk, x)
        return self.prediction(g, h)

