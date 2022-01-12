import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from data.dataqm93 import GraphReader
import time
import matplotlib.pyplot as plt
from  modeltest.model4 import NKModel, MLPPredictor, collate
test_loader = DataLoader(GraphReader('../data/r11test.txt', 123), batch_size=1, shuffle=False,
                         collate_fn=collate)
# print(len(data_loader))
# model_key = Model(14, 28, 5)
# savapath = "E://pythonPreoject/MAGCN/propTestData1"
model = NKModel(19, 32, 5)
model_key = MLPPredictor(19, 1)
PATH = "../saveModel/M4_11_params.pkl"
model.load_state_dict(torch.load(PATH))
model.eval()

PATHK = "../saveModel/M4_11K_params.pkl"
model_key.load_state_dict(torch.load(PATHK))
model_key.eval()
loss_func = nn.L1Loss()
loss_func2 = nn.MSELoss(reduce=True, size_average=True)
total_loss = 0
total_loss1 = 0
total_loss2 = 0
total_loss3 = 0
total_losskeyr = 0
tloss = []
total_time = 0
count = 0
number = 0

def cal(true, predi):
    result = [abs(predi.detach().numpy()[i] - true.detach().numpy()[i]) for i in range(len(predi))]
    return result

for iter, (graph, graph_key, label) in enumerate(test_loader):
    since = time.time()
    label2 = graph.edata['label']
    pred = model(graph, graph_key, graph.ndata['feature'])
    nk_mask = graph.edata['nk_mask']
    k_mask = graph.edata['key_mask']
    pred = pred.view(-1)
    label2 = label2.view(-1)

    pred_key = model_key(graph_key, graph_key.ndata['feature'])
    label2_key = graph_key.edata['label']
    pred_key = pred_key.view(-1)
    label2_key = label2_key.view(-1)
    pred[k_mask] = pred_key
    loss1 = loss_func(pred[nk_mask], label2[nk_mask])
    loss2 = loss_func(pred_key, label2_key)
    lossrsme = loss_func2(pred, label2)**0.5
    losskeyr = loss_func2(pred_key, label2_key) ** 0.5
    number += len(pred)
    loss = (loss1 + loss2)/2.0
    ld1 = cal(pred_key, label2_key)
    tloss.extend(ld1)

    test_time = time.time()-since
    total_time += test_time
    print('file:', label)

    print("keylosss:", loss2, "nklosss:", loss1)
    print('test{}, loss {:.4f}, atom_count{}, {:.4f}ms'.format(iter, loss, len(label2), test_time))
    total_loss += loss
    total_loss1 += loss1
    total_loss2 += loss2
    total_loss3 += lossrsme
    total_losskeyr += losskeyr
    s = '%06d' % label
    pred1 = pred.detach().numpy()

print('average MAE：{:.4f}，average RMSE：{:.4f}'.format(total_loss/len(test_loader), total_loss3/len(test_loader)))

