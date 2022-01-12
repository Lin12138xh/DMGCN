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
from  modeltest.model4 import NKModel, MLPPredictor, collate


data_loader = DataLoader(GraphReader('../data/r11train.txt', 123), batch_size=2048, shuffle=True,
                         collate_fn=collate)
test_loader = DataLoader(GraphReader('../data/r11test.txt', 123), batch_size=1, shuffle=False,
                         collate_fn=collate)

model = NKModel(19, 32, 5)
model_key = MLPPredictor(19, 1)
# model = MLPPredictor(14, 1)
loss_func = nn.L1Loss()
opt = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.01},
    {'params': model_key.parameters(), 'lr': 0.01}
])
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=90, gamma=0.1)
model.train()
model_key.train()
since = time.time()
epoch_losses = []
loss_func2 = nn.MSELoss()
last20=0
for epoch in range(200):
    epoch_loss = 0
    epoch_time = time.time()

    for iter, (graph, graph_key, label) in enumerate(data_loader):
        label2 = graph.edata['label']
        nk_mask = graph.edata['nk_mask']
        k_mask = graph.edata['key_mask']
        pred = model(graph, graph_key, graph.ndata['feature'])

        label2_key = graph_key.edata['label']
        pred_key = model_key(graph_key, graph_key.ndata['feature'])

        pred = pred.view(-1)
        label2 = label2.view(-1)
        pred_key = pred_key.view(-1)
        label2_key = label2_key.view(-1)
        pred[k_mask] = pred_key
        loss1 = loss_func(pred[nk_mask].float(), label2[nk_mask].float())
        loss2 = loss_func2(pred_key.float(), label2_key.float())**0.5
        loss =0.4*loss2 + 0.6*loss1

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= len(data_loader)
    epoch_elapsed = time.time() - epoch_time
    print('Epoch {}, epoch_loss {:.4f}, time:{:.0f}m {:.0f}s'.format(epoch, epoch_loss, epoch_elapsed//60, epoch_elapsed%60))
    print('-'*10)
    epoch_losses.append(epoch_loss)
time_elapsed = time.time() - since
print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
save_dir = "E://pythonPreoject/MSGCN/saveModel"
save_filename = 'M4_11_params.pkl'
save_filenamek = 'M4_11k_params.pkl'
PATH = os.path.join(save_dir, save_filename)
PATHK = os.path.join(save_dir, save_filenamek)
torch.save(model.state_dict(), PATH)
torch.save(model_key.state_dict(), PATHK)
plt.plot(epoch_losses)
plt.xlabel("Epoch Number")
plt.ylabel("Epoch_loss")
fig1 = plt.gcf()
fig1.set_size_inches(4.8, 4.2)
fig1.savefig('figure1.tif', dpi=300)
plt.show()
model.eval()
model_key.eval()
total_loss = 0
total_loss1 = 0
total_loss2 = 0
total_loss3 = 0
total_losskeyr = 0
total_time = 0
count = 0
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
    loss = (loss1 + loss2)/2.0
    lossrsme = loss_func2(pred, label2)**0.5
    losskeyr = loss_func2(pred_key, label2_key)**0.5
    print(label)
    test_time = time.time()-since
    total_time += test_time

    print('test{}, loss {:.4f}, loss {:.4f}, atom_count{}, {:.4f}ms'.format(iter, loss, lossrsme, len(label2), test_time))
    print('-' * 20)
    total_loss += loss
    total_loss1 += loss1
    total_loss2 += loss2
    total_loss3 += lossrsme
    total_losskeyr += losskeyr

print('average MAE：{:.4f}，average rmse：{:.4f}'.format(total_loss/len(test_loader), total_loss3/len(test_loader)))
# print('平均时间:', total_time/len(test_loader))