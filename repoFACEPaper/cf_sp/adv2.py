#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:33:49 2019

@author: A897WD
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as npo
plt.style.use('seaborn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from advertorch.attacks import L1PGDAttack, LinfPGDAttack, L2PGDAttack, LBFGSAttack, CarliniWagnerL2Attack

def plot_decision_boundary(X, y, clf):
    fig, ax = plt.subplots()
    h = 0.25
    
    xmin, ymin = npo.min(X, axis=0)
    xmax, ymax = npo.max(X, axis=0)
    xx, yy = npo.meshgrid(npo.arange(xmin, xmax, h),
                         npo.arange(ymin, ymax, h))

    cm = plt.cm.Blues
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
       
    cm = plt.cm.RdYlBu
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    

    newx = npo.c_[xx.ravel(), yy.ravel()]
    newx = Variable(torch.tensor(newx, dtype=torch.float32))
    Z = clf.predict(newx)[:, 1].detach().numpy()
    #newx = Variable(torch.from_numpy(newx)).float()
    #newx = Variable(torch.from_numpy(newx).type(torch.FloatTensor).to(device))
    
    #outputs = clf(newx)
    #Z = clf.predict(newx)
    #Z = torch.softmax(outputs.data, 1)[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, 
                levels=20, 
                cmap=cm, 
                alpha=.8)

    #plt.colorbar(contour_plot, ax=ax)

    ax.grid(color='k', 
            linestyle='-', 
            linewidth=0.50, 
            alpha=0.75)
       
    ax.scatter(X[:, 0], X[:, 1], c=y, 
           cmap=cm_bright,
           edgecolors='k',
           zorder=2)
    
    return ax

m = nn.Softmax(dim=0)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)         
        self.fc3 = nn.Linear(hidden_size, hidden_size)  
        self.fc4 = nn.Linear(hidden_size, hidden_size)         
        self.fc5 = nn.Linear(hidden_size, hidden_size)  
        self.fc6 = nn.Linear(hidden_size, num_classes)  
    
        self.relu1 = nn.ReLU() 
        self.relu2 = nn.ReLU() 
        self.relu3 = nn.ReLU() 
        self.relu4 = nn.ReLU() 
        self.relu5 = nn.ReLU() 
        
    def get_layers(self, x):
        out1 = self.relu(self.fc1(x))        
        out2 = self.relu(self.fc2(out1))
        out3 = self.relu(self.fc3(out2))
        out4 = self.relu(self.fc4(out3))
        out5 = self.relu(self.fc5(out4))
        out6 = self.fc6(out5)
        return (out1, out2, out3, out4, out5, out6)
    
    def predict(self, x):
        #x = Variable(torch.tensor(x).type(torch.FloatTensor).to(device))
        return m(self.forward(x))
    
    def predict_proba(self, x):
        #x = Variable(torch.tensor(x).type(torch.FloatTensor).to(device))
        return self.forward(x)
    
    def forward(self, x):
        out1 = self.relu1(self.fc1(x))        
        out2 = self.relu2(self.fc2(out1))
        out3 = self.relu3(self.fc3(out2))
        out4 = self.relu4(self.fc4(out3))
        out5 = self.relu5(self.fc5(out4))
        out6 = self.fc6(out5)
        return out6

class net_supp(object):
    def __init__(self):       
        # Device configuration
        
        # Hyper-parameters 
        input_size = 2
        hidden_size = 10
        num_classes = 2

        #lb_enc = OneHotEncoder()
        #y = lb_enc.fit_transform(y_.reshape(-1, 1)).toarray()
        self.model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        
    def fit(self, X, y):
        num_epochs = 100
        batch_size = 50
        learning_rate = 1e-3
        self.criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  

        xtrain, xtest, ytrain, ytest = train_test_split(X, y)
        
        xtrain = Variable(torch.from_numpy(xtrain).type(torch.FloatTensor).to(device))
        ytrain = Variable(torch.from_numpy(ytrain).long().to(device))
            
        
        xtest = Variable(torch.from_numpy(xtest).type(torch.FloatTensor).to(device))
        ytest = Variable(torch.from_numpy(ytest).long().to(device))
            
            
        for epoch in range(num_epochs):
            # X is a torch Variable
            permutation = torch.randperm(xtrain.size()[0])
        
            for i in range(0,xtrain.size()[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = xtrain[indices], ytrain[indices]
        
                # in case you wanted a semi-full example
                outputs = self.model.forward(batch_x)
                loss = self.criterion(outputs,batch_y)
        
                loss.backward()
                optimizer.step()
                
            if (i%20 == 0):
                with torch.no_grad():
                    correct = 0
                    total = 0
                    outputs = self.model(xtest)
                    _, predicted = torch.max(outputs.data, 1)
                    total += ytest.size(0)
                    correct += (predicted == ytest).sum().item()
        return self.model
    
    def get_adv(self, x, target_sc, penalty, mad, threshold = 0.50):
        mad = Variable(torch.tensor(mad, dtype=torch.float).to(device))
        cln_data = Variable(torch.tensor(x.reshape(1, -1), dtype=torch.float).to(device))
        if target_sc == 0:
            target = Variable(torch.tensor(npo.array([1, 0]).reshape(1, -1), dtype=torch.float, device=device))
            #target = torch.tensor([[1, 0]], dtype=torch.float).to(device)
        else:
            target = Variable(torch.tensor(npo.array([0, 1]).reshape(1, -1)).type(torch.FloatTensor).to(device))
        #target = torch.tensor(npo.array([1, 0]).reshape(1, -1), dtype=torch.float, device=device)

        pr = self.model.predict(cln_data)[0][target_sc]
        #target = target.detach().clone()
        delta = Variable(torch.zeros_like(cln_data).to(device))
        delta.requires_grad_(True)
# =============================================================================
#         delta = torch.tensor(npo.array([1, 1]).reshape(1, -1), 
#                              dtype=torch.float, 
#                              device=device, 
#                              requires_grad=True)
# =============================================================================

        counter = 0
        #loss_fn = nn.CrossEntropyLoss(reduction="sum")
        loss_fn = nn.MSELoss(reduction="sum")
        step_size = 1e-3 
# =============================================================================
#         grads = {}
#         def save_grad(name):
#             def hook(grad):
#                 grads[name] = grad
#             return hook
# =============================================================================
        #fig, ax = plt.subplots()
        #ax.scatter(t[0][0], t[0][1], marker='x', s=300)
        while pr < threshold:
            counter += 1
            input_var = cln_data + delta
            output = self.model.forward(input_var)
            loss0 = loss_fn(output, target)
            #print(output, target)
            #loss0 = loss_fn(output, target)
            loss1 = torch.tensor(0., requires_grad=True)
            for idx in range(2):
                loss1 = loss1 + torch.abs(delta[0][idx]) / mad[idx]
                
            #print(loss0, loss1)
            loss = -loss0 - penalty * loss1
            #delta.register_hook(grads['delta'])
            
            loss.backward()
            #delta = delta + step_size * grads['delta']
            grad = delta.grad.data
            qn = torch.norm(grad, p=2, dim=1).detach()
            grad = grad.div(qn.expand_as(grad))
            delta.data = delta.data + step_size * grad
            delta.grad.data.zero_()
            #t = (cln_data + delta).detach().numpy()
            #ax.scatter(t[0][0], t[0][1], marker='x', s=300)
            input_var = cln_data + delta
            pr = self.model.predict(input_var)[0][target_sc]
            if counter > 1e5: break
        print(pr)
        return (cln_data + delta).detach().numpy()
    
    def get_adv_(self, x, target, penalty, mad):
        adversary = L1PGDAttack(
                    self.model, 
                    loss_fn=nn.MSELoss(reduction="sum"),
                    #num_classes=2,
                    eps=3.,
                    nb_iter=250,
                    clip_min=0, 
                    clip_max=10.0,
                    targeted=True,
                    penalty=penalty,
                    mad=mad
                    )
        cln_data = Variable(torch.from_numpy(x.reshape(1, -1)).type(torch.FloatTensor).to(device))
        target = Variable(torch.from_numpy(npo.array([target])).float().to(device))
        return adversary.perturb(cln_data, target)


# =============================================================================
# ax = plot_decision_boundary(X, y, model)
# ax.scatter(xtrain[0, 0], xtrain[0, 1], marker='x')
# cln_data = xtrain[0, :].reshape(1, -1)
# true_label = torch.tensor([1]) * ytrain[0]
# target = torch.tensor([1]) * int(not(ytrain[0]))
# from advertorch.attacks import LinfPGDAttack
# 
# 
# 
# adv_untargeted = adversary.perturb(cln_data, true_label)
# ax.scatter(adv_untargeted[0][0], adv_untargeted[0][1], marker='x')
# adversary.targeted = True
# adv_targeted = 
# ax.scatter(adv_targeted[0][0], adv_targeted[0][1], marker='x')
# =============================================================================

# =============================================================================
# X, y = datasets.make_moons(n_samples = 1000,
#                            noise = 0.10)
# mdl = net_supp()
# mdl.fit(X, y)
# mdl.get_adv(X[0, :], int(not(y[0])))
# =============================================================================
