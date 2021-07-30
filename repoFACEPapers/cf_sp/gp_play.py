#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:24:51 2019

@author: A897WD
"""

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

train_x = torch.linspace(0, 1, 10)
train_y = torch.sign(torch.cos(train_x * (4 * math.pi))).add(1).div(2)


from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

from gpytorch.mlls.variational_elbo import VariationalELBO

class gp_model(object):
    def __init__(self):
        pass
    def fit(self, train_x, train_y):
        # Initialize model and likelihood
        self.model = GPClassificationModel(train_x)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        
        
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the amount of training data
        mll = VariationalELBO(self.likelihood, self.model, train_y.numel())
        
        training_iter = 50
        for i in range(training_iter):
            # Zero backpropped gradients from previous iteration
            optimizer.zero_grad()
            # Get predictive output
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()
        return self
    
    def predict(self, test_x):
        # Go into eval mode
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            # Test x are regularly spaced by 0.01 0,1 inclusive
            # Get classification predictions
            observed_pred = self.likelihood(self.model(test_x))
        return observed_pred
            