from grad_expl import Explainer2

from sklearn.neural_network import MLPClassifier as NN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import sys
sys.path.append(r'C:\Users\rp13102\Documents\GitHub\experiment\ghent')
from sklearn.ensemble import RandomForestClassifier as RFC

import itertools
#from gcs import get_gcs
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def plot_density_kde(X, y, ax, density_estimator):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))

    cm = plt.cm.Blues
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
    
    Z = density_estimator.score_samples(newx)

    Z = Z.reshape(xx.shape)
    contour_plot = ax.contourf(xx, yy, np.exp(Z), 
                                 levels=20,
                                 cmap=cm, 
                                 alpha=.8)
    plt.colorbar(contour_plot, ax=ax)
    
    ax.scatter(X[:, 0], X[:, 1], c=y, 
                  cmap=cm_bright,
                  edgecolors='k',
                  zorder=2)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)

    #plt.xticks(())
    #plt.yticks(())
    ax.set_title('Shortest Path - Density Based Distances (DBD)')

    return mdl

def plot_decision_boundary(X, y, ax, clf, title='GPC'):
    h = 0.50
    xmin = np.min(X[:, 0])
    xmax = np.max(X[:, 0])
    
    ymin = np.min(X[:, 1])
    ymax = np.max(X[:, 1])
    extra = 1
    xx, yy = np.meshgrid(np.arange(xmin - extra, xmax + extra, h),
                         np.arange(ymin - extra, ymax + extra, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])
    
    #plt.figure()
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.yticks(())
    
    newx = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict_proba(newx)[:, 1]

    Z = Z.reshape(xx.shape)
    contour_plot = ax.contourf(xx, yy, Z, 
                               levels=20, 
                               cmap=cm, 
                               alpha=.8)

    plt.colorbar(contour_plot, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, 
               cmap=cm_bright,
               edgecolors='k',
               alpha=0.50,
               zorder=2)
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)
    ax.set_title(title)
    return xx, yy

def get_data(val):  
    val0, val1 = val
    n1 = 200
    t1 = 10 * np.random.random_sample(n1) - 0.50
    t0 = np.random.normal(0, 0.40, n1)
    x1 = np.vstack((t0, t1)).T
    y1 = np.ones(n1)
    
    n2 = 200
    t1 = 6 * np.random.random_sample(n2) + val1 #- 0.50
    t0 = np.random.normal(0.50, 0.50, n2)
    x2 = np.vstack((t1, t0)).T
    y2 = np.zeros(n2)
    
    mean = [3.50, 8.00]
    sigma = 0.50
    cov = sigma * np.identity(2)
    n3 = 100
    x3 = np.random.multivariate_normal(mean, cov, n3) + np.array([val0, 0])
    y3 = np.zeros(n3)
    
    X = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3))
    
    return X, y


#
val = [1, 0]

VALUES_0 = [-0.50, 0, 1]
VALUES_1 = [-0.50, 0, 2]
VALUES = list(itertools.product(*[VALUES_0, VALUES_1]))

VALUES = [[0, 0]]

for gamma in [10]:
#for rho in [1/2, 1, 3, 5]:#, 3, 5, 10]:
    #fig, ax_list_ = plt.subplots(nrows=3, ncols=3)
    #ax_list = [item for sublist in ax_list_ for item in sublist]
    
    for axis_index, val in enumerate(VALUES):
        #ax = ax_list[axis_index]
        fig, ax = plt.subplots()
        np.random.seed(99)
        X, y = get_data(val)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        #mdl = GPC(max_iter_predict=10000)
        #mdl = RFC(n_estimators=100)
        mdl = NN(max_iter=10000)
        mdl.fit(X, y)
    
        xx, yy = plot_decision_boundary(X, y, ax, mdl, title='NNC')
        
        
        xpl = Explainer2(mdl, X, y, 
                         step = 1e-4, 
                         n_steps = 100,
                         alpha = None,
                         sc_threshold = 0.0056,
                         sc_percentile = None,
                         norm_bool = 1,
                         step_size = 1,
                         threshold = None,
                         distance_measure='fro',
                         bool_return_grad = True,
                         rho = None,
                         whattouse='lap_dist',
                         gamma = gamma,
                         delta = None,
                         epsilon = None,
                         norm_bool_2=False,
                         radius = 1)
        
        starting_point = np.array([-1.00, 1.50])
        #starting_point = np.array([0.0, 8.0])
        target_instance = np.array([4.0, 0.0])
        clss = 1
        successes = []
        newx = np.c_[xx.ravel(), yy.ravel()]
        pr = mdl.predict(newx)
        L = np.where(y == 1)[0]
        J = np.where(pr == 1)[0]
        x_to_use = X
    
        I = list(set(L).intersection(J))
        #Z = mdl.predict_proba(newx)[:, 1]
        #Z = Z.reshape(xx.shape)
        U = np.zeros(x_to_use.shape)
        C = np.zeros((x_to_use.shape[0], 1))
        
    
        indices = []
        for idx in range(x_to_use.shape[0]): 
            print(idx)
            try:
                test_index = idx
                starting_point = x_to_use[test_index]
    
                target_class = np.array([1, 0])    
                p=xpl.explain(starting_point, 
                          ax,
                          target_class,
                          target_instance = target_instance
                          )
            
                if not isinstance(p, int):
                    v = np.linalg.norm(p)
                    p /= v
                    
                    U[idx] = p.ravel()
                    C[idx] = v
                    indices.append(idx)
    
                else:
                    U[idx] = np.array([0, 0])
                    C[idx] = 0
            except:
                pass
            
            #plt.quiver(starting_point[0], starting_point[1], p[0], p[1])
    
            
        #C = C - np.min(C)
        #C = C / np.max(C)
        #C = sigmoid(C)
        colors = C.ravel()
        colormap = cm.inferno
        norm = Normalize()
        norm.autoscale(colors)  
        ax.quiver(x_to_use[indices, 0], x_to_use[indices, 1], 
                  U[indices, 0], U[indices, 1], 
                   color=colormap(norm(colors)))

    
