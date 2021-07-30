from grad_expl import Explainer2

from sklearn.neural_network import MLPClassifier as NN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier as RFC

import matplotlib as mpl
import itertools
import sys
sys.path.append(r'C:\Users\rp13102\Documents\GitHub\experiment\ghent')

#from gcs import get_gcs

def plot_density_kde(X, y, ax, mdl):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))

    cm = plt.cm.Blues
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
    
    Z = mdl.score_samples(newx)

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


def plot_decision_boundary(X, y, ax, clf, title='GPC'):
    h = 0.1
    xmin = np.min(X[:, 0])
    xmax = np.max(X[:, 0])
    
    ymin = np.min(X[:, 1])
    ymax = np.max(X[:, 1])
    extra = 0.50
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
    
#    newx = np.c_[xx.ravel(), yy.ravel()]
#    Z = clf.predict_proba(newx)[:, 1]
#
#    Z = Z.reshape(xx.shape)
#    contour_plot = ax.contourf(xx, yy, Z, 
#                               levels=20, 
#                               cmap=cm, 
#                               alpha=.8)
# 
#    plt.colorbar(contour_plot, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, 
               cmap=cm_bright,
               edgecolors='k',
               alpha=0.50,
               zorder=1)
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)
    ax.set_title(title)
    
def get_data(val):  
    val0, val1 = val
    t = 3
    n1 = t*100; n2 = t*100; n3 = t*50
    t1 = 10 * np.random.random_sample(n1) - 0.50
    t0 = np.random.normal(0, 0.40, n1)
    x1 = np.vstack((t0, t1)).T
    y1 = np.ones(n1)
    
    t1 = 6 * np.random.random_sample(n2) + val1 #- 0.50
    t0 = np.random.normal(0.50, 0.50, n2)
    x2 = np.vstack((t1, t0)).T
    y2 = np.zeros(n2)
    
    mean = [3.50, 8.00]
    sigma = 0.50
    cov = sigma * np.identity(2)
    x3 = np.random.multivariate_normal(mean, cov, n3) + np.array([val0, 0])
    y3 = np.zeros(n3)
    
    X = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3))
    
    return X, y


float_formatter = lambda x: "%.2f" % x
float_formatter_arr = lambda x: np.array([float_formatter(item) for item in x])
scaler = StandardScaler()

VALUES_0 = [0, 1]
VALUES_1 = [1, 2]
VALUES = list(itertools.product(*[VALUES_0, VALUES_1]))

VALUES = [[0, 1]]
# =============================================================================
# X, y, cols, codes = get_gcs()
# X = X.astype(float)
# X = scaler.fit_transform(X)
# =============================================================================
for stepsize in [0.05, 0.10, 0.25]:#[1,3,5,10]:
    for gamma in [0.001, 0.01]:
        for sim_step_size in [0.01, 0.1, 1, 3]:
            #fig, ax_ = plt.subplots(nrows=2, ncols=2)
            #ax__ = [item for sublist in ax_ for item in sublist]
            
            #ax = ax_
            
            epsilon = 0.01
            for I, val in enumerate(VALUES):
                #ax = ax__[I]
                fig, ax = plt.subplots()

                #mdl = NN(max_iter=10000)
                np.random.seed(99)

                X, y = get_data(val)
                #X = X - np.mean(X, axis=0)
                #X = scaler.fit_transform(X)
                mdl = GPC(1.0 * RBF(1.0))
                #mdl = RFC(n_estimators=300)
                mdl.fit(X, y)
                
                plot_decision_boundary(X, y, ax, mdl, title='Neural Network Classification')
                
                
                xpl = Explainer2(mdl, X, y, 
                                 step = 1e-2, 
                                 n_steps = 500,
                                 alpha = None,
                                 sc_threshold = 'a',
                                 sc_percentile = None,
                                 norm_bool = 1,
                                 step_size = stepsize,
                                 threshold = None,
                                 distance_measure='fro',
                                 bool_return_grad = 0,
                                 rho = None,
                                 whattouse='projected',
                                 gamma = gamma,
                                 delta = None,
                                 epsilon = epsilon,
                                 norm_bool_2=False,
                                 sigma= None,
                                 radius = 'a',
                                 sim_step_size=sim_step_size)
                
                #starting_point = np.array([-1.00, 1.25])
                starting_point = np.array([0.0, 8.0])
                target_instance = np.array([4.0, 0.0])
                clss = 1
                successes = []
                starting_points = np.array([[0, 10],
                                            [0, 9],
                                            [0, 8],
                                            [0, 7],
                                            [0, 6],
                                            [0, 5],
                                            [0, 4],
                                            [0, 3],
                                            [0, 2],
                                            [0, 0]], dtype=float)
                for idx in range(1):#starting_points.shape[0]): 
                    #test_index = I[idx]
                    #starting_point = X[test_index, :].reshape(1, -1)
                    #target_class = int(not y[test_index])
                    p=xpl.explain(starting_point, 
                                  ax,
                                  np.array([1, 0]),
                                  toplot=True
                                  )
                
                ax.set_title([epsilon, gamma, sim_step_size])
                #ax[I].set_title(val)
                # =============================================================================
                #     pr = mdl.predict_proba(p.reshape(1, -1))
                #     s = float_formatter_arr(np.ravel(scaler.inverse_transform(p.reshape(1, -1))))
                #     sp = float_formatter_arr(scaler.inverse_transform(X[test_index, :]))
                # 
                #     print(sp)
                #     print(s)
                #     print(pr)
                #     print('\n')
                # =============================================================================


