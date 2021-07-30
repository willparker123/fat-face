from grad_expl import Explainer2

from sklearn.neural_network import MLPClassifier as NN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
import matplotlib as mpl
import itertools
import sys
#sys.path.append(r'C:\Users\rp13102\Documents\GitHub\experiment\ghent')

#from gcs import get_gcs

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
               zorder=1)
    ax.grid(color='k', linestyle='-', linewidth=0.50, alpha=0.75)
    ax.set_title(title)
    
def get_data(val):  
    val0, val1 = val
    n1 = 100
    t1 = 10 * np.random.random_sample(n1) - 0.50
    t0 = np.random.normal(0, 0.40, n1)
    x1 = np.vstack((t0, t1)).T
    y1 = np.ones(n1)
    
    n2 = 100
    t1 = 6 * np.random.random_sample(n2) + val1 #- 0.50
    t0 = np.random.normal(0, 0.50, n2)
    x2 = np.vstack((t1, t0)).T
    y2 = np.zeros(n2)
    
    mean = [3.50, 8.00]
    sigma = 0.50
    cov = sigma * np.identity(2)
    n3 = 50
    x3 = np.random.multivariate_normal(mean, cov, n3) + np.array([val0, 0])
    y3 = np.zeros(n3)
    
    X = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3))
    
    return X, y


float_formatter = lambda x: "%.2f" % x
float_formatter_arr = lambda x: np.array([float_formatter(item) for item in x])
scaler = StandardScaler()

VALUES_0 = [-0.50, 0]
VALUES_1 = [-0.50, 1]
VALUES = list(itertools.product(*[VALUES_0, VALUES_1]))

# =============================================================================
# X, y, cols, codes = get_gcs()
# X = X.astype(float)
# X = scaler.fit_transform(X)
# =============================================================================

use_all = 0
#VALUES = [[1, 0]]
np.random.seed(99)
gammas = np.logspace(-1, 1, 5)
deltas = np.logspace(-1, 0, 4)
#gammas = [0.01, 0.1]
#deltas = [0.01, 0.1]
#gammas = [1]
#deltas = [0.50]
params_list = list(itertools.product(*[gammas, deltas]))

X, y = get_data([1, 0])
n_samples = X.shape[0]
X = scaler.fit_transform(X)
mdl = GPC(1.0 * RBF(1.0))
mdl.fit(X, y)
pr = mdl.predict(X)    
J = np.where(pr == 1)[0] 
K = np.where(y == 1)[0]
I = list(set(J).intersection(K))
scores = {item: 0 for item in params_list}
for params in params_list:
    print(params)
    gamma, delta = params
    xpl = Explainer2(mdl, X, y, 
                     step = 1e-6, 
                     n_steps = 500,
                     alpha = None,
                     sc_threshold = 0.0056,
                     sc_percentile = None,
                     norm_bool = 1,
                     step_size = 0.1,
                     threshold = 0.25,
                     distance_measure='fro',
                     bool_return_grad = False,
                     rho = 4,
                     gamma = gamma,
                     norm_bool_2=1,
                     whattouse=5,
                     delta = delta, 
                     lambda_1 = None,
                     lambda_2 = None)
    
    for idx, value in enumerate(I):
        if idx%20 == 0:
            print('\n')
            print(idx)
            for item, val in scores.items():
                print(item, val)
        if idx > 100:
            break
        starting_point = X[value, :].reshape(1, -1)
        #fig, ax = plt.subplots()
        #ax.scatter(X[:, 0], X[:, 1],c=y)
        #target_class = int(not y[test_index])
        p=xpl.explain(starting_point,
                      None,
                      np.array([1, 0]),
                      target_instance = None,
                      toplot = False,
                      )
        pr = mdl.predict_proba(p.reshape(1, -1))
       # print(pr)
        if pr[0][0] >= 0.50:
            scores[params] += 1
            