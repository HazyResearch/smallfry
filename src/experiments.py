import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
import matplotlib.pyplot as plt
import compress
import utils

# inspect spectrum of ppmi matrix
def ppmi_spectrum():
    coo = load_npz('C:\\Users\\avnermay\\Babel_Files\\smallfry\\coo.npz')
    R = np.loadtxt('C:\\Users\\avnermay\\Babel_Files\\smallfry\\R.txt')
    dim = coo.shape[0]
    n = 5000
    D = coo.sum()
    w = np.asarray(np.reshape(coo.sum(axis=1),(dim,1)))
    c = np.asarray(np.reshape(coo.sum(axis=0),(1,dim)))
    coo = np.array(coo[:n,:n].todense())
    # for k > 1 negative samples, should divide inside by k
    num_zeros = np.sum(w[:n]*c[:,:n]==0.0)
    pmi = np.log(D * coo / (w[:n]*c[:,:n]))
    ppmi = np.maximum(pmi,0)
    u,s,v = np.linalg.svd(ppmi)
    plt.figure(1)
    plt.subplot(121)
    plt.plot(s)
    plt.yscale('log')
    plt.subplot(122)
    plt.plot(np.abs(R))
    plt.yscale('log')
    plt.show()
    # display('test')

def deltas_vs_precision_and_lambda(gaussian):
    n = 1000
    d = 30
    lim = 1.0/np.sqrt(d)
    X = (np.random.randn(d,n) * 0.5 * lim if gaussian else
        np.random.uniform(low=-lim, high=lim, size=(n,d)))
    K = X @ X.T
    bs = [1,2,4,8,16]
    lambdas = [2**(-32), 2**(-24), 2**(-16), 2**(-12), 2**(-8),2**(-4), 2**(-2), 2**0, 2**1, 2**2, 2**4, 2**8, 2**12, 2**16, 2**20]
    delta1s = np.zeros((len(bs), len(lambdas)))
    delta2s = np.zeros((len(bs), len(lambdas)))
    # Gather delta1 and delta2 for different precisions and lambdas
    for i_b,b in enumerate(bs):
        Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=gaussian, stochastic_round=True)
        Kq = Xq @ Xq.T
        for i_l,lam in enumerate(lambdas):
            delta1s[i_b,i_l], delta2s[i_b,i_l], _ = utils.delta_approximation(K, Kq, lambda_ = lam)
    plt.figure(1)
    plt.subplot(121)
    plot_deltas(delta1s, bs, lambdas, n, d, 'abs(Delta_1)', gaussian)
    plt.subplot(122)
    plot_deltas(delta2s, bs, lambdas, n, d, 'abs(Delta_2)', gaussian)
    plt.show()
    print(1)

def plot_deltas(delta_results, bs, lambdas, n, d, ylabel, gaussian, xlog=True, ylog=True):
    nplams = np.array(lambdas)
    x = 2**16 * nplams
    gaussian_str = 'Gaussian' if gaussian else 'Uniform'
    for i_b,b in enumerate(bs):
        if np.alltrue(delta_results[i_b,:] >= 0):
            print('b={} all positive --- {},{}'.format(b, gaussian_str, ylabel))
        else:
            print('b={} contains negative values --- {},{}'.format(b, gaussian_str, ylabel))
        plt.plot(2**b * nplams, np.abs(delta_results[i_b,:]), marker='o', linestyle='dashed')
        #plt.plot(2.0**b/((2.0**b-1)**2 * nplams), delta2s[i_b,:] - 1.0/((2**b-1)**2 * nplams) )
        # plt.plot(nplams, delta2s[i_b,:])
    plt.plot(x, np.sqrt(2*np.log(n/0.1)/d) * 5 * n / x, marker='o', linestyle='dashed')
    plt.legend(['1','2','4','8','16','bound'])
    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    plt.xlabel('2^b * lambda')
    plt.ylabel(ylabel)
    title = '{}: {} vs. {}'.format(gaussian_str, ylabel, '2^b * lambda')
    plt.title(title)

def clipping_effect():
    n = 1000
    d = 30
    # lim = 1.0/np.sqrt(d)
    # X = np.random.randn(d,n) * 0.5 * lim
    X = np.random.randn(d,n)
    K = X @ X.T
    # We are trying to understanding plateau at small 2^b * lam, so take very small lam
    lam = 2**(-32)
    max_r = np.max(X)
    rs = np.linspace(0,max_r,num=20)
    delta1s = np.zeros(len(rs))
    delta2s = np.zeros(len(rs))
    for i_r,r in enumerate(rs):
        Xq = np.clip(X, -r, r)
        Kq = Xq @ Xq.T
        delta1s[i_r], delta2s[i_r], _ = utils.delta_approximation(K, Kq, lambda_ = lam)
    plt.figure(1)
    # plt.subplot(121)
    plt.plot(rs, 1 - delta1s, marker='o', linestyle='dashed', color='b')
    plt.plot(rs, 1 + delta2s, marker='o', linestyle='dashed', color='b')
    plt.xlabel('clip value')
    plt.legend(['1-Delta1','1+Delta2'])
    plt.title('Effect of clipping on Delta1 and Delta2 for random Gaussian data.')
    # plt.subplot(122)
    # plt.plot(rs, delta2s)
    # plt.xlabel('clip value')
    # plt.ylabel('Delta_2')
    plt.show()
    print(1)

def clipping_with_quantization_effect():
    n = 1000
    d = 30
    X = np.random.randn(d,n)
    K = X @ X.T
    bs = [1,2,4,8,32]
    colors = ['b','g','r','y','k']
    # We are trying to understanding plateau at small 2^b * lam, so take very small lam
    lam = 2**(-32)
    max_r = np.max(X)
    rs = np.linspace(0,max_r,num=20)
    delta1s = np.zeros((len(bs),len(rs)))
    delta2s = np.zeros((len(bs),len(rs)))
    for i_b,b in enumerate(bs):
        for i_r,r in enumerate(rs):
            Xq = compress._compress_uniform(X, b, r, stochastic_round=False)
            Kq = Xq @ Xq.T
            delta1s[i_b,i_r], delta2s[i_b,i_r], _ = utils.delta_approximation(K, Kq, lambda_ = lam)

    plt.figure(1)
    # plt.subplot(121)
    legend = []
    for i_b,b in enumerate(bs):
        plt.plot(rs, 1.0/(1 - delta1s[i_b,:]), marker='o', linestyle='dashed', color=colors[i_b])
        plt.plot(rs, 1 + delta2s[i_b,:], marker='x', linestyle='dashed', color=colors[i_b])
        legend.append('1/(1-Delta1) (b={})'.format(b))
        legend.append('1+Delta2 (b={})'.format(b))
    plt.xlabel('clip value')
    plt.legend(legend)
    plt.title('Effect of clipping + quantization on (Delta1,Delta2) for Gaussian data.')
    plt.yscale('log')
    plt.show()
    print(1)

if __name__ == '__main__':
    # ppmi_spectrum()
    # deltas_vs_precision_and_lambda(False)
    # deltas_vs_precision_and_lambda(True)
    # clipping_effect()
    clipping_with_quantization_effect()