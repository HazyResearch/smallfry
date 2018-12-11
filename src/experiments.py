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

def deltas_vs_precision_and_lambda():
    n = 1000
    d = 30
    lim = 1.0/np.sqrt(d)
    X = np.random.uniform(low=-lim, high=lim, size=(n,d))
    K = X @ X.T
    bs = [1,2,4,8,16]
    lambdas = [2**(-32), 2**(-24), 2**(-16), 2**(-12), 2**(-8),2**(-4), 2**(-2), 2**0, 2**1, 2**2, 2**4, 2**8, 2**12, 2**16, 2**20]
    nplams = np.array(lambdas)
    x = 2**4 * nplams
    delta1s = np.zeros((len(bs), len(lambdas)))
    delta2s = np.zeros((len(bs), len(lambdas)))
    for i_b,b in enumerate(bs):
        Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=False, stochastic_round=True)
        Kq = Xq @ Xq.T
        for i_l,lam in enumerate(lambdas):
            delta1s[i_b,i_l], delta2s[i_b,i_l], _ = utils.delta_approximation(K, Kq, lambda_ = lam)
    plt.figure(1)
    plt.subplot(121)
    for i_b,b in enumerate(bs):
        plt.plot(2**b * nplams, delta1s[i_b,:])
        # plt.plot(2.0**b/((2.0**b-1)**2 * nplams), delta1s[i_b,:])
        # plt.plot(nplams, delta1s[i_b,:])
    plt.plot(x, np.sqrt(2*np.log(n/0.1)/d) * 5 * n / x)
    plt.legend(['1','2','4','8','16','bound'])
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(122)
    for i_b,b in enumerate(bs):
        plt.plot(2**b * nplams, delta2s[i_b,:])
        #plt.plot(2.0**b/((2.0**b-1)**2 * nplams), delta2s[i_b,:] - 1.0/((2**b-1)**2 * nplams) )
        # plt.plot(nplams, delta2s[i_b,:])
    plt.plot(x, np.sqrt(2*np.log(n/0.1)/d) * 5 * n / x)
    plt.legend(['1','2','4','8','16','bound'])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    print(1)

if __name__ == '__main__':
    # ppmi_spectrum()
    deltas_vs_precision_and_lambda()