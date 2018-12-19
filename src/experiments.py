import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
import matplotlib.pyplot as plt
import compress
import utils
from plotter import save_plot

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

def flat_spectrum_vs_generalization(tight):
    # We show that when the regularizer is small relative to the smallest
    # non-zero eigenvalue of XX^T, the generalization performance of the
    # regularized model is close to that of the optimal model (we are
    # considering the case of fixed design regression with no noise, so lambda=0
    # is optimal, and gets 0 loss).  We plot the normalized MSE loss
    # (||Y-Y_pred||^2/||Y||^2)) as a function of a = lambda/eig_min
    # (this is the same 'a' as in Thm 1).
    # We consider 2 cases: 
    # (1) Y is well aligned with eigenvector of smallest eigenvalue (tight bound).
    # (1) Y is well aligned with eigenvector of largest eigenvalue (loose bound).
    X = np.random.randn(1000,2)
    sigmas = [.0001, .001, .01, .1, 1, 10]
    lambdas = np.logspace(-6,8,15)
    results = np.zeros((len(sigmas),len(lambdas)))
    a_vals = np.zeros((len(sigmas),len(lambdas)))
    c_vals = np.zeros((len(sigmas),len(lambdas)))
    for s_i,sigma in enumerate(sigmas):
        X_curr = np.vstack((sigma * X[:,0], sigma * X[:,1])).T
        Y = X_curr[:,1] if tight else X_curr[:,0]
        for l_i,lam in enumerate(lambdas):
            w = np.linalg.inv(lam * np.eye(2) + X_curr.T @ X_curr) @ (X_curr.T @ Y)
            Y_pred = X_curr @ w
            results[s_i,l_i] = np.sum((Y-Y_pred)**2)/np.sum(Y**2)
            eigs,_ = np.linalg.eigh(X_curr.T @ X_curr)
            eigs = np.sort(eigs)
            eig_min = eigs[0]
            eig_max = eigs[1]
            a_vals[s_i,l_i] = lam/eig_min
            c_vals[s_i,l_i] = eig_max/eig_min
    plt.figure(1)
    for i,_ in enumerate(sigmas):
        plt.plot(a_vals[i,:],results[i,:])
    a = np.logspace(-11,0,40)
    plt.plot(a,a**2, marker='o', linestyle='dashed')
    # plt.plot(a, a**2 / (c_vals[0,0] + a)**2, marker='o', linestyle='dashed')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['sigma = .0001', 'sigma = .001', 'sigma = .01', 'sigma = .1', 'sigma = 1', 'sigma = 10', 'a^2 (upper bound)'])
    # plt.legend(['sigma = .0001', 'sigma = .001', 'sigma = .01', 'sigma = .1', 'sigma = 1', 'sigma = 10', 'a^2 (upper bound)', 'a^2/(c+a)^2 (lower bound)'])
    plt.xlabel('a (equal to lambda/eig_min)')
    plt.ylabel('Normalized MSE (||Y-Y_pred||^2/||Y||^2)')
    tight_str = 'tight' if tight else 'loose'
    plt.title('Normalized MSE vs. a (bound is {})'.format(tight_str))
    save_plot('micro_large_sigma_min.pdf')
    # plt.show()
    # print(1)

# gaussian: if true, use gaussian random data.  Else use uniform random data.
# use_adapt: if true, compare adaptive and non-adaptive.  Else compare stoch/det.
def deltas_vs_precision_and_lambda(gaussian, use_adapt):
    gaussian_str = 'gaussian' if gaussian else 'uniform'
    adapt_str = 'adapt' if use_adapt else 'nonadapt'
    n = 1000
    d = 30
    lim = 1.0/np.sqrt(d)
    X = (np.random.randn(n,d) * 0.5 * lim if gaussian else
        np.random.uniform(low=-lim, high=lim, size=(n,d)))
    K = X @ X.T
    bs = [1,2,8]
    # bs = [1,2,4,8,32]
    lambdas = [2**(-32), 2**(-24), 2**(-16), 2**(-12), 2**(-8),2**(-4), 2**(-2), 2**0, 2**1, 2**2, 2**4, 2**8, 2**12, 2**16, 2**20]
    # these bools control either stoch/det, or adaptive/non-adaptive (depending on use_adapt flag)
    bools = [False,True]
    delta1s = np.zeros((len(bools), len(bs), len(lambdas)))
    delta2s = np.zeros((len(bools), len(bs), len(lambdas)))

    # Gather delta1 and delta2 for different precisions and lambdas
    for i_bl,bool_ in enumerate(bools):
        for i_b,b in enumerate(bs):
            # Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=gaussian, stochastic_round=stoch)
            if use_adapt: # adapt vs. non-adapt
                Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=bool_, stochastic_round=False)
            else: # stoch vs. det
                Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=gaussian, stochastic_round=bool_)
            Kq = Xq @ Xq.T
            for i_l,lam in enumerate(lambdas):
                delta1s[i_bl,i_b,i_l], delta2s[i_bl,i_b,i_l], _ = utils.delta_approximation(K, Kq, lambda_ = lam)
    plt.figure(1)
    plot_deltas_v2(delta1s, bs, lambdas, n, d, 'Delta_1', gaussian, use_adapt)
    save_plot('micro_{}_{}_delta1_vs_2_b_lambda.pdf'.format(gaussian_str, adapt_str))
    plt.figure(2)
    plot_deltas_v2(delta2s, bs, lambdas, n, d, 'Delta_2', gaussian, use_adapt)
    save_plot('micro_{}_{}_delta2_vs_2_b_lambda.pdf'.format(gaussian_str, adapt_str))
    # plt.subplot(211)
    # plot_deltas_v2(delta1s, bs, lambdas, n, d, 'Delta_1', gaussian, use_adapt)
    # plt.subplot(212)
    # plot_deltas_v2(delta2s, bs, lambdas, n, d, 'Delta_2', gaussian, use_adapt)
    # plt.show()
    # print(1)
    # plt.figure(1)
    # plt.subplot(221)
    # plot_deltas(delta1s[0,:,:], bs, lambdas, n, d, 'Delta_1', gaussian, False)
    # plt.subplot(222)
    # plot_deltas(delta2s[0,:,:], bs, lambdas, n, d, 'Delta_2', gaussian, False)
    # plt.subplot(223)
    # plot_deltas(delta1s[1,:,:], bs, lambdas, n, d, 'Delta_1', gaussian, True)
    # plt.subplot(224)
    # plot_deltas(delta2s[1,:,:], bs, lambdas, n, d, 'Delta_2', gaussian, True)
    # plt.show()
    # print(1)

def plot_deltas(delta_results, bs, lambdas, n, d, ylabel, gaussian, stoch, xlog=True, ylog=True):
    nplams = np.array(lambdas)
    x = 2**16 * nplams
    gaussian_str = 'Gaussian' if gaussian else 'Uniform'
    stoch_str = 'Stochastic' if stoch else 'Deterministic'
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
    title = '{} ({}): {} vs. {}'.format(gaussian_str, stoch_str, ylabel, '2^b * lambda')
    plt.title(title)

def plot_deltas_v2(delta_results, bs, lambdas, n, d, ylabel, gaussian, use_adapt, xlog=True, ylog=True):
    nplams = np.array(lambdas)
    x = 2**16 * nplams
    gaussian_str = 'Gaussian' if gaussian else 'Uniform'
    bool_str = 'Adapt/Non-Adapt' if use_adapt else 'Stoch/Det.'
    bool_strs = ['Non-Adapt','Adapt'] if use_adapt else ['Det','Stoch']
    leg = []
    colors = ['b','g','r','y','k']
    for i_b,b in enumerate(bs):
        plt.plot(2**b * nplams, np.abs(delta_results[0,i_b,:]), marker='o', linestyle='dashed', color=colors[i_b])
        plt.plot(2**b * nplams, np.abs(delta_results[1,i_b,:]), marker='x', linestyle='dashed', color=colors[i_b])
        leg.append('b={} ({})'.format(b, bool_strs[0]))
        leg.append('b={} ({})'.format(b, bool_strs[1]))
        #plt.plot(2.0**b/((2.0**b-1)**2 * nplams), delta2s[i_b,:] - 1.0/((2**b-1)**2 * nplams) )
        # plt.plot(nplams, delta2s[i_b,:])
    plt.plot(x, np.sqrt(2*np.log(n/0.1)/d) * 5 * n / x, marker='s', linestyle='dashed', color='m')
    leg.append('bound')
    plt.legend(leg)
    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    plt.xlabel('2^b * lambda')
    plt.ylabel(ylabel)
    title = '{} ({}): {} vs. {}'.format(gaussian_str, bool_str, ylabel, '2^b * lambda')
    plt.title(title)

def clipping_effect():
    n = 1000
    d = 30
    # lim = 1.0/np.sqrt(d)
    # X = np.random.randn(n,d) * 0.5 * lim
    X = np.random.randn(n,d)
    K = X @ X.T
    eigs,_ = np.linalg.eigh(X.T @ X)
    eigs = np.sort(eigs)
    eig_min = eigs[0]
    lam = eig_min/10
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
    plt.plot(rs, 1 - delta1s, marker='o', linestyle='dashed')
    plt.plot(rs, 1 + delta2s, marker='o', linestyle='dashed')
    plt.xlabel('clip value')
    plt.legend(['1-Delta1','1+Delta2'])
    plt.title('Effect of clipping on Delta1 and Delta2 for random Gaussian data.')
    # plt.subplot(122)
    # plt.plot(rs, delta2s)
    # plt.xlabel('clip value')
    # plt.ylabel('Delta_2')
    plt.show()
    print(1)

def clipping_effect2():
    n = 1000
    d = 30
    # lim = 1.0/np.sqrt(d)
    # X = np.random.randn(n,d) * 0.5 * lim
    X = np.random.randn(n,d)
    eigs,_ = np.linalg.eigh(X.T @ X)
    eigs = np.sort(eigs)
    max_r = np.max(X)
    rs = np.linspace(0,max_r,num=8)
    plt.figure(1)
    plt.plot(np.arange(d), eigs)
    leg = ['Orig. Spectrum']
    for r in rs:
        Xc = np.clip(X, -r, r)
        eigs_c,_ = np.linalg.eigh(Xc.T @ Xc)
        eigs_c = np.sort(eigs_c)
        plt.plot(np.arange(d), eigs_c)
        leg.append('Clipped (r={:.2f})'.format(r))
    plt.legend(leg)
    plt.title('Clipped vs. unclipped spectra')
    plt.show()
    print(1)

def clipping_with_quantization_effect():
    n = 1000
    d = 30
    X = np.random.randn(n,d)
    eigs,_ = np.linalg.eigh(X.T @ X)
    eigs = np.sort(eigs)
    eig_min = eigs[0]
    K = X @ X.T
    bs = [1,2,4,32]
    colors = ['b','g','r','y','k']
    # We are trying to understanding plateau at small 2^b * lam, so take very small lam
    lam = eig_min/10
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
        # plt.plot(rs, delta1s[i_b,:], marker='o', linestyle='dashed', color=colors[i_b])
        # plt.plot(rs, delta2s[i_b,:], marker='x', linestyle='dashed', color=colors[i_b])
        # legend.append('Delta1 (b={})'.format(b))
        # legend.append('Delta2 (b={})'.format(b))
        plt.plot(rs, 1 - delta1s[i_b,:], marker='o', linestyle='dashed', color=colors[i_b])
        plt.plot(rs, 1 + delta2s[i_b,:], marker='x', linestyle='dashed', color=colors[i_b])
        legend.append('1-Delta1 (b={})'.format(b))
        legend.append('1+Delta2 (b={})'.format(b))
        # plt.plot(rs, 1/(1 - delta1s[i_b,:]), marker='o', linestyle='dashed', color=colors[i_b])
        # plt.plot(rs, np.maximum(delta2s[i_b,:],0.001), marker='x', linestyle='dashed', color=colors[i_b])
        # legend.append('1/(1-Delta1) (b={})'.format(b))
        # legend.append('max(Delta2,0.001) (b={})'.format(b))
    plt.xlabel('clip value')
    plt.legend(legend)
    plt.title('Effect of clipping + quantization on (Delta1,Delta2) for Gaussian data.')
    plt.yscale('log')
    save_plot('deltas_vs_clip_and_quant.pdf')
    # plt.show()
    # print(1)

if __name__ == '__main__':
    # ppmi_spectrum()

    # MICRO #1: Delta1 and Delta2 vs. 2^b lambda. Can specify whether data should
    # be Gaussian or uniform, and whether range should be chosen adaptively.
    # deltas_vs_precision_and_lambda(False, False)

    # MICRO #2: Show that when sigma_min is large, one can use a large regularizer.
    flat_spectrum_vs_generalization(True)

    # clipping_effect2()

    # This experiment looks at the effect of clipping and quantization on
    # Delta1 and Delta2.
    # clipping_with_quantization_effect()
