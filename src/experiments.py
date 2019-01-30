import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
import matplotlib.pyplot as plt
import compress
import utils
import pathlib
import math

def save_plot(plot_filename):
    plot_file = str(pathlib.PurePath(utils.get_git_dir(),
        'paper', 'figures', plot_filename))
    plt.savefig(plot_file)
    plt.close()

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
    base_sing_vals = np.linalg.svd(X, compute_uv=False)
    base_eigs = base_sing_vals**2
    # eig_min = base_eigs[-1]
    eig_max = base_eigs[0]

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
    plot_deltas_v2(delta1s, bs, lambdas, n, d, 'Delta_1', gaussian, use_adapt, eig_max=eig_max)
    save_plot('micro_{}_{}_delta1_vs_2_b_lambda.pdf'.format(gaussian_str, adapt_str))
    plt.figure(2)
    plot_deltas_v2(delta2s, bs, lambdas, n, d, 'Delta_2', gaussian, use_adapt, eig_max=eig_max)
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

def plot_deltas_v2(delta_results, bs, lambdas, n, d, ylabel, gaussian, use_adapt, eig_min=-1, eig_max=-1, xlog=True, ylog=True):
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
    if eig_min != -1:
        for b in bs:
            plt.plot(2**b * nplams, nplams/eig_min)
            leg.append('lam/eig_min (b={})'.format(b))
            # plt.plot(2**b * nplams, eig_min/(eig_min + nplams))
            # leg.append('eig_min/(eig_min + nplams) (b={})'.format(b))
    if eig_max != -1:
        for b in bs:
            plt.plot(2**b * nplams, eig_max/(eig_max + nplams))
            leg.append('eig_max/(eig_max + nplams) (b={})'.format(b))   

    plt.legend(leg)
    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    plt.xlabel('2^b * lambda')
    plt.ylabel(ylabel)
    title = '{} ({}): {} vs. {}'.format(gaussian_str, bool_str, ylabel, '2^b * lambda')
    plt.title(title)
    plt.ylim(10**-3,10**3)

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

def delta_experiment():
    n = 1000
    d = 25
    eps = 2*10**(-1)
    X = (np.random.rand(n,d) * 2 - 1) * 1/np.sqrt(d)
    X2 = X + np.random.randn(n,d)* eps
    #X2 = X[:,:d-1] + np.random.randn(n,d-1)* eps
    K = X@X.T
    K2 = X2 @ X2.T

    base_sing_vals = np.linalg.svd(X, compute_uv=False)
    base_eigs = base_sing_vals**2
    eig_min = base_eigs[-1]
    lambdas = [eig_min/100, eig_min/10, eig_min, eig_min*10, eig_min*100]
    for lam in lambdas:
        delta1,delta2,_ = utils.delta_approximation(K,K2,lambda_=lam)
        delta1_lower_bound = eig_min/(eig_min + lam)
        # assert delta1 >= delta1_lower_bound
        print(delta1,delta2,delta1_lower_bound)

def deltas_vs_precision(gaussian):
    # gaussian_str = 'gaussian' if gaussian else 'uniform'
    # adapt_str = 'adapt' if use_adapt else 'nonadapt'
    n = 1000
    d = 30
    lim = 1.0/np.sqrt(d)
    X = (np.random.randn(n,d) * 0.5 * lim if gaussian else
        np.random.uniform(low=-lim, high=lim, size=(n,d)))
    K = X @ X.T

    base_sing_vals = np.linalg.svd(X, compute_uv=False)
    base_eigs = base_sing_vals**2
    eig_min = base_eigs[-1]

    b_list = [1,2,4,8,16,32]
    a_list = [10**(-2),1,10**2]

    delta1s = np.zeros((len(a_list), len(b_list)))
    delta2s = np.zeros((len(a_list), len(b_list)))
    
    for j,b in enumerate(b_list):
        Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=False, stochastic_round=True)
        Kq = Xq @ Xq.T
        for i,a in enumerate(a_list):
            lam = a * eig_min
            delta1s[i,j], delta2s[i,j], _ = utils.delta_approximation(K, Kq, lambda_ = lam)

    plt.rc('text', usetex=True)
    plt.figure()
    leg = []
    for i,a in enumerate(a_list):
        plt.plot(b_list, delta1s[i,:], label=r'$\lambda / \sigma_{min} = '+ str(a) + '$')
        # leg.append(r'$\lambda / \sigma_\min = {}$'.format(a))
    plt.xlabel(r'Precision $(b)$')
    plt.ylabel(r'$\Delta_{}$'.format(1))
    # plt.yscale('log')
    plt.xscale('log')
    plt.xlim([1, 32])
    plt.ylim([0, 1])
    plt.title(r'$\Delta_{}$ vs. Precision'.format(1))
    plt.grid()
    plt.xticks(b_list,b_list)
    plt.legend(loc="upper right")
    # plt.show()
    save_plot("Delta1_vs_precision.pdf")

    plt.figure()
    leg = []
    for i,a in enumerate(a_list):
        plt.plot(b_list, delta2s[i,:], label=r'$\lambda / \sigma_{min} = '+ str(a) + '$')
        # leg.append(r'$\lambda / \sigma_\min = {}$'.format(a))
    plt.xlabel(r'Precision $(b)$')
    plt.ylabel(r'$\Delta_{}$'.format(2))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([1, 32])
    plt.ylim([1e-14, 1e2])
    plt.title(r'$\Delta_{}$ vs. Precision'.format(2))
    plt.grid()
    plt.xticks(b_list,b_list)
    plt.legend(loc="lower left")
    # plt.show()
    save_plot("Delta2_vs_precision.pdf")


def eigenspace_overlap():
    # gaussian_str = 'gaussian' if gaussian else 'uniform'
    # adapt_str = 'adapt' if use_adapt else 'nonadapt'
    n = 1000
    d = 30
    lim = 1.0/np.sqrt(d)
    X = np.random.uniform(low=-lim, high=lim, size=(n,d))
    K = X @ X.T
    base_sing_vals = np.linalg.svd(X, compute_uv=False)
    base_eigs = base_sing_vals**2
    eig_min = base_eigs[-1]
    eig_max = base_eigs[0]
    print('eig_min = {}, eig_max = {}'.format(eig_min,eig_max))
    U,S,V = np.linalg.svd(X,full_matrices=False)
    # Uq,Sq,Vq = np.linalg.svd(Xq,full_matrices=False)
    # print(U.shape)
    # C = Uq.T @ U

    w = np.ones(d)
    y = X@w
    y_norm = np.sum(y**2)
    bs = [1,2,3,4,5,6,7,8,32]
    leg = []
    for b in bs:
        leg.append('b={}'.format(b))
    ks = [5,10,15,20,25,30]
    lambdas = [eig_min/100, eig_min, eig_max]
    delta1s = np.zeros((len(bs),len(ks),len(lambdas)))
    delta2s = np.zeros((len(bs),len(ks),len(lambdas)))
    # specs_X = np.zeros((len(bs),len(ks)))
    # frobs_X = np.zeros((len(bs),len(ks)))
    specs_K = np.zeros((len(bs),len(ks)))
    frobs_K = np.zeros((len(bs),len(ks)))
    eig_overlap = np.zeros((len(bs),len(ks)))

    rel_err = np.zeros((len(bs),len(ks)))
    for i_b,b in enumerate(bs):
        for i_k,k in enumerate(ks):
            Xq,_,_ = compress.compress_uniform(X[:,:k], b, adaptive_range=False, stochastic_round=True)
            Uq,Sq,Vq = np.linalg.svd(Xq,full_matrices=False)
            Kq = Xq @ Xq.T
            y_pred = Xq @ np.linalg.inv(Xq.T @ Xq) @ Xq.T @ y
            rel_err[i_b,i_k] = np.sum((y_pred-y)**2) / y_norm
            # print('b={},k={}, relative error = {}'.format(b, k, rel_err[i_b,i_k]))
            for i_lam,lam in enumerate(lambdas):
                delta1s[i_b,i_k,i_lam], delta2s[i_b,i_k,i_lam],_ = utils.delta_approximation(K,Xq @ Xq.T, lambda_=lam)
            # specs_X[i_b,i_k] = np.linalg.norm(X-Xq,2)
            # frobs_X[i_b,i_k] = np.linalg.norm(X-Xq)
            specs_K[i_b,i_k] = np.linalg.norm(K-Kq,2)
            frobs_K[i_b,i_k] = np.linalg.norm(K-Kq)
            eig_overlap[i_b,i_k] = np.linalg.norm(Uq.T @ U)
    

    for i_lam in range(3):
        plt.figure(1)
        # plt.subplot(231+i_lam)
        for i_b,b in enumerate(bs):
            plt.scatter(1/(1-delta1s[i_b,:,i_lam]), rel_err[i_b,:])
        plt.title('Relative error vs. Delta1 (lam = {:.2f})'.format(lambdas[i_lam]))
        plt.legend(leg)
        plt.xscale('log')
        save_plot('Error_vs_delta1_trans_{}.pdf'.format(i_lam))

        plt.figure(1)
        # plt.subplot(231+i_lam)
        for i_b,b in enumerate(bs):
            plt.scatter(delta1s[i_b,:,i_lam], rel_err[i_b,:])
        plt.title('Relative error vs. Delta1 (lam = {:.2f})'.format(lambdas[i_lam]))
        plt.legend(leg)
        plt.xscale('log')
        save_plot('Error_vs_delta1_{}.pdf'.format(i_lam))

        plt.figure(1)
        # plt.subplot(234+i_lam)
        for i_b,b in enumerate(bs):
            plt.scatter(delta2s[i_b,:,i_lam], rel_err[i_b,:])
        plt.title('Relative error vs. Delta2 (lam = {:.2f})'.format(lambdas[i_lam]))
        plt.legend(leg)
        plt.xscale('log')
        save_plot('Error_vs_delta2_{}.pdf'.format(i_lam))
    # save_plot('Error_vs_delta1_delta2.pdf')

    plt.figure(1)
    # metrics = ['Spectral (X)', 'Frob (X)', 'Spectral (K)', 'Frob (K)']
    # results = [specs_X, frobs_X, specs_K, frobs_K]
    metrics = ['Spectral (K)', 'Frob (K)']
    results = [specs_K, frobs_K]
    for i_m,metric in enumerate(metrics):
        result = results[i_m]
        plt.subplot(121 + i_m)
        for i_b,b in enumerate(bs):
            plt.scatter(result[i_b,:], rel_err[i_b,:])
        plt.title('Relative error vs. {}'.format(metric))
        plt.legend(leg)
    save_plot('Error_vs_frob_spec.pdf')

    plt.figure(1)
    for i_b,b in enumerate(bs):
        plt.scatter(eig_overlap[i_b,:], rel_err[i_b,:])
    plt.title('Relative error vs. eig-overlap')
    plt.legend(leg)
    save_plot('Error_vs_eig_overlap.pdf')


# from plotter import latexify_setup_fig, latexify_finalize_fig, default_latexify_config
# from plotter import get_embedtype_name_map, get_legend_name_map
def eigenspace_overlap_micro():
    # gaussian_str = 'gaussian' if gaussian else 'uniform'
    # adapt_str = 'adapt' if use_adapt else 'nonadapt'
    # ns = [10000,1000,100]
    # ns = [10000,3000,1000]
    # latexify_config = default_latexify_config
    # embedtype_name_map = get_embedtype_name_map()
    # latexify_config["minor_tick_off"] = True
    # ax = latexify_setup_fig(latexify_config)
    # ns = [100000]
    ns = [200,400,800,1600]
    ds = [10,25,50,100]
    bs = [1,2,4,8,16,32]
    plt.figure(1)
    leg = []
    full_matrices = False
    for n in ns:
        for d in ds:
            lim = 1.0/np.sqrt(d)
            X = np.random.uniform(low=-lim, high=lim, size=(n,d))
            U,S,_ = np.linalg.svd(X,full_matrices=full_matrices)
            leg.append('n = {}, d = {}'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 1)'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 2)'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 3)'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 4)'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 5)'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 6)'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 7)'.format(n,d))
            # leg.append('n = {}, d = {} (Bound 8)'.format(n,d))
            overlaps = np.zeros(len(bs))
            # bounds = np.zeros(len(bs))
            # bounds2 = np.zeros(len(bs))
            # bounds3 = np.zeros(len(bs))
            # bounds4 = np.zeros(len(bs))
            # bounds5 = np.zeros(len(bs))
            # bounds6 = np.zeros(len(bs))
            # bounds7 = np.zeros(len(bs))
            # bounds8 = np.zeros(len(bs))
            s_max = S[0]
            s_min = S[d-1]
            for i,b in enumerate(bs):
                Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=False, stochastic_round=True)
                C = Xq - X
                Uq,_,_ = np.linalg.svd(Xq,full_matrices=full_matrices)
                overlaps[i] = np.linalg.norm(Uq[:,:d].T @ U[:,:d])**2 / d
                print('b = {}, overlap = {}'.format(b,overlaps[i]))
                # prec = 2**b - 1
                # H = Xq @ Xq.T - X @ X.T
                # bounds[i] = max(0, d - np.linalg.norm(Uq[:,d:].T @ H @ U[:,:d])**2/s_min**4) / d
                # bounds2[i] = max(0, d - np.linalg.norm(H)**2/s_min**4) / d
                # bounds3[i] = max(0, d - 16 * n * (s_max + np.sqrt(n)/prec)**2 / (prec**2 * s_min)) / d
                # bounds[i] = (d - np.linalg.norm(Uq[:,d:].T @ H @ U[:,:d])**2/s_min**4) / d
                # bounds2[i] = (d - np.linalg.norm(H)**2/s_min**4) / d
                # bounds3[i] = (d - (4*n**2/(d * (2**b-1)**2))/s_min**4) / d
                # bounds4[i] = (d - (2 * np.linalg.norm(C @ X.T) + np.linalg.norm(C @ C.T))**2/s_min**4) / d
                # bounds5[i] = (d - (2 * np.linalg.norm(X,2) * np.linalg.norm(C) + np.linalg.norm(C) * np.linalg.norm(C,2))**2/s_min**4) / d
                # bounds6[i] = (d - (2 * np.linalg.norm(X,2) * np.linalg.norm(C) + np.linalg.norm(C)**2)**2/s_min**4) / d
                # bounds7[i] = (d - (16 * n / prec**2) * ( s_max + np.sqrt(n)/prec )**2 / s_min**4) / d
                # B = 2 * math.log(2 * n) * (s_max  + 1/3) / prec
                # bounds8[i] = (d - d * (2*B + (1/prec)**2)**2 / s_min**4 ) / d
            plt.plot(bs,1-overlaps, '-o')
            # plt.plot(bs,bounds, '-o')
            # plt.plot(bs,bounds2, '-o')
            # plt.plot(bs,bounds3, '-o')
            # plt.plot(bs,bounds4, '-o')
            # plt.plot(bs,bounds5, '-o')
            # plt.plot(bs,bounds6, '-o')
            # plt.plot(bs,bounds7, '-o')
            # plt.plot(bs,bounds8, '-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(leg)
    # plt.yscale('log')
    plt.xticks(bs,bs)
    plt.ylim(0,1)
    plt.xlim(1,32)
    # plt.title('Normalized eigenvector overlap vs. Precision (b)')
    plt.title(r'1-Eigenspace Overlap vs. Precision')
    plt.xlabel(r'Precision (b)')
    plt.ylabel(r'1-Eigenspace overlap (E)')
    # plt.legend(leg)
    # latexify_finalize_fig(ax, latexify_config)
    save_plot('micro_eig_overlap_vs_precision.pdf')
    

# from plotter import latexify_setup_fig, latexify_finalize_fig, default_latexify_config
# from plotter import get_embedtype_name_map, get_legend_name_map
def eigenspace_overlap_micro2():
    ns = [200,400,800,1600]
    ds = [10,25,50,100]
    bs = [1,2,4,8,16,32]
    plt.figure(1)
    leg = []
    full_matrices = False
    for n in ns:
        for d in ds:
            lim = 1.0/np.sqrt(d)
            X = np.random.uniform(low=-lim, high=lim, size=(n,d))
            U,S,_ = np.linalg.svd(X,full_matrices=full_matrices)
            leg.append('n = {}, d = {}'.format(n,d))
            overlaps = np.zeros(len(bs))
            s_max = S[0]
            s_min = S[d-1]
            for i,b in enumerate(bs):
                Xq,_,_ = compress.compress_uniform(X, b, adaptive_range=False, stochastic_round=True)
                Uq,_,_ = np.linalg.svd(Xq,full_matrices=full_matrices)
                overlaps[i] = np.linalg.norm(Uq[:,:d].T @ U[:,:d])**2 / d
                print('n = {}, d = {}, b = {}, overlap = {}'.format(n, d, b, overlaps[i]))
            plt.plot(bs,1-overlaps, '-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(leg)
    plt.xticks(bs,bs)
    plt.ylim(0,1)
    plt.xlim(1,32)
    plt.title(r'1-Eigenspace Overlap vs. Precision')
    plt.xlabel(r'Precision (b)')
    plt.ylabel(r'1-Eigenspace overlap (E)')
    save_plot('micro_eig_overlap_vs_precision.pdf')



if __name__ == '__main__':
    # ppmi_spectrum()

    # MICRO #1: Delta1 and Delta2 vs. 2^b lambda. Can specify whether data should
    # be Gaussian or uniform, and whether range should be chosen adaptively.
    #deltas_vs_precision_and_lambda(False, False)

    # MICRO #2: Show that when sigma_min is large, one can use a large regularizer.
    # flat_spectrum_vs_generalization(True)

    # clipping_effect2()

    # This experiment looks at the effect of clipping and quantization on
    # Delta1 and Delta2.
    # clipping_with_quantization_effect()

    # This looks at deltas between two similar matrices.
    # delta_experiment()

    # Plot deltas vs. precision for fixed dimension, several lambdas:
    # deltas_vs_precision(False)

    # Eigenvector overlap micros
    eigenspace_overlap_micro2()
    # eigenspace_overlap()
