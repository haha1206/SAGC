import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
import numpy as np
from W_Construct import ancher_W
from data_loader import load_mat
from sklearn import metrics
import warnings
import time
warnings.filterwarnings("ignore",category=Warning)


def initialize_q_v_u(K, n_min, n_max):
    """
    Initialize q, v, u for Sinkhorn DB-OT, given kernel K and column bounds.
    K: (n, c)
    n_min, n_max: float or ndarray (c,)
    """
    n, c = K.shape
    delta = 1e-8
    s = np.sum(K, axis=0)  # (c,) column sums

    n_min = np.full(c, n_min) if np.isscalar(n_min) else n_min
    n_max = np.full(c, n_max) if np.isscalar(n_max) else n_max

    q = np.maximum(n_min / (s + delta), 1.0)
    v = np.minimum(n_max / (s + delta), 1.0)

    Kvq = K @ (q * v)  # (n,)
    u = 1.0 / (Kvq + delta)

    return u, q, v

def sinkhorn_knopp_entropy(U, n_min, n_max, epsilon=0.001, max_iter=100, tol=1e-6, verbose=False):

    K = np.exp(U/ epsilon)  # Gibbs kernel

    u, q, v = initialize_q_v_u(K, n_min, n_max)

    for it in range(max_iter):
        u_prev = u.copy()

        # Update u: row normalization
        Kvq = K @ (q * v)
        u = 1.0 / (Kvq)

        # Update q: enforce lower bound
        Ku = u @ K
        q = np.maximum(n_min / (Ku * v ), 1)

        # Update v: enforce upper bound
        KTu = K.T @ u
        KTuq = KTu * q
        v = np.minimum(n_max / (KTuq), 1)

        # Convergence check
        delta = np.linalg.norm(u - u_prev, 1)
        if verbose:
            print(f"Iter {it+1}: Δu = {delta:.2e}")
        if delta < tol:
            break
    # Final transport plan (i.e., assignment matrix)
    F = (u[:, None] * K) * (q * v)[None, :]
    return F

def discrete_rounding(F_soft, n_min, n_max):

    n, c = F_soft.shape
    labels = np.argmax(F_soft, axis=1)
    counts = np.bincount(labels, minlength=c)


    for cls in range(c):
        while counts[cls] > n_max:
            idx = np.where(labels == cls)[0]
            i = idx[np.argmin(F_soft[idx, cls])] 
            F_soft[i, cls] = -np.inf
            labels[i] = np.argmax(F_soft[i])
            counts = np.bincount(labels, minlength=c)


    for cls in range(c):
        while counts[cls] < n_min:
            donor = np.argmax(counts)
            if counts[donor] <= n_min or donor == cls:
                break
            idx = np.where(labels == donor)[0]
            i = idx[np.argmax(F_soft[idx, cls])] 
            labels[i] = cls
            counts = np.bincount(labels, minlength=c)

    F_hard = np.zeros_like(F_soft)
    F_hard[np.arange(n), labels] = 1
    return F_hard


def SGR(W, n_iter, n_max, n_min, c, init_method='svd', random_state=3):
    """
    SGR clustering.

    Parameters
    ----------
    W : ndarray, shape (n, m)
        Anchor graph.
    n_iter : int
        Number of iterations.
    n_max : int or array-like
        Upper bound of cluster size.
    n_min : int or array-like
        Lower bound of cluster size.
    c : int
        Number of clusters.
    init_method : str
        Initialization method for U:
        - 'svd': SVD initialization
        - 'random': random orthogonal initialization
        - 'kmeans': KMeans-based initialization
    random_state : int
        Random seed.

    Returns
    -------
    y_pred, F, U, s, OBJ
    """

    n, m = W.shape
    rng = np.random.RandomState(random_state)

    # =========================================================
    # 1. SVD initialization
    # =========================================================
    if init_method == 'svd':

        U = np.linalg.svd(
            W,
            full_matrices=False
        )[2][:c].T

    # =========================================================
    # 2. Random initialization
    # =========================================================
    elif init_method == 'random':

        # Random matrix: m x c
        U_random = rng.randn(m, c)

        # QR decomposition guarantees U^T U = I
        U, _ = np.linalg.qr(U_random)

        # Ensure correct dimension
        U = U[:, :c]

    else:
        raise ValueError(
            f"Unknown initialization method: {init_method}. "
            f"Choose from ['svd', 'random']."
        )

    # Optimization
    F, U, s, OBJ = opt(
        W,
        U,
        n_iter,
        n_max,
        n_min
    )

    y_pred = np.argmax(F, axis=1) + 1

    return y_pred, F, U, s, OBJ

def opt(W, U, NITER,n_max,n_min):
    s = 5
    OBJ = []
    obj = 0
    for Iter in range(NITER):
        obj_prev = obj
    #update F
        H = 2*s*U.T@W.T
        F = sinkhorn_knopp_entropy(H.T,n_min,n_max)
        F = discrete_rounding(F, n_min, n_max)
    #update U
        H1 = 2*s*F.T@W
        U1, _, VT = np.linalg.svd(H1, full_matrices=False)
        U = VT.T @ U1.T  # (m x c)

    #update s
        numerator = np.trace(F.T @ W @ U)  # Tr(F^T B U)
        denominator = np.trace(U @ F.T @ F @ U.T)  # Tr(U^T F^T F U)

        if denominator < 1e-8:
            denominator = 1e-8

        s = numerator / denominator
        obj = np.square(np.linalg.norm(W-s*F@U.T))
        OBJ.append(obj)
        if np.abs(obj_prev - obj)<1e-8:
            break
    return F,U,s,OBJ

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)

from sklearn.metrics.cluster import contingency_matrix

def F_score(labels_true, labels_pred):
    cm = contingency_matrix(labels_true, labels_pred)

    tp = np.sum(cm * (cm - 1)) / 2

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    fp = np.sum(col_sum * (col_sum - 1)) / 2 - tp
    fn = np.sum(row_sum * (row_sum - 1)) / 2 - tp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)

    return 2 * precision * recall / (precision + recall + 1e-12)


def NE(y_pred):
    n = len(y_pred)
    ar, num = np.unique(y_pred, return_counts=True)
    c = len(ar)
    ne = 0
    for i in range(c):
        ne = ne + (num[i]/n)*np.log(num[i]/n)
    return (-1/np.log(c))*ne

if __name__=="__main__":
    X,GT = load_mat('data/JAFFE.mat')
    GT = GT.reshape(np.max(GT.shape), )
    n_cluster = len(np.unique(GT))
    N = X.shape[0]
    print(N)
    # X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    m = n_cluster*5
    print(m)
    GT = GT.reshape(np.max(GT.shape), )
    c = n_cluster
    Max_list = [np.floor(1.0 * N / c), np.floor(1.5 * N / c), np.floor(2.0 * N / c), np.floor(2.5 * N / c),
                np.floor(5.0 * N / c)]
    Min_list = [np.floor(0.8 * N / c), np.floor(0.6 * N / c), np.floor(0.4 * N / c), np.floor(0.2 * N / c)]
    print(Max_list)
    print(Min_list)
    print(np.unique(GT, return_counts=True)[1])
    for i in range(5):
        for j in range(4):
            ACC_res = []
            NMI_res = []
            Purtiy_res = []
            ARI_res = []
            F1_res = []
            for inter in range(1):
                W = ancher_W(X, m, 5)
                t1 = time.time()
                y_pred,F,U,s,OBJ= SGR(W, 10,Max_list[i],Min_list[j],c,init_method="random")
                t2 = time.time()
                print(OBJ)
                t2 = time.time()
                print(t2-t1)
                print(np.unique(y_pred, return_counts=True)[1])
                print(NE(y_pred))
                ACC = acc(GT,y_pred)
                NMI = metrics.normalized_mutual_info_score(GT,y_pred)
                Purity = purity_score(GT,y_pred)
                ARI = metrics.adjusted_rand_score(GT, y_pred)
                F1_score = F_score(GT, y_pred)
                print('clustering accuracy: {}, NMI: {}, Purity: {},ARI:{},F1:{}'.format(ACC,NMI,Purity,ARI,F1_score))
                print(Max_list[i],Min_list[j])
                ACC_res.append(ACC)
                NMI_res.append(NMI)
                Purtiy_res.append(Purity)
                ARI_res.append(ARI)
                F1_res.append(F1_score)

