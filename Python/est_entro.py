import numpy as np
import scipy.io as sio
import scipy.sparse as ssp

poly_entro = None

def est_entro_JVHW(samp):
    """Proposed JVHW estimate of Shannon entropy (in bits) of the input sample

    This function returns a scalar JVHW estimate of the entropy of samp when
    samp is a vector, or returns a row vector containing the JVHW estimate of
    each column of samp when samp is a matrix.

    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each column
               of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    # The order of polynomial is no more than 22 because otherwise floating-point error occurs
    order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
    global poly_entro
    if poly_entro is None:
        poly_entro = sio.loadmat('poly_coeff_entro.mat')['poly_entro']
    coeff = poly_entro[order-1, 0][0]

    f = fingerprint(samp)

    prob = np.arange(1, f.shape[0] + 1) / n

    # Piecewise linear/quadratic fit of c_1
    V1 = np.array([0.3303, 0.4679])
    V2 = np.array([-0.530556484842359, 1.09787328176926, 0.184831781602259])
    f1nonzero = f[0] > 0
    c_1 = np.zeros(wid)

    with np.errstate(divide='ignore', invalid='ignore'):
        if n >= order and f1nonzero.any():
            if n < 200:
                c_1[f1nonzero] = np.polyval(V1, np.log(n / f[0, f1nonzero]))
            else:
                n2f1_small = f1nonzero & (np.log(n / f[0]) <= 1.5)
                n2f1_large = f1nonzero & (np.log(n / f[0]) > 1.5)
                c_1[n2f1_small] = np.polyval(V2, np.log(n / f[0, n2f1_small]))
                c_1[n2f1_large] = np.polyval(V1, np.log(n / f[0, n2f1_large]))

            # make sure nonzero threshold is higher than 1/n
            c_1[f1nonzero] = np.maximum(c_1[f1nonzero], 1 / (1.9 * np.log(n)))

        prob_mat = entro_mat(prob, n, coeff, c_1)

    return np.sum(f * prob_mat, axis=0) / np.log(2)


def est_entro_JVHW_from_fingerprint_dict(fingerprint):
    """`fingerprint` should be a dict mapping frequencies to how often that
    frequency occurs in the profile."""

    fingerprint = {(k if isinstance(k, tuple) else (k,0)): v for k, v in fingerprint.items()}
    shape = tuple(np.max(np.array(list(fingerprint.keys())), axis=0) + 1)
    f_dok = ssp.dok_matrix(shape, dtype=int)
    for k, v in fingerprint.items():
        f_dok[k] = v
    n = ssp.csr_matrix(np.arange(1, f_dok.shape[0]+1)).dot(f_dok).max()
    wid = f_dok.shape[1]

    order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
    global poly_entro
    if poly_entro is None:
        poly_entro = sio.loadmat('poly_coeff_entro.mat')['poly_entro']
    coeff = poly_entro[order-1, 0][0]

    f_csr = f_dok.tocsr()
    fnonzero_rows = sorted(list(set(f_csr.nonzero()[0])))

    prob_dok = ssp.dok_matrix((f_csr.shape[0], 1))
    prob_dok[fnonzero_rows,0] = (np.array(fnonzero_rows)+1)/n
    prob_csr = prob_dok.tocsr()

    # Piecewise linear/quadratic fit of c_1
    V1 = np.array([0.3303, 0.4679])
    V2 = np.array([-0.530556484842359, 1.09787328176926, 0.184831781602259])
    f_row1 = f_csr[0].toarray().squeeze(0)
    f1nonzero = f_row1 > 0
    c_1 = np.zeros(wid)

    with np.errstate(divide='ignore', invalid='ignore'):
        if n >= order and f1nonzero.any():
            if n < 200:
                c_1[f1nonzero] = np.polyval(V1, np.log(n / f[0, f1nonzero]))
            else:
                n2f1_small = f1nonzero & (np.log(n / f_row1) <= 1.5)
                n2f1_large = f1nonzero & (np.log(n / f_row1) > 1.5)
                c_1[n2f1_small] = np.polyval(V2, np.log(n / f_row1[n2f1_small]))
                c_1[n2f1_large] = np.polyval(V1, np.log(n / f_row1[n2f1_large]))

            # make sure nonzero threshold is higher than 1/n
            c_1[f1nonzero] = np.maximum(c_1[f1nonzero], 1 / (1.9 * np.log(n)))

        prob_mat = ssp.lil_matrix(f_csr.shape)
        prob_mat[fnonzero_rows] = entro_mat(prob_csr.data, n, coeff, c_1)

    return np.array(f_csr.multiply(prob_mat).sum(axis=0)).squeeze() / np.log(2)


def entro_mat(x, n, g_coeff, c_1):
    # g_coeff = {g0, g1, g2, ..., g_K}, K: the order of best polynomial approximation,
    K = len(g_coeff) - 1
    thres = 4 * c_1 * np.log(n) / n
    T, X = np.meshgrid(thres, x)
    ratio = np.minimum(np.maximum(2 * X / T - 1, 0), 1)
    q = np.arange(K).reshape((1, 1, K))
    g = g_coeff.reshape((1, 1, K + 1))
    MLE = - X * np.log(X) + 1 / (2 * n)
    polyApp = np.sum(np.concatenate((T[..., None], ((n * X)[..., None]  - q) / (T[..., None] * (n - q))), axis=2).cumprod(axis=2) * g, axis=2) - X * np.log(T)
    polyfail = np.isnan(polyApp) | np.isinf(polyApp)
    polyApp[polyfail] = MLE[polyfail]
    output = ratio * MLE + (1 - ratio) * polyApp
    return np.maximum(output, 0)


def est_entro_MLE(samp):
    """Maximum likelihood estimate of Shannon entropy (in bits) of the input
    sample

    This function returns a scalar MLE of the entropy of samp when samp is a
    vector, or returns a (row-) vector consisting of the MLE of the entropy
    of each column of samp when samp is a matrix.

    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each
               column of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    f = fingerprint(samp)
    prob = np.arange(1, f.shape[0] + 1) / n
    prob_mat = - prob * np.log2(prob)
    return prob_mat.dot(f)


def formalize_sample(samp):
    samp = np.array(samp)
    if np.any(samp != np.fix(samp)):
        raise ValueError('Input sample must only contain integers.')
    if samp.ndim == 1 or samp.ndim == 2 and samp.shape[0] == 1:
        samp = samp.reshape((samp.size, 1))
    return samp

def fingerprint(samp):
    """A memory-efficient algorithm for computing fingerprint when wid is
    large, e.g., wid = 100
    """
    wid = samp.shape[1]

    d = np.r_[
        np.full((1, wid), True, dtype=bool),
        np.diff(np.sort(samp, axis=0), 1, 0) != 0,
        np.full((1, wid), True, dtype=bool)
    ]

    f_col = []
    f_max = 0

    for k in range(wid):
        a = np.diff(np.flatnonzero(d[:, k]))
        a_max = a.max()
        hist, _ = np.histogram(a, bins=a_max, range=(1, a_max + 1))
        f_col.append(hist)
        if a_max > f_max:
            f_max = a_max

    return np.array([np.r_[col, [0] * (f_max - len(col))] for col in f_col]).T
