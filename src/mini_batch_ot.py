import numpy as np
import ot

def mini_batch(data, batch_size):
    """
    Select a subset of sample space according to measure
    with np.random.choice

    Parameters
    ----------
    - data : ndarray(N, d)
    - batch_size : int
        batch size 'm'

    Returns
    -------
    - minibatch : ndarray(ns, nt)
        minibatch of samples
    - sub_weights : ndarray(m,)
        distribution weights of the minibatch
    - id_batch : ndarray(N_data,)
        index of minibatch elements
    """
    id_ = np.random.choice(np.shape(data)[0], batch_size, replace=False)
    sub_weights = ot.unif(batch_size)
    return data[id_], sub_weights, id_

def update_plan(pi, pi_minibatch, id_a, id_b):
    """
    Update the full mini batch transportation matrix

    Parameters
    ----------
    - pi : ndarray(ns, nt)
        full minibatch transportation matrix
    - pi_mb : ndarray(m, m)
        minibatch transportation plan
    - id_a : ndarray(m)
        selected samples from source
    - id_b : ndarray(m)
        selected samples from target

    Returns
    -------
    - pi : ndarray(ns, nt)
        updated transportation matrix
    """
    for i, i2 in enumerate(id_a):
        for j, j2 in enumerate(id_b):
            pi[i2, j2] += pi_minibatch[i][j]
    return pi

def compute_incomplete_plan(xs, xt, a, b, bs, K, C, lambd=1e-1, method="exact"):
    """
    Compute the minibatch gamma with stochastic source and target

    Parameters
    ----------
    - xs : ndarray(ns, d)
        source data
    - xt : ndarray(nt, d)
        target data
    - a : ndarray(ns)
        source distribution weights
    - b : ndarray(nt)
        target distribution weights
    - bs : int
        batch size
    - K : int
        number of batch couples
    - C : ndarray(ns, nt)
        cost matrix
    - lambda : float
        entropic reg parameter
    - method : char
        name of method (entropic or emd)

    Returns
    -------
    - incomplete_pi : ndarray(ns, nt)
        incomplete minibatch OT matrix
    """
    incomplete_pi = np.zeros((np.shape(xs)[0], np.shape(xt)[0]))
    for i in range(K):
        # Select a source and target mini-batch couple
        sub_xs, sub_weights_a, id_a = mini_batch(xs, bs)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, bs)

        # compute ground cost between minibatches
        mb_C = C[id_a, :][:, id_b]
        # The minibatch Cost could be computed on the fly instead of using the full-size ground cost

        # Solve the OT problem between minibatches
        if method == "exact":
            G0 = ot.emd(sub_weights_a, sub_weights_b, mb_C.copy())

        elif method == "entropic":
            G0 = ot.sinkhorn(sub_weights_a, sub_weights_b, mb_C, lambd)

        # Update the transport plan
        incomplete_pi = update_plan(incomplete_pi, G0, id_a, id_b)

    return (1 / K) * incomplete_pi