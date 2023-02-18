import numba as nb
import numpy as np
from numba import njit


@nb.vectorize(["float32(float32)", "float64(float64)"], nopython=True)
def numba_sigmoid(x):
    return 1 / (1 + np.exp(-x)) if x >= 0 else 1 - 1 / (1 + np.exp(x))


@nb.vectorize(["float32(float32)", "float64(float64)"], nopython=True)
def numba_logsigmoid(x):
    return -np.log(1 + np.exp(-x)) if x >= 0 else x - np.log(1 + np.exp(x))


@nb.njit("uint32(float64[:], float64)")
def numba_bisect_left(arr, x):
    lo, hi = 0, arr.shape[0]
    while hi > lo:
        mid = (lo + hi) >> 1
        if arr[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo


@nb.njit("uint32[:](float64[:], uint64)")
def numba_sample(cdf, size):
    out = np.empty(size, dtype=np.uint32)
    for i in range(size):
        out[i] = numba_bisect_left(cdf, np.random.rand())
    return out


@njit
def skipgram_negative_sampling(
    input_embs: np.ndarray,
    output_embs: np.ndarray,
    target_idx: int,
    context_indices: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    update: bool = False,
    return_loss: bool = False,
):
    # context predicts target, following Google's Word2Vec C code
    n_context = context_indices.shape[0]
    context_embs = input_embs[context_indices]
    target_emb = output_embs[target_idx]
    z = context_embs @ target_emb  # (n_context,)

    grad_z = (numba_sigmoid(z) - labels) / n_context
    grad_inputs = grad_z.reshape(-1, 1) * target_emb.reshape(1, -1)
    grad_output = grad_z @ context_embs

    if update:
        for i in range(n_context):
            input_embs[context_indices[i]] -= grad_inputs[i] * lr
        output_embs[target_idx] -= grad_output * lr

    loss = None
    if return_loss:
        loss = -labels * numba_logsigmoid(z) - (1 - labels) * numba_logsigmoid(-z)
        loss = loss.mean()

    return loss, grad_inputs, grad_output
