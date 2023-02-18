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
    context_idx: np.ndarray,
    target_indices: np.ndarray,
    lr: float = 0.01,
    update: bool = False,
    return_loss: bool = False,
):
    # context predicts target, following Google's Word2Vec C code
    context_emb = input_embs[context_idx]
    target_embs = output_embs[target_indices]

    z = target_embs @ context_emb
    grad_z = numba_sigmoid(z)
    grad_z[0] -= 1

    grad_input = grad_z @ target_embs
    grad_outputs = grad_z.reshape(-1, 1) * context_emb.reshape(1, -1)

    if update:
        input_embs[context_idx] -= grad_input * lr
        for i in range(target_indices.shape[0]):
            output_embs[target_indices[i]] -= grad_outputs[i] * lr

    loss = None
    if return_loss:
        loss = -numba_logsigmoid(z[0]) - numba_logsigmoid(-z[1:]).sum()

    return loss, grad_input, grad_outputs
