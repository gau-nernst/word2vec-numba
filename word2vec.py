from time import time

import numba as nb
import numpy as np


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


@nb.njit("uint32[:](float64[:], uint64, uint32[:])")
def numba_sample(cdf, size, out):
    for i in range(size):
        out[i] = numba_bisect_left(cdf, np.random.rand())
    return out


@nb.njit(["(float32,float32[:],float32[:])", "(float64,float64[:],float64[:])"])
def saxpy(a, x, y):
    for i in range(x.shape[0]):
        y[i] += a * x[i]


@nb.njit
def skipgram_negative_sampling_step(
    input_embs: np.ndarray,
    output_embs: np.ndarray,
    context_idx: np.ndarray,
    target_indices: np.ndarray,
    lr: float = 0.025,
    update: bool = False,
    return_loss: bool = False,
    return_grad: bool = False,
):
    # NOTE: context predicts target, following Google's Word2Vec C code
    # NOTE: computing loss and allocating memory for gradients is expensive
    # only compute them for diagnostic and monitoring
    context_emb = input_embs[context_idx]  # this is a view
    target_embs = output_embs[target_indices]  # this is a copy
    z = target_embs @ context_emb

    # NOTE: z will be modified in-place, so loss needs to be calculated here
    loss = None
    if return_loss:
        loss = -numba_logsigmoid(z[0]) - numba_logsigmoid(-z[1:]).sum()

    # NOTE: y_hat = sigmoid(z); dloss / dz = y_hat - y
    for i in range(z.shape[0]):
        z[i] = numba_sigmoid(z[i])
    z[0] -= 1

    grad_input = z @ target_embs
    grad_outputs = None
    if return_grad:
        grad_outputs = z.reshape(-1, 1) * context_emb.reshape(1, -1)

    if update:
        # NOTE: order of update is important, since they are in-place operations
        for i in range(target_indices.shape[0]):
            saxpy(-lr * z[i], context_emb, output_embs[target_indices[i]])
        saxpy(-lr, grad_input, context_emb)

    return loss, grad_input, grad_outputs


@nb.njit
def skipgram_negative_sampling_train(
    stream: np.ndarray,
    input_embs: np.ndarray,
    output_embs: np.ndarray,
    unigram_cdf: np.ndarray,
    window_size: int = 10,
    negative: int = 5,
    lr: float = 0.025,
    lr_anneal_interval: int = 10_000,
):
    interval = 10_000
    current_lr = lr
    target_indices = np.empty(1 + negative, dtype=np.uint32)
    with nb.objmode(time0="float64"):
        time0 = time.time()

    for stream_pos in range(stream.shape[0]):
        target_indices[0] = stream[stream_pos]

        if (stream_pos + 1) % lr_anneal_interval == 0:
            current_lr = lr * (1 - stream_pos / stream.shape[0])
            current_lr = max(current_lr, lr * 0.0001)

        truncated_window_size = np.random.randint(window_size) + 1
        window_start = max(stream_pos - truncated_window_size, 0)
        window_end = min(stream_pos + truncated_window_size + 1, stream.shape[0])

        for window_idx in range(window_start, window_end):
            if window_idx == stream_pos:
                continue

            numba_sample(unigram_cdf, negative, target_indices[1:])
            loss, _, _ = skipgram_negative_sampling_step(
                input_embs,
                output_embs,
                stream[window_idx],
                target_indices,
                lr=current_lr,
                update=True,
                return_loss=(stream_pos + 1) % interval == 0,
            )

        if (stream_pos + 1) % interval == 0:
            with nb.objmode(time0="float64"):
                progress = (stream_pos + 1) / stream.shape[0] * 100
                throughput = int(interval / (time.time() - time0))
                print(
                    "\rlr: {:.2e}  Progress: {:.2f}%  Loss: {:.4f}  Words/sec: {:,}".format(
                        current_lr, progress, loss, throughput
                    ),
                    end="",
                )
                time0 = time.time()
