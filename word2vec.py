import time

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


@nb.njit(["(float32,float32[:],float32[:])", "(float64,float64[:],float64[:])"])
def axpy(a, x, y):
    for i in range(x.shape[0]):
        y[i] += a * x[i]


@nb.njit
def binary_cross_entropy(in_emb, out_emb, label, grad_in_emb, grad_out_emb, lr, return_loss):
    z = np.vdot(in_emb, out_emb)
    g = (label - numba_sigmoid(z)) * lr  # this is -dloss / dz * lr
    axpy(g, out_emb, grad_in_emb)
    axpy(g, in_emb, grad_out_emb)
    return -label * numba_logsigmoid(z) - (1 - label) * numba_logsigmoid(-z) if return_loss else 0.0


@nb.njit
def skipgram_negative_sampling_step(input_embs, output_embs, in_idx, out_idx, noise_cdf, negative, lr, return_loss):
    grad_in_emb = np.zeros(input_embs.shape[1], dtype=input_embs.dtype)
    in_emb = input_embs[in_idx]  # this is a view
    out_emb = output_embs[out_idx]

    # accumulate gradients for in_emb. directly update out_emb
    loss = binary_cross_entropy(in_emb, out_emb, 1.0, grad_in_emb, out_emb, lr, return_loss)
    for _ in range(negative):
        noise_idx = numba_bisect_left(noise_cdf, np.random.rand())
        if noise_idx == out_idx:
            continue
        out_emb = output_embs[noise_idx]
        loss += binary_cross_entropy(in_emb, out_emb, 0.0, grad_in_emb, out_emb, lr, return_loss)

    axpy(1.0, grad_in_emb, in_emb)
    return loss


@nb.njit
def skipgram_negative_sampling_train(
    stream: np.ndarray,
    input_embs: np.ndarray,
    output_embs: np.ndarray,
    noise_cdf: np.ndarray,
    window_size: int = 10,
    negative: int = 5,
    lr: float = 0.025,
    lr_anneal_interval: int = 10_000,
):
    # NOTE: context predicts target, following Google's Word2Vec C code
    # NOTE: computing loss is expensive
    # only compute them for diagnostic and monitoring
    interval = 10_000
    current_lr = lr
    with nb.objmode(time0="float64"):
        time0 = time.time()

    for stream_pos in range(stream.shape[0]):
        if (stream_pos + 1) % lr_anneal_interval == 0:
            current_lr = lr * (1 - stream_pos / stream.shape[0])
            current_lr = max(current_lr, lr * 0.0001)

        truncated_window_size = np.random.randint(window_size) + 1
        window_start = max(stream_pos - truncated_window_size, 0)
        window_end = min(stream_pos + truncated_window_size + 1, stream.shape[0])

        for window_idx in range(window_start, window_end):
            if window_idx == stream_pos:
                continue

            loss = skipgram_negative_sampling_step(
                input_embs,
                output_embs,
                stream[window_idx],
                stream[stream_pos],
                noise_cdf,
                negative,
                current_lr,
                (stream_pos + 1) % interval == 0,
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
