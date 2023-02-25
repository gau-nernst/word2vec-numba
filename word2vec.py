import logging

import numba as nb
import numpy as np


LOGGER = logging.getLogger(__name__)


@nb.vectorize(["float32(float32)", "float64(float64)"], nopython=True, fastmath=True)
def numba_sigmoid(x):
    return 1 / (1 + np.exp(-x)) if x >= 0 else 1 - 1 / (1 + np.exp(x))


@nb.vectorize(["float32(float32)", "float64(float64)"], nopython=True)
def numba_logsigmoid(x):
    return -np.log(1 + np.exp(-x)) if x >= 0 else x - np.log(1 + np.exp(x))


@nb.njit("int64(float64[::1], float64)")
def numba_bisect_left(arr, x):
    lo, hi = 0, arr.shape[0]
    while hi > lo:
        mid = (lo + hi) >> 1
        if arr[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo


@nb.njit(["(float32,float32[::1],float32[::1])", "(float64,float64[::1],float64[::1])"])
def axpy(a, x, y):
    for i in range(x.shape[0]):
        y[i] += a * x[i]


@nb.njit
def binary_cross_entropy(in_emb, out_emb, label, grad_in_emb, grad_out_emb, lr, return_loss=False):
    z = np.vdot(in_emb, out_emb)
    g = (label - numba_sigmoid(z)) * lr  # this is -dloss / dz * lr
    axpy(g, out_emb, grad_in_emb)
    axpy(g, in_emb, grad_out_emb)
    return -label * numba_logsigmoid(z) - (1 - label) * numba_logsigmoid(-z) if return_loss else 0.0


@nb.njit
def skipgram_ns_step(input_embs, output_embs, in_idx, out_idx, noise_cdf, negative, lr):
    in_emb = input_embs[in_idx]  # this is a view
    out_emb = output_embs[out_idx]
    grad_in_emb = np.zeros_like(in_emb)

    # accumulate gradients for in_emb. directly update out_emb
    binary_cross_entropy(in_emb, out_emb, 1.0, grad_in_emb, out_emb, lr)
    for _ in range(negative):
        noise_idx = numba_bisect_left(noise_cdf, np.random.rand())
        if noise_idx == out_idx:
            continue
        out_emb = output_embs[noise_idx]
        binary_cross_entropy(in_emb, out_emb, 0.0, grad_in_emb, out_emb, lr)

    axpy(1.0, grad_in_emb, in_emb)


@nb.njit(parallel=True)
def skipgram_ns_batch(stream, input_embs, output_embs, noise_cdf, window_size, negative, lr):
    for stream_pos in nb.prange(stream.shape[0]):
        truncated_window_size = np.random.randint(window_size) + 1
        window_start = max(stream_pos - truncated_window_size, 0)
        window_end = min(stream_pos + truncated_window_size + 1, stream.shape[0])

        for window_idx in range(window_start, window_end):
            if window_idx == stream_pos:
                continue

            # NOTE: context predicts target, following Google's Word2Vec C code
            skipgram_ns_step(
                input_embs,
                output_embs,
                stream[window_idx],  # context word
                stream[stream_pos],  # center word
                noise_cdf,
                negative,
                lr,
            )


def train_word2vec(
    filename: str,
    emb_dim: int,
    window_size: int = 5,
    negative: int = 5,
    lr: float = 0.025,
    min_count: int = 5,
    batch_size: int = 10_000,
    n_workers: int = 1,
):
    token_ids, vocab, noise_cdf = prepare_data(filename, min_count)
    n_tokens = token_ids.shape[0]
    n_vocab = len(vocab)
    LOGGER.info(f"Number of tokens: {n_tokens:,}")
    LOGGER.info(f"Vocabulary size: {n_vocab:,}")

    embs_shape = (n_vocab, emb_dim)
    rng = np.random.default_rng()
    input_embs = (rng.random(embs_shape, dtype=np.float32) - 0.5) / emb_dim
    output_embs = np.zeros(embs_shape, dtype=np.float32)

    nb.set_num_threads(n_workers)
    i = 0
    while i < n_tokens:
        batch = token_ids[i : min(i + batch_size, n_tokens)]
        _lr = lr * (1 - i / n_tokens)
        skipgram_ns_batch(batch, input_embs, output_embs, noise_cdf, window_size, negative, _lr)
        i += batch_size

    return input_embs, vocab


def prepare_data(filename: str, min_count: int):
    def tokens_from_file(filename):
        with open(filename) as f:
            for line in f:
                for token in line.split():
                    yield token

    vocab = dict()
    for token in tokens_from_file(filename):
        vocab[token] = vocab.get(token, 0) + 1
    vocab = {k: v for k, v in vocab.items() if v >= min_count}
    token_to_id = {k: idx for idx, k in enumerate(vocab.keys())}

    noise_cdf = np.empty(len(vocab))
    for i, count in enumerate(vocab.values()):
        noise_cdf[i] = count * 0.75
    noise_cdf /= noise_cdf.sum()
    np.cumsum(noise_cdf, out=noise_cdf)

    token_ids = np.empty(sum(vocab.values()), dtype=np.uint32)
    i = 0
    for token in tokens_from_file(filename):
        if token in token_to_id:
            token_ids[i] = token_to_id[token]
            i += 1

    return token_ids, list(vocab.keys()), noise_cdf
