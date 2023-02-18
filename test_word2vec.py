import numba as nb
import numpy as np
from functools import partial

from word2vec import numba_logsigmoid, numba_sample, numba_sigmoid, skipgram_negative_sampling


def _check_ufunc_dtype(func):
    assert isinstance(func(0), np.float64)
    assert isinstance(func(0.0), np.float64)
    assert isinstance(func(np.float32(0)), np.float32)


def test_numba_sigmoid():
    _check_ufunc_dtype(numba_sigmoid)
    assert np.allclose(numba_sigmoid(0), 0.5)
    assert np.allclose(numba_sigmoid(np.log(3)), 0.75)

    # numerical stability
    assert np.allclose(numba_sigmoid(800), 1)
    assert np.allclose(numba_sigmoid(-800), 0)

    x = np.random.randn(10, 20)
    assert numba_sigmoid(x).shape == x.shape


def test_numba_logsigmoid():
    _check_ufunc_dtype(numba_logsigmoid)
    assert np.allclose(numba_logsigmoid(0), -np.log(2))

    # numerical stability
    assert np.allclose(numba_logsigmoid(800), 0)
    assert np.allclose(numba_logsigmoid(-800), -800)

    x = np.random.randn(10, 20)
    assert numba_logsigmoid(x).shape == x.shape


def test_numba_sample():
    np.random.seed(0)
    nb.njit(lambda x: np.random.seed(x))(0)

    N = 10
    p = np.arange(N) / N
    p /= p.sum()
    cdf = np.cumsum(p)

    assert np.allclose(np.random.choice(N, size=N, p=p), numba_sample(cdf, N))


def grad_check(func, x, eps=1e-4):
    _, grad = func(x)
    loss_left, _ = func(x - eps)
    loss_right, _ = func(x + eps)
    _grad = (loss_right - loss_left) / eps / 2
    assert np.allclose(_grad, grad)


def test_skipgram_negative_sampling_grad():
    input_embs = np.random.randn(4, 10)
    output_embs = np.random.randn(4, 10)
    target_idx = 0
    context_indices = np.array([1, 2, 3])
    labels = np.array([1, 0, 0])

    def input_wrapper(x, emb_idx):
        _input_embs = input_embs.copy()
        _input_embs[context_indices[0], emb_idx] += x
        loss, grad_inputs, _ = skipgram_negative_sampling(
            _input_embs,
            output_embs,
            target_idx,
            context_indices,
            labels,
            return_loss=True,
        )
        return loss, grad_inputs[0, emb_idx]

    for i in range(10):
        grad_check(partial(input_wrapper, emb_idx=i), 0)

    def output_wrapper(x, emb_idx):
        _output_embs = input_embs.copy()
        _output_embs[target_idx, emb_idx] += x
        loss, _, grad_output = skipgram_negative_sampling(
            input_embs,
            _output_embs,
            target_idx,
            context_indices,
            labels,
            return_loss=True,
        )
        return loss, grad_output[emb_idx]

    for i in range(10):
        grad_check(partial(output_wrapper, emb_idx=i), 0)
