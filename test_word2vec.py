from functools import partial

import numpy as np

from word2vec import binary_cross_entropy, numba_logsigmoid, numba_sigmoid


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


def grad_check(func, eps=1e-4):
    _, grad = func(0)
    loss_left, _ = func(-eps)
    loss_right, _ = func(eps)
    _grad = (loss_right - loss_left) / eps / 2
    assert np.allclose(_grad, grad)


def test_skipgram_negative_sampling_grad():
    emb_dim = 10
    embs = np.random.randn(2, emb_dim)

    def wrapper(x, row_idx, col_idx, label):
        grad_embs = np.zeros((2, emb_dim))
        embs[row_idx, col_idx] += x
        loss = binary_cross_entropy(embs[0], embs[1], label, grad_embs[0], grad_embs[1], 1.0, True)
        embs[row_idx, col_idx] -= x
        return loss, -grad_embs[row_idx, col_idx]

    for row_idx in range(2):
        for label in (0.0, 1.0):
            for col_idx in range(emb_dim):
                grad_check(partial(wrapper, row_idx=row_idx, col_idx=col_idx, label=label))
