from functools import partial

import numba as nb
import numpy as np

from word2vec import numba_logsigmoid, numba_sample, numba_sigmoid, skipgram_negative_sampling_step


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

    np_samples = np.random.choice(N, size=N, p=p)
    nb_samples = numba_sample(cdf, N, np.empty(N, dtype=np.uint32))
    assert np.allclose(np_samples, nb_samples)


def grad_check(func, eps=1e-4):
    _, grad = func(0)
    loss_left, _ = func(-eps)
    loss_right, _ = func(eps)
    _grad = (loss_right - loss_left) / eps / 2
    assert np.allclose(_grad, grad)


def test_skipgram_negative_sampling_grad():
    emb_dim = 10
    input_embs = np.random.randn(4, emb_dim)
    output_embs = np.random.randn(4, emb_dim)
    context_idx = 0
    target_indices = np.array([1, 2, 3])

    func = partial(
        skipgram_negative_sampling_step,
        input_embs,
        output_embs,
        context_idx,
        target_indices,
        return_loss=True,
        return_grad=True,
    )

    for j in range(emb_dim):

        def wrapper(x):
            input_embs[context_idx, j] += x
            loss, grad_input, _ = func()
            input_embs[context_idx, j] -= x
            return loss, grad_input[j]

        grad_check(wrapper)

    for i, target_idx in enumerate(target_indices):
        for j in range(emb_dim):

            def wrapper(x):
                output_embs[target_idx, j] += x
                loss, _, grad_outputs = func()
                output_embs[target_idx, j] -= x
                return loss, grad_outputs[i, j]

            grad_check(wrapper)

    input_before = input_embs[context_idx].copy()
    outputs_before = output_embs[target_indices].copy()
    _, grad_input, grad_outputs = skipgram_negative_sampling_step(
        input_embs, output_embs, context_idx, target_indices, lr=0.025, update=True, return_grad=True
    )
    assert not np.allclose(input_before, input_embs[context_idx])
    input_after = input_before - grad_input * 0.025
    assert np.allclose(input_after, input_embs[context_idx])

    assert not np.allclose(outputs_before, output_embs[target_indices])
    outputs_after = outputs_before - grad_outputs * 0.025
    assert np.allclose(outputs_after, output_embs[target_indices])
