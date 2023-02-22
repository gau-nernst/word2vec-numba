# Word2Vec with Numba

References:

- https://arxiv.org/abs/1301.3781
- https://arxiv.org/abs/1310.4546
- https://code.google.com/archive/p/word2vec/
- https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_inner.pyx
- https://web.stanford.edu/class/cs224n/

## Learnings

- Although Word2Vec paper describes skip-gram as center word predicts context words, Google C code actually use each context word to predict center word. Most online resources seem to not be aware of this. FastText documentation describes skip-gram as context word predicts center word. The overall objective is still the same though, since if (A, B) is a skip-gram, (B, A) will also be a skip-gram. However, since Word2Vec uses stochastic gradient descent with batch size = negative + 1, there will probably be some differences.
- Google C code (and Gensim) uses lookup table to compute sigmoid (and log-sigmoid).
- Numba likes for loops and in-place operations. Allocating new memory is pretty expensive, especially for intermediate/temporary results.
- We can call scipy's BLAS inside numba functions. saxpy can be used to speed up SGD update (Gensim approach, but with Cython). Although scipy's saxpy is faster than naive for-loop implementations, when used in conjunction with the whole training script, scipy's saxpy is a bit slower. Not very sure why. (profiling numba is not easy...)
- For small, simple jit functions, it's a good practice to add type annotation. This ensures the input and return values have the correct data types that you want. Sometimes, it is also faster.
- When adding type annotations for numba, remember to use `[::1]` (C layout) instead of `[:]` (any layout) for arrays. It gives a huge speed boost. One reason is perhaps numba will only SIMD for contiguous memory.
- Numba type annotation does not play well with default arguments. If you need default arguments, it's best not to add type annotations.
