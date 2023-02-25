# Word2Vec with Numba

References:

- https://arxiv.org/abs/1301.3781
- https://arxiv.org/abs/1310.4546
- https://code.google.com/archive/p/word2vec/
- https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_inner.pyx
- https://web.stanford.edu/class/cs224n/

Features:

- Only skip-gram negative sampling is implemented
- Multi-threading

On Macbook Air M1 (plugged in), 1 pass over text8 dataset (~17M tokens), embedding size 100, negative sampling 5, window size 5.

Thread count | 1 | 2 | 4
-------------|---|---|---
This (tokens/s) | 123,843 | 242,302 | 464,412
Gensim (tokens/s) | 145,142 | 281,340 | 517,985

NOTE: Gensim throughput is reported by looking at their info logging. I'm not familiar with gensim internals. It seems like they do some trimming of the sentences (fewer tokens in total) so measuring training time is not accurate for comparison.

## Learnings

- Although Word2Vec paper describes skip-gram as center word predicts context words, Google C code actually use each context word to predict center word. Most online resources seem to not be aware of this. FastText documentation describes skip-gram as context word predicts center word. The overall objective is still the same though, since if (A, B) is a skip-gram, (B, A) will also be a skip-gram. However, since Word2Vec uses stochastic gradient descent with batch size = negative + 1, there will probably be some differences.
- Google C code (and Gensim) uses lookup table to compute sigmoid (and log-sigmoid). It doesn't seem to be faster than just computing sigmoid directly. I guess modern computers can do the calculations fairly quickly.
- Numba likes for loops and in-place operations. Allocating new memory is pretty expensive, especially for intermediate/temporary results.
- We can call scipy's BLAS inside numba functions. saxpy can be used to speed up SGD update (Gensim approach, but with Cython). Although scipy's saxpy is faster than naive for-loop implementations, when used in conjunction with the whole training script, scipy's saxpy is a bit slower. Not very sure why. (profiling numba is not easy...)
- For small, simple jit functions, it's a good practice to add type annotation. This ensures the input and return values have the correct data types that you want. Sometimes, it is also faster.
- When adding type annotations for numba, remember to use `[::1]` (contiguous layout) instead of `[:]` (any layout) for arrays. It gives a huge speed boost. One reason is perhaps numba will only generate SIMD instructions for contiguous memory.
- Numba type annotation does not play well with default arguments. If you need default arguments, it's best not to add type annotations.
- There are 2 approaches for multi-threading with numba: (1) use `numba.jit(parallel=True)`, or (2) use `numba.jit(nogil=True)` together with Python's `threading` module (`concurrent.futures.ThreadPoolExecutor` works too). The first approach can be thread-safe, but it is not guaranteed (see [here](https://numba.pydata.org/numba-doc/latest/user/parallel.html)). The second approach is definitely not thread-safe from what I understand, since we can't use Python `threading.Lock` inside numba jit functions.
- Word vectors updates in Google C code (and Gensim) are not thread-safe (no locking mechanism). However, since it is unlikely that any 2 threads will modify a word vector at the same time (at least true for not-so-common words), the impact is small. See [Gensim's note](https://rare-technologies.com/parallelizing-word2vec-in-python/) and [Word2Vec Google Groups' discussion](https://groups.google.com/g/word2vec-toolkit/c/NLvYXU99cAM/m/rryQhcaxKSQJ).
