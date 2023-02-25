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

On Macbook Air M1 (plugged in), 1 pass over text8 dataset (~17M tokens, ~12.5M tokens after sub-sampling frequent words), embedding size 100, negative sampling 5, window size 5.

Thread count | 1 | 2 | 4
-------------|---|---|---
This (tokens/s) | 116,859 | 240,516 | 446,646
Gensim (tokens/s) | 145,142 | 281,340 | 517,985

## Set up environment

```bash
conda create -n word2vec python=3.10
conda activate word2vec
conda install numpy scipy numba tqdm
```

## Basic usage

```python
import numpy as np

from word2vec import train_word2vec

embs, vocab = train_word2vec("text8", 100, batch_size=10_000, n_workers=4)

np.save("embs.npy", embs)
with open("vocab.txt", "w") as f:
    for token in vocab:
        f.write(token + "\n")
```

Only text file input is supported. Next line character (`\n`) is treated as white space instead of sentence separator. This is to keep the code simple.

## About Word2Vec

There are 2 models for Word2Vec: continuous-bag-of-words (CBOW) or skip-gram. To train either type of Word2Vec, we can use hierarchical softmax or negative sampling loss. I only implement skip-gram + negative sampling loss combination.

Negative sampling loss is a modified version of noise-contrastive-estimation (NCE). Instead of learning a particular distribution function directly (i.e. multi-class classification), NCE proposes learning a Bernoulli distribution (i.e. binary classification): given a mixture of the original distribution and a noise distribution, what is the probability that a sample comes from the original distribution. Since Word2Vec does not care about the actual skip-gram distribution, the probability mentioned previously is modelled directly by the inner product between two word vectors.

The loss function is simply binary cross entropy. The loss function (and its back-propagation) can be implemented in fewer than 10 lines. The challenging part is to implement the **sampling** logic.

- Frequent words sub-sampling: For words appearing more than 0.1% of the time, there is a non-zero probability that the word will be **removed** from the text. This also means that the surrounding words will have larger context while excluding the removed word.
- Skip-gram sampling: For each word, we sample skip-grams `(word, context_word)` for all `context_word`s within the context window.
- Negative sampling: These are samples from the **noise distribution**. Word2Vec uses unigram global distribution raised to the power of 0.75.

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
