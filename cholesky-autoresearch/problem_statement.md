# Batched dense Cholesky factorization

This is the fixed competition contract, derived from the [pinned official problem](https://github.com/gpu-mode/reference-kernels/tree/f83cb92a534cc67fa98d7e9813e7f8238abdb4e0/problems/linalg/cholesky_py).

## Task

Implement `custom_kernel(data)` in `submission.py`.

Input `A` is a CUDA `torch.float32` tensor shaped `(batch, n, n)`. Every matrix is symmetric positive definite up to FP32 roundoff. Return a CUDA FP32 tensor `L` with the same shape and device, strictly positive diagonal, and lower-triangular structure such that:

```text
A = L @ L.T
```

Correctness is property-based, not an elementwise comparison with one library implementation.

## Exact checker

For every matrix, with `eps = torch.finfo(torch.float32).eps` and `scale = ||A||₁` clamped to the smallest positive FP32 value, the checker requires:

- output is a tensor with the exact input shape, FP32 dtype, and device;
- every output value is finite;
- every diagonal value is strictly positive;
- `||triu(L, 1)||₁ <= 8 * n * eps * scale`;
- `||L @ L.T - A||₁ <= 20 * n * eps * scale`.

The reconstruction multiplication is evaluated with TF32 disabled.

## Input coverage

Inputs include dense covariance-like matrices, planted spectra, exact diagonal matrices, damped low-rank matrices, scaled rows and columns, and tridiagonal SPD matrices. `cond` is a deterministic dynamic-range control, not an exact condition number.

Released correctness rows:

```text
dense:       (16,32,2) (16,64,2) (16,128,2) (8,256,2)
             (4,512,2) (2,1024,2) (1,2048,2)
spectrum:    (32,32,5) (16,64,5) (8,128,5)
diagonal:    (32,32,5) (16,64,5) (8,128,5)
lowrank:     (4,256,4) (2,1024,4)
rowscale:    (4,512,4)
tridiagonal: (4,512,1)
```

Each tuple is `(batch, n, cond)`. Service runs may reseed inputs.

## Benchmark and score

Target GPU: NVIDIA B200. All released benchmark rows are dense with `cond=2`:

```text
(4096,32) (1024,64) (256,128) (64,256)
(16,512) (640,512) (4,1024) (60,1024)
(2,2048) (8,2048) (1,4096) (2,4096)
(1,8192) (1,16384) (1,32768)
```

Each tuple is `(batch, n)`. Passing submissions are ranked by the geometric mean of runtime across all 15 rows, so every row has equal log-space weight.

The evaluator prepares up to 256 MiB of inputs per row, performs an untimed correctness warmup, disturbs L2 before timed repeats, times all prepared calls with CUDA events, and rechecks every returned output against preserved originals. Compilation is outside the scored interval; launches, allocations, copies, synchronization, and library calls inside `custom_kernel` count.

## Legitimate-solution boundary

The submission must compute a valid Cholesky factor for arbitrary fresh inputs satisfying this contract. Shape specialization and data-dependent algorithms are allowed. The following are not legitimate optimizations:

- hardcoding outputs for public seeds, cases, or call order;
- caching answers by tensor identity, storage address, evaluator reuse, or previous calls;
- mutating inputs or returning outputs that alias mutable storage reused by another call;
- monkey-patching the checker, evaluator, CUDA-event timing, random seeds, or framework functions to bypass validation or measurement;
- exploiting benchmark mechanics instead of performing the requested factorization.

Legal CUDA, PyTorch, Triton, and vendor-library mechanisms may be used when the competition environment permits them. The current `submission.py` is the official `torch.linalg.cholesky_ex` starter baseline.
