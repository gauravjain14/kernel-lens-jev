"""Kernel Lens demo. A deliberately incomplete mask for an irregular row width.
Intent: row-wise FP32 softmax; N may be any positive size, including 513.
Try adding mask=cols < N to the store, then watch the next assessment.
This file is an inspection example, not a runnable benchmark.
"""
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * stride + cols
    values = tl.load(X + offsets, mask=cols < N, other=-float("inf"))
    shifted = values - tl.max(values, axis=0)
    numerators = tl.exp(shifted)
    probabilities = numerators / tl.sum(numerators, axis=0)
    tl.store(Y + offsets, probabilities)
