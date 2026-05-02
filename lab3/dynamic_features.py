import numpy as np

def stack_context(features, context=3):
    """
    Stack frames [n-3, ..., n, ..., n+3]
    using mirrored boundaries.
    """

    N, D = features.shape
    stacked = []

    for n in range(N):
        frames = []

        for offset in range(-context, context + 1):
            idx = n + offset

            if idx < 0:
                idx = -idx
            elif idx >= N:
                idx = 2 * N - idx - 2

            frames.append(features[idx])

        stacked.append(np.concatenate(frames))

    return np.array(stacked)