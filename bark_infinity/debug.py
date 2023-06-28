# benchmark.py

import numpy as np
from time import time
import torch


def numpy_benchmark():
    np.random.seed(0)  # for reproducibility

    size = 4096
    A, B = np.random.random((size, size)), np.random.random((size, size))
    C, D = np.random.random((size * 1280,)), np.random.random(
        (size * 1280,)
    )  # increase vector size for benchmark
    E = np.random.random((int(size / 2), int(size / 4)))
    F = np.random.random((int(size / 2), int(size / 2)))
    F = np.dot(F, F.T)
    G = np.random.random((int(size / 2), int(size / 2)))
    H = np.random.random((size, size))
    I = np.random.random((int(size), int(size)))

    print("\nNUMPY CONFIGURATION:")
    print(np.show_config())

    print("\nNUMPY BENCHMARK RESULTS:")

    t0 = time()
    # Matrix multiplication
    N = 20
    t = time()
    for i in range(N):
        np.dot(A, B)
    delta = time() - t
    print(f"1. Dotted two {size}x{size} matrices in {delta / N:.3f} s.")
    del A, B

    # Vector multiplication
    N = 5000
    t = time()
    for i in range(N):
        np.dot(C, D)
    delta = time() - t
    print(f"2. Dotted two vectors of length {size * 1280} in {1e3 * delta / N:.3f} ms.")
    del C, D

    # Singular Value Decomposition (SVD)
    N = 3
    t = time()
    for i in range(N):
        np.linalg.svd(E, full_matrices=False)
    delta = time() - t
    print(f"3. SVD of a {size // 2}x{size // 4} matrix in {delta / N:.3f} s.")
    del E

    # Cholesky Decomposition
    N = 3
    t = time()
    for i in range(N):
        np.linalg.cholesky(F)
    delta = time() - t
    print(f"4. Cholesky decomposition of a {size // 2}x{size // 2} matrix in {delta / N:.3f} s.")

    # Eigendecomposition
    t = time()
    for i in range(N):
        np.linalg.eig(G)
    delta = time() - t
    print(f"5. Eigendecomposition of a {size // 2}x{size // 2} matrix in {delta / N:.3f} s.")

    # compute covariance matrix
    N = 10
    t = time()
    for i in range(N):
        np.dot(H.T, H)
    delta = time() - t
    print(f"6. Computing Covariance Matrix of a {size}x{size} matrix in {delta / N:.4f} s.")

    # compute inverse matrix
    N = 3
    t = time()
    for i in range(N):
        np.linalg.inv(I)
    delta = time() - t
    print(f"7. Inverse Matrix of a {size}x{size} matrix in {delta / N:.4f} s.")

    # Gradient calculation
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)
    learning_rate = 1e-6

    t = time()
    for _ in range(10000):
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)
        loss = np.square(y_pred - y).sum()
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    delta = time() - t
    print(f"8. Gradient calculation time: {delta:.3f} s.")

    J = np.random.rand(size * 1280)
    t = time()
    N = 5
    for _ in range(N):
        sorted_indices = np.argsort(J)[::-1]
        cumulative_probs = np.cumsum(sorted_indices)
        sorted_indices_to_remove = cumulative_probs > np.random.rand()
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        J[sorted_indices[sorted_indices_to_remove]] = -np.inf
    delta = time() - t
    print(
        f"9. np.argsort and np.cumsum on a vector of length {size*1280} in {1e3 * delta / N:.3f} ms."
    )
    del J

    K, L = np.random.random((size, 1)), np.random.random((1, size))
    t = time()
    N = 200
    for _ in range(N):
        M = K * L
    delta = time() - t
    print(f"10. Broadcasting two vectors of length {size} in {1e3 * delta / N:.3f} ms.")
    del K, L, M

    N = np.random.random((size, size))
    indices = np.random.randint(size, size=(size,))
    t = time()
    M = 200
    for _ in range(M):
        O = N[indices, :]
    delta = time() - t
    print(f"11. Indexing a {size}x{size} matrix in {1e3 * delta / M:.3f} ms.")
    del N, O

    P = np.random.random((size, size))
    t = time()
    M = 100
    for _ in range(M):
        s = np.sum(P)
    delta = time() - t
    print(f"12. Sum reduction of a {size}x{size} matrix in {1e3 * delta / M:.3f} ms.")
    del P

    Q = np.random.random((size, size))
    R = torch.tensor(Q)

    # Numpy to PyTorch
    t = time()
    N = 100
    for _ in range(N):
        R = torch.from_numpy(Q)
    delta = time() - t
    print(
        f"13. Conversion of a Numpy {size}x{size} matrix to PyTorch tensor in {1e3 * delta / N:.3f} ms."
    )

    # PyTorch to Numpy
    t = time()
    for _ in range(N):
        Q_new = R.numpy()
    delta = time() - t
    print(
        f"14. Conversion of a PyTorch tensor {size}x{size} to Numpy array in {1e3 * delta / N:.3f} ms."
    )
    del Q, R

    # Benchmark for conversion operations
    Q = np.random.random((size, size)).astype(np.float32)
    R = torch.tensor(Q)

    # Numpy to PyTorch with forced copy via type conversion
    t = time()
    N = 100
    for _ in range(N):
        R = torch.tensor(Q, dtype=torch.float64)
    delta = time() - t
    print(
        f"15. Conversion of a Numpy {size}x{size} matrix to PyTorch tensor with forced copy in {1e3 * delta / N:.3f} ms."
    )

    # PyTorch to Numpy with forced copy via operation that doesn't change data
    t = time()
    for _ in range(N):
        Q_new = (R + 0).numpy()
    delta = time() - t
    print(
        f"16. Conversion of a PyTorch tensor {size}x{size} to Numpy array with forced copy in {1e3 * delta / N:.3f} ms."
    )
    del Q, R

    t1 = time()
    print(f"\nTotal time: {t1 - t0:.3f}s \n\n")


if __name__ == "__main__":
    numpy_benchmark()
