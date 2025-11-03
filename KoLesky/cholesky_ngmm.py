import sys
import numpy as np
import os
import scipy.sparse as sparse
import scipy.linalg
from tqdm import tqdm
from scipy.spatial import KDTree

# script_dir = os.path.dirname(__file__)
# # Append the directory to conditional-knn
# kolexky_dir = os.path.join(os.path.dirname(script_dir),'conditional-knn','KoLesky')
# print(f"Appending KoLesky directory to sys.path: {kolexky_dir}")
# if kolexky_dir not in sys.path:
#     sys.path.append(kolexky_dir)
from . import cknn, ordering
from .typehints import (
    CholeskyFactor,
    CholeskySelect,
    Empty,
    GlobalSelect,
    Grouping,
    Kernel,
    LengthScales,
    Matrix,
    Ordering,
    Points,
    Select,
    Sparse,
    Sparsity,
)
from .cholesky import __mult_cholesky, __cholesky_subsample

def cholesky_joint(
    x_train: Points,
    x_test: Points,
    kernel: Kernel,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
    useMPI: bool = False
) -> CholeskyFactor:
    """Computes Cholesky of the joint covariance."""
    if useMPI:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0
    if rank == 0:
        print(f"Cholesky joint ordering, useMPI={useMPI}")
    x, order, lengths = __joint_order(x_train, x_test, p=p)
    if rank == 0:
        print(f"Cholesky joint grouping, useMPI={useMPI}")
    sparsity, groups = __cholesky_kl(x, lengths, rho, lambd)
    if useMPI:
        comm.Barrier()
        if rank == 0:
            print("Starting MPI Cholesky factorization")
    return __mult_cholesky(x, kernel, sparsity, groups), order


def cholesky_joint_subsample(
    x_train: Points,
    x_test: Points,
    kernel: Kernel,
    s: float,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
    select: Select = cknn.chol_select,
    useMPI: bool = False
) -> CholeskyFactor:
    """Cholesky of the joint covariance with subsampling."""
    # standard geometric algorithm
    if useMPI:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0
    if rank == 0:
        print(f"Cholesky joint ordering, useMPI={useMPI}")
    x, order, lengths = __joint_order(x_train, x_test, p=p)
    if rank == 0:
        print(f"Cholesky joint grouping, useMPI={useMPI}")
    sparsity, groups = __cholesky_kl(x, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    if rank == 0:
        print(f"Cholesky joint candidate sparsity, useMPI={useMPI}")
    candidate_sparsity = sparsity_pattern(x, lengths, s * rho)
    if rank == 0:
        print(f"Cholesky joint subsampling, useMPI={useMPI}")
    if useMPI:
        comm.Barrier()
        if rank == 0:
            print("Starting MPI Cholesky subsampling")
    return (
        __cholesky_subsample(
            x, kernel, sparsity, candidate_sparsity, groups, select,
            useMPI=useMPI
        ),
        order,
    )


def __joint_order(
    x_train: Points, x_test: Points, p: int = 1
) -> tuple[Points, Ordering, LengthScales]:
    """Return the joint ordering and length scale."""
    # Don't include the eq_id for ordering
    x_train_sub = x_train[:, 1:]
    x_test_sub = x_test[:, 1:]
    train_order, train_lengths = ordering.p_reverse_maximin(x_train_sub, p=p)
    # initialize test point ordering with training points
    test_order, test_lengths = ordering.p_reverse_maximin(x_test_sub, x_train_sub, p=p)
    # put testing points before training points (after in transpose)
    x = np.vstack((x_test[test_order], x_train[train_order]))
    order = np.append(test_order, x_test.shape[0] + train_order)
    lengths = np.append(test_lengths, train_lengths)
    return x, order, lengths

def __cholesky_kl(
    x: Points,
    lengths: LengthScales,
    rho: float,
    lambd: float | None,
) -> tuple[Sparsity, Grouping]:
    """Computes Cholesky given pre-ordered points and length scales."""
    sparsity = sparsity_pattern(x, lengths, rho)
    groups, sparsity = (
        ([[i] for i in range(len(x))], sparsity)
        if lambd is None
        else ordering.supernodes(sparsity, lengths, lambd)
    )
    return sparsity, groups

def sparsity_pattern(x: Points, lengths: LengthScales, rho: float) -> Sparsity:
    """Compute the sparity pattern given the ordered x."""
    # O(n log^2 n + n s)
    x_sub = x[:, 1:]  # exclude eq_id for distance calculation
    tree, offset, length_scale = KDTree(x_sub), 0, lengths[0]
    sparsity = {}
    for i in range(len(x)):
        # length scale doubled, re-build tree to remove old points
        if lengths[i] > 2 * length_scale:
            tree, offset, length_scale = KDTree(x_sub[i:]), i, lengths[i]
        sparsity[i] = [
            offset + j
            for j in tree.query_ball_point(x_sub[i], rho * lengths[i])
            if offset + j >= i
        ]

    return sparsity