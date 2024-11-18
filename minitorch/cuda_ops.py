# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i >= out_size:
            return

        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)

        out_pos = index_to_position(out_index, out_strides)
        in_pos = index_to_position(in_index, in_strides)

        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)  # shared memory
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # global index
    pos = cuda.threadIdx.x  # thread pos

    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0

    cuda.syncthreads()

    # tree based reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # Write result to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)  # a_storage??
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x  # pos without global id

        to_index(out_pos, out_shape, out_index)  # writing out_pos to out_index
        cache[pos] = reduce_value  # Initialize the cache with the start value

        reduce_size = a_shape[reduce_dim]  # size of dimension to be reduced

        # loading into shared memory
        for i in range(
            pos, reduce_size, THREADS_PER_BLOCK
        ):  # start from pos, end at reduce_size, step by threads per block to maximize parallelization
            a_index = cuda.local.array(
                MAX_DIMS, numba.int32
            )  # create a local index array for tensor 'a'
            for d in range(len(out_shape)):  # copy the output index to the input index
                a_index[d] = out_index[d]
            # Set the reduction dimension index
            a_index[reduce_dim] = i
            a_pos = index_to_position(
                a_index, a_strides
            )  # Compute the pos in a_storage
            # Apply the reduction function
            cache[pos] = fn(cache[pos], a_storage[a_pos])

        cuda.syncthreads()

        # parallel reduction
        stride = THREADS_PER_BLOCK // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride = stride // 2

        # Write the final reduced value to the output storage
        if pos == 0:
            out_index[reduce_dim] = (
                0  # reset the reduced dimension to 0 to align with index in out_shape
            )
            o_pos = index_to_position(out_index, out_strides)  # pos in out storage
            out[o_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # shared memory for matrices A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # calculate thread indices within the block
    thread_x = cuda.threadIdx.x  # col
    thread_y = cuda.threadIdx.y  # row

    # load into shared memory
    if thread_y < size and thread_x < size:
        a_shared[thread_y, thread_x] = a[
            thread_y * size + thread_x
        ]  # row index* stride + cur col
    else:
        a_shared[thread_y, thread_x] = 0.0

    if thread_y < size and thread_x < size:
        b_shared[thread_y, thread_x] = b[thread_y * size + thread_x]
    else:
        b_shared[thread_y, thread_x] = 0.0

    cuda.syncthreads()

    tmp = 0.0

    # dot product of row i of A and column j of B
    for k in range(size):
        tmp += a_shared[thread_y, k] * b_shared[k, thread_x]

    # write
    if thread_y < size and thread_x < size:
        out[thread_y * size + thread_x] = tmp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # Calculate the number of tiles needed to cover the K dimension
    temp = 0.0

    # Number of tiles along the K dimension (shared dimension)
    K = a_shape[-1]
    num_tiles = (K + BLOCK_DIM - 1) // BLOCK_DIM

    # Adjust batch indices for broadcasting
    a_batch = batch if a_shape[0] > 1 else 0
    b_batch = batch if b_shape[0] > 1 else 0

    for tile_idx in range(num_tiles):
        # Global indices for elements in A and B
        a_i = i
        a_j = tile_idx * BLOCK_DIM + pj

        b_i = tile_idx * BLOCK_DIM + pi
        b_j = j

        # Load data into shared memory with boundary checks
        if a_i < a_shape[-2] and a_j < a_shape[-1]:
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index[0] = a_batch
            a_index[1] = a_i
            a_index[2] = a_j
            a_pos = index_to_position(a_index, a_strides)
            a_shared[pi, pj] = a_storage[a_pos]
        else:
            a_shared[pi, pj] = 0.0

        if b_i < b_shape[-2] and b_j < b_shape[-1]:
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index[0] = b_batch
            b_index[1] = b_i
            b_index[2] = b_j
            b_pos = index_to_position(b_index, b_strides)
            b_shared[pi, pj] = b_storage[b_pos]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize to make sure the sub-matrices are loaded
        cuda.syncthreads()

        # Compute partial dot product
        for k in range(BLOCK_DIM):
            temp += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize before loading the next tile
        cuda.syncthreads()

    # Write the result to global memory if within bounds
    if i < out_shape[-2] and j < out_shape[-1]:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_index[0] = batch
        out_index[1] = i
        out_index[2] = j
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = temp

    # ntiles = (a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM

    # for tile in range(ntiles):
    #     # global index for a and b
    #     a_row = i
    #     a_col = tile * BLOCK_DIM + pj
    #     b_row = tile * BLOCK_DIM + pi
    #     b_col = j

    #     # Calculate pos for a[i][j]
    #     if a_row < a_shape[-2] and a_col < a_shape[-1]:
    #         a_ind = cuda.local.array(MAX_DIMS, numba.int32)
    #         a_ind[0] = a_batch_stride
    #         a_ind[1] = a_row
    #         a_ind[2] = a_col
    #         a_pos = index_to_position(a_ind, a_strides)
    #         a_shared[pi, pj] = a_storage[a_pos]
    #     else:
    #         a_shared[pi, pj] = 0.0

    #     # Calculate pos for b[i][k]
    #     if b_row < b_shape[-2] and b_col < b_shape[-1]:
    #         b_ind = cuda.local.array(MAX_DIMS, numba.int32)
    #         b_ind[0] = b_batch_stride
    #         b_ind[1] = b_row
    #         b_ind[2] = b_col
    #         b_pos = index_to_position(b_ind, b_strides)
    #         b_shared[pi, pj] = b_storage[b_pos]
    #     else:
    #         b_shared[pi, pj] = 0.0

    #     # sync to ensure all data is loaded into shared memory
    #     cuda.syncthreads()

    #     tmp = 0.0  # init acc for the output element

    #     # perform dot product for this tile
    #     for t in range(BLOCK_DIM):
    #         tmp += a_shared[pi, t] * b_shared[t, pj]

    #     # sync before loading the next tile
    #     cuda.syncthreads()

    # # calc the pos in out and write the accumulated result to the output tensor
    # if i < out_shape[-2] and j < out_shape[-1]:
    #     out_index = cuda.local.array(MAX_DIMS, numba.int32)
    #     out_index[0] = batch
    #     out_index[1] = i
    #     out_index[2] = j
    #     pos = index_to_position(out_index, out_strides)
    #     out[pos] = tmp


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
