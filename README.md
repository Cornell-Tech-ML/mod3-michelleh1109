# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1 and 3.2 Parallel Computing Diagnostics Script
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/mic
hellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/michellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        eq = np.array_equal(out_strides, in_strides) and np.array_equal(     | 
            out_shape, in_shape                                              | 
        )                                                                    | 
                                                                             | 
        if eq:                                                               | 
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   | 
        else:                                                                | 
            for i in prange(len(out)):---------------------------------------| #3
                out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------| #0
                in_index: Index = np.zeros(MAX_DIMS, np.int32)---------------| #1
                to_index(i, out_shape, out_index)                            | 
                                                                             | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                o = index_to_position(out_index, out_strides)                | 
                j = index_to_position(in_index, in_strides)                  | 
                out[o] = fn(in_storage[j])                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/michellehui/Cornell
Tech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (180) is hoisted out of 
the parallel loop labelled #3 (it will be performed before the loop is executed 
and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/michellehui/Cornell
Tech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (181) is hoisted out of 
the parallel loop labelled #3 (it will be performed before the loop is executed 
and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/mic
hellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (215)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/michellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (215) 
----------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                     | 
        out: Storage,                                                             | 
        out_shape: Shape,                                                         | 
        out_strides: Strides,                                                     | 
        a_storage: Storage,                                                       | 
        a_shape: Shape,                                                           | 
        a_strides: Strides,                                                       | 
        b_storage: Storage,                                                       | 
        b_shape: Shape,                                                           | 
        b_strides: Strides,                                                       | 
    ) -> None:                                                                    | 
        eq_stride = np.array_equal(out_strides, a_strides) and np.array_equal(    | 
            a_strides, b_strides                                                  | 
        )                                                                         | 
        eq_shape = np.array_equal(out_shape, a_shape) and np.array_equal(         | 
            a_shape, b_shape                                                      | 
        )                                                                         | 
                                                                                  | 
        if eq_stride and eq_shape:                                                | 
            for i in prange(len(out)):--------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                           | 
        else:                                                                     | 
            for i in prange(len(out)):--------------------------------------------| #8
                out_index: Index = np.zeros(MAX_DIMS, np.int32)-------------------| #4
                a_index: Index = np.zeros(MAX_DIMS, np.int32)---------------------| #5
                b_index: Index = np.zeros(MAX_DIMS, np.int32)---------------------| #6
                to_index(i, out_shape, out_index)                                 | 
                o = index_to_position(out_index, out_strides)                     | 
                                                                                  | 
                broadcast_index(out_index, out_shape, a_shape, a_index)           | 
                j = index_to_position(a_index, a_strides)                         | 
                                                                                  | 
                broadcast_index(out_index, out_shape, b_shape, b_index)           | 
                k = index_to_position(b_index, b_strides)                         | 
                                                                                  | 
                out[o] = fn(a_storage[j], b_storage[k])                           | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/michellehui/Cornell
Tech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (238) is hoisted out of 
the parallel loop labelled #8 (it will be performed before the loop is executed 
and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/michellehui/Cornell
Tech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (239) is hoisted out of 
the parallel loop labelled #8 (it will be performed before the loop is executed 
and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/michellehui/Cornell
Tech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (240) is hoisted out of 
the parallel loop labelled #8 (it will be performed before the loop is executed 
and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/michellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py 
(276)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/michellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (276) 
---------------------------------------------------------------|loop #ID
    def _reduce(                                               | 
        out: Storage,                                          | 
        out_shape: Shape,                                      | 
        out_strides: Strides,                                  | 
        a_storage: Storage,                                    | 
        a_shape: Shape,                                        | 
        a_strides: Strides,                                    | 
        reduce_dim: int,                                       | 
    ) -> None:                                                 | 
        reduce_size = a_shape[reduce_dim]                      | 
                                                               | 
        for i in prange(len(out)):-----------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, np.int32)----| #9
            to_index(i, out_shape, out_index)                  | 
            o = index_to_position(out_index, out_strides)      | 
                                                               | 
            for s in range(reduce_size):                       | 
                out_index[reduce_dim] = s                      | 
                j = index_to_position(out_index, a_strides)    | 
                out[o] = fn(out[o], a_storage[j])              | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/michellehui/Cornell
Tech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (288) is hoisted out of 
the parallel loop labelled #10 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/mich
ellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (300)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/michellehui/CornellTech/workspace/mod3-michelleh1109/minitorch/fast_ops.py (300) 
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              | 
    out: Storage,                                                                         | 
    out_shape: Shape,                                                                     | 
    out_strides: Strides,                                                                 | 
    a_storage: Storage,                                                                   | 
    a_shape: Shape,                                                                       | 
    a_strides: Strides,                                                                   | 
    b_storage: Storage,                                                                   | 
    b_shape: Shape,                                                                       | 
    b_strides: Strides,                                                                   | 
) -> None:                                                                                | 
    """NUMBA tensor matrix multiply function.                                             | 
                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                           | 
                                                                                          | 
    ```                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                     | 
    ```                                                                                   | 
                                                                                          | 
    Optimizations:                                                                        | 
                                                                                          | 
    * Outer loop in parallel                                                              | 
    * No index buffers or function calls                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                | 
                                                                                          | 
                                                                                          | 
    Args:                                                                                 | 
    ----                                                                                  | 
        out (Storage): storage for `out` tensor                                           | 
        out_shape (Shape): shape for `out` tensor                                         | 
        out_strides (Strides): strides for `out` tensor                                   | 
        a_storage (Storage): storage for `a` tensor                                       | 
        a_shape (Shape): shape for `a` tensor                                             | 
        a_strides (Strides): strides for `a` tensor                                       | 
        b_storage (Storage): storage for `b` tensor                                       | 
        b_shape (Shape): shape for `b` tensor                                             | 
        b_strides (Strides): strides for `b` tensor                                       | 
                                                                                          | 
    Returns:                                                                              | 
    -------                                                                               | 
        None : Fills in `out`                                                             | 
                                                                                          | 
    """                                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                | 
                                                                                          | 
    assert (np.array_equal(a_shape[-1], b_shape[-2]) )                                    | 
                                                                                          | 
    batch_size = out_shape[0] if len(out_shape) == 3 else 1                               | 
    M = out_shape[-2]  # Number of rows in output                                         | 
    N = out_shape[-1]  # Number of columns in output                                      | 
    K = a_shape[-1]    # Shared dimension                                                 | 
                                                                                          | 
    out_batch_stride = out_strides[0] if len(out_shape) == 3 else 0                       | 
                                                                                          | 
    # Outer loop in parallel over batches                                                 | 
    for batch in prange(batch_size):------------------------------------------------------| #11
        # Calculate batch offsets                                                         | 
        a_batch_offset = batch * a_batch_stride if a_batch_stride > 0 else 0              | 
        b_batch_offset = batch * b_batch_stride if b_batch_stride > 0 else 0              | 
        out_batch_offset = batch * out_batch_stride if out_batch_stride > 0 else 0        | 
                                                                                          | 
        # Loop over rows and columns of the output matrix                                 | 
        for i in range(M):                                                                | 
            for j in range(N):                                                            | 
                # Compute the position in the output storage                              | 
                out_idx = out_batch_offset + i * out_strides[-2] + j * out_strides[-1]    | 
                                                                                          | 
                # Accumulator for the dot product                                         | 
                sum_val = 0.0                                                             | 
                                                                                          | 
                # Inner loop over the shared dimension                                    | 
                for k in range(K):                                                        | 
                    # Compute positions in `a_storage` and `b_storage`                    | 
                    a_idx = a_batch_offset + i * a_strides[-2] + k * a_strides[-1]        | 
                    b_idx = b_batch_offset + k * b_strides[-2] + j * b_strides[-1]        | 
                                                                                          | 
                    # Multiply and accumulate                                             | 
                    sum_val += a_storage[a_idx] * b_storage[b_idx]                        | 
                                                                                          | 
                # Assign the computed value to the output storage                         | 
                out[out_idx] = sum_val                                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
