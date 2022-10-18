#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "mmio.h"

#define BlockDim 1024
#define ITER 3

template <class T>
__device__ T warp_reduce (T val)
{
  /**
   *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
   *  the value of the val variable from the thread at lane X+offset of the same warp.
   *  The data exchange is performed between registers, and more efficient than going
   *  through shared memory, which requires a load, a store and an extra register to
   *  hold the address.
   */
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync (FULL_WARP_MASK, val, offset);

  return val;
}

template <typename data_type, typename index_type, index_type bs>
__global__ void spmv_bcsr_kernel(
    index_type num_block_rows;
    index_type block_size;
    const index_type * __restrict__ row_ptr;
    const index_type * __restrict__ col_ids;
    const data_type * __restrict__ data;
    const data_type * __restrict__ x;
    data_type * y)
{
    const index_type idx = blockIdx.x * blockDim.x + threadIdx.x; // thread id
    const index_type row = (idx / 32) % bs;
    // threads in a warp accessing the same row & different cols to reuse shared memory
    // 
    const index_type lane = idx % 32; 
    const index_type block_row = (idx / 32) % bs;
    const index_type first_block = row_ptr[block_row];
    const index_type last_block = row_ptr[block_row + 1];

    data_type local_out = 0.0;

    if(row < block_size && block_row < num_block_rows) 
    {
        for(index_type loc_col = lane; loc_col < block_size * (last_block - first_block); loc_col += 32)
        {
            const index_type block = first_block + loc_col / block_size;
            const index_type c = loc_col % block_size;
            const index_type col = col_ids[block] * block_size + c;
            local_out += x[col] * data[block * block_size * block_size + row * block_size + c];
        }
    }
    local_out = warp_reduce(local_out);

    if(row < block_size && block_row < num_block_rows && lane == 0)
        y[block_row * block_size + row] = local_out;

}

template <typename T>
void spmv_bcsr(MatrixInfo<T> * mat,T *vector,T *out) 
{

}