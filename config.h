#ifndef _CONFIG_H_
#define _CONFIG_H_

// main config

// debug printf
#define PRINTF 0
#define SH_MEM 1

#define GMEM_LOCATION(x, y, z)

// threads per block

#define SHMEM_CUBE_SIZE 8
#define SHMEM_MESH_SIZE 12

// calculate memory jump(stride) in Z-space

#define SHM_STRIDE_Z (4 + SHMEM_CUBE_SIZE) * (4 + SHMEM_CUBE_SIZE)

// kernel config

#define DIMENSION 3

#define SDATA(index) cutilBankChecker(sdata, index)
#define BLOCK 8
#define LOCAL_SWEEPS 50
#define NCOLUMNS ((2 * BLOCK) + 4)
#define SIZE_X (BLOCK + 4)
#define SIZE_Y ((2 * BLOCK) + 4)
#define N 1

#endif
