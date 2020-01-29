#include <iostream>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define DSIZE 1000000				//Data size
#define nTPB 256					//nTPB should be a power-of-2 //Number of threads per block
#define MAX_KERNEL_BLOCKS 30		//Maximum kernel blocks
#define MAX_BLOCKS ((DSIZE/nTPB)+1)	//Maximum blocks
#define MIN(a,b) ((a > b)? b:a)		//Function to return the minium
#define MIN_VAL -1.0				//Value minium to compare the values

//--------------------------------------------------------------------
#include <time.h>
#include <sys/time.h>
unsigned long long dtime_usec(unsigned long long prev){
	#define USECPSEC 1000000ULL
	timeval tv1;
	gettimeofday(&tv1,0);
	return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}
//--------------------------------------------------------------------

__device__ volatile int blk_vals[MAX_BLOCKS];
__device__ volatile int blk_idxs[MAX_BLOCKS];
__device__ int blk_num = 0;

__global__ 
void max_id_kernel(const int *data, const int dsize, int *result, int seed){
	__shared__ volatile int vals[nTPB];
	__shared__ volatile int idxs[nTPB];
	__shared__ volatile int last_block;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	last_block = 0;
	int my_val = MIN_VAL;
	int my_idx = -1;

	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
				0, /* the sequence number is only important with multiple cores */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&state);

	// sweep from global memory [all data in jumps]
	while (idx < dsize){
		if (data[idx] > my_val){
			my_val = data[idx]; 
			my_idx = idx;
		}
		else if(data[idx] == my_val){
			if(curand(&state) % 2 == 0){ 
				my_idx = idx;
			}
		}
		idx += blockDim.x * gridDim.x;
	}

	//[Compare all the values in the shared memory]
	// populate shared memory
	vals[threadIdx.x] = my_val;
	idxs[threadIdx.x] = my_idx;
	__syncthreads();
	// sweep in shared memory ????
	for (int i = (nTPB>>1); i > 0; i>>=1){
		if (threadIdx.x < i){
			if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
				vals[threadIdx.x] = vals[threadIdx.x + i]; 
				idxs[threadIdx.x] = idxs[threadIdx.x + i]; 
			}
			else if(vals[threadIdx.x] == vals[threadIdx.x + i]){
				if(curand(&state) % 2 == 0){
					idxs[threadIdx.x] = idxs[threadIdx.x + i];
				}
			}
		}
		__syncthreads();
	}

	// perform block-level reduction [save the max value in the array of block values]
	if (!threadIdx.x){ //threadIdx.x == 0
		blk_vals[blockIdx.x] = vals[0];
		blk_idxs[blockIdx.x] = idxs[0];
		if (atomicAdd(&blk_num, 1) == gridDim.x - 1) // then I am the last block
			last_block = 1;
	}
	__syncthreads();
	
	if (last_block){ //can be a new kernel [compare all values in the blocks]
		idx = threadIdx.x;
		my_val = MIN_VAL;
		my_idx = -1;
		while(idx < gridDim.x){
			if (blk_vals[idx] > my_val) {
				my_val = blk_vals[idx]; 
				my_idx = blk_idxs[idx];
			}
			else if(blk_vals[idx] == my_val){
				if(curand(&state) % 2 == 0){ 
					my_idx = blk_idxs[idx];
				}
			}
			idx += blockDim.x;
		}
		
		// populate shared memory
		vals[threadIdx.x] = my_val;
		idxs[threadIdx.x] = my_idx;
		__syncthreads();
		// sweep in shared memory
		for (int i = (nTPB>>1); i > 0; i>>=1){
			if (threadIdx.x < i)
				if (vals[threadIdx.x] < vals[threadIdx.x + i]){
					vals[threadIdx.x] = vals[threadIdx.x+i]; 
					idxs[threadIdx.x] = idxs[threadIdx.x+i];
				}
				else if(vals[threadIdx.x] == vals[threadIdx.x + i]){
					if(curand(&state) % 2 == 0){ 
						idxs[threadIdx.x] = idxs[threadIdx.x+i];
					}
				}
			__syncthreads();
		}
		if (!threadIdx.x)
			*result = idxs[0];
	}
}

int main(){

	int *d_vector, *h_vector;
	h_vector = new int[DSIZE];
	
	for (int i = 0; i < DSIZE; i++){
		h_vector[i] = 0;//rand()/(int)RAND_MAX;
	}

	h_vector[9] = 10;  // create definite ma
	h_vector[19] = 10;  // create definite max 
	h_vector[39] = 10;  // create definite ma
	h_vector[59] = 10;  // create definite ma
	h_vector[69] = 10;  // create definite ma
	h_vector[79] = 10;  // create definite ma
	h_vector[89] = 10;  // create definite ma
	h_vector[99] = 10;  // create definite ma
	h_vector[109] = 10;  // create definite ma
	h_vector[119] = 10;  // create definite ma
	h_vector[129] = 10;  // create definite ma
	h_vector[139] = 10;  // create definite ma
	h_vector[149] = 10;  // create definite ma
	h_vector[159] = 10;  // create definite ma

	cudaMalloc(&d_vector, DSIZE * sizeof(int));
	cudaMemcpy(d_vector, h_vector, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	int max_index = 0, *d_max_index;

	unsigned long long dtime; 

	dtime = dtime_usec(0);
	cudaMalloc(&d_max_index, sizeof(int));
	max_id_kernel<<<MIN(MAX_KERNEL_BLOCKS, ((DSIZE+nTPB-1)/nTPB)), nTPB>>>(d_vector, DSIZE, d_max_index, time(NULL));
	cudaMemcpy(&max_index, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
	dtime = dtime_usec(dtime);
	
	std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: " << max_index << std::endl;

	return 0;
}