#ifndef REDUCE_H__
#define REDUCE_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DATA_BLOCK_SIZE 1024
#define WARP_SIZE 32
#define CORR 5
#define MAX_KERNEL_BLOCKS 4096


#define cuda_check_error(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void chk_cuda_error(const char *mensaje)
{
	 cudaError_t error;
	 cudaDeviceSynchronize();
	 error = cudaGetLastError();
	 if(error != cudaSuccess)
	 {
	 	printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	 	printf("\npulsa INTRO para finalizar...");
	 	fflush(stdin);
		char tecla = getchar();
	 	exit(EXIT_FAILURE);
	 }
}
//Segmented reduction-----------------------

/**
	Reduction in a block
*/
/*__device__
void block_reduction (
	int* s_data,
	int* s_owner,
	int* g_output, //array dataI 
	curandState_t state)
{
	unsigned int thid = threadIdx.x; //id in the block
	unsigned int stride = 1;
	unsigned int i, ai, bi, imax, lo, ro;
	unsigned int B = blockDim.x; // size of cudablock //number of threads per block 

	for(unsigned int d = B; d > 0; d >>= 1){ //reduce d in power of two
		__syncthreads();
		if(thid < d){
			i = 2 * stride * thid;
			ai = i;
			bi = ai + stride;
			ai += (ai >> 5); //log2(nBanks)=5
			bi += (bi >> 5); //log2(nBanks)=5
			//Reduction
			lo = s_owner[ai]; //left owner
			ro = s_owner[bi]; //right owner
			
			if(lo!=ro){
				g_output[ro] = op(g_output[ro], s_data[bi], state);
			}
			else{
				s_data[ai] = op(s_data[ai], s_data[bi], state);
			}
		}
		stride <<=1;
	}
	//The result is in s_data[0]
}*/


/*
__global__
void lp_s_reduce(
				int* sptr,
				int* ownersBlocks,
				int* W,
				int sizeW,
				int* I,
				int nNodes,
				int* dataBlock,
				unsigned int seed)
{
	curandState_t state;
	curand_init(seed, // the seed controls the sequence of random values that are produced 
				0,    // the sequence number is only important with multiple cores 
				0,    // the offset is how much extra we advance in the sequence for each call, can be 0 
				&state);
	__shared__ int s_data[DATA_BLOCK_SIZE];
	__shared__ int s_owner[DATA_BLOCK_SIZE];

	int idx = threadIdx.x;
	int iW = 0, owner, incr;

	while(idx < DATA_BLOCK_SIZE) {
		iW = (blockIdx.x * DATA_BLOCK_SIZE) + idx;
		if(iW < sizeW){
			s_data[idx] =  W[iW];
			if(iW == 0){
				s_owner[idx] = 0;
			}
			else if(iW == sizeW - 1 || iW > sptr[nNodes - 1]){
				s_owner[idx] = nNodes - 1; 
			}
			else{ //find the owner 
				owner = nNodes / 2;
				incr = owner / 2;
				s_owner[idx] = 0;

				while(owner >= 0 && owner < nNodes){
					if(iW < sptr[owner]){
						if(idx == 1){
							printf("< idx: %d, iW: %d, owner:%d, sptr[owner - 1]: %d, sptr[owner]: %d, sptr[owner + 1]: %d\n", 
									  idx, 
									  iW, 
									  owner, 
									  sptr[owner - 1],
									  sptr[owner],
									  sptr[owner + 1]);
						}

						if(iW >= sptr[owner - 1]){
							s_owner[idx] = owner - 1;
							break;
						}
						else{
							owner -= (incr);
							incr = incr / 2;
						}
					}		
					else if(iW > sptr[owner]){
						//printf("> idx: %d, iW: %d, owner:%d, sptr[owner]: %d\n", idx, iW, owner, sptr[owner]);
						if(iW < sptr[owner + 1]){
							s_owner[idx] = owner;
							break;
						}
						if(iW == sptr[owner + 1]){
							s_owner[idx] = owner + 1;
							break;
						}
						else{
							owner += incr;
							incr = incr / 2;
						}
					}
					else{ //== 
						//printf("== idx: %d, iW: %d, owner:%d, sptr[owner]: %d\n", idx, iW, owner, sptr[owner]);
						s_owner[idx] = owner;
						break;
					}
				}
			}
		}
		else{
			s_data[idx] = 0;
			s_owner[idx] = 0;
		}

		//printf("primer while idx:%d\n", idx);
		idx += blockDim.x;
	}
	__syncthreads();

	printf("ENTRO A WARP REDUCTION idx:%d\n", idx);
	warp_reduction(s_data, I, s_owner, DATA_BLOCK_SIZE, state);
	
	if(idx == 0){
		dataBlock[blockIdx.x] = s_data[0];
		ownersBlocks[blockIdx.x] = s_data[0];
	}
}*/


/**
	Find de maximun of two values
*/
__device__
int op(
	int data1, 
	int data2, 
	curandState_t state)
{
	int max = 0;

	if(data1 > data2){
		max = data1;
	}
	else if(data2 > data1){
		max = data2;
	}
	else{
		int randBand = curand(&state) % 2;
		if(randBand == 0){
			max = data1;
		}
		else{
			max = data2;
		}
	}

	return max;
}

__device__
void s_reducePair_toTheLeft(
	int* g_output,
	unsigned int oLeft, 
	int &dLeft,
	unsigned int oRight, 
	int dRight,
	curandState_t state)
{
	if(oLeft != oRight)
		g_output[oRight] = op(g_output[oRight], dRight, state);
	else 
		dLeft = op(dLeft, dRight, state);
}

__device__
void tb_reduceWarp( 
				int* s_data,
				int* g_output,
				int* s_owner,
				unsigned int thid,
				unsigned int lane,
				int& target,
				int& owner_target,
				curandState_t state)
{
	if( !(lane & 1) ) // lane % 2 == 0
		s_reducePair_toTheLeft(g_output, s_owner[thid], s_data[thid], s_owner[thid + 1], s_data[thid + 1], state); 
		//s_data[thid] = op(s_data[thid], s_data[thid + 1], state); 
	if( !(lane & 3) ) // lane % 4 == 0
		s_reducePair_toTheLeft(g_output, s_owner[thid], s_data[thid], s_owner[thid + 2], s_data[thid + 2], state); 
		//s_data[thid] = op(s_data[thid], s_data[thid + 2], state); 
	if( !(lane & 7) ) // lane % 8 == 0
		s_reducePair_toTheLeft(g_output, s_owner[thid], s_data[thid], s_owner[thid + 4], s_data[thid + 4], state); 
		//s_data[thid] = op(s_data[thid], s_data[thid + 4], state); 
	if( !(lane & 15) ) // lane % 16 == 0
		s_reducePair_toTheLeft(g_output, s_owner[thid], s_data[thid], s_owner[thid + 8], s_data[thid + 8], state); 
		//s_data[thid] = op(s_data[thid], s_data[thid + 8], state); 
	if( !(lane & 31) ){ // lane % 32 == 0
		s_reducePair_toTheLeft(g_output, s_owner[thid], s_data[thid], s_owner[thid + 16], s_data[thid + 16], state);
		target = s_data[thid];
		owner_target = s_owner[thid];
	} 
		//target = op(s_data[thid], s_data[thid + 16], state);	   
}

// ***************************************
//D = 1024 = 2 ^ 10, B=256 = 2 ^ 8 => Nwarps=8
__device__
void warp_reduction (
						int* s_data,
						int* g_output,
						int* s_owner,
						int n_s_data, 
					    curandState_t state)
{
	int threadsInWarp = WARP_SIZE;
	int Nwarps = blockDim.x / threadsInWarp; 	 //256 / 32
	__shared__ int s_result[WARP_SIZE];   //32 results
	__shared__ int s_result_owner[WARP_SIZE];
	unsigned int thid = threadIdx.x; //				[0 - 255]
	unsigned int warpid = thid >> CORR; //thid / 32 	[0 - 7]
	unsigned int lane = thid & (threadsInWarp - 1);   //thid % 32	[0 - 31]
	
	int steps = n_s_data / blockDim.x; //4
	//Reduce s_data[kB+warpid*32, kB+(warpid+1)*32) and store the result into s_result[k*Nwarps+warpid]
	for(unsigned int k = 0; k < steps; k++){
		tb_reduceWarp(
					s_data, 
					g_output,
					s_owner,
					thid + (k * blockDim.x), 
					lane, 
					s_result[warpid + (k * Nwarps)],
					s_result_owner[warpid + (k * Nwarps)],
					state);
	}
	__syncthreads();
	
	if(warpid == 0){ //threads [0 - 31]
		tb_reduceWarp(
					s_result, 
					g_output,
					s_result_owner,
					thid, 
					lane, 
					s_data[0],
					s_owner[0],
					state);
	}

	__syncthreads();//save the last value
	if(thid == 0){
		g_output[s_owner[0]] = op(g_output[s_owner[0]], s_data[0], state);
	}
	//The final result is in s_data[0]
}

__global__
void lp_s_reduce_blocks(
				int* owners,
				int* ownersBlock,
				int* W,
				int sizeW,
				int* I,
				int nNodes,
				int* dataBlock,
				unsigned int seed)
{
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
				0,    /* the sequence number is only important with multiple cores */
				0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&state);

	__shared__ int s_data[DATA_BLOCK_SIZE];
	__shared__ int s_owner[DATA_BLOCK_SIZE];

	int idx = threadIdx.x;
	int iW = 0;

	while(idx < DATA_BLOCK_SIZE) {
		iW = (blockIdx.x * DATA_BLOCK_SIZE) + idx;
		if(iW < sizeW){
			s_data[idx] =  W[iW];
			s_owner[idx] = owners[iW];
		}
		else{
			s_data[idx] = 0;
			s_owner[idx] = 0;
		}

		idx += blockDim.x;
	}
	__syncthreads();

	warp_reduction(s_data, I, s_owner, DATA_BLOCK_SIZE, state);
	
	if(idx == 0){
		dataBlock[blockIdx.x] = s_data[0];
		ownersBlock[blockIdx.x] = s_owner[0];
	}
}

__global__
void get_size_W(
	int* F_s, 
	int nEdges, 
	int *sizeW)
{
	*sizeW = F_s[nEdges - 1] + 1;
} 


__global__
void compute_sptr_owners(
	int* sptr,
	int* owners,
	int nNodes,
	int sizeW)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int b, e;

	while(idx < nNodes){
		b = sptr[idx];
		e = (idx + 1 < nNodes)? sptr[idx + 1]: sizeW;
		
		for(int i = b; i < e; i++){
			owners[i] = idx;	
		}
		
		idx += blockDim.x * gridDim.x;
	}
}



int getSizeW(int* d_F_s, int nEdges){
	
	int sizeW = 0;
	int *d_sizeW;
	cudaMalloc(&d_sizeW, sizeof(int));
	get_size_W<<<1,1>>>(d_F_s, nEdges, d_sizeW);
	cudaMemcpy(&sizeW, d_sizeW, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_sizeW);

	return sizeW;
}

void segmented_reduce(
	int* d_sptr,
	int* d_W,
	int* d_I,
	int nNodes,
	unsigned int seed,
	int* d_F_s,
	int nEdges)
{
	int sizeW = getSizeW(d_F_s, nEdges);
	int nTPB = 256;
	//printf("sizeW %d\n", sizeW);
	int block_plus = (sizeW % DATA_BLOCK_SIZE == 0)? 0: 1;
	int nBlocks = (sizeW / DATA_BLOCK_SIZE) + block_plus; //calcular el numero de datos

	int* d_owners; 
	int* d_owners_b;
	int* d_dataBlock;
	
	cuda_check_error(cudaMalloc(&d_owners, sizeW * sizeof(int)));
	cuda_check_error(cudaMalloc(&d_owners_b, nBlocks * sizeof(int)));
	cuda_check_error(cudaMalloc(&d_dataBlock, nBlocks * sizeof(int)));

	int nTPB_N = 1024;
	int nBNodes = nNodes / nTPB_N + ((nNodes % nTPB_N == 0)? 0 : 1);
	int nBlocksNodes = (MAX_KERNEL_BLOCKS > nBNodes)? nBNodes:MAX_KERNEL_BLOCKS; //MIN(MAX_KERNEL_BLOCKS, nBNodes);

	compute_sptr_owners<<<nBlocksNodes, nTPB_N>>>(d_sptr, d_owners, nNodes, sizeW);
	chk_cuda_error("Kernel compute_sptr_owners invocation");
	cuda_check_error(cudaDeviceSynchronize());

	//printf("Entro a primera reduccion nBlocks:%d, nTPB:%d\n", nBlocks, nTPB);
	//lp_s_reduce<<<nBlocks, nTPB>>>(d_sptr, d_owners_b, d_W, sizeW, d_I, nNodes, d_dataBlock, seed);
	lp_s_reduce_blocks<<<nBlocks, nTPB>>>(d_owners, d_owners_b, d_W, sizeW, d_I, nNodes, d_dataBlock, seed);
	chk_cuda_error("Kernel lp_s_reduce invocation");
	cuda_check_error(cudaDeviceSynchronize());
	//printf("Salio de primera reduccion\n");

	//printf("primer condicional para segunda reduccion\n");
	if(nBlocks > DATA_BLOCK_SIZE){
		//printf("Entro a segunda reduccion\n");
		sizeW = nBlocks;
		int* d_owners_blocks; 
		cuda_check_error(cudaMalloc(&d_owners_blocks, nBlocks * sizeof(int)));

		while(nBlocks > DATA_BLOCK_SIZE){
			cuda_check_error(cudaMemcpy(d_W, d_dataBlock, sizeof(int) * nBlocks, cudaMemcpyDeviceToDevice));
			cuda_check_error(cudaDeviceSynchronize());
			
			nBlocks = sizeW / DATA_BLOCK_SIZE + (sizeW % DATA_BLOCK_SIZE == 0)? 0: 1;
			
			lp_s_reduce_blocks<<<nBlocks, nTPB>>>(d_owners_b, d_owners_blocks, d_W, sizeW, d_I, nNodes, d_dataBlock, seed);
			chk_cuda_error("Kernel lp_s_reduce_blocks invocation");

			cuda_check_error(cudaMemcpy(d_owners_b, d_owners_blocks, sizeof(int) * nBlocks, cudaMemcpyDeviceToDevice));
			sizeW = nBlocks;
		}

		cudaFree(d_owners_blocks);
		//printf("Salio de segunda reduccion\n");
	}
	//printf("acabo primer condicional para segunda reduccion\n");

	cudaFree(d_owners);
	cudaFree(d_owners_b);
	cudaFree(d_dataBlock);
	//printf("Acabo segmented_reduce\n");
}

#endif
