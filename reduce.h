#ifndef REDUCE_H__
#define REDUCE_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

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

/**
	Find de maximun of two values
*/
__device__
int op(
	int data1, 
	int data2, 
	int idata1,
	int idata2,
	int &imax,
	curandState_t state)
{
	int max = 0;

	if(data1 > data2){
		max = data1;
		imax = idata1;
	}
	else if(data2 > data1){
		max = data2;
		imax = idata2;
	}
	else{
		int randBand = curand(&state) % 2;
		if(randBand == 0){
			max = data1;
			imax = idata1;
		}
		else{
			max = data2;
			imax = idata2;
		}
	}

	return max;
}

__device__
void s_reducePair_toTheLeft(
	int* g_output,
	int* g_output_i,
	unsigned int oLeft, 
	int &dLeft,
	int &iLeft,
	unsigned int oRight, 
	int dRight,
	int iRight,
	curandState_t state)
{
	int imax = 0;
	if(oLeft != oRight){

		/*printf("BEFORE: g_output[oRight]:%d, dRight:%d, g_output_i[oRight]:%d, iRight:%d, imax:%d \n", 
				g_output[oRight], 
				dRight, 
				g_output_i[oRight], 
				iRight, 
				imax);*/

		g_output[oRight] = op(g_output[oRight], dRight, g_output_i[oRight], iRight, imax, state);
		g_output_i[oRight] = imax;

		/*printf("AFTER: g_output[oRight]:%d, dRight:%d, g_output_i[oRight]:%d, iRight:%d, imax:%d \n", 
				g_output[oRight], 
				dRight, 
				g_output_i[oRight], 
				iRight, 
				imax);*/
	}
	else{ 
		dLeft = op(dLeft, dRight, iLeft, iRight, imax, state);
		iLeft = imax;
	}

}

__device__
void tb_reduceWarp( 
				int* s_data,
				int* s_owner,
				int* s_wi,
				int* g_output,
				int* g_output_i,
				unsigned int thid,
				unsigned int lane,
				int& target,
				int& owner_target,
				int& iw_target,
				curandState_t state)
{
	if( !(lane & 1) ) // lane % 2 == 0
		s_reducePair_toTheLeft(g_output, g_output_i, s_owner[thid], s_data[thid], s_wi[thid], s_owner[thid + 1], s_data[thid + 1], s_wi[thid + 1], state);   
	if( !(lane & 3) ) // lane % 4 == 0
		s_reducePair_toTheLeft(g_output, g_output_i, s_owner[thid], s_data[thid], s_wi[thid], s_owner[thid + 2], s_data[thid + 2], s_wi[thid + 2], state); 
	if( !(lane & 7) ) // lane % 8 == 0
		s_reducePair_toTheLeft(g_output, g_output_i, s_owner[thid], s_data[thid], s_wi[thid], s_owner[thid + 4], s_data[thid + 4], s_wi[thid + 4], state); 
	if( !(lane & 15) ) // lane % 16 == 0
		s_reducePair_toTheLeft(g_output, g_output_i, s_owner[thid], s_data[thid], s_wi[thid], s_owner[thid + 8], s_data[thid + 8], s_wi[thid + 8], state);  
	if( !(lane & 31) ){ // lane % 32 == 0
		s_reducePair_toTheLeft(g_output, g_output_i, s_owner[thid], s_data[thid], s_wi[thid], s_owner[thid + 16], s_data[thid + 16], s_wi[thid + 16], state);
		target = s_data[thid];
		owner_target = s_owner[thid];
		iw_target = s_wi[thid];
	} 
		//target = op(s_data[thid], s_data[thid + 16], state);	   
}

// ***************************************
//D = 1024 = 2 ^ 10, B=256 = 2 ^ 8 => Nwarps=8
__device__
void warp_reduction (
						int* s_data,
						int* s_owner,
						int* s_iw,
						int* g_output,
						int* g_output_i,
						int n_s_data, 
					    curandState_t state)
{
	int threadsInWarp = WARP_SIZE;
	int Nwarps = blockDim.x / threadsInWarp; 	 //256 / 32
	__shared__ int s_result[WARP_SIZE];   //32 results
	__shared__ int s_result_owner[WARP_SIZE];
	__shared__ int s_result_iw[WARP_SIZE];

	unsigned int thid = threadIdx.x; //				[0 - 255]
	unsigned int warpid = thid >> CORR; //thid / 32 	[0 - 7]
	unsigned int lane = thid & (threadsInWarp - 1);   //thid % 32	[0 - 31]
	
	int steps = n_s_data / blockDim.x; //4
	//Reduce s_data[kB+warpid*32, kB+(warpid+1)*32) and store the result into s_result[k*Nwarps+warpid]
	for(unsigned int k = 0; k < steps; k++){
		tb_reduceWarp(
					s_data, 
					s_owner,
					s_iw,
					g_output,
					g_output_i,
					thid + (k * blockDim.x), 
					lane, 
					s_result[warpid + (k * Nwarps)],
					s_result_owner[warpid + (k * Nwarps)],
					s_result_iw[warpid + (k * Nwarps)],
					state);
	}
	__syncthreads();
	
	if(warpid == 0){ //threads [0 - 31]
		tb_reduceWarp(
					s_result, 
					s_result_owner,
					s_result_iw,
					g_output,
					g_output_i,
					thid, 
					lane, 
					s_data[0],
					s_owner[0],
					s_iw[0],
					state);
	}

	__syncthreads();//save the last value
	if(thid == 0){
		int imax = 0;
		g_output[s_owner[0]] = op(g_output[s_owner[0]], s_data[0], 
								  g_output_i[s_owner[0]], s_iw[0], 
								  imax, state);
		g_output_i[s_owner[0]] = imax;
	}
	//The final result is in s_data[0]
}

__global__
void lp_s_reduce(
				int* data,			//data to reduce
				int* owners,		//array of owners			
				int* data_i,			//indexs os the real data
				int* maxs,			//result of segment reduce
				int* maxs_i,		//result of indices to segmente reduce
				int sizeData,		//size of data
				int* dataBlock,		//the new data to blocks
				int* ownersBlock,	//results of new owners blocks
				int* dataIBlocks,		//new indexs to the real data
				unsigned int seed)	//seed to random number
{	
	curandState_t state;
	curand_init(seed, 0, 0, &state);

	__shared__ int s_data[DATA_BLOCK_SIZE];
	__shared__ int s_owner[DATA_BLOCK_SIZE];
	__shared__ int s_data_i[DATA_BLOCK_SIZE];

	int idx = threadIdx.x;
	int i = 0;

	while(idx < DATA_BLOCK_SIZE) {
		i = (blockIdx.x * DATA_BLOCK_SIZE) + idx;
		if(i < sizeData){
			s_data[idx] =  data[i];
			s_owner[idx] = owners[i];
			s_data_i[idx] = (data_i)? data_i[idx]: i;
		}
		else{
			s_data[idx] = 0;
			s_owner[idx] = 0;
			s_data_i[idx] = -1;
		}

		idx += blockDim.x;
	}
	__syncthreads();
	idx = threadIdx.x;

	warp_reduction(s_data, s_owner, s_data_i, maxs, maxs_i, DATA_BLOCK_SIZE, state);

	if(idx == 0){
		dataBlock[blockIdx.x] = s_data[0];
		ownersBlock[blockIdx.x] = s_owner[0];
		dataIBlocks[blockIdx.x] = s_data_i[0];
	}
}

__global__
void compute_data_owners(
	int* segments,
	int* owners,
	int nSegments,
	int sizeData)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int b, e;

	while(idx < nSegments){
		b = segments[idx];
		e = (idx + 1 < nSegments)? segments[idx + 1]: sizeData;
		
		for(int i = b; i < e; i++){
			owners[i] = idx;	
		}
		
		idx += blockDim.x * gridDim.x;
	}
}


__global__
void print_array_gpu(int* array, int nelements){
	for(int i = 0; i < nelements; i++){
		printf("array[%d]:%d \n",i, array[i]);
	}
}

void compute_owners(
				int* d_segments, 
				int* d_owners, 
				int nSegments, 
				int sizeData){
	int nTPB_N = 1024;
	int nBNodes = nSegments / nTPB_N + ((nSegments % nTPB_N == 0)? 0 : 1);
	int nBlocksNodes = (MAX_KERNEL_BLOCKS > nBNodes)? nBNodes:MAX_KERNEL_BLOCKS; //MIN(MAX_KERNEL_BLOCKS, nBNodes);
	compute_data_owners<<<nBlocksNodes, nTPB_N>>>(d_segments, d_owners, nSegments, sizeData);
	chk_cuda_error("Kernel compute_data_owners invocation");
	cuda_check_error(cudaDeviceSynchronize());
}


void printArrayGPU(int* array, int size){
	print_array_gpu<<<1,1>>>(array, size);
	chk_cuda_error("Kernel print_array_gpu invocation");
	cuda_check_error(cudaDeviceSynchronize());
}

void printarrayCPU(int arg[], int length) {
	for (int n=0; n<length; ++n)
		printf("%d ",  arg[n]);
	printf("\n");
}

void segmented_reduce(
	int* d_segments,
	int* d_data,
	int* d_maxs,
	int* d_maxs_i,
	int sizeData,
	int nSegments,
	unsigned int seed)
{
	int nTPB = 256;
	int block_plus = (sizeData % DATA_BLOCK_SIZE == 0)? 0: 1;
	int nBlocks = (sizeData / DATA_BLOCK_SIZE) + block_plus; //calcular el numero de datos

	int* d_owners; 
	int* d_ownersBlock;
	int* d_dataBlock;
	int* d_dataiBlock;

	cuda_check_error(cudaMalloc(&d_owners, sizeData * sizeof(int)));
	cuda_check_error(cudaMalloc(&d_ownersBlock, nBlocks * sizeof(int)));
	cuda_check_error(cudaMalloc(&d_dataBlock, nBlocks * sizeof(int)));
	cuda_check_error(cudaMalloc(&d_dataiBlock, nBlocks * sizeof(int)));

	compute_owners(d_segments, d_owners, nSegments, sizeData);

	printf("nBlocks:%d, nSegments:%d, sizeData: %d \n", nBlocks, nSegments, sizeData);

	//printf("\nd_data\n");
	//printArrayGPU(d_data, sizeData);

	//printf("\nd_owners\n");
	//printArrayGPU(d_owners, sizeData);
	
	lp_s_reduce<<<nBlocks, nTPB>>>(
								d_data,
								d_owners,
								nullptr,
								d_maxs, 
								d_maxs_i, 
								sizeData,   
								d_dataBlock,
								d_ownersBlock, 
								d_dataiBlock, 
								seed);
	chk_cuda_error("Kernel lp_s_reduce invocation");
	cuda_check_error(cudaDeviceSynchronize());


	//printf("\ndata_blocks \n");
	//printArrayGPU(d_data_blocks, nBlocks);
	//printf("owners_blocks \n");
	//printArrayGPU(d_owners_blocks, nBlocks);
	//printf("datai_blocks \n");
	//printArrayGPU(d_datai_blocks, nBlocks);

	if(nBlocks > 1){
		sizeData = nBlocks;

		int* d_data_blocks;
		int* d_owners_blocks; 
		int* d_datai_blocks;

		cuda_check_error(cudaMalloc(&d_data_blocks, nBlocks * sizeof(int)));
		cuda_check_error(cudaMalloc(&d_owners_blocks, nBlocks * sizeof(int)));
		cuda_check_error(cudaMalloc(&d_datai_blocks, nBlocks * sizeof(int)));

		while(nBlocks > 1){//DATA_BLOCK_SIZE){
			cuda_check_error(cudaMemcpy(d_data_blocks, d_dataBlock, sizeof(int) * nBlocks, cudaMemcpyDeviceToDevice));
			cuda_check_error(cudaMemcpy(d_owners_blocks, d_ownersBlock, sizeof(int) * nBlocks, cudaMemcpyDeviceToDevice));
			cuda_check_error(cudaMemcpy(d_datai_blocks, d_dataiBlock, sizeof(int) * nBlocks, cudaMemcpyDeviceToDevice));				
			cuda_check_error(cudaDeviceSynchronize());

			printf("\n data_blocks \n");
			printArrayGPU(d_data_blocks, nBlocks);
			printf("\n owners_blocks \n");
			printArrayGPU(d_owners_blocks, nBlocks);
			printf("\n datai_blocks \n");
			printArrayGPU(d_datai_blocks, nBlocks);
			printf("\n d_maxs \n");
			printArrayGPU(d_maxs, nSegments);
			printf("\n d_maxs_i \n");
			printArrayGPU(d_maxs_i, nSegments);

			nBlocks = sizeData / DATA_BLOCK_SIZE + ((sizeData % DATA_BLOCK_SIZE == 0)? 0: 1);

			printf("nBlocks:%d, nSegments:%d, sizeData: %d \n", nBlocks, nSegments, sizeData);
			lp_s_reduce<<<nBlocks, nTPB>>>(
										d_data_blocks,
										d_owners_blocks,										
										d_datai_blocks,
										d_maxs, 
										d_maxs_i,
										sizeData,
										d_dataBlock,
										d_ownersBlock,          
										d_dataiBlock, 
										seed);
			chk_cuda_error("Kernel lp_s_reduce_blocks");
			
			sizeData = nBlocks;
		}

		cudaFree(d_data_blocks);
		cudaFree(d_owners_blocks);
		cudaFree(d_datai_blocks);
	}

	//printf("\nd_maxs\n");
	//printArrayGPU(d_maxs, nSegments);
	//printf("\nd_maxs_i\n");
	//printArrayGPU(d_maxs_i, nSegments);

	cudaFree(d_owners);
	cudaFree(d_ownersBlock);
	cudaFree(d_dataBlock);
	cudaFree(d_dataiBlock);
}


//FOR TESTS
__global__
void initarrays(int* segments, 
				int* data,
				int size_data,
				int n_segments,
				unsigned int seed){
	//Example 1
	//-----------------------------	
	int segmentSize = size_data / n_segments;
	for(int i = 0;i < n_segments; i++){
		segments[i] = i * segmentSize;
	}


	curandState_t state;
	curand_init(seed, 0, 0, &state);

	for(int i = 0; i < size_data; i++){
		data[i] = curand(&state) % size_data; 
	}
	//------------------------------
}


//FOR COMPARISION
__global__
void lp_reduce(
			int* segments,
			int* data,
			int* maxI,
			int nSegments,
			int sizeData,
			unsigned int seed
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int index, nextIndex, numberMax, countMax, tam;
	int *maxIndexs = nullptr;

	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
				0,    /* the sequence number is only important with multiple cores */
				0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&state);
		
	while(idx < nSegments){
		numberMax = -1;
		countMax = 0;

		index = segments[idx];
		nextIndex = (idx + 1 < nSegments)? segments[idx + 1]:sizeData; 
		tam = (nextIndex - index < 0)? 1 : nextIndex - index;
		
		if(tam > 1){
			maxIndexs = new int[tam];

			for(int edgei = index; edgei < nextIndex; edgei++){
				if(numberMax < data[edgei]){
					countMax = 0;
					numberMax = data[edgei];
					maxIndexs[countMax] = edgei;
					countMax++;
				}
				else if(numberMax == data[edgei]){
					maxIndexs[countMax] = edgei;
					countMax++;
				}
			}
			maxI[idx] = maxIndexs[curand(&state) % countMax];
			delete[] maxIndexs;
		}
		else if(tam == 1){
			maxI[idx] = index;	
		}
		
		idx += blockDim.x * gridDim.x;		
	}
}

#endif
