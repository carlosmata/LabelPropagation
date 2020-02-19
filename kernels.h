//------------------------------------Kernels-------------------------------------------------------
#include "graphMethods.h"
#include <curand.h>
#include <curand_kernel.h>
#define MIN_VAL -1.0				//Value minium to compare the values
//---------------Kernels Centrality-------------------------
__global__ 
void bc_bfs_kernel(
				int *d_v, 
				int *d_e, 
				int *d_d, 
				int *d_sigma,
				unsigned int *d_p, 
				bool *d_continue, 
				int *d_dist, 
				int n_count, 
				int e_count)
{	
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
  	if(tid < e_count){  /* for each edge (u, w) */
		int u = d_v[tid];
		int w = d_e[tid];
		if(d_d[u] == *d_dist){
			if(d_d[w] == -1){
				*d_continue = true;
				d_d[w] = *d_dist + 1;
			}
		  if(d_d[w] == *d_dist + 1){
		  	
		    unsigned long long  bit = (unsigned long long)(w * n_count + u);
		    //unsigned int aux = d_p[BIT_INT(bit)];
		    atomicOr(&d_p[BIT_INT(bit)], (unsigned int) BIT_IN_INT(bit));
		    //printf("[%d]:bit:%llu, (ant)d_p[bit]:%u, BIT_IN_INT(bit):%u  (desp)d_p[bit]:%u\n", tid, bit, aux, (unsigned int) BIT_IN_INT(bit), d_p[BIT_INT(bit)]);
		    
		    atomicAdd(&d_sigma[w], d_sigma[u]);
		  }
		}
	}
}

__global__ 
void bc_bfs_back_prop_kernel(
							int *d_v, 
							int *d_e,
							int *d_d, 
							int *d_sigma, 
							float *d_delta, 
							unsigned int *d_p, 
							int *d_dist, 
							int n_count, 
							int e_count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < e_count){
		/* for each edge (u, w) */
		int u = d_v[tid];
		int w = d_e[tid];
		if(d_d[u] == (*d_dist - 1)){
			unsigned long long  bit = (unsigned long long)(u * n_count + w);
			
			if((d_p[BIT_INT(bit)] & (unsigned int)BIT_IN_INT(bit)) != 0){
				atomicAdd(&d_delta[w], 1.0f * d_sigma[w] / d_sigma[u] * (1.0f + d_delta[u]));
			}
		}
	}
}

__global__ 
void bc_bfs_back_sum_kernel(
						int s, 
						int *d_dist, 
						int *d_d, 
						float *d_delta, 
						float *d_bc, 
						int n_count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < n_count){
		if(tid != s && d_d[tid] == (*d_dist - 1)){
	  		d_bc[tid] += d_delta[tid];
		}
	}
}

//---------------------------------------------------------

//--------------Kernels Communities------------------------

//------------------------Version 1------------------------------------------
//Kernel 1 Code parallel to get the permutation of the nodes
__global__
void lp_permutation_kernel(
							int *nodes,			//Array of nodes
							int n_count,		//Number of nodes
							unsigned int seed	//Seed to generate random numbers
						)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int aux;

	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
				0, /* the sequence number is only important with multiple cores */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&state);

	int rand_pos = curand(&state) % n_count; //random( n_count, seed);
	if(rand_pos < n_count && rand_pos > 0){
		aux = atomicExch(&nodes[tid], nodes[rand_pos]);
		atomicExch(&nodes[rand_pos], aux);
	}
} 

//Kernel 2 Code parallel to count the numbers of labes in node's edges
__device__ 
void warpReduce(volatile int* sdata, int tid) {
		//sdata[tid] = sdata[tid]; 
		sdata[tid] += sdata[tid + 16]; 
		sdata[tid] += sdata[tid +  8]; 
		sdata[tid] += sdata[tid +  4]; 
		sdata[tid] += sdata[tid +  2]; 
		sdata[tid] += sdata[tid +  1]; 
}
__global__ 
void lp_count_labels_kernel(
							int *d_histo_labels,			// Histogram labels
							const int *neighbours_labels,	// Neighbour's labels to count the ocurrences
							const int SIZE_LABELS,			// Number of total labels
							const int neighbours_count		// Size of array of neighbour's labels
						)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;	// Id of thread in the total blocks
	int threadsPerBlock = blockDim.x;					// Number of threads per block
	int tid = threadIdx.x;								// Id of the Thread in the block
	int numBlocks = gridDim.x;							// Number of blocks in the grid
	int dataBlock = threadsPerBlock * numBlocks;		// Number of total threads
	extern __shared__ int histo[];	// Histogram of the block
	
	int cols = threadsPerBlock;// + threadsPerBlock/2;
	
	//Initialice the part of histagram
	for( int i = 0; i < SIZE_LABELS; i++){
		histo[i * cols + tid] = 0;
	}
	//__syncthreads();
	
	//Create the histograms
	for( int i = myId; i < neighbours_count; i += dataBlock ){
		histo[(neighbours_labels[i]) * cols + tid]++;
	}
	//__syncthreads();
	
	//Make the reductions
	for( int i = 0; i < SIZE_LABELS; i++) {
		warpReduce( &histo[i * cols + 0], tid );
		if( tid == 0 ){
			d_histo_labels[i + SIZE_LABELS * blockIdx.x] = histo[i * cols + 0]; //
		} 
		//__syncthreads();
	}
}


//Kernel 3 Code parallel to get the maximum of the label
__global__ 
void lp_max_label_kernel(
					const int *n_labels,	//Array of labels [histogram]
					const int dsize,		//Size of array labels
					int *maxIdlabel, 		//Id of the max ocurrences labels
					//Auxiliars
					int seed, 				//Seed to generate random numbers
					volatile int *blk_vals,	//Auxiliar array vals in blocks
					volatile int *blk_idxs,	//Auxiliar array indexs in blocks
					int *blk_num,			//Variable to know the last block
					const int nTPB			//Number of threads per block
					)
{
	extern __shared__ volatile int vals[]; //vals[nTPB]
	extern __shared__ volatile int idxs[]; //vals[nTPB]
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

	// sweep from global memory [all labels in jumps]
	while (idx < dsize){
		if (n_labels[idx] > my_val){
			my_val = n_labels[idx]; 
			my_idx = idx;
		}
		else if(n_labels[idx] == my_val){
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
		if (atomicAdd(blk_num, 1) == gridDim.x - 1) // then I am the last block
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
			*maxIdlabel = idxs[0];
	}
}
//----------------------------------------------------------------------------

//-------------------------Version 2------------------------------------------
__device__
int lp_get_maximum_label(
					int node,
					int* tails, 
					int* indexs,
					int* labels,
					curandState_t state,
					const int nNodes,
					const int nEdges
					)
{
	//Get their neighboors
	int neighbor = -1;
	int index = indexs[node];
	int nextIndex = (node + 1 < nNodes)? indexs[node + 1]:nEdges; 
	int tamLabels = (nextIndex - index < 0)?1 : nextIndex - index; 

	int *labelsNeighbours = new int[tamLabels];
	int *countersLabels = new int[tamLabels];
	int posLabelN = -1;
	int itLabelN = 0;

	for(int i = 0; i < tamLabels; i++){
		labelsNeighbours[i] = -1;
		countersLabels[i] = 0;
	}

	//Count the labels
	for(int tail = index; tail < nextIndex; tail++){
		neighbor = tails[tail];//get the neighbor
		if(neighbor < nNodes){ //the neightbor exist
			//find if the label exist en the labelsNeighbor
			posLabelN = -1;

			for(int n = 0; n < tamLabels; n++){ //find label
				if(labels[neighbor] == labelsNeighbours[n]){
					posLabelN = n;
					break;
				}
			}

			if(posLabelN == -1){//new label
				labelsNeighbours[itLabelN] = labels[neighbor];
				countersLabels[itLabelN] = 1;
				itLabelN++;
			}
			else{
				countersLabels[posLabelN]++;
			}
		}
	}

	//Find the Maximum
	int numberMax = -1;
	int *maximumLabels = new int[tamLabels];
	int indexMaximumLabels = 0;

	for(int i = 0;i < itLabelN; i++){
		if(numberMax < countersLabels[i]){
			indexMaximumLabels = 0;
			numberMax = countersLabels[i];
			maximumLabels[indexMaximumLabels] = labelsNeighbours[i];
			indexMaximumLabels++;
		}
		else if(numberMax == countersLabels[i]){
			maximumLabels[indexMaximumLabels] = labelsNeighbours[i];
			indexMaximumLabels++;
		}
	}

	

	//Select a label at random
	int posRandom = curand(&state) % indexMaximumLabels;
	int maximumLabel = maximumLabels[posRandom];

	delete[] labelsNeighbours;
	delete[] countersLabels;
	delete[] maximumLabels;

	return maximumLabel;
}


__global__
void lp_copy_array(
				int* array1,
				int* array2,
				int tam
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < tam){
		array1[idx] = array2[idx];
		idx += blockDim.x * gridDim.x;
	}
}


__global__
void lp_compute_maximum_labels_kernel(
						int* nodes,					//Array of nodes (permutation)
						int* tails,					//edges
						int* indexs,				//edges
						int* labels,				//Array of label's nodes 
						int* labels_aux,				//Array of label's nodes 
						bool* thereAreChanges,		//flag
						int seed,					//time(NULL)
						const int nNodes,			//number of nodes
						const int nEdges			//number of edges
						)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	//gridDim.x		-----	Number of blocks
	//threadIdx.x	-----	id of thread in the block
	//blockDim.x	-----	Number of threads per block
	//blockIdx.x	-----	id of the block in the grid
	
	if(idx < nNodes){
		int node, maximumLabel;
		curandState_t state;
		curand_init(seed, /* the seed controls the sequence of random values that are produced */
					0,    /* the sequence number is only important with multiple cores */
					0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&state);

		while(idx < nNodes){
			node = nodes[idx];
			maximumLabel = lp_get_maximum_label(node, tails, indexs, labels_aux, state, nNodes, nEdges);
			if(maximumLabel != labels[node]){
				atomicExch(&labels[node], maximumLabel);
				//labels[node] = maximumLabel;
				*thereAreChanges = true;
			}
			idx += blockDim.x * gridDim.x;
		}
	}
}


//----------------------------------------------------------------------------



