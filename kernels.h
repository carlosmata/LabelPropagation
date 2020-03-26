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
void lp_permutation_kernel_1(
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
__global__
void lp_permutation_kernel(
							int *nodes,			//Array of nodes
							int n_count,		//Number of nodes
							unsigned int seed	//Seed to generate random numbers
						)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(idx < n_count){
		int aux;
		curandState_t state;
		curand_init(seed, /* the seed controls the sequence of random values that are produced */
					0, /* the sequence number is only important with multiple cores */
					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&state);

		while(idx < n_count){
			int rand_pos = curand(&state) % n_count; //random( n_count, seed);
			if(rand_pos < n_count && rand_pos > 0){
				aux = atomicExch(&nodes[idx], nodes[rand_pos]);
				atomicExch(&nodes[rand_pos], aux);
			}
		}
	}
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

//Init the labels and nodes
__global__
void lp_init_arrays(
				int* labels,
				int* nodes,
				int nNodes
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nNodes){
		labels[idx] = idx;
		nodes[idx] = idx;
		idx += blockDim.x * gridDim.x;
	}
}

__device__
int lp_get_maximum_label(
					int node,
					const int* tails, 
					const int* indexs,
					int* labels,
					curandState_t state,
					const int nNodes,
					const int nEdges
					)
{
	//Get their neighboors
	int index = indexs[node];
	int nextIndex = (node + 1 < nNodes)?indexs[node + 1]:nEdges; 
	int nEdgesNode = (nextIndex - index < 0)?1 : nextIndex - index; 

	int *labelsEdges = new int[nEdgesNode];
	int *sumsLabels = new int[nEdgesNode];
	
	for(int i = 0; i < nEdgesNode; i++){
		labelsEdges[i] = -1;
		sumsLabels[i] = 0;
	}

	//Count the labels
	int posLabel = -1;
	int nLabelsNode = 0;
	int edge = -1;
	for(int tail = index; tail < nextIndex; tail++){
		edge = tails[tail];//get the neighbor
		if(edge < nNodes){ //the neightbor exist
			//find if the label exist en the labelsNeighbor
			posLabel = -1;
			for(int edgei = 0; edgei < nEdgesNode; edgei++){
				if(labels[edge] == labelsEdges[edgei]){
					posLabel = edgei;
					sumsLabels[edgei]++;
					break;
				}
			}

			if(posLabel == -1){//new label, the label never find it
				labelsEdges[nLabelsNode] = labels[edge];
				sumsLabels[nLabelsNode] = 1;
				nLabelsNode++;
			}
		}
	}
	//Find the Maximum
	int numberMax = -1;
	int *maximumLabels = new int[nLabelsNode];
	int nMaxLabels = 0;

	for(int i = 0;i < nLabelsNode; i++){
		if(numberMax < sumsLabels[i]){
			nMaxLabels = 0;
			numberMax = sumsLabels[i];
			maximumLabels[nMaxLabels] = labelsEdges[i];
			nMaxLabels++;
		}
		else if(numberMax == sumsLabels[i]){
			maximumLabels[nMaxLabels] = labelsEdges[i];
			nMaxLabels++;
		}
	}

	//Select a label at random
	int maximumLabel = maximumLabels[curand(&state) % nMaxLabels];

	delete[] labelsEdges;
	delete[] sumsLabels;
	delete[] maximumLabels;

	return maximumLabel;
}


__global__
void lp_compute_maximum_labels_kernel(
						bool synch,
						int* nodes,					//Array of nodes (permutation)
						int* tails,					//edges
						int* indexs,				//edges
						int* labels,				//Array of label's nodes 
						int* labels_aux,			//Array of label's nodes 
						int* thereAreChanges,		//flag
						unsigned int seed,					//time(NULL)
						const int nNodes,			//number of nodes
						const int nEdges,			//number of edges
						const int totalNodes		//number of total nodes
						)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	//gridDim.x		-----	Number of blocks
	//threadIdx.x	-----	id of thread in the block
	//blockDim.x	-----	Number of threads per block
	//blockIdx.x	-----	id of the block in the grid
	
	if(idx < nNodes){
		int node;
		int maximumLabel;

		curandState_t state;
		curand_init(seed, /* the seed controls the sequence of random values that are produced */
					0,    /* the sequence number is only important with multiple cores */
					0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&state);

		while(idx < nNodes){
			node = nodes[idx];

			maximumLabel = lp_get_maximum_label(node, tails, indexs, labels_aux, state, totalNodes, nEdges);
			
			if(maximumLabel != labels[node]){
				atomicExch(&labels[node], maximumLabel);
				atomicAdd(thereAreChanges, 1);

				//labels[node] = maximumLabel;
				//*thereAreChanges = *thereAreChanges + 1;
			}
			idx += blockDim.x * gridDim.x;
		}
	}
}

__global__
void lp_compute_maximum_labels_semi(
						const int* nodes,			//Array of nodes (permutation)
						const int* tails,			//edges
						const int* indexs,			//edges
						int* labels,				//Array of label's nodes 
						int* changes,				//flag
						const unsigned int seed,	//time(NULL)
						const int nNodes,			//number of nodes
						const int nEdges,			//number of edges
						const int beginData,		//number of data to compute
						const int endData
						)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	//gridDim.x		-----	Number of blocks
	//threadIdx.x	-----	id of thread in the block
	//blockDim.x	-----	Number of threads per block
	//blockIdx.x	-----	id of the block in the grid
	idx = beginData + idx;
	if(idx < endData){
		int node, maxLabel;

		curandState_t state;
		curand_init(seed, /* the seed controls the sequence of random values that are produced */
					0,    /* the sequence number is only important with multiple cores */
					0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&state);

		while(idx < endData){
			node = nodes[idx];
			//printf("%d ", node);

			maxLabel = lp_get_maximum_label(node, tails, indexs, labels, state, nNodes, nEdges);
			
			if(maxLabel != labels[node]){
				labels[node] = maxLabel;
				atomicAdd(changes, 1);
			}
			idx += blockDim.x * gridDim.x;
		}
	}
}

//-------------------------Version 3----------------------------------
//Init the labels
__global__
void lp_init_labels(
				int* labels,
				int nNodes
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nNodes){
		labels[idx] = idx;
		idx += blockDim.x * gridDim.x;
	}
}


//Gather		 //Create the array labels_vertex
__global__
void lp_gather(
			const int* labels,
			int* labels_vertex,
			const int* tails,
			const int nEdges
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nEdges){
		labels_vertex[idx] = labels[tails[idx]];
		idx += blockDim.x * gridDim.x;
	}
}

//Segmented sort //Sort the subarrays
//------------------------------------
__global__
void lp_sort(
			int* indexs,
			int* labels_vertex,
			int nNodes,
			int nEdges
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int index, nextIndex, temp;

	while(idx < nNodes){
		index = indexs[idx];
		nextIndex = (idx + 1 < nNodes)? indexs[idx + 1]:nEdges; 

		for(int edgei = index; edgei < nextIndex - 1; edgei++){
			for(int edgej = index; edgej < nextIndex - 1 - (edgei - index); edgej++){
				if(labels_vertex[edgej] > labels_vertex[edgej + 1]){
					temp = labels_vertex[edgej];
					labels_vertex[edgej] = labels_vertex[edgej + 1];
					labels_vertex[edgej + 1] = temp;
				}
			}
		}
		idx += blockDim.x * gridDim.x;
	}
}

//------------------------------------

//Calculates boundaries F
__global__
void lp_init_boundaries_1(
				const int* labels_vertex,
				int* F,
				const int nEdges
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nEdges){
		if(idx == nEdges - 1){
			F[idx] = 1;
		}
		else{
			F[idx] = (labels_vertex[idx] != labels_vertex[idx + 1])? 1 : 0;
		}
		idx += blockDim.x * gridDim.x;
	}
}
//Calculates boundaries in indexs
__global__
void lp_init_boundaries_2(
				int* indexs,
				int* F,
				int nNodes
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nNodes){
		if(idx > 0){
			F[indexs[idx] - 1] = 1;
		}
		idx += blockDim.x * gridDim.x;
	}
}

//Compute S
__global__
void lp_init_S(
				int* S,
				int* F_scan,
				int nEdges
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nEdges - 1){
		if(F_scan[idx] != F_scan[idx + 1]){
			S[F_scan[idx]] = idx;
		}
		idx += blockDim.x * gridDim.x;
	}
}
//Compute Sptr
__global__
void lp_init_Sptr(
				int* F_scan,
				int* indexs,
				int* Sptr,
				int nNodes
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nNodes){
		Sptr[idx] = F_scan[indexs[idx]];
		idx += blockDim.x * gridDim.x;
	}
}
//Compute W
__global__
void lp_init_W(
				int* S,
				int* W,
				int* F_s,
				int nEdges
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < F_s[nEdges - 1] + 1){
		if(idx != 0){
			W[idx] = S[idx] - S[idx - 1];
		}
		else{
			W[idx] = S[idx] + 1;
		}
		idx += blockDim.x * gridDim.x;
	}
}
//Segmented reduce
//---------------------
__global__
void lp_reduce(
			int* sptr,
			int* W,
			int* I,
			int nNodes,
			unsigned int seed,
			int* F_s,
			int nEdges
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
		
	while(idx < nNodes){
		numberMax = -1;
		countMax = 0;

		index = sptr[idx];
		nextIndex = (idx + 1 < nNodes)? sptr[idx + 1]:F_s[nEdges - 1] + 1; 
		tam = (nextIndex - index < 0)? 1 : nextIndex - index;;
		maxIndexs = new int[tam];

		for(int edgei = index; edgei < nextIndex; edgei++){
			if(numberMax < W[edgei]){
				countMax = 0;
				numberMax = W[edgei];
				maxIndexs[countMax] = edgei;
				countMax++;
			}
			else if(numberMax == W[edgei]){
				maxIndexs[countMax] = edgei;
				countMax++;
			}
		}
		I[idx] = maxIndexs[curand(&state) % countMax];

		idx += blockDim.x * gridDim.x;
		delete[] maxIndexs;
	}
}
//---------------------

//Computes Labels
__global__
void lp_compute_labels(
				int* labels,
				int* labels_vertex,
				int* S,
				int* I,
				int nNodes
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nNodes){
		labels[idx] = labels_vertex[S[I[idx]]];
		idx += blockDim.x * gridDim.x;
	}
}

//Count changes
__global__
void lp_compare_labels(
				int* labels,
				int* labels_ant,
				int* numberChanges,
				int nNodes
	)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	while(idx < nNodes){
		if(labels[idx] != labels_ant[idx]){
			atomicAdd(numberChanges, 1);
		}
		idx += blockDim.x * gridDim.x;
	}
}

//----------------------------------------------------------------------------



