//------------------------------------Kernels-------------------------------------------------------
#include "graphMethods.h"
#include <curand.h>
#include <curand_kernel.h>

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
__device__ 
int random(int MAX, unsigned int seed) {
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	we will store a random state for every thread  */
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
				0, /* the sequence number is only important with multiple cores */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&state);
	/* curand works like rand - except that it takes a state as a parameter */
	return curand(&state) % MAX;
}

__global__
void lp_permutation_kernel(
							int *nodes,
							int n_count,
							unsigned int seed
						)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int aux;

	int rand_pos = random( n_count, seed);
	if(rand_pos < n_count && rand_pos > 0){
		aux = atomicExch(&nodes[tid], nodes[rand_pos]);
		atomicExch(&nodes[rand_pos], aux);
	}
} 

//Kernel 2 Code parallel to count the numbers of labes in node's edges
__global__
void lp_count_labels_kernel(
							int *tails,
							int *labels,
							int *neighbours_labels,
							int *counters_labels,
							int *it_label,
							int index,
							int n_count,
							int neighbours_count
							)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tail = index + tid;

	int neighbor = tails[tail];//get the neighbor
	if(neighbor < n_count){ //the neightbor exist
		//find if the label exist en the labelsNeighbor
		int pos = -1;
		for(int n = 0; n < neighbours_count; n++){
			if(labels[neighbor] == neighbours_labels[n]){
				pos = n;
				break;
			}
		}
		if(pos == -1){//new label
			//------------------[Standalone]---------------------
			neighbours_labels[*it_label] = labels[neighbor];
			counters_labels[*it_label] = 1;
			atomicAdd(it_label, 1);
			//-------------------------------------------------
		}
		else{
			atomicAdd(&counters_labels[pos], 1);
		}
	}
}

//Kernel 3 Code parallel to get the maximum of the label
__global__
void lp_find_maximum_kernel(
							int numberMax,
							int indexMaximumLabels,
							int *countersLabels,
							int *maximumLabels,
							int *labelsNeighbours
							)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(numberMax < countersLabels[tid]){
		indexMaximumLabels = 0;
		numberMax = countersLabels[tid];
		maximumLabels[indexMaximumLabels] = labelsNeighbours[tid];
		indexMaximumLabels++;
	}
	else if(numberMax == countersLabels[tid]){
		maximumLabels[indexMaximumLabels] = labelsNeighbours[tid];
		indexMaximumLabels++;
	}

}
//----------------------------------------------------------------------------

//-------------------------Version 2------------------------------------------

//----------------------------------------------------------------------------



