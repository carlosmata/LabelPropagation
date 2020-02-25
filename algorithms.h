#include "kernels.h"
#include <stdlib.h>
#include <bits/stdc++.h>
#include "community_measures.cu"

using namespace std;
#define MIN(a,b) ((a > b)? b:a)		//Function to return the minium

int MAX_THREADS_PER_BLOCK = 256;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//----------------------------------Algorithms--------------------------------------------

//----------------------------------------------------------------------------------------
//-------------------------------Centrality Sequential------------------------------------
float* brandesSequential(
				float* costs, 
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//Stack
	int S[nNodes];   		//pop_back
	int s_pointer = 0;
	//Queue
	int Q[nEdges];			//pop_front
	int init_q, last_q;
	
	int sigma[nNodes]; 		//arreglo
	int d[nNodes];			//distancias [arreglo]
	float delta[nNodes];	//arreglo

	float *centrality = new float[nNodes]; 
	//float centrality[nNodes];// = new float[nNodes]; 

	for(int i = 0;i < nNodes; i++){
		centrality[i] = 0;
	}

	int index, nextIndex, v, w;

	for(int s = 0; s < nNodes; s++){

		for(int t = 0; t < nNodes; t++){
			sigma[t] = 0;
			d[t] = -1;
			delta[t] = 0.0;
		}
		sigma[s] = 1;
		d[s] = 1;
		s_pointer = 0;
		last_q = init_q = 0;

		Q[last_q] = s;
		last_q++;

		while(init_q != last_q){ //!Q.isempty()
			v = Q[init_q];
			init_q++;

			S[s_pointer] = v;
			s_pointer++;
			index = indexs[v];
			nextIndex = (v + 1 < nNodes)?indexs[v + 1]:nEdges; //getNextIndex(v, indexs, nNodes, nEdges);

			for(int tail = index; tail < nextIndex; tail++){
				w = tails[tail];
				if(w < nNodes){
					if(d[w] < 0){
						Q[last_q] = w;
						last_q++;

						d[w] = d[v] + 1;
					}
					
					if(d[w] == d[v] + 1){
						sigma[w] += sigma[v];
					}
				}
			}
		} 
		
		int vp;
		for(int i = s_pointer-1; i >= 0 ; i--){
			w = S[i];

			index = indexs[w];
			nextIndex = (w + 1 < nNodes)?indexs[w + 1]:nEdges; //getNextIndex(w, indexs, nNodes, nEdges);

			for(int tail = index; tail < nextIndex; tail++){
				vp = tails[tail];
				if(d[vp] == (d[w] - 1)){
					delta[vp] += (sigma[vp]/(float)sigma[w])*(1+delta[w]);
				}
			}		
			if(w != s){
				centrality[w] += delta[w]; 
			}
		}   
	}

	return centrality;
}
//-------------------------------Centrality Parallel-------------------------------------
void brandesParallel( 
				int* nodes,
				int* edges,
				float* centrality,
				const int nNodes,
				const int nEdges)
{
	cout<< "Brandes Parallel:" << endl;
	int sizeInt = sizeof(int);
	int sizeBool = sizeof(bool);
	int sizeIntEdges = sizeof(int) * nEdges;
	int sizeIntNodes = sizeof(int) * nNodes;
	int sizeFloatNodes = sizeof(float) * nNodes;
	unsigned long long total_bits = (unsigned long long)nNodes * nNodes;
	unsigned int num_of_ints = BITS_TO_INTS(total_bits);
	int sizeNumInts = sizeof(unsigned int) * num_of_ints;


	//Graph data
	int *d_nodes;
	int *d_edges;
	cudaCheckError(cudaMalloc((void **)&d_nodes, sizeIntEdges));
  	cudaCheckError(cudaMalloc((void **)&d_edges, sizeIntEdges));
  	cudaCheckError(cudaMemcpy(d_nodes, nodes, sizeIntEdges, cudaMemcpyHostToDevice));
  	cudaCheckError(cudaMemcpy(d_edges, edges, sizeIntEdges, cudaMemcpyHostToDevice));

  	//Brandes algorithm Device data
  	int *d_d;
  	int *d_sigma;
  	float *d_delta;
  	unsigned int *d_p; 	//Predecessor arry (n * n)
  	int *d_dist; 
  	float *d_centrality;

	cudaCheckError(cudaMalloc((void **)&d_d, sizeIntNodes));
	cudaCheckError(cudaMalloc((void **)&d_sigma, sizeIntNodes)); 
	cudaCheckError(cudaMalloc((void **)&d_delta, sizeFloatNodes)); 
	cudaCheckError(cudaMalloc((void **)&d_p, sizeNumInts)); 
	cudaCheckError(cudaMalloc((void **)&d_dist, sizeInt));
	cudaCheckError(cudaMalloc((void **)&d_centrality, sizeFloatNodes)); 

	cudaCheckError(cudaMemcpy(d_centrality, centrality, sizeFloatNodes, cudaMemcpyHostToDevice));
	
	//Brandes algorithm Host data
	int *h_d = (int*)malloc(sizeIntNodes);
	int h_sigma_0 = 1;

	for(int i = 0; i < nNodes; i++){
		//----------------------------Initialization----------------------------------------
		for(int j = 0; j < nNodes; j++){
			h_d[j]=-1;
		}
		
		h_d[i]=0;
		cudaCheckError(cudaMemcpy(d_d, h_d, sizeIntNodes, cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemset(d_sigma, 0, sizeIntNodes));
		cudaCheckError(cudaMemcpy(&d_sigma[i],&h_sigma_0, sizeInt, cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemset(d_delta, 0, sizeFloatNodes));
		cudaCheckError(cudaMemset(d_p, 0, sizeNumInts));

		//-----------------------Compute nBlocks, nThreads-----------------------------------
		int threads_per_block = nEdges;
		int blocks = 1;
		if(nEdges > MAX_THREADS_PER_BLOCK){
			blocks = (int)ceil(nEdges/(float)MAX_THREADS_PER_BLOCK); 
			threads_per_block = MAX_THREADS_PER_BLOCK; 
		}
		dim3 grid(blocks);
		dim3 threads(threads_per_block);
		
		int threads_per_block2 = nNodes;
		int blocks2 = 1;
		if(nNodes > MAX_THREADS_PER_BLOCK){ 
			blocks2 = (int)ceil(nNodes/(double)MAX_THREADS_PER_BLOCK); 
			threads_per_block2 = MAX_THREADS_PER_BLOCK; 
		}
		dim3 grid2(blocks2);
		dim3 threads2(threads_per_block2);
		//----------------------------------------------------------------------------------

		bool h_continue;
		bool *d_continue;
		int h_dist = 0;
		cudaMalloc((void **)&d_continue, sizeBool);
		cudaCheckError(cudaMemset(d_dist, 0, sizeInt));
		//--------------------------------Breath First Search-------------------------------  
		do{
			h_continue = false;
			cudaCheckError(cudaMemcpy(d_continue, &h_continue, sizeBool, cudaMemcpyHostToDevice));
			//Número de aristas
			bc_bfs_kernel<<<grid, threads>>>(d_nodes, d_edges, d_d, d_sigma, d_p, d_continue, d_dist, nNodes, nEdges);
			check_CUDA_Error("Kernel bc_bfs_kernel invocation");
			cudaCheckError( cudaPeekAtLastError() );
			cudaCheckError( cudaDeviceSynchronize() );

			h_dist++; 
			cudaCheckError(cudaMemcpy(d_dist, &h_dist, sizeInt, cudaMemcpyHostToDevice));
			cudaCheckError(cudaMemcpy(&h_continue, d_continue, sizeBool, cudaMemcpyDeviceToHost));
		}while( h_continue );   

		cudaFree(d_continue);
		//--------------------------------Back propagation-----------------------------------
		h_continue = false;
		cudaCheckError(cudaMemcpy(&h_dist, d_dist, sizeInt, cudaMemcpyDeviceToHost));
		do{
			//Número de aristas
			bc_bfs_back_prop_kernel<<<grid, threads>>>(d_nodes, d_edges, d_d, d_sigma, d_delta, d_p, d_dist, nNodes, nEdges);
			check_CUDA_Error("Kernel bc_bfs_back_prop_kernel invocation");
			cudaCheckError( cudaPeekAtLastError() );
			cudaCheckError( cudaDeviceSynchronize() );
			
			//Número de nodos
			bc_bfs_back_sum_kernel<<<grid2, threads2>>>(i, d_dist, d_d,  d_delta, d_centrality, nNodes);
			check_CUDA_Error("Kernel bc_bfs_back_sum_kernel invocation");
			cudaCheckError( cudaPeekAtLastError() );
			cudaCheckError( cudaDeviceSynchronize() );
			h_dist--;
			
			cudaCheckError(cudaMemcpy(d_dist, &h_dist, sizeInt, cudaMemcpyHostToDevice));
		}while(h_dist > 1);
	}

	cudaCheckError(cudaMemcpy(centrality, d_centrality, sizeFloatNodes, cudaMemcpyDeviceToHost));
	free(h_d);
	cudaFree(d_nodes);
	cudaFree(d_edges);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_p);
	cudaFree(d_dist);
}
//---------------------------------------------------------------------------------------
//_----------------------------Communities Sequential------------------------------------
/**
	Compute the community detection running the label propagation algorithm

	Parameters:
		labelsNeigbours:
		label:
		totalNeighbours: Size of the array neigh
*/
int findLabel(
			int* labelsNeighbours,
			int label,
			int totalNeighbours)
{
	for(int n = 0; n < totalNeighbours; n++){
		if(label == labelsNeighbours[n]){
			return n;
		}
	}

	return -1;
}

/**
	Returns the label occurring with the highest frequency among neighbours.
	
	Parameters.
	costs: Costs of the edges
	tails: Array of neighbors of the nodes
	indexs: Indexs of tails array
	nNodes: Number of nodes
	nEdges: Number of edges
*/
int getMaximumLabel(int node,
					int* tails, 
					int* indexs,
					int* labels, 
					const int nNodes,
					const int nEdges,
					int *numberLabelMax)
{
	//Get their neighboors
	int neighbor = -1;
	int index = indexs[node];
	int nextIndex = (node + 1 < nNodes)?indexs[node + 1]:nEdges; 
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
			posLabelN = findLabel(labelsNeighbours, labels[neighbor], tamLabels);
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
	int posRandom = rand() % indexMaximumLabels;
	int maximumLabel = maximumLabels[posRandom];
	*numberLabelMax = numberMax;

	delete[] labelsNeighbours;
	delete[] countersLabels;
	delete[] maximumLabels;

	return maximumLabel;
}

/**
	Compute a new permutation to the sent array
	Pameters.
		nodes: Array to get the permutation
		nNodes: Size of the array
*/
void getPermutation(int* nodes, int nNodes)
{
	int newPos = 0; //0 to tam array 
	int aux;
	for(int i = 0;i < nNodes; i++){
		//swap
		newPos = rand() % nNodes;
		if(newPos < nNodes && newPos > 0){
			aux = nodes[i];
			nodes[i] = nodes[newPos];
			nodes[newPos] = aux;
		}
	}
}

void printarray (int arg[], int length) {
	for (int n=0; n<length; ++n)
		cout << arg[n] << ' ';
	cout << '\n';
}

/**
	Apply the Label propagation algorithm in the sequential way

	Parameters.
	costs: Costs of the edges
	tails: Array of neighbors of the nodes
	indexs: Indexs of tails array
	nNodes: Number of nodes
	nEdges: Number of edges
*/
int* labelPropagationSequential(
				float* costs, 
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges,
				bool synchronous)
{
	int *labels = new int[nNodes];
	int *countLabels = new int[nNodes];
	int *labelsAux = new int[nNodes];
	int *nodes = new int[nNodes];
	bool thereAreChanges = true;
	int maximumLabel = -1;
	int node;

	/* initialize random seed: */
	double seed = time(NULL);
	srand (seed);

	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		nodes[i] = i;
		countLabels[i] = 0;
	}

	cout<< "Begin Label propagation sequential ";
	if(synchronous){
		cout<< "synchronous:" << endl; 
	}
	else{
		cout<< "asynchronous:" << endl; 
	}

	int t = 0;
	int numberLabelMax = 0;

	//int changes = 0;
	//int t2 = 0;
	//float mod = 0;
	//int com = 0, com2 = 0;
	//int minchange = nNodes;

	while(thereAreChanges){//until a node dont have the maximum of their neightbors
		//mod = getModularity(tails, indexs, nNodes, nEdges, labels);
		//com = countCommunities(labels, nNodes);
		//printf("%d \t %f \t %d \t %d\n", t, mod, com, changes);
		//printf("%d \t %d \t %d\n", t, com, changes);
		//if(changes < minchange && changes != 0){
		//	minchange = changes;
		//	com2 = com;
		//	t2 = t;
		//}
		//changes = 0;

		thereAreChanges = false;

		if(!synchronous){ //asynchronous
			getPermutation(nodes, nNodes); //Optionally: delete nodes with 1 edge and 0 edges
		}
		
		//printarray(labels, nNodes);
		if(synchronous){
			for(int i = 0; i < nNodes; i++){
				labelsAux[i] = labels[i];
			}
		}

		for(int i = 0; i < nNodes; i++){ //random permutation of Nodes
			node = nodes[i];
			//find the maximum label of their neightbors
			if(synchronous){
				maximumLabel = getMaximumLabel(node, tails, indexs, labelsAux, nNodes, nEdges, &numberLabelMax);
			}
			if(!synchronous){ //asynchronous
				maximumLabel = getMaximumLabel(node, tails, indexs, labels, nNodes, nEdges, &numberLabelMax);
			}

			if(maximumLabel != labels[node] && countLabels[node] != numberLabelMax){
				labels[node] = maximumLabel;
				countLabels[node] = numberLabelMax;
				thereAreChanges = true;
				//changes++;
			}
		}
		t++;
		if(t > nNodes)
			break;
	}

	//printf("it %d, minchange %d, communities %d",t2 , minchange, com2);
	return labels;
}


//-----------------------------Communities Parallel---------------------------------------

/**
	Apply the Label propagation algorithm in the parallel way

	Parameters.
	costs: Costs of the edges
	tails: Array of neighbors of the nodes
	indexs: Indexs of tails array
	nNodes: Number of nodes
	nEdges: Number of edges
*/
int* labelPropagationParallel(
				float* costs, 
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//int NUMBER_OF_THREADS = 32;
	//int SHARED_MEMORY_SIZE = (32 + 16) * nNodes * 4;
	//Number of threads
	int nTPB = MAX_THREADS_PER_BLOCK;											//Threads in a block  256
	int MAX_KERNEL_BLOCKS = 30;													//Max blocks in a Grid
	//int numberOfThreadsPerBlock = nTPB 
	int numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, ((nNodes + nTPB - 1)/nTPB));	//Blocks in a Grid

	int *labels = new int[nNodes];
	int *nodes = new int[nNodes];
	bool thereAreChanges = true;
	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		nodes[i] = i;
	}

	//GPU memory
	int *d_labels;
	int *d_nodes;
	int *d_tails;
	int *d_indexs;
	bool *d_thereAreChanges;

	cudaMalloc(&d_labels, nNodes * sizeof(int));
	cudaMalloc(&d_nodes, nNodes * sizeof(int));
	cudaMalloc(&d_tails, nEdges * sizeof(int));
	cudaMalloc(&d_indexs, nNodes * sizeof(int));
	cudaMalloc(&d_thereAreChanges, sizeof(bool));

	cudaMemcpy(d_tails, tails, nEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indexs, indexs, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);

	int t = 0;
	long seed = time(NULL);
	while(thereAreChanges){//until a node dont have the maximum label of their neightbors

		thereAreChanges =  false;
		
		getPermutation(nodes, nNodes); //obtener doble permutacion

		cudaMemcpy(d_nodes, nodes, nNodes * sizeof(int), cudaMemcpyHostToDevice);
		/*lp_permutation_kernel<<<numberOfBlocks, nTPB>>>(
															d_nodes,			//Array of nodes
															nNodes,		//Number of nodes
															seed	//Seed to generate random numbers
														);*/
		/*lp_copy_array<<<numberOfBlocks, nTPB>>>(
												d_labels_aux,
												d_labels,
												nNodes
												);*/

		cudaMemcpy(d_thereAreChanges, &thereAreChanges, sizeof(bool), cudaMemcpyHostToDevice);
		
		//cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);

		//Parallel
		lp_compute_maximum_labels_kernel<<<numberOfBlocks, nTPB>>>(
																	d_nodes, 
																	d_tails, 
																	d_indexs, 
																	d_labels,
																	d_labels, 
																	//d_labels_aux,
																	d_thereAreChanges, 
																	seed, 
																	nNodes,
																	nEdges,
																	nNodes
																);

		cudaMemcpy(&thereAreChanges, d_thereAreChanges, sizeof(bool), cudaMemcpyDeviceToHost);
		t++;
		//cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		if(t > nNodes)
			break;
	}
	cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_labels);
	cudaFree(d_nodes);
	cudaFree(d_tails);
	cudaFree(d_indexs);
	cudaFree(d_thereAreChanges);

	delete[] nodes;

	return labels;
}

int* labelPropagationParallel_V2(
				float* costs, 
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//int NUMBER_OF_THREADS = 32;
	//int SHARED_MEMORY_SIZE = (32 + 16) * nNodes * 4;

	//Number of threads
	int nTPB = MAX_THREADS_PER_BLOCK;											//Threads in a block  256
	int MAX_KERNEL_BLOCKS = 30;													//Max blocks in a Grid
	//int numberOfThreadsPerBlock = nTPB 
	//int numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, ((nNodes + nTPB - 1)/nTPB));	//Blocks in a Grid

	int *labels = new int[nNodes];
	//int *nodes = new int[nNodes];

	int tamLow = nNodes / 2;
	int tamHigh = (nNodes % 2 == 0)? nNodes / 2: (nNodes / 2) + 1;

	int blocksLow = MIN(MAX_KERNEL_BLOCKS, ((tamLow + nTPB - 1)/nTPB));
	int blocksHigh = MIN(MAX_KERNEL_BLOCKS, ((tamHigh + nTPB - 1)/nTPB));

	int *nodesLow = new int[tamLow];
	int *nodesHigh = new int[tamHigh];

	bool thereAreChanges = true;
	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		if(i < tamLow){
			nodesLow[i] = i;
		}
		else{
			nodesHigh[i - tamLow] = i;
		}
		
	}

	//GPU memory
	int *d_labels;
	int *d_nodesLow;
	int *d_nodesHigh;
	int *d_tails;
	int *d_indexs;
	bool *d_thereAreChanges;

	cudaMalloc(&d_labels, nNodes * sizeof(int));

	cudaMalloc(&d_nodesLow, tamLow * sizeof(int));
	cudaMalloc(&d_nodesHigh, tamHigh * sizeof(int));

	cudaMalloc(&d_tails, nEdges * sizeof(int));
	cudaMalloc(&d_indexs, nNodes * sizeof(int));
	cudaMalloc(&d_thereAreChanges, sizeof(bool));

	cudaMemcpy(d_tails, tails, nEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indexs, indexs, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);

	int t = 0;
	long seed = time(NULL);
	while(thereAreChanges){//until a node dont have the maximum label of their neightbors

		thereAreChanges =  false;
		
		getPermutation(nodesLow, tamLow); 
		getPermutation(nodesHigh, tamHigh); 

		cudaMemcpy(d_thereAreChanges, &thereAreChanges, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(d_nodesLow, nodesLow, tamLow * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_nodesHigh, nodesHigh, tamHigh * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);

		//Parallel
		lp_compute_maximum_labels_kernel<<<blocksLow, nTPB>>>(
																	d_nodesLow, 
																	d_tails, 
																	d_indexs, 
																	d_labels,
																	d_labels, 
																	//d_labels_aux,
																	d_thereAreChanges, 
																	seed, 
																	tamLow,
																	nEdges,
																	nNodes
																);
		check_CUDA_Error("Kernel nodes Low invocation");
		cudaCheckError(cudaDeviceSynchronize());

		lp_compute_maximum_labels_kernel<<<blocksHigh, nTPB>>>(
																	d_nodesHigh, 
																	d_tails, 
																	d_indexs, 
																	d_labels,
																	d_labels, 
																	//d_labels_aux,
																	d_thereAreChanges, 
																	seed, 
																	tamHigh,
																	nEdges,
																	nNodes
																);
		check_CUDA_Error("Kernel nodes High invocation");
		cudaCheckError(cudaDeviceSynchronize());

		cudaMemcpy(&thereAreChanges, d_thereAreChanges, sizeof(bool), cudaMemcpyDeviceToHost);
		t++;
		//cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		if(t > nNodes)
			break;
	}
	cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_labels);
	cudaFree(d_nodesLow);
	cudaFree(d_nodesHigh);
	cudaFree(d_tails);
	cudaFree(d_indexs);
	cudaFree(d_thereAreChanges);

	delete[] nodesLow;
	delete[] nodesHigh;

	return labels;
}
//----------------------------------------------------------------------------------------
