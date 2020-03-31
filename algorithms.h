#include "kernels.h"
#include <stdlib.h>
#include <bits/stdc++.h>
#include "community_measures.cu"
#include "scan.h"

using namespace std;
#define MIN(a,b) ((a > b)? b:a)		//Function to return the minium
#define MAX(a,b) ((a < b)? b:a)		//Return the maximum of two numbers

#define MAX_THREADS_PER_BLOCK 256
#define MAX_KERNEL_BLOCKS 4096
#define MAX_ITERATION 500

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
	int maximumLabel = maximumLabels[rand() % nMaxLabels];

	delete[] labelsEdges;
	delete[] sumsLabels;
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
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges,
				bool synchronous)
{
	cout<< "seq ";
	if(synchronous){
		cout<< "sync "; 
	}
	else{
		cout<< "async "; 
	}

	int *labels = new int[nNodes];
	int *labelsAux = new int[nNodes];
	int *nodes = new int[nNodes];
	int changes = 1;
	int maximumLabel = -1;
	int node;
	int t = 0;
	int com = 0, comAnt = 0;
	int res = 0, resAnt = -1;

	unsigned int seed = time(NULL);
	srand (seed);

	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		nodes[i] = i;
	}
	int maxIteration = MIN(nNodes, MAX_ITERATION);
	
	/*float *dataY = new float[maxIteration];
	float* results = nullptr;
	float learning_rate = 0.1f;
	int maxitergd = 1000;
	float min_error = 0.4f;*/

	while(changes){//until a node dont have the maximum of their neightbors
		//mod = getModularity(tails, indexs, nNodes, nEdges, labels);
		changes = 0;

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
				maximumLabel = getMaximumLabel(node, tails, indexs, labelsAux, nNodes, nEdges);
			}
			if(!synchronous){ //asynchronous
				maximumLabel = getMaximumLabel(node, tails, indexs, labels, nNodes, nEdges);
			}

			if(maximumLabel != labels[node]){
				labels[node] = maximumLabel;
				changes++;
			}
		}
		t++;
		com = countCommunities(labels, nNodes);
		//cout << " t:" << t << " changes:" << changes << " communities:" << com << endl;
		
		/*dataY[t] = com;
		if(t >= 10){
			results = gradient_descent(t, dataY, learning_rate, maxitergd, min_error); //results[0] = beta, results[1] = error	
			if(results[1] < min_error){
				break;
			}

		}*/
		
		res = comAnt - com;
		if(res == 0 && resAnt ==0){ break; }
		comAnt = com;
		resAnt = res;

		if(t > maxIteration){
			break;
		}
	}

	delete[] labelsAux;
	delete[] nodes;
	//delete[] dataY;
	//delete[] results;
	
	return labels;
}

//--------------------------Graph colors----------------------------------------
int* getGraphColors(
					int* nodes,
					int* tails, 
					int* indexs, 
					int* numberOfColors,
					int* colors,
					const int nNodes,
					const int nEdges
					)
{
	int index;
	int nextIndex; 
	int neighbor;
	int minimumcolor;
	int *colorsAux = new int[nNodes];

	for(int i = 0; i < nNodes; i++){
		colors[i] = -1;
		colorsAux[i] = -1;
	}

	for(int nodei = 0; nodei < nNodes; nodei++){
		minimumcolor = 0;
		//Visit all edges
		index = indexs[nodei];
		nextIndex = (nodei + 1 < nNodes)?indexs[nodei + 1]:nEdges; 
		for(int tail = index; tail < nextIndex; tail++){
			neighbor = tails[tail];//get the neighbor
			if(colorsAux[neighbor] == minimumcolor){
				minimumcolor++;
			}
		}
		//Set the color
		colorsAux[nodei] = minimumcolor;
	}

	*numberOfColors = countCommunities(colorsAux, nNodes);

	int pos = 0;
	int node = 0;
	int* ptrcolors = new int[*numberOfColors];
	for(int colori = 0; colori < *numberOfColors; colori++){
		ptrcolors[colori] = pos;
		for(int i = 0; i < nNodes; i++){
			node = i; //nodes[i];
			if(colorsAux[node] == colori){
				colors[pos] = node;
				pos++;
			}
		}
	}


	//test
	/*int *idcolors = getCommunities(colors, nNodes);

	for(int i = 0; i < *numberOfColors; i++){
		cout<<idcolors[i]<<", ";
	}
	cout<< endl;*/
	delete[] colorsAux;

	return ptrcolors;
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
int* labelPropagationSemiSynchSeq(
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	cout<< "seq semisync ";
	int *labels = new int[nNodes];
	int changes = 1;
	int maxLabel = -1;
	int node;
	int com = 0, comAnt = 0;
	int res = 0, resAnt = -1;
	int nColors;
	int t = 0;

	unsigned int seed = time(NULL);
	srand (seed);

	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
	}

	int *colors = new int[nNodes];
	int *ptrcolors = getGraphColors(nullptr, tails, indexs, &nColors, colors, nNodes,nEdges);
	//cout << "Number of colors:" << nColors << endl;

	int maxIteration = MIN(nNodes, MAX_ITERATION);
	int index, nextIndex;
	
	/*float *dataY = new float[maxIteration];
	float* results = nullptr;
	float learning_rate = 0.1f;
	int maxitergd = 1000;
	float min_error = 0.4f;*/

	while(changes){
		changes = 0;

		for(int colori = 0; colori < nColors; colori++){
			//cout << "color:" << colori << "[ ";
			
			index = ptrcolors[colori];
			nextIndex = (colori + 1 < nColors)?ptrcolors[colori + 1]: nNodes; 
			for(int i = index; i < nextIndex; i++){
				node = colors[i];//get the neighbor
				//cout<< node << " ";
				maxLabel = getMaximumLabel(node, tails, indexs, labels, nNodes, nEdges);
				if(maxLabel != labels[node]){
					labels[node] = maxLabel;
					changes++;
				}
			}
			//cout << "]" << endl;
		}
		t++;
		com = countCommunities(labels, nNodes);
		res = comAnt - com;
		//cout << " t:" << t << " changes:" << changes << " communities:" << com << endl;
		/*dataY[t] = com;
		if(t >= 10){
			results = gradient_descent(t, dataY, learning_rate, maxitergd, min_error); //results[0] = beta, results[1] = error	
			if(results[1] < min_error){
				break;
			}

		}*/

		if(res == 0 && resAnt ==0){ break; }
		comAnt = com;
		resAnt = res;

		if(t > maxIteration){
			break;
		}
	}

	delete[] colors;
	delete[] ptrcolors;
	//delete[] dataY;
	//delete[] results;

	return labels;
}


//-----------------------------Communities Parallel---------------------------------------

/**
	Apply the Label propagation algorithm synchronous in the parallel way

	Parameters.
	costs: Costs of the edges
	tails: Array of neighbors of the nodes
	indexs: Indexs of tails array
	nNodes: Number of nodes
	nEdges: Number of edges
*/
int* LPParallelSynchronous(
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//int NUMBER_OF_THREADS = 32;
	//int SHARED_MEMORY_SIZE = (32 + 16) * nNodes * 4;
	int nTPB = MAX_THREADS_PER_BLOCK;											//Threads in a block  256
	int nBlocks = nNodes / nTPB + ((nNodes % nTPB == 0)? 0 : 1); 
	int numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, nBlocks);	//Blocks in a Grid

	int *labels = new int[nNodes];
	int *nodes = new int[nNodes];
	int thereAreChanges = 1;
	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		nodes[i] = i;
	}

	cout<< "par sync ";

	//GPU memory
	int *d_labels;
	int *d_labels_ant;
	int *d_nodes;
	int *d_tails;
	int *d_indexs;
	int *d_thereAreChanges;

	cudaMalloc(&d_labels, nNodes * sizeof(int));
	cudaMalloc(&d_labels_ant, nNodes * sizeof(int));
	cudaMalloc(&d_nodes, nNodes * sizeof(int));
	cudaMalloc(&d_tails, nEdges * sizeof(int));
	cudaMalloc(&d_indexs, nNodes * sizeof(int));
	cudaMalloc(&d_thereAreChanges, sizeof(int));

	cudaMemcpy(d_tails, tails, nEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indexs, indexs, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, nodes, nNodes * sizeof(int), cudaMemcpyHostToDevice);

	int t = 0;
	unsigned int seed = time(NULL);

	int com = 0, comAnt = 0;
	int res = 0, resAnt = -1;
	int maxIteration = MIN(nNodes, MAX_ITERATION);
	
	/*float *dataY = new float[maxIteration];
	float* results = nullptr;
	float learning_rate = 0.1f;
	int maxitergd = 1000;
	float min_error = 0.4f;*/

	while(thereAreChanges > 0){//until a node dont have the maximum label of their neightbors
		//thereAreChanges =  0;
		//cudaMemcpy(d_thereAreChanges, &thereAreChanges, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_thereAreChanges, 0, sizeof(int));

		lp_copy_array<<<numberOfBlocks, nTPB>>>(
												d_labels_ant,
												d_labels,
												nNodes
												);
		//cudaMemcpy(d_labels_ant, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
		
		//Parallel
		lp_compute_maximum_labels_kernel<<<numberOfBlocks, nTPB>>>(
																	true,
																	d_nodes, 
																	d_tails, 
																	d_indexs, 
																	d_labels,
																	d_labels_ant, 
																	d_thereAreChanges, 
																	seed, 
																	nNodes,
																	nEdges,
																	nNodes
																);
		cudaDeviceSynchronize();

		cudaMemcpy(&thereAreChanges, d_thereAreChanges, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		t++;

		com = countCommunities(labels, nNodes);
		res = comAnt - com;
		//cout << " t:" << t << " changes:" << thereAreChanges << " communities:" << com << endl;
		/*dataY[t] = com;
		if(t >= 10){
			results = gradient_descent(t, dataY, learning_rate, maxitergd, min_error); //results[0] = beta, results[1] = error	
			if(results[1] < min_error){
				break;
			}

		}*/

		if(res == 0 && resAnt ==0){ break; }
		comAnt = com;
		resAnt = res;

		if(t > maxIteration){
			break;
		}
	}
	//cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_labels);
	cudaFree(d_labels_ant);
	cudaFree(d_nodes);
	cudaFree(d_tails);
	cudaFree(d_indexs);
	cudaFree(d_thereAreChanges);

	delete[] nodes;
	//delete[] dataY;
	//delete[] results;

	return labels;
}


/**
	Apply the Label propagation algorithm in the parallel way

	Parameters.
	costs: Costs of the edges
	tails: Array of neighbors of the nodes
	indexs: Indexs of tails array
	nNodes: Number of nodes
	nEdges: Number of edges
*/
int* LPParallelAsynchronous(
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//int NUMBER_OF_THREADS = 32;
	//int SHARED_MEMORY_SIZE = (32 + 16) * nNodes * 4;
	//Number of threads
	int nTPB = MAX_THREADS_PER_BLOCK;											//Threads in a block  256
	int nBlocks = nNodes / nTPB + ((nNodes % nTPB == 0)? 0 : 1); 
	int numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, nBlocks);	//Blocks in a Grid

	int *labels = new int[nNodes];
	int *nodes = new int[nNodes];
	int thereAreChanges = 1;
	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		nodes[i] = i;
	}

	cout<< "par async ";

	//GPU memory
	int *d_labels;
	int *d_labels_ant;
	int *d_nodes;
	int *d_tails;
	int *d_indexs;
	int *d_thereAreChanges;

	cudaMalloc(&d_labels, nNodes * sizeof(int));
	cudaMalloc(&d_labels_ant, nNodes * sizeof(int));
	cudaMalloc(&d_nodes, nNodes * sizeof(int));
	cudaMalloc(&d_tails, nEdges * sizeof(int));
	cudaMalloc(&d_indexs, nNodes * sizeof(int));
	cudaMalloc(&d_thereAreChanges, sizeof(int));

	cudaMemcpy(d_tails, tails, nEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indexs, indexs, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, nodes, nNodes * sizeof(int), cudaMemcpyHostToDevice);

	int t = 0;
	unsigned int seed = time(NULL);
	int com = 0, comAnt = 0;
	int res = 0, resAnt = -1;

	int maxIteration = MIN(nNodes, MAX_ITERATION);
	
	/*float *dataY = new float[maxIteration];
	float* results = nullptr;
	float learning_rate = 0.1f;
	int maxitergd = 1000;
	float min_error = 0.4f;*/

	while(thereAreChanges > 0){//until a node dont have the maximum label of their neightbors
		//thereAreChanges =  0;
		//cudaMemcpy(d_thereAreChanges, &thereAreChanges, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_thereAreChanges, 0, sizeof(int));
		
		getPermutation(nodes, nNodes);
		cudaMemcpy(d_nodes, nodes, nNodes * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_labels_ant, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToDevice);

		//Parallel
		lp_compute_maximum_labels_kernel<<<numberOfBlocks, nTPB>>>(
																	false,
																	d_nodes, 
																	d_tails, 
																	d_indexs, 
																	d_labels,
																	d_labels, 
																	d_thereAreChanges, 
																	seed, 
																	nNodes,
																	nEdges,
																	nNodes
																);
		cudaDeviceSynchronize();

		cudaMemcpy(&thereAreChanges, d_thereAreChanges, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		t++;

		com = countCommunities(labels, nNodes);
		res = comAnt - com;
		//cout << " t:" << t << " changes:" << thereAreChanges << " communities:" << com << endl;
		/*dataY[t] = com;
		if(t >= 10){
			results = gradient_descent(t, dataY, learning_rate, maxitergd, min_error); //results[0] = beta, results[1] = error	
			if(results[1] < min_error){
				break;
			}

		}*/

		if(res == 0 && resAnt ==0){ break; }
		comAnt = com;
		resAnt = res;

		if(t > maxIteration){
			break;
		}
	}
	//cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_labels);
	cudaFree(d_labels_ant);
	cudaFree(d_nodes);
	cudaFree(d_tails);
	cudaFree(d_indexs);
	cudaFree(d_thereAreChanges);

	delete[] nodes;
	//delete[] dataY;
	//delete[] results;

	return labels;
}

/**
	Apply the Label propagation algorithm in the parallel way

	Parameters.
	costs: Costs of the edges
	tails: Array of neighbors of the nodes
	indexs: Indexs of tails array
	nNodes: Number of nodes
	nEdges: Number of edges
*/
int* LPParallelSemySynchronous(
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//int NUMBER_OF_THREADS = 32;
	//int SHARED_MEMORY_SIZE = (32 + 16) * nNodes * 4;
	//Number of threads
	int nTPB = MAX_THREADS_PER_BLOCK;											//Threads in a block  256
	int nBlocks = nNodes / nTPB + ((nNodes % nTPB == 0)? 0 : 1); 
	int numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, nBlocks);	//Blocks in a Grid

	int *labels = new int[nNodes];
	int changes = 1;
	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
	}

	cout<< "par semisync ";
	
	int nColors;
	int *colors = new int[nNodes];
	int *ptrcolors = getGraphColors(nullptr, tails, indexs, &nColors, colors, nNodes,nEdges);
	//cout << "Number of colors:" << nColors << endl;
	int nNodesColori = 0;

	//GPU memory
	int *d_labels;
	int *d_nodes;
	int *d_tails;
	int *d_indexs;
	int *d_changes;

	cudaMalloc(&d_labels, nNodes * sizeof(int));
	cudaMalloc(&d_nodes, nNodes * sizeof(int));
	cudaMalloc(&d_tails, nEdges * sizeof(int));
	cudaMalloc(&d_indexs, nNodes * sizeof(int));
	cudaMalloc(&d_changes, sizeof(int));

	cudaMemcpy(d_tails, tails, nEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indexs, indexs, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, colors, nNodes * sizeof(int), cudaMemcpyHostToDevice);

	int t = 0;
	unsigned int seed = time(NULL);

	int com = 0, comAnt = 0;
	int res = 0, resAnt = -1;

	int maxIteration = MIN(nNodes, MAX_ITERATION);
	int index, nextIndex;
	int tamLow, tamHigh;

	/*float *dataY = new float[maxIteration];
	float* results = nullptr;
	float learning_rate = 0.1f;
	int maxitergd = 1000;
	float min_error = 0.4f;*/

	while(changes > 0){//until a node dont have the maximum label of their neightbors
		//thereAreChanges =  0;
		//cudaMemcpy(d_thereAreChanges, &thereAreChanges, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_changes, 0, sizeof(int));
		

		for(int colori = 0; colori < nColors; colori++){
			//printf("color: %d [ ", colori );
 
			index = ptrcolors[colori];
			nextIndex = (colori + 1 < nColors)?ptrcolors[colori + 1]: nNodes;
			nNodesColori = (nextIndex - index < 0)?1 : nextIndex - index;

			tamLow = nNodesColori / 2;
			nBlocks = (tamLow / 2)  / nTPB + ((tamLow % nTPB == 0)? 0 : 1); 
			numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, nBlocks);

			lp_compute_maximum_labels_semi<<<numberOfBlocks, nTPB>>>(
																		d_nodes, 
																		d_tails, 
																		d_indexs, 
																		d_labels, 
																		d_changes, 
																		seed, 
																		nNodes,
																		nEdges,
																		index,
																		index + (nNodesColori / 2) 
																	);
			cudaDeviceSynchronize();

			tamHigh = (nNodesColori % 2 == 0)? nNodesColori / 2: (nNodesColori / 2) + 1;
			nBlocks = (tamHigh / 2)  / nTPB + ((tamHigh % nTPB == 0)? 0 : 1); 
			numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, nBlocks);
			lp_compute_maximum_labels_semi<<<numberOfBlocks, nTPB>>>(
																		d_nodes, 
																		d_tails, 
																		d_indexs, 
																		d_labels, 
																		d_changes, 
																		seed, 
																		nNodes,
																		nEdges,
																		index + (nNodesColori / 2),
																		nextIndex
																	);

			//TODO: Partir en 2

			//printf("]\n");
			cudaDeviceSynchronize();
		}

		cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		t++;

		com = countCommunities(labels, nNodes);
		res = comAnt - com;
		//cout << " t:" << t << " changes:" << changes << " communities:" << com << endl;
		/*dataY[t] = com;
		if(t >= 10){
			results = gradient_descent(t, dataY, learning_rate, maxitergd, min_error); //results[0] = beta, results[1] = error	
			if(results[1] < min_error){
				break;
			}

		}*/

		if(res == 0 && resAnt ==0){ break; }
		comAnt = com;
		resAnt = res;

		if(t > maxIteration){
			break;
		}
	}
	//cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_labels);
	cudaFree(d_nodes);
	cudaFree(d_tails);
	cudaFree(d_indexs);
	cudaFree(d_changes);

	delete[] ptrcolors;
	delete[] colors;
	//delete[] dataY;
	//delete[] results;

	return labels;
}


/**
	Apply the Label propagation algorithm in the parallel way -- version 2

	Parameters.
	costs: Costs of the edges
	tails: Array of neighbors of the nodes
	indexs: Indexs of tails array
	nNodes: Number of nodes
	nEdges: Number of edges
*/
int* LPParallel_V2(
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//CPU memory
	int *labels = new int[nNodes];
	int numberChanges = 1;
	int t = 0;
	unsigned int seed = time(NULL);
	int com = 0, comAnt = 0;
	int res = 0, resAnt = -1;

	//GPU memory
	int *d_labels;		//size:nNodes
	int *d_labels_ant;	//size:nNodes
	int *d_labels_v;	//size:nEdges
	
	int *d_F;			//size:nEdges
	int *d_F_s;			//size:nEdges
	
	int *d_S;			//size:F[n - 1] + 1
	int *d_W;			//size:F[n - 1] + 1
	int *d_S_ptr;		//size:nNodes
	int *d_I;			//size:nNodes

	int *d_tails;		//id size:nEdges
	int *d_indexs;		//ptr size:nNodes
	int *d_numberChanges;

	cudaMalloc(&d_labels, nNodes * sizeof(int));
	cudaMalloc(&d_labels_ant, nNodes * sizeof(int));
	cudaMalloc(&d_labels_v, nEdges * sizeof(int));
	
	cudaMalloc(&d_F, nEdges * sizeof(int));
	cudaMalloc(&d_F_s, nEdges * sizeof(int));

	cudaMalloc(&d_S, nEdges * sizeof(int));
	cudaMalloc(&d_W, nEdges * sizeof(int));
	cudaMalloc(&d_S_ptr, nNodes * sizeof(int));
	cudaMalloc(&d_I, nNodes * sizeof(int));
	cudaMalloc(&d_tails, nEdges * sizeof(int));
	cudaMalloc(&d_indexs, nNodes * sizeof(int));
	cudaMalloc(&d_numberChanges, sizeof(int));

	cudaMemcpy(d_tails, tails, nEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indexs, indexs, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	
	//Number of threads and blocks
	int nTPB = MAX_THREADS_PER_BLOCK;											//Threads in a block  256
	int nBNodes = nNodes / nTPB + ((nNodes % nTPB == 0)? 0 : 1); 
	int nBEdges = nEdges / nTPB + ((nEdges % nTPB == 0)? 0 : 1);
	int nBlocksNodes = MIN(MAX_KERNEL_BLOCKS, nBNodes);							//Blocks in a Grid
	int nBlocksEdges = MIN(MAX_KERNEL_BLOCKS, nBEdges);

	cout<< "par async2 ";

	//cout << "nBlocksNodes: " << nBlocksNodes << " nTPB: " << nTPB << endl; 
	lp_init_labels<<<nBlocksNodes, nTPB>>>(d_labels, nNodes);
	check_CUDA_Error("Kernel lp_init_labels invocation");
	cudaCheckError( cudaDeviceSynchronize() );

	int maxIteration = MIN(nNodes, MAX_ITERATION);
	int* aux = new int[nEdges];
	int* auxNodes = new int[nNodes];

	/*float *dataY = new float[maxIteration];
	float* results = nullptr;
	float learning_rate = 0.1f;
	int maxitergd = 1000;
	float min_error = 0.4f;*/

	while(numberChanges > 0){//until a node dont have the maximum label of their neightbors	 
		lp_copy_array<<<nBlocksNodes, nTPB>>>(d_labels_ant, d_labels, nNodes);
		check_CUDA_Error("Kernel lp_copy_array invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		lp_gather<<<nBlocksEdges, nTPB>>>(d_labels, d_labels_v, d_tails, nEdges);
		check_CUDA_Error("Kernel lp_gather invocation");
		cudaCheckError( cudaDeviceSynchronize() );
		
		lp_sort<<<nBlocksNodes, nTPB>>>(d_indexs, d_labels_v, nNodes, nEdges);
		check_CUDA_Error("Kernel lp_sort invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		lp_init_boundaries_1<<<nBlocksEdges, nTPB>>>(d_labels_v, d_F, nEdges);
		check_CUDA_Error("Kernel lp_init_boundaries_1 invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		lp_init_boundaries_2<<<nBlocksNodes, nTPB>>>(d_indexs, d_F, nNodes);
		check_CUDA_Error("Kernel lp_init_boundaries_2 invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		//Scan
		sum_scan_blelloch(d_F_s, d_F, nEdges);
		//cudaCheckError( cudaDeviceSynchronize() );
		
		//TESTTTT
		//cudaMemcpy(aux, d_F_s, nEdges * sizeof(int), cudaMemcpyDeviceToHost);
		//printarray(aux, nEdges);


		lp_init_S<<<nBlocksEdges, nTPB>>>(d_S, d_F_s, nEdges);
		check_CUDA_Error("Kernel lp_init_S invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		lp_init_Sptr<<<nBlocksNodes, nTPB>>>(d_F_s, d_indexs, d_S_ptr, nNodes);
		check_CUDA_Error("Kernel lp_init_Sptr invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		lp_init_W<<<nBlocksEdges, nTPB>>>(d_S, d_W, d_F_s, nEdges);
		cudaDeviceSynchronize();

		lp_reduce<<<nBlocksNodes, nTPB>>>(d_S_ptr, d_W, d_I, nNodes, seed, d_F_s, nEdges);
		check_CUDA_Error("Kernel lp_reduce invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		lp_compute_labels<<<nBlocksNodes, nTPB>>>(d_labels, d_labels_v, d_S, d_I, nNodes);
		check_CUDA_Error("Kernel lp_compute_labels invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		/*cudaMemcpy(auxNodes, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		printarray(auxNodes, nNodes);

		cudaMemcpy(auxNodes, d_labels_ant, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		printarray(auxNodes, nNodes);*/

		cudaMemset(d_numberChanges, 0, sizeof(int));
		lp_compare_labels<<<nBlocksNodes, nTPB>>>(d_labels, d_labels_ant, d_numberChanges, nNodes);
		check_CUDA_Error("Kernel lp_compute_labels invocation");
		cudaCheckError( cudaDeviceSynchronize() );

		cudaMemcpy(&numberChanges, d_numberChanges, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

		t++;
		com = countCommunities(labels, nNodes);
		res = comAnt - com;
		//cout << " t:" << t << " changes:" << numberChanges << " communities:" << com << endl;
		/*dataY[t] = com;
		if(t >= 10){
			results = gradient_descent(t, dataY, learning_rate, maxitergd, min_error); //results[0] = beta, results[1] = error	
			if(results[1] < min_error){
				break;
			}

		}*/

		if(res == 0 && resAnt ==0){ break; }
		comAnt = com;
		resAnt = res;

		if(t > maxIteration){
			break;
		}
	}

	cudaFree(d_labels);
	cudaFree(d_labels_ant);
	cudaFree(d_labels_v);
	cudaFree(d_F);
	cudaFree(d_F_s);
	cudaFree(d_S);
	cudaFree(d_W);
	cudaFree(d_S_ptr);
	cudaFree(d_I);
	cudaFree(d_tails);
	cudaFree(d_indexs);
	cudaFree(d_numberChanges);

	delete[] aux;
	delete[] auxNodes;
	//delete[] dataY;
	//delete[] results;

	return labels;
}

int* LPParallelAsynchronous_v2(
				int* tails, 
				int* indexs, 
				const int nNodes,
				const int nEdges)
{
	//int NUMBER_OF_THREADS = 32;
	//int SHARED_MEMORY_SIZE = (32 + 16) * nNodes * 4;
	//Number of threads
	int nTPB = MAX_THREADS_PER_BLOCK;											//Threads in a block  256
	int nBlocks = nNodes / nTPB + ((nNodes % nTPB == 0)? 0 : 1); 
	int numberOfBlocks = MIN(MAX_KERNEL_BLOCKS, nBlocks);	//Blocks in a Grid

	int *labels = new int[nNodes];
	int *nodes = new int[nNodes];
	int thereAreChanges = 1;
	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		nodes[i] = i;
	}

	cout<< "par async (version 2)" << endl;

	//GPU memory
	int *d_labels;
	int *d_labels_ant;
	int *d_nodes;
	int *d_tails;
	int *d_indexs;
	int *d_thereAreChanges;

	cudaMalloc(&d_labels, nNodes * sizeof(int));
	cudaMalloc(&d_labels_ant, nNodes * sizeof(int));
	cudaMalloc(&d_nodes, nNodes * sizeof(int));
	cudaMalloc(&d_tails, nEdges * sizeof(int));
	cudaMalloc(&d_indexs, nNodes * sizeof(int));
	cudaMalloc(&d_thereAreChanges, sizeof(int));

	cudaMemcpy(d_tails, tails, nEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indexs, indexs, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels, labels, nNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, nodes, nNodes * sizeof(int), cudaMemcpyHostToDevice);

	int t = 0;
	unsigned int seed = time(NULL);
	int com = 0, comAnt = 0;
	int res = 0, resAnt = -1;

	int maxIteration = MIN(nNodes, MAX_ITERATION);
	
	/*float *dataY = new float[maxIteration];
	float* results = nullptr;
	float learning_rate = 0.1f;
	int maxitergd = 1000;
	float min_error = 0.4f;*/

	while(thereAreChanges > 0){//until a node dont have the maximum label of their neightbors
		//thereAreChanges =  0;
		//cudaMemcpy(d_thereAreChanges, &thereAreChanges, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_thereAreChanges, 0, sizeof(int));
		
		getPermutation(nodes, nNodes);
		cudaMemcpy(d_nodes, nodes, nNodes * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_labels_ant, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToDevice);

		//Parallel
		lp_compute_maximum_labels_kernel<<<numberOfBlocks, nTPB>>>(
																	false,
																	d_nodes, 
																	d_tails, 
																	d_indexs, 
																	d_labels,
																	d_labels, 
																	d_thereAreChanges, 
																	seed, 
																	nNodes,
																	nEdges,
																	nNodes
																);
		cudaDeviceSynchronize();

		cudaMemcpy(&thereAreChanges, d_thereAreChanges, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);
		t++;

		com = countCommunities(labels, nNodes);
		res = comAnt - com;
		//cout << " t:" << t << " changes:" << thereAreChanges << " communities:" << com << endl;
		/*dataY[t] = com;
		if(t >= 10){
			results = gradient_descent(t, dataY, learning_rate, maxitergd, min_error); //results[0] = beta, results[1] = error	
			if(results[1] < min_error){
				break;
			}

		}*/

		if(res == 0 && resAnt ==0){ break; }
		comAnt = com;
		resAnt = res;

		if(t > maxIteration){
			break;
		}
	}
	//cudaMemcpy(labels, d_labels, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_labels);
	cudaFree(d_labels_ant);
	cudaFree(d_nodes);
	cudaFree(d_tails);
	cudaFree(d_indexs);
	cudaFree(d_thereAreChanges);

	delete[] nodes;
	//delete[] dataY;
	//delete[] results;

	return labels;
}
//----------------------------------------------------------------------------------------
