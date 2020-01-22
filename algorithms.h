#include "kernels.h"
#include <stdlib.h>
#include <bits/stdc++.h>

using namespace std;

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
//----------------------------------------B Centrality Algorithms--------------------------------------------
//----------------------------------------Secuential---------------------------------------------------------
//---------------------Centrality-----------------------------
__host__ 
void dijkstraSequential(
				float* costs, 
				int* tails, 
				int* indexs, 
				float* centrality,
				const int nNodes,
				const int nEdges)
{
	int cost = 0, totalcost = 0;
	int node = 0, endpoint = 0;
	int index, nextIndex;
	
	int *visited = new int[nNodes]; 
	int *distance = new int[nNodes];  
	List *parents = new List[nNodes];


	for(int source = 0; source < nNodes; source++){
		//Inicializar arrays
		for(int i = 0; i < nNodes; i++){
			visited[i] = 0;
			distance[i] = INF;
			parents[i].clear();
		}
		distance[source] = 0;
		
	 	//Iterate in the node
		while ((node = getSmallDistance(visited, distance, nNodes)) != -1) {
			index = indexs[node];
			nextIndex = (node + 1 < nNodes)?indexs[node + 1]:nEdges;//getNextIndex(node, indexs, nNodes, nEdges);

			if(index != -1){
				for (int i = index; i < nextIndex ; i++) { //regresar el tamaño de los endpoints
					endpoint = tails[i];
					
					if (endpoint != -1 && visited[endpoint] != 1) {
						cost = costs[i];
						totalcost = cost + distance[node];
						if (totalcost < distance[endpoint]) { //Add only one path
							distance[endpoint] = totalcost;
							parents[endpoint].clear();
							parents[endpoint].push_back(node);
						}
						else if (totalcost == distance[endpoint]) { //Add other shortest path
							parents[endpoint].push_back(node);
						}
					}
				}
			}
			visited[node]= 1;
		}

		//Calcular la centralidad
		for (int nodo_j = 0; nodo_j < nNodes; nodo_j++) {
			computeCentralityPath(source, nodo_j, parents, centrality);
		}
	}

	delete[] visited;
	delete[] distance;
	delete[] parents;
}

__host__
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

float* GPUFANSequential(
					float* costs, 
					int* tails, 
					int* indexs, 
					const int nNodes,
					const int nEdges)
{
	int n_nodes = nNodes;
	int index, nextIndex;
	float *centrality = new float[nNodes]; 

	for(int i = 0;i < nNodes; i++){
		centrality[i] = 0;
	}

	for(int i=0; i<n_nodes; i++){
		stack<int> s;
		list<int> l;
		vector<list<int> > vli(n_nodes, l);
		vector<int> sigma(n_nodes, 0);
		vector<int> d(n_nodes,-1);
		
		sigma[i] = 1;
		d[i] = 0;
		
		queue<int> q;

		q.push(i);
		while(!q.empty()){
			int v = q.front();
			q.pop();
			s.push(v);

			index = indexs[v];
			nextIndex = (v + 1 < nNodes)?indexs[v + 1]:nEdges; //getNextIndex(v, indexs, n_nodes, nEdges);
			for(int tail = index; tail < nextIndex; tail++){
				int w = tails[tail];;
				if(d[w] < 0){
					q.push(w);
					d[w]=d[v] + 1;
				}
				if(d[w] == d[v] + 1){
					sigma[w]+=sigma[v];
					vli[w].push_back(v);
				}
			}
		}
		vector<float> delta(n_nodes,0);
		while(!s.empty()){
			int w = s.top();
			s.pop();
			list<int>::iterator lit;
			for(lit=vli[w].begin(); lit!=vli[w].end(); lit++){
				int v = *lit;
				delta[v] += (1.0 * sigma[v] / sigma[w] * (1 + delta[w]));
			}
			if(w!=i)
				centrality[w]+=delta[w];
		}
	}

	return centrality;
}

float* hybric_BCSequential( 
					float* costs, 
					int* tails, 
					int* indexs, 
					const int nNodes,
					const int nEdges)
{
	//g.R = (rows)    indexs
	//g.C = (columns) tails 

	//std::vector<float> bc(nNodes,0);

	float *bc = new float[nNodes]; 

	for(int i = 0;i < nNodes; i++){
		bc[i] = 0;
	}

	int end = nNodes;
	//std::set<int>::iterator it = source_vertices.begin();

	for(int k=0; k<end; k++)
	{
		int i = k;
		std::queue<int> Q;
		std::stack<int> S;
		std::vector<int> d(nNodes,INT_MAX);
		d[i] = 0;
		std::vector<unsigned long long> sigma(nNodes,0);
		sigma[i] = 1;
		std::vector<float> delta(nNodes,0);
		Q.push(i);

		while(!Q.empty())
		{
			int v = Q.front();
			Q.pop();
			S.push(v);
			int start = indexs[v];/*g.R[v]*/
			int end = (v + 1 < nNodes)?indexs[v + 1]:nEdges;/*g.R[v+1]*/
			for(int j = start; j < end; j++)
			{
				int w = tails[j]; //g.C[j];
				if(d[w] == INT_MAX)
				{
					Q.push(w);
					d[w] = d[v] + 1;
				}
				if(d[w] == (d[v] + 1))
				{
					sigma[w] += sigma[v];
				}
			}
		}

		while(!S.empty())
		{
			int w = S.top();
			S.pop();
	
			int start = indexs[w];/*g.R[v]*/
			int end = (w + 1 < nNodes)?indexs[w + 1]:nEdges;/*g.R[v+1]*/
			for(int j = start; j < end; j++)
			{
				int v = tails[j];///g.C[j];
				if(d[v] == (d[w] - 1))
				{
					delta[v] += (sigma[v]/(float)sigma[w])*(1+delta[w]);
				}
			}	
			
			if(w != i)
			{
				bc[w] += delta[w];
			}
		}
	}

	/*for(int i=0; i < nNodes; i++)
	{
		bc[i] /= 2.0f; //Undirected edges are modeled as two directed edges, but the scores shouldn't be double counted.
	}*/
	return bc;
}
//-------------------------------------------------------------
//_-----------------Communities--------------------------------
/**
	Compute the community detection running the label propagation algorithm
*/
int findLabel(int* labelsNeighbors,
			  int label,
			  int totalNeighbors){
	for(int n = 0; n < totalNeighbors; n++){
		if(label == labelsNeighbors[n]){
			return n;
		}
	}

	return -1;
}

int getMaximumLabel(int node,
					int* tails, 
					int* indexs,
					int* labels, 
					const int nNodes,
					const int nEdges){
	//Get their neighboors
	int maximumLabel = -1;
	int neighbor = -1;
	int index = indexs[node];
	int nextIndex = (node + 1 < nNodes)?indexs[node + 1]:nEdges; 
	int tamLabels = (nextIndex - index < 0)?1 : nextIndex - index; 

	int *labelsNeighbors = new int[tamLabels];
	int *countersLabels = new int[tamLabels];
	int posLabelN = -1;
	int itLabelN = 0;
	for(int i = 0; i < tamLabels; i++){
		labelsNeighbors[i] = -1;
		countersLabels[i] = 0;
	}

	//find the maximum label of their neighbors

	//Count the labels
	for(int tail = index; tail < nextIndex; tail++){
		neighbor = tails[tail];//get the neighbor
		if(neighbor < nNodes){ //the neightbor exist
			//find if the label exist en the labelsNeighbor
			posLabelN = findLabel(labelsNeighbors, labels[neighbor], tamLabels);
			if(posLabelN == -1){//new label
				labelsNeighbors[itLabelN] = labels[neighbor];
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
	for(int i = 0;i < itLabelN; i++){
		if(numberMax < countersLabels[i]){
			numberMax = countersLabels[i];
			maximumLabel = labelsNeighbors[i];
		}
		//else if(numberMax == countersLabels[i]){
			//add new maximum label
		//}

	}

	delete[] labelsNeighbors;
	delete[] countersLabels;
	return maximumLabel;
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
				const int nEdges)
{
	int *labels = new int[nNodes];
	int *nodes = new int[nNodes];
	bool thereAreChanges = true;
	int maximumLabel = -1;
	int node;

	/* initialize random seed: */
	srand (time(NULL));

	//set the community to each node
	for(int i = 0;i < nNodes; i++){
		labels[i] = i;
		nodes[i] = i;
	}

	sort(nodes, nodes + nNodes);
	int t = 0;
	while(thereAreChanges){//until a node dont have the maximum of their neightbors
		printf("Iteracion %d \n [", t);
		for(int i = 0;i < nNodes; i++){
			printf("%d, ",labels[i]);
		}
		printf("]\n");

		thereAreChanges =  false;
		//Optionally: delete nodes with 1 edge and 0 edges
		next_permutation(nodes, nodes + nNodes);

		for(int i = 0;i < nNodes; i++){
			cout << nodes[i] << ",";
		}
		cout << endl;

		for(int i = 0; i < nNodes; i++){ //random permutation of Nodes
			node = nodes[i];
			//find the maximum label of their neightbors
			maximumLabel = getMaximumLabel(node, tails, indexs, labels, nNodes, nEdges);
			if(maximumLabel != labels[node]){
				labels[node] = maximumLabel;
				thereAreChanges = true;
			}
		}
		t++;
	}
	return labels;
}
//-------------------------------------------------------------



//---------------------------------------------------Parallel--------------------------------------------

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


int bc_bfs(int n_count, int e_count, int * h_v, int *h_e, float *h_bc){
	/*cout<< "nodos:"<< n_count<<endl;
	cout<< "edges:"<< e_count<<endl;

	cout<<"nodes: ["<<endl;
	for(int n = 0; n < e_count; n++){
		cout<< ", "<< h_v[n];
	}
	cout<<"]"<<endl;

	cout<<"edges: ["<<endl;
	for(int n = 0; n < e_count; n++){
		cout<< ", "<< h_e[n];
	}
	cout<<"]"<<endl;

	cout<<h_bc<<endl;*/

  int *d_v, *d_e;
  cudaCheckError(cudaMalloc((void **)&d_v, sizeof(int)*e_count));
  cudaCheckError(cudaMalloc((void **)&d_e, sizeof(int)*e_count));

  cudaCheckError(cudaMemcpy(d_v, h_v, sizeof(int)*e_count, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_e, h_e, sizeof(int)*e_count, cudaMemcpyHostToDevice));

  int *d_d, *d_sigma;
  float *d_delta;
  /* use unsigned int array to implement bit array */
  unsigned int *d_p; /* two dimensional predecessor  array (nxn)*/ 
  int *d_dist;
  float *d_bc;

  cudaCheckError(cudaMalloc((void **)&d_d, sizeof(int)*n_count));
  cudaCheckError(cudaMalloc((void **)&d_sigma, sizeof(int)*n_count)); 
  cudaCheckError(cudaMalloc((void **)&d_delta, sizeof(float)*n_count)); 
  unsigned long long total_bits=(unsigned long long)n_count*n_count;
  unsigned int num_of_ints=BITS_TO_INTS(total_bits);

  cudaCheckError(cudaMalloc((void **)&d_p, sizeof(unsigned int)*num_of_ints)); 
  cudaCheckError(cudaMalloc((void **)&d_dist, sizeof(int)));
  cudaCheckError(cudaMalloc((void **)&d_bc, sizeof(float)*n_count)); 

  cudaCheckError(cudaMemcpy(d_bc, h_bc, sizeof(float)*n_count, cudaMemcpyHostToDevice));

  int *h_d;
  int h_sigma_0=1;
  h_d=(int *)malloc(sizeof(int)*n_count);

  for(int i=0; i<n_count; i++){
    for(int j=0; j<n_count; j++)
      h_d[j]=-1;
    h_d[i]=0;
    cudaCheckError(cudaMemcpy(d_d, h_d, sizeof(int)*n_count, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(d_sigma, 0, sizeof(int)*n_count));
    cudaCheckError(cudaMemcpy(&d_sigma[i],&h_sigma_0, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(d_delta, 0, sizeof(int)*n_count));
    cudaCheckError(cudaMemset(d_p, 0, sizeof(unsigned int)*num_of_ints));
    int threads_per_block=e_count;
    int blocks=1;
    if(e_count>MAX_THREADS_PER_BLOCK){
      blocks = (int)ceil(e_count/(float)MAX_THREADS_PER_BLOCK); 
      threads_per_block = MAX_THREADS_PER_BLOCK; 
    }
    dim3 grid(blocks);
    dim3 threads(threads_per_block);
    int threads_per_block2=n_count;
    int blocks2=1;
    if(n_count>MAX_THREADS_PER_BLOCK){ 
      blocks2 = (int)ceil(n_count/(double)MAX_THREADS_PER_BLOCK); 
      threads_per_block2 = MAX_THREADS_PER_BLOCK; 
    }
    dim3 grid2(blocks2);
    dim3 threads2(threads_per_block2);

    bool h_continue;
    bool *d_continue;
    cudaMalloc((void **)&d_continue, sizeof(bool));
    int h_dist=0;
    cudaCheckError(cudaMemset(d_dist, 0, sizeof(int)));
    // BFS  
    do{
      h_continue=false;
      cudaCheckError(cudaMemcpy(d_continue, &h_continue, sizeof(bool), cudaMemcpyHostToDevice));
      bc_bfs_kernel<<<grid,threads>>>(d_v, d_e, d_d, d_sigma, d_p, d_continue, d_dist, n_count, e_count);
      check_CUDA_Error("Kernel bc_bfs_kernel invocation");
      cudaCheckError(cudaDeviceSynchronize());
      h_dist++; 
      cudaCheckError(cudaMemcpy(d_dist, &h_dist, sizeof(int), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));
    }while(h_continue);   
    //cout << "h_dist: " << h_dist;

    h_continue=false;
    //Back propagation
    cudaCheckError(cudaMemcpy(&h_dist, d_dist, sizeof(int), cudaMemcpyDeviceToHost));
    do{
      bc_bfs_back_prop_kernel<<<grid, threads>>>(d_v, d_e, d_d, d_sigma, d_delta, d_p, d_dist, n_count, e_count);
      check_CUDA_Error("Kernel bc_bfs_back_prop_kernel invocation");
      cudaCheckError(cudaDeviceSynchronize());
      bc_bfs_back_sum_kernel<<<grid2, threads2>>>(i, d_dist, d_d,  d_delta, d_bc, n_count);
      check_CUDA_Error("Kernel bc_bfs_back_sum_kernel invocation");
      cudaCheckError(cudaDeviceSynchronize());
      h_dist--;
      cudaCheckError(cudaMemcpy(d_dist, &h_dist, sizeof(int), cudaMemcpyHostToDevice));
    }while(h_dist>1);
  }
  cudaCheckError(cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));

  /*for(int i= 0; i < n_count; i++){
  	cout << "d_bc: " << h_bc[i]<< endl;
  }*/


  free(h_d);
  cudaFree(d_v);
  cudaFree(d_e);
  cudaFree(d_d);
  cudaFree(d_sigma);
  cudaFree(d_delta);
  cudaFree(d_p);
  cudaFree(d_dist);
  return 0;
}
