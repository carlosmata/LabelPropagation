//------------------------------------Graph methods----------------------------------------------------
#define INF 2147483647;
#ifndef INT_BIT
#define INT_BIT 32
#endif
#define BIT_INT(bit)         ((bit) / INT_BIT)								/* array index for character containing bit */
#define BIT_IN_INT(bit)      (1 << (INT_BIT - 1 - ((bit)  % INT_BIT)))		/* position of bit within character */
#define BITS_TO_INTS(bits)   ((((bits) - 1) / INT_BIT) + 1)  

__host__ __device__
int getNextIndex(int source, int *indexs, int nNodes, int nEdges) {
	
	for (int i = source + 1; i < nNodes; i++) {
		if (indexs[i] > indexs[source] &&
			indexs[i] < nEdges) {
			return indexs[i];
		}
	}

	return nEdges;
}

int verifyDeviceCUDA(){
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp devProps;

	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
		std::cout << "Chosen Device: " << devProps.name << std::endl;
		std::cout << "Compute Capability: " << devProps.major << "." << devProps.minor << std::endl;
		std::cout << "Number of Streaming Multiprocessors: " << devProps.multiProcessorCount << std::endl;
		std::cout << "Size of Global Memory: " << devProps.totalGlobalMem/(float)(1024*1024*1024) << " GB" << std::endl;

		int max_threads_per_block = devProps.maxThreadsPerBlock;
		int number_of_SMs = devProps.multiProcessorCount;

		std::cout << "maxThreadsPerBlock: "<< max_threads_per_block << std::endl;
		std::cout << "number_of_SMs: "<< number_of_SMs << std::endl << std::endl;
	}
	
	return deviceCount;
}

void check_CUDA_Error(const char *mensaje)
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


__host__ 
int getSmallDistance(int *visited, int *distance, int nNodes) {
	int smallDistance = -1;
	int mini = INF;
	for (int node_i = 0; node_i < nNodes; node_i++) {
		if (visited[node_i] != 1 && distance[node_i] < mini) {
			mini = distance[node_i];
			smallDistance = node_i;
		}
	}

	return smallDistance;
}

__host__ 
void computeCentralityPath(int source, int tail, List parents[], float* centrality) {

	int parent, i = 0;
	List queue;
	//float incremento = 1.0 / parents[tail].size();

	int paths = parents[tail].size();

	if (tail != source) {
		do{
			for(int parent_i = 0; parent_i < parents[tail].size(); parent_i++){
				parent = parents[tail].at(parent_i);
				if (parent != source) {
					queue.push_back(parent);
					//centrality[parent] += incremento;
				}
			}
			tail = queue.at(i); 
			if(tail != -1 && tail != source )
				paths += (parents[tail].size() - 1);
			i++;
		}while(tail != -1 && tail != source);
	}

	float incremento = 1.0 / paths;
	for(int i = 0; i < queue.size(); i++){
		centrality[queue.at(i)] += incremento;
	}
}
//-----------------------------------------------------------------------------------------------------------

