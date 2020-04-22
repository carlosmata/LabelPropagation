#include "reduce.h"
#include <stdlib.h>
#include <bits/stdc++.h>

using namespace std;
//-----------------------------------------------main-------------------------------------------------------------

int* segmentedReduceV2(
					int *d_segments, 
					int* d_data, 
					int size_data, 
					int n_segments, 
					unsigned int seed)
{
	int *s_maxs = new int[n_segments];
	int *s_i = new int[n_segments];
	int* d_s_maxs; 	//Out N = n_segments
	int* d_s_i;	   	//Out N = n_segments
	
	cudaMalloc(&d_s_maxs, n_segments * sizeof(int));
	cudaMalloc(&d_s_i, n_segments * sizeof(int));

	segmented_reduce(d_segments, d_data, d_s_maxs, d_s_i, size_data, n_segments, seed);
	cudaMemcpy(s_maxs, d_s_maxs, n_segments * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(s_i, d_s_i, n_segments * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nmaxs\n");
	printarrayCPU(s_maxs, n_segments);
	printf("\nmaxs_i\n");
	printarrayCPU(s_i, n_segments);

	delete[] s_maxs;
	//delete[] s_i;
	cudaFree(d_s_maxs);
	cudaFree(d_s_i);

	return s_i;
}

int* segmentedReduceV1(
					int* d_segments, 
					int* d_data, 
					int size_data, 
					int n_segments, 
					unsigned int seed)
{
	int *s_i = new int[n_segments];
	int* d_s_i;	   	//Out N = n_segments

	int nTPB = 256;
	int nBNodes = n_segments / nTPB + ((n_segments % nTPB == 0)? 0 : 1); 
	int nBlocksNodes = (4096 < nBNodes)? 4096:nBNodes;	

	cudaMalloc(&d_s_i, n_segments * sizeof(int));

	lp_reduce<<<nBlocksNodes, nTPB>>>(d_segments, d_data, d_s_i, n_segments, size_data,seed);
	cudaMemcpy(s_i, d_s_i, n_segments * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nmaxs_i\n");
	printarrayCPU(s_i, n_segments);

	cudaFree(d_s_i);

	return s_i;
}

int compare(int *res1, int *res2, int* data, int size_data, int tam){
	for(int i=0; i<tam; i++){
		if(data[res1[i]] != data[res2[i]]){
			printf("\nsegment: %d\n", i);
			printf("indexs %d:%d\n", res1[i], res2[i]);
			printf("values %d:%d\n",data[res1[i]],data[res2[i]]);

			printf("values:\n");
			int pos = i * (size_data/tam);
			for (int j = 0; j < (size_data/tam); j++)
			{ 
				printf("data[%d]:%d \n",  pos + j, data[pos + j]);
			}


			printf("It's different\n");
			return 0;		
		}
	}

	printf("It's OK\n");
	return 1;
}


int main(int argc, char **argv)
{	
	int* d_segments;//In 
	int* d_data;	//In
	int seed = time(NULL); //In
	int size_data;	//In
	int n_segments; //In

	size_data = 1000000;
	n_segments = 100;

	if(argc > 2){
		size_data = atoi(argv[1]);
		n_segments = atoi(argv[2]);
	}

	cudaMalloc(&d_segments, n_segments * sizeof(int));
	cudaMalloc(&d_data, size_data * sizeof(int));
	initarrays<<<1,1>>>(d_segments, d_data, size_data, n_segments, seed);
	/////
	int *data = new int[size_data];
	int *res1 = segmentedReduceV2(d_segments, d_data, size_data, n_segments, seed);
	int *res2 = segmentedReduceV1(d_segments, d_data, size_data, n_segments, seed);

	cudaMemcpy(data, d_data, size_data * sizeof(int), cudaMemcpyDeviceToHost);
	compare(res1, res2, data, size_data, n_segments);
	/////

	cudaFree(d_segments);
	cudaFree(d_data);

	delete[] res1;
	delete[] res2;
	delete[] data;
	return 0;
}