#include <iostream>
#include "Graph.h"
#include "List.h"
#include <bits/stdc++.h>
#include <chrono>
#include <cuda_runtime.h>

#include "algorithms.h"

using namespace std;

//---------------------------------------------Call methods--------------------------------------------
/**
	Get the grade of a node
*/
int getGrade(
			int node, 
			int *indexs,
			int nNodes, 
			int nEdges){
	int index = indexs[node];
	int nextIndex = (node + 1 < nNodes)?indexs[node + 1]:nEdges; 
	int tamLabels = (nextIndex - index < 0)?1 : nextIndex - index; 

	return tamLabels;
}

/**
	Get the value of the Matrix of adjacency 
*/
int getAij(
			int nodei, 
			int nodej,
			int *indexs,
			int *tails,
			int nNodes, 
			int nEdges){
	int neighbor = -1;
	int index = indexs[nodei];
	int nextIndex = (nodei + 1 < nNodes)?indexs[nodei + 1]:nEdges; 

	for(int tail = index; tail < nextIndex; tail++){
		neighbor = tails[tail];//get the neighbor
		if(neighbor == nodej)
			return 1;
	}

	return 0;
}
/**
	Get the modularity of a result of the label propagation algorithm
	in other words its a measure of quality of the algorithm
*/
float getModularity(Graph *g, int *labels){
	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();
	int *edges = g->getTails();
	int *indexs = g->getIndexs();

	int m = nEdges / 2; //Undirected graph
	float sum = 0;
	float delta = 0;
	for(int i = 0; i < nNodes; i++){
		for(int j = 0; j < nNodes; j++){
			delta = (labels[i] == labels[j])?1 : 0;
			if(delta == 1){
				sum += ((getAij(i, j, indexs, edges, nNodes, nEdges) - 
						getGrade(i, indexs, nNodes, nEdges) * getGrade(j, indexs, nNodes, nEdges) / 
						(2.0f * m)
						) /** delta */);
			}
		}
	}

	float modularity = (1.0f / (2.0f* m)) * sum;
	return modularity;
}


/**
	Count the number of communities in the sent labels
*/
int countCommunities(int *labels, int nNodes){
	int totalLabels[nNodes];
	int posLabelN;
	int itLabelN = 0;

	for(int i = 0;i < nNodes; i++){
		totalLabels[i] = -1;
	}

	for(int i = 0;i < nNodes; i++){
		posLabelN = -1;
		//find label
		for(int n = 0; n < nNodes; n++){ //find label
			if(labels[i] == totalLabels[n]){
				posLabelN = n;
				break;
			}
		}
		if(posLabelN == -1){//new label
			totalLabels[itLabelN] = labels[i];
			itLabelN++;
		}
	}

	return itLabelN;
}

/**
	Print the centrality 
*/
void printCentrality(Graph *g, int nNodes, float *centralityGraph, bool directed){
	float value;
	string name = "";


	cout << endl;
	for (int node_i = 0; node_i < nNodes; node_i++) {
		value = (directed)? centralityGraph[node_i]:centralityGraph[node_i] / 2;

		name = g->getName(node_i); 
		cout << "Node: " << node_i
			 << ":" << name
			 << " Centrality:" 
			 << value << endl;
	}
}

/**
	Print the communities computed
*/
void printCommunities(Graph *g, int nNodes, int *communities){
	/*float value;
	string name = "";

	cout << endl;
	for (int node_i = 0; node_i < nNodes; node_i++) {
		value = communities[node_i];

		name = g->getName(node_i); 
		cout << "Node: " << node_i
			 << ":" << name
			 << " Community:" 
			 << value << endl;
	}*/

	cout << "\nModularity: "<< getModularity(g, communities);
	cout << "\nNumber of communities: " << countCommunities(communities, nNodes) << endl;
}

/**
	Compute the betweenness centrality in a parallel way
*/
void centrality_parallel_brandes(Graph *g){
	verifyDeviceCUDA();
	
	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();
	printf("nNodes:%d\n", nNodes);
	float centralityGraph[nNodes];
	//memset(centralityGraph, 0,  sizeof(float) * nNodes);
	
	for(int i = 0; i < nNodes; i++){
		centralityGraph[i] = 0;
	}

	//Data GPU
	//float* costs =  g->getCosts();
	int* tails = g->getTails();
	//int* indexs = g->getIndexs();
	int* nodes = g->getNodesArray();	

	/*for(int i = 0; i < nNodes; i++){
		cout << nodes[i] << "-->" << tails[i] << endl;
	}*/ 

	auto start = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);


	brandesParallel(nodes, tails, centralityGraph, nNodes, nEdges);
	//bc_bfs(nNodes,nEdges, nodes, tails, centralityGraph);
	
	auto end = chrono::high_resolution_clock::now();
	double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-9;
	
	//printCentrality(g, nNodes, centralityGraph, false);

	cout << "Parallel brandes time taken by program is : " << fixed
		 << time_taken << setprecision(9);
	cout << " sec" << endl;

	//delete[] nodes;
}
/**
	Compute the beteweenness centrality  in a sequential way
*/
void centrality_sequential_brandes(Graph *g){
	cout << "Secuencial" << endl;
	
	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();
	printf("nNodes:%d\n", nNodes);
	printf("nEdges:%d\n", nEdges);
	
	//-----------------------------Begin time------------------------------------------------
	auto start = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);

	float* centralityG1 = brandesSequential(g->getCosts(), g->getTails(), g->getIndexs(), nNodes, nEdges);

	auto end = chrono::high_resolution_clock::now();
	double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-9;
	//---------------------------------------------------------------------------------------
	
	//printCentrality(g, nNodes, centralityG1, false);

	delete[] centralityG1;

	cout << "Secuencial Brandes time taken by program is : " << fixed
		 << time_taken << setprecision(9);
	cout << " sec" << endl;
}
/**
	Compute the label propagation in a sequential way
*/
void label_propagation_sequential(Graph *g){
	cout << "Label Propagation Secuential" << endl;

	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();
	printf("nNodes:%d\n", nNodes);
	printf("nEdges:%d\n", nEdges);

	//-----------------------------Begin time to algorithm------------------------------------------------
	auto start = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);
	
	int* labels = labelPropagationSequential(g->getCosts(), g->getTails(), g->getIndexs(), nNodes, nEdges);

	auto end = chrono::high_resolution_clock::now();
	double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-9;
	//---------------------------------------------------------------------------------------
	
	printCommunities(g, nNodes, labels);

	delete[] labels;

	cout << "Secuencial Label propagation time taken by program is : " << fixed
		 << time_taken << setprecision(9);
	cout << " sec" << endl;
}

/**
	Compute the label propagation in a parallel way
*/
void label_propagation_parallel(Graph *g){
	cout << "Label Propagation Parallel" << endl;

	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();
	printf("nNodes:%d\n", nNodes);
	printf("nEdges:%d\n", nEdges);

	//-----------------------------Begin time to algorithm------------------------------------------------
	auto start = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);
	
	int* labels = labelPropagationParallel(g->getCosts(), g->getTails(), g->getIndexs(), nNodes, nEdges);

	auto end = chrono::high_resolution_clock::now();
	double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-9;
	//---------------------------------------------------------------------------------------
	
	printCommunities(g, nNodes, labels);

	delete[] labels;

	cout << "Parallel Label propagation time taken by program is : " << fixed
		 << time_taken << setprecision(9);
	cout << " sec" << endl;
}

/**
	Print the sender graph 
*/
void printGraph(Graph *g){
	int *tails = g->getTails();
	int *indexs = g->getIndexs();
	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();
	int start, end;
	string name;

	/*for(int i = 0; i < nNodes; i++){
		cout<< "[" << indexs[i] << "],";
	}
	cout << endl;*/

	for(int i = 0; i < nNodes; i++){
		name = g->getName(i);
		start = indexs[i];
		end = (i + 1 < nNodes)? indexs[i + 1]: nEdges;
		
		cout<< "[" << i << "]Node: "<< name << endl;
		for(int j = start; j < end; j++){
			cout << "," <<  g->getName(tails[j]) << "("<< tails[j] << ")";
		}
		cout<< endl;

	}
}

//-----------------------------------------------main--------------------------------------------------------------

int main(int argc, char **argv)
{
	string filename = "datasets/karate_test.txt";
	int type = 2; //1-directed, 2-undirected, 3-NET extension
	int sorted = 0;

	if(argc == 2){//Add the filename of the datasets
		filename = argv[1];
	}
	if(argc == 3){//Add the type of the filename
		filename = argv[1];
		type = atoi(argv[2]);
	}
	if(argc == 4){//Add the type of the filename and a sorted way desc-asc
		filename = argv[1];
		type = atoi(argv[2]);
		sorted = atoi(argv[3]);
	}

	Graph *g = new Graph(filename, type, sorted);
	if(g->getNumberNodes() > 0)
	{
		cout << "Dataset: " << filename << endl;
		cout << "Number of Nodes: " << g->getNumberNodes() << endl;
		cout << "Number of Edges: " << g->getNumberEdges() << endl;

		//centrality_sequential_brandes(g);
		//centrality_parallel_brandes(g);
		//printGraph(g);
		//label_propagation_sequential(g);
		label_propagation_parallel(g);
	}
	else
		cout << "Data null in the dataset";
		
	delete[] g->getCosts();
	delete[] g->getTails();
	delete[] g->getIndexs();
	delete g;
}


