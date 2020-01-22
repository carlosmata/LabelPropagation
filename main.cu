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
	Print the centrality 
*/
void printCentrality(Graph *g, int nNodes, float *centralityGraph, bool directed){
	float value;
	string name = "";

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
void printCommunities(Graph *g, int nNodes, int *communities, bool directed){
	float value;
	string name = "";

	for (int node_i = 0; node_i < nNodes; node_i++) {
		value = (directed)? communities[node_i]:communities[node_i] / 2;

		name = g->getName(node_i); 
		cout << "Node: " << node_i
			 << ":" << name
			 << " Community:" 
			 << value << endl;
	}
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
	cout << "Label Propagation Secuencial" << endl;

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
	
	printCommunities(g, nNodes, labels, false);

	delete[] labels;

	cout << "Secuencial Label propagation time taken by program is : " << fixed
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
    string filename = "datasets/paper/com-amazon.ungraph.txt";
    int type = 3;

    if(argc == 2){
    	filename = argv[1];
    }
    if(argc == 3){
    	filename = argv[1];
    	type = atoi(argv[2]);
    }

    Graph *g = new Graph(filename, type);
	
    if(g->getNumberNodes() > 0)
    {
    	cout << "Dataset: " << filename << endl;
		cout << "Number of Nodes: " << g->getNumberNodes() << endl;
		cout << "Number of Edges: " << g->getNumberEdges() << endl;

		//centrality_sequential_brandes(g);
		//centrality_parallel_brandes(g);
    }
    else
        cout << "Data null in the dataset";
		
	delete[] g->getCosts();
    delete[] g->getTails();
    delete[] g->getIndexs();
    delete g;
}


