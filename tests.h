#include <iostream>
#include "Graph.h"
#include "List.h"
#include <bits/stdc++.h>
#include <chrono>
#include <cuda_runtime.h>

#include "algorithms.h"
//#include "community_measures.cu"

using namespace std;

//---------------------------------------------Call methods--------------------------------------------

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
void printCommunities(Graph *g, int nNodes, int *labels, string truedata, double time_taken){
	/*float value;
	string name = "";

	cout << endl;
	for (int node_i = 0; node_i < nNodes; node_i++) {
		value = labels[node_i];

		name = g->getName(node_i); 
		cout << "Node: " << node_i
			 << ":" << name
			 << " Community:" 
			 << value << endl;
	}*/

	float modularity = getModularity(g->getTails(), g->getIndexs(), g->getNumberNodes(), g->getNumberEdges(), labels);
	float NMI = -1;
	if(truedata != ""){
		int *realLabels = g->getRealCommunities(truedata);
		/*
		for(int i = 0; i < nNodes; i++){
			cout << i << "\t" << realLabels[i] << endl;
		}

		cout << "communities calculated:"<< endl;
		for(int i = 0; i < nNodes; i++){
			cout << i << "\t" << labels[i] << endl;
		}*/

		if(realLabels != nullptr){
			NMI = getNMI(labels, realLabels, g->getNumberNodes());
		}

		delete[] realLabels;
	}
	
	int nLabels = countCommunities(labels, nNodes);
	int nEdges = g->getNumberEdges();

	cout << nNodes << "\t" 
	     << nEdges << "\t"   
	     << nLabels << "\t" 
	     << modularity << "\t" 
	     << NMI  << "\t"
	     << time_taken << setprecision(5) << endl; 
}

/**
	Compute the betweenness centrality in a parallel way
*/
void centrality_parallel_brandes(Graph *g){
	verifyDeviceCUDA();
	
	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();

	cout << "nNodes:" << nNodes << endl;
	float centralityGraph[nNodes];
	//memset(centralityGraph, 0,  sizeof(float) * nNodes);
	
	for(int i = 0; i < nNodes; i++){
		centralityGraph[i] = 0;
	}

	//Data GPU
	int* tails = g->getTails();
	int* nodes = g->getNodesArray();	 

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

	delete[] nodes;
}
/**
	Compute the beteweenness centrality  in a sequential way
*/
void centrality_sequential_brandes(Graph *g){
	cout << "Secuencial" << endl;
	
	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();
	cout << "nNodes:" << nNodes << endl;
	cout << "nEdges:" << nEdges << endl;
	
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
void label_propagation_sequential(Graph *g, string truedata, int mode){
	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();

	//-----------------------------Begin time to algorithm------------------------------------------------
	auto start = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);
	
	int* labels = nullptr;
	switch(mode){
		case 0://synchronous
			labels = labelPropagationSequential(g->getTails(), g->getIndexs(), nNodes, nEdges, true);
		break;
		case 1://asynchronous
			labels = labelPropagationSequential(g->getTails(), g->getIndexs(), nNodes, nEdges, false);
		break;
		case 2://semi-synchronous
			labels = labelPropagationSemiSynchSeq(g->getTails(), g->getIndexs(), nNodes, nEdges);
		break;
		case 3://asynchronous
			labels = labelPropagationSequential(g->getTails(), g->getIndexs(), nNodes, nEdges, false);
		break;
	}

	auto end = chrono::high_resolution_clock::now();
	double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-9;
	//---------------------------------------------------------------------------------------
	printCommunities(g, nNodes, labels, truedata, time_taken);
	//g->saveCommunitiesinFile("output.groups", labels); 	//print in a file

	delete[] labels;

	//cout << "Secuencial Label propagation time taken by program is : " << fixed
	//	 << time_taken << setprecision(9);
	//cout << " sec" << endl;
}
/**
	Compute the label propagation in a parallel way
*/
void label_propagation_parallel(Graph *g, string truedata, int mode){
	//cout << "LP Parallel" << endl;

	int nNodes = g->getNumberNodes();
	int nEdges = g->getNumberEdges();

	//-----------------------------Begin time to algorithm------------------------------------------------
	auto start = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);
	
	int* labels = nullptr;
	switch(mode){
		case 0: //Synchronous mode
			labels = LPParallelSynchronous(g->getTails(), g->getIndexs(), nNodes, nEdges);
		break;
		case 1: //Asynchronous mode
			labels = LPParallelAsynchronous(g->getTails(), g->getIndexs(), nNodes, nEdges);
		break;
		case 2: //SemySynchronous mode
			labels = LPParallelSemySynchronous(g->getTails(), g->getIndexs(), nNodes, nEdges);
		break;
		case 3: //Asynchronous 2 mode
			labels = LPParallel_V2(g->getTails(), g->getIndexs(), nNodes, nEdges);
		break;
	}

	//cout << "end calculation parallel" << endl;
	auto end = chrono::high_resolution_clock::now();
	double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-9;
	//---------------------------------------------------------------------------------------
	printCommunities(g, nNodes, labels, truedata, time_taken);

	delete[] labels;
	//cout << "Parallel Label propagation time taken by program is : " << fixed
	//	 << time_taken << setprecision(9);
	//cout << " sec" << endl;
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

void testLabelGraph(Graph *g, string truedata, int mode){
	label_propagation_sequential(g, truedata, mode);
	label_propagation_parallel(g, truedata, mode);
}

void testGraph(string filename, int type, int sorted, string truedata, int mode){
	Graph *g = new Graph(filename, type, sorted);
	if(g->getNumberNodes() > 0){
		cout << "Dataset: " << filename << endl;
		//printGraph(g);
		cout << "\t Nodes"<< "\t" 
		<< "Edges" << "\t"  
		<< "Com"   << "\t" 
		<< "Mod"   << "\t" 
		<< "NMI"   << "\t"
		<< "Time"  << endl; 
		testLabelGraph(g, truedata, mode);
	}
	else{
		cout << "Data null in the dataset";
	}
		
	delete[] g->getCosts();
	delete[] g->getTails();
	delete[] g->getIndexs();
	delete g;
}


void testGraphAllModes(string filename, int type, int sorted, string truedata){
	Graph *g = new Graph(filename, type, sorted);
	if(g->getNumberNodes() > 0){
		cout << "Dataset: " << filename << endl;
		//printGraph(g);
		cout << "\t Nodes"<< "\t" 
		<< "Edges" << "\t"  
		<< "Com"   << "\t" 
		<< "Mod"   << "\t" 
		<< "NMI"   << "\t"
		<< "Time"  << endl; 
		testLabelGraph(g, truedata, 3);
		testLabelGraph(g, truedata, 2);
		testLabelGraph(g, truedata, 1);
		testLabelGraph(g, truedata, 0);
	}
	else{
		cout << "Data null in the dataset";
	}
		
	delete[] g->getCosts();
	delete[] g->getTails();
	delete[] g->getIndexs();
	delete g;
}

void testNETFilesAllModes(int sorted){
	int type = 3;
	testGraphAllModes("datasets/converted/karate.net", type, sorted, "");
	testGraphAllModes("datasets/converted/lesmiserables.net", type, sorted, "");
	testGraphAllModes("datasets/converted/football.net", type, sorted, "");
	testGraphAllModes("datasets/converted/4adjnoun.net", type, sorted, "");
	testGraphAllModes("datasets/converted/5powergrid.net", type, sorted, "");
	testGraphAllModes("datasets/converted/4internet.net", type, sorted, "");
}


void testNMIFilesAllModes(int sorted){
	int type = 2;
	testGraphAllModes("datasets/true-data/email/email-Eu-core.txt", type, sorted, 
					  "datasets/true-data/email/email-Eu-core-department-labels.txt");
	testGraphAllModes("datasets/true-data/grass_web/grass_web.pairs", type, sorted, 
					  "datasets/true-data/grass_web/grass_web.labels");
	testGraphAllModes("datasets/true-data/karate/karate_edges_77.txt", type, sorted, 
					  "datasets/true-data/karate/karate_groups.txt");
	testGraphAllModes("datasets/true-data/terrorists/terrorist.pairs", type, sorted, 
					  "datasets/true-data/terrorists/terrorist.groups");
}

void testMediumFilesAllModes(int sorted){
	int type = 2;
	testGraphAllModes("datasets/com-amazon.ungraph.txt", type, sorted, "");
	testGraphAllModes("datasets/com-dblp.ungraph.txt", type, sorted, "");
}

void testBigFilesAllModes(int sorted){
	int type = 2;
	testGraphAllModes("datasets/com-youtube.ungraph.txt", type, sorted, "");
}

void testAllFilesAllModes(int sorted){

	cout << " NET FILES" << endl;
	testNETFilesAllModes(sorted);

	cout << "NMI FILES" << endl;
	testNMIFilesAllModes(sorted);

	cout << "MEDIUM FILES" << endl;
	testMediumFilesAllModes(sorted);

	cout << "BIG FILES" << endl;
	testBigFilesAllModes(sorted);
}

void testNETFiles(int mode, int sorted){

	if(mode == 4){
		testNETFilesAllModes(sorted);
	}
	else{
		int type = 3;
		testGraph("datasets/converted/karate.net", type, sorted, "", mode);
		testGraph("datasets/converted/lesmiserables.net", type, sorted, "", mode);
		testGraph("datasets/converted/football.net", type, sorted, "", mode);
		testGraph("datasets/converted/4adjnoun.net", type, sorted, "", mode);
		testGraph("datasets/converted/5powergrid.net", type, sorted, "", mode);
		testGraph("datasets/converted/4internet.net", type, sorted, "", mode);
	}
}

void testNMIFiles(int mode, int sorted){
	
	if(mode == 4){
		testNMIFilesAllModes(sorted);
	}
	else{
		int type = 2;
		testGraph("datasets/true-data/email/email-Eu-core.txt", type, sorted, 
						  "datasets/true-data/email/email-Eu-core-department-labels.txt", mode);
		testGraph("datasets/true-data/grass_web/grass_web.pairs", type, sorted, 
						  "datasets/true-data/grass_web/grass_web.labels", mode);
		testGraph("datasets/true-data/karate/karate_edges_77.txt", type, sorted, 
						  "datasets/true-data/karate/karate_groups.txt", mode);
		testGraph("datasets/true-data/terrorists/terrorist.pairs", type, sorted, 
					  "datasets/true-data/terrorists/terrorist.groups", mode);
	}
}

void testMediumFiles(int mode, int sorted){
	
	if(mode == 4){
		testMediumFilesAllModes(sorted);
	}
	else{
		int type = 2;
		testGraph("datasets/com-amazon.ungraph.txt", type, sorted, "", mode);
		testGraph("datasets/com-dblp.ungraph.txt", type, sorted, "", mode);
	}
}

void testBigFiles(int sorted, int mode){

	if(mode == 4){
		testBigFilesAllModes(sorted);
	}
	else{
		int type = 2;
		testGraph("datasets/com-youtube.ungraph.txt", type, sorted, "", mode);
	}
}

void testAllFiles(int mode, int sorted){
	if(mode == 4){
		testAllFilesAllModes(sorted);
	}
	else{
		cout << " NET FILES" << endl;
		testNETFiles(sorted, mode);

		cout << "NMI FILES" << endl;
		testNMIFiles(sorted, mode);

		cout << "MEDIUM FILES" << endl;
		testMediumFiles(sorted, mode);

		cout << "BIG FILES" << endl;
		testBigFiles(sorted, mode);
	}
}

