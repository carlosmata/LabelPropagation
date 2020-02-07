#ifndef GRAPH_H
#define GRAPH_H

#include <fstream>
#include <map>
#include <list>
#include <queue>
#include <stack>
#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include "List.h"

#include <set>
#include <functional>

using namespace std;

class Graph
{
    public:
        __host__ Graph(string filename, int type, int sorted);
        __host__ Graph(int numberNodes);
        __host__ Graph(float *costs, int *tails, int *indexs, int nEdges, int nNodes);
        __host__ ~Graph();
		
        //Graph
        __host__ int getNumberNodes();
        __host__ int getNumberEdges();
        __host__ void computeCentrality();
        __host__ void computePartCentrality(int start, int end);
        __host__ void computeCentralityPathFloydWharshall(
                                                List paths[], 
                                                float* centrality, 
                                                int source, int tail, 
                                                int nNodes);
        __host__ void computeCentralityPathFloydWharshall_2(
                                                List paths[], 
                                                float* centrality, 
                                                int i, 
                                                int j, 
                                                int nNodes);
        __host__ void printCentrality();
        __host__ void dijkstra(int source);
        __host__ void dijkstraDevice(int source);
        __host__ void floydWharshall(
                                    float* costs, 
                                    int* tails, 
                                    int* indexs, 
                                    float* centrality,
                                    const int nNodes,
                                    const int nEdges);
        __host__ float* getCosts();
        __host__ int* getTails();
        __host__ int* getIndexs();
        __host__ int* getNodesArray();
		__host__ void setNodeData(float *costs, int *tails, int *indexs, int nEdges, int nNodes);
		//Node 
		__host__ float getCentrality(int source);
        __host__ string getName(int id);
		
        __host__ string removeWhiteSpaces(string s);

        //To centrality
        float *centrality = nullptr;
        map<string, int> nodes;
        int numberNodes;
		int numberEdges;
		
    protected:

    private:
        int inf;

        //CSC (compress sparce column)
        float *edges_cost = nullptr;
        int *edges_tail = nullptr;
        int *indexs = nullptr;

        //Graph
        __host__ bool addEdges(string filename);
        __host__ vector<string> split (string s, string delimiter);
        __host__ bool createFromFile(string filename, int directed, int sorted);
        __host__ bool createFromFileNET(string filename, int sorted);
        __host__ int readNumberOfNodes(string filename);
        __host__ __device__ int getSmallDistance(bool *visited, int *distance);
        __host__ __device__ void computeCentralityPath(int source, int tail, float incremento, List parents[]);


        //Node
        __host__ __device__ float* getEdgesCost(int source);
        __host__ __device__ int* getEdgesEdpoints(int source);
        __host__ __device__ int getNextIndex(int source);
        __host__ void resetCentrality(int source);
        __host__ __device__ void incrementCentrality(int source, float increment);
        
};

#endif // GRAPH_H
