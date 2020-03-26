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
        Graph(string filename, int type, int sorted);
        ~Graph();

        //Graph
        int getNumberNodes();
        int getNumberEdges();
        float* getCosts();
        int* getTails();
        int* getIndexs();
        int* getNodesArray();
        void setNodeData(float *costs, int *tails, int *indexs, int nEdges, int nNodes);
        float* getEdgesCost(int source);
        int* getEdgesEdpoints(int source);

		//Node 
        string getName(int id);
        int getId(string name);

        string removeWhiteSpaces(string s);
        int* getRealCommunities(string truedata);

        bool saveCommunitiesinFile(string filename, int* labels);
        int* getCommunities(int *labels, int* numCommunities);
        //int* renameLabels(int* labels);

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
        bool addEdges(string filename);
        vector<string> split (string s, string delimiter);
        bool createFromFile(string filename, int directed, int sorted);
        bool createFromFileNET(string filename, int sorted);
        
};

#endif // GRAPH_H
