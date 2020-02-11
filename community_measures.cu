#include "Graph.h" 

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
    Get the normalized mutual information (NMI) value of the values
    TODO:
    	Leer archivo para devolver el arreglo trueLabels
    	Implementar getCommunities (obtener los nombres de las comunidades)
    	Implementar compareCommunities (comparar dos arreglos)
		Implementar sumRow y sumColumn
*/
float getNMI(int* labesCalculated, int* trueLabels, int nNodes){
    float denom = 0f;
    float nume = 0f;
    float sumNi = 0f;
    float sumNj = 0f;

    int cA = countCommunities(labesCalculated, nNodes); //Number of real community
    int cB = countCommunities(trueLabels, nNodes);		//Number of community calculated
    int N[cA][cB];

    int* realCommunities = getCommunities(trueLabels, nNodes);
    int* estCommunities = getCommunities(trueLabels, nNodes);

    //Create the confusion matrix
    for(int i = 0; i < cA; i++){
    	for(int j = 0; j < cB; j++){
    		N[i][j] = compareCommunities(trueLabels, 
    									 labesCalculated, 
    									 nNodes, 
    									 realCommunities[i], 
    									 estCommunities[j]);
    	}
    }

    //denominator
    float Ni, Nj;
    for(int i = 0; i < cA; i++){
    	Ni = sumRow(N, i);
    	for(int j = 0; j < cB; j++){
    		Nj = sumColumn(N, j);
    		denom += N[i][j] * log10((N[i][j] * nNodes) / (Ni * Nj));
    	}
    }
    denom = -2 * denom;

    //sumNi
    for(int i = 0; i < cA; i++){
    	Ni = sumRow(N, i);
    	sumNi += (Ni * log10(Ni / nNodes));
    }

    //sumNj
    for(int j = 0; j < cB; j++){
    	Nj = sumColumn(N, j);
    	sumNj += (Nj * log10(Nj / nNodes));
    }

    float nmi = (sumNi + sumNj > 0)? denom / (sumNi + sumNj): 0;

    return nmi;
}