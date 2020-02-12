#include "Graph.h" 
#include <math.h>
#include <unordered_set>

/**
	Get the grade of a node
*/
int getGrade(
        int node, 
        int *indexs,
        int nNodes, 
        int nEdges)
{
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
        int nEdges)
{
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
    Return an array thar contains the communities labels
*/
int* getCommunities(int *labels, int nNodes){
    std::unordered_set<int> s(labels, labels + nNodes);
    int* communities = new int[s.size()];

    int i = 0;
    for(int community: s){
        communities[i] = community;
        i++;
    }

    return communities;
}

/**
    Return the number of nodes in the first array with label i that have 
    the same labelj in the second array
*/
int compareCommunities(
                    int* trueLabels,
                    int* labelsCalculated,
                    int nNodes,
                    int labeli,
                    int labelj)
{
    int Nij = 0;

    for(int i = 0;i < nNodes; i++){
        if(trueLabels[i] == labeli && labelsCalculated[i] == labelj){
            Nij++;
        }
    }

    return Nij;
}

/**
    Return the sum of a column
*/
int sumColumn(int** matrix, int column, int rows){
     int sum = 0;
     for(int i = 0;i < rows; i++){
        sum += matrix[i][column];
     }
     return sum;
}

/**
    Return the sum of a row
*/
int sumRow(int** matrix, int row, int columns){
    int sum = 0;
    for(int j = 0;j < columns; j++){
        sum += matrix[row][j];
    }
    return sum;
}


/**
    Get the normalized mutual information (NMI) value of the values
*/
float getNMI(int* labelsCalculated, int* trueLabels, int nNodes){
    if(nNodes == 0)
        return 0;

    int cA = countCommunities(trueLabels, nNodes); //Number of real community ---   rows
    int cB = countCommunities(labelsCalculated, nNodes);		//Number of community calculated  columns
    int** N = new int*[cA];
    for(int i = 0; i < cA; ++i)
        N[i] = new int[cB];

    int* realCommunities = getCommunities(trueLabels, nNodes);
    int* estCommunities = getCommunities(trueLabels, nNodes);

    //Create the confusion matrix
    for(int i = 0; i < cA; i++){
    	for(int j = 0; j < cB; j++){
    		N[i][j] = compareCommunities(trueLabels, 
    									 labelsCalculated, 
    									 nNodes, 
    									 realCommunities[i], 
    									 estCommunities[j]);
    	}
    }

    /*cout << "\ncommunities true: " <<  cA << endl;
    cout << "communities calculated: " <<  cB << endl;

    cout << "communities true:";
    for(int c = 0; c < cA; c++){
        cout << realCommunities[c] << ",";
    }
    cout << endl;

    cout << "communities calculated:";
    for(int c = 0; c < cB; c++){
        cout << estCommunities[c] << ",";
    }
    cout << endl;

    cout << "Confusion matrix" << endl;
    for(int i = 0; i < cA; i++){
        for(int j = 0; j < cB; j++){
            cout << N[i][j] << "\t";
        }
        cout << endl;
    }*/

    //denominator
    float denom = 0;
    float Ni, Nj;
    float aux1, aux2, aux3, aux4; 
    for(int i = 0; i < cA; i++){
    	Ni = sumRow(N, i, cB);
    	for(int j = 0; j < cB; j++){
    		Nj = sumColumn(N, j, cA);
            aux1 = N[i][j] * nNodes;
            aux2 = (Ni * Nj);
            aux3 = (aux2 != 0)? aux1 / aux2: 0;
            aux4 = (aux3 != 0)? N[i][j] * log10( aux3 ): 0;
            denom += aux4;
    	}
    }
    denom = (-2) * denom;
    //cout << "\nDenom: " << denom << endl;

    //sumNi
    float sumNi = 0;
    for(int i = 0; i < cA; i++){
    	Ni = sumRow(N, i, cB);
        aux1 = Ni / nNodes;
        aux2 = (aux1 != 0)? Ni * log10(aux1): 0;
    	sumNi += aux2;
    }
    //cout << "sumNi: " << sumNi << endl;

    //sumNj
    float sumNj = 0;
    for(int j = 0; j < cB; j++){
    	Nj = sumColumn(N, j, cA);
        aux1 = Nj / nNodes;
        aux2 = (aux1 != 0)? Nj * log10(aux1): 0;
    	sumNj += aux2;
    }
    //cout << "sumNj: " << sumNj << endl;

    float nmi = (sumNi + sumNj != 0)? denom / (sumNi + sumNj): 0;

    for (int i = 0; i < cA; ++i)
        delete [] N[i];
    delete [] N;

    return nmi;
}