#include "Graph.h"
//-----------------------------Graph-----------------------------------------------
__host__
Graph::Graph(string filename, int type, int sorted) {
	this->inf = 2147483647; 

	if(type == 1){ //txt directed
		this->createFromFile(filename, 1/*directed*/, sorted);
	}
	else if(type == 2){ //txt undirected 
		this->createFromFile(filename, 0/*undirected*/, sorted);
	}
	else if(type == 3){ //net extension 
		this->createFromFileNET(filename, sorted);
	}
	else{ //txt #FromNodeId ToNodeId
		int n = this->readNumberOfNodes(filename);

		//Allocate
	    this->indexs = new int[n]; 
	    this->centrality = new float[n];
		
		//Create the nodes
		this->numberNodes = n;
		for (int i = 0; i < numberNodes; i++) {
			this->indexs[i] = -1;
			this->centrality[i] = 0.0;
		}
		//create the nodes
		this->addEdges(filename);
	}


}

__host__
Graph::Graph(int numberNodes) {
	this->inf = 2147483647;
	this->numberNodes = numberNodes;
	
    this->indexs = new int[numberNodes];
    this->centrality = new float[numberNodes];
    
	for (int i = 0; i < numberNodes; i++) {
		this->indexs[i] = -1; 
		this->centrality[i] = 0.0; 
	}
}

__host__
Graph::Graph(float* costs, int* tails, int* indexs, int nEdges, int nNodes) {
	this->inf = 2147483647;
	this->numberNodes = nNodes;
	this->numberEdges = nEdges;

    this->centrality = new float[nNodes];
    
	for (int i = 0; i < nNodes; i++) {
		this->centrality[i] = 0.0; 
	}

	this->setNodeData(costs, tails, indexs, nEdges, nNodes);
}

__host__ 
Graph::~Graph() {
	//delete[] edges_cost;
    //delete[] edges_tail;
    //delete[] indexs;
    delete[] centrality;
}
/**
	Get the number of nodes
*/
__host__
int Graph::getNumberNodes() {
	return this->numberNodes;
}
/**
	Set the data's nodes 
*/
__host__
void Graph::setNodeData(float* costs, int* tails, int* indexs, int nEdges, int nNodes){    
    //this->edges_cost = new float[nEdges]; 
    //this->edges_tail = new int[nEdges]; 
    //this->indexs = new int[nNodes]; 
    
    //for(int i = 0; i < nEdges; i++){
    //	this->edges_cost[i] = costs[i];
    //	this->edges_tail[i] = tails[i];
    //}
    
    //for(int i = 0; i < nNodes; i++){
    //	this->indexs[i] = indexs[i];
    //}
    this->edges_cost = costs; 
    this->edges_tail = tails;
    this->indexs = indexs; 
    printf("Assignated data Graph\n");
}
/*
	Get the array of costs of the nodes
*/
__host__
float* Graph::getCosts(){
	return this->edges_cost;
}
/*
	Get the array of tails nodes
*/
__host__
int* Graph::getTails(){
	return this->edges_tail;
}
/*
	Get the indexs of the nodes
*/
__host__
int* Graph::getIndexs(){
	return this->indexs;
}

/*
	Get the array of graph's nodes
*/
__host__ int* Graph::getNodesArray(){
	int *v = new int[this->numberEdges];
	int start, end;

	for(int i = 0; i < this->numberNodes; i++){
		start = this->indexs[i];
		end = (i < this->numberNodes - 1)? this->indexs[i + 1] : this->numberEdges;

		for(int j = start; j < end; j++){
			v[j] = i;
		}
	}

	return v;
}
/*
	Get the number of edges to the graph
*/
__host__
int Graph::getNumberEdges(){
	return this->numberEdges;
}

/**
	--FILE--
	Add all the edges from a dataset
*/
__host__
bool Graph::addEdges(string filename) {
	string line, number = "";
	ifstream dataset;
	size_t found;
	string stringFind = "FromNodeId";

	int counter = 1, nodei, nodej;

	dataset.open(filename);
	
	//Datas
	vector<float> costs;
	vector<int> tails;
	
	//
	vector<int> edges;

	//Iterators
	vector<int>::iterator it_edges;
	vector<int>::iterator it_endpoints;
	vector<float>::iterator it_costs;

	if (dataset.is_open()) {
		while (getline(dataset, line)) {
			if (line.length() > 0) {
				if (line.at(0) == '#') {
					found = line.find(stringFind);
					if (found != string::npos) { //The word is inside of the line
						//Add edges
						while (getline(dataset, line)) {
							//read the nodei and the nodej
							counter = 1;
							nodei = nodej = this->numberNodes;

							for (unsigned int i = 0; i < line.length(); i++) {
								if (isdigit(line[i])) {
									number = "";
									while (isdigit(line[i]) && i < line.length()) {
										number += line[i];
										i++;
									}
								}
								if (counter == 1)
									nodei = stoi(number);
								else if (counter == 2)
									nodej = stoi(number);
								else
									break;

								counter++;
							}

							//Add the edge
							if (nodei < this->numberNodes && nodej < this->numberNodes) {
								//Insertar ordenado las aristas
								if (edges.size() == 0 || nodei > edges.back()) {
									edges.push_back(nodei);
									tails.push_back(nodej);
									costs.push_back(1);
								}
								else {
									for (unsigned int i = 0; i < edges.size(); i++) {
										if (nodei <= edges[i]) {
											it_edges = edges.begin() + i;
											it_endpoints = tails.begin() + i;
											it_costs = costs.begin() + i;

											edges.insert(it_edges, nodei);
											tails.insert(it_endpoints, nodej);
											costs.insert(it_costs, 1);
											break;
										}
									}
								}
							}
							//////
						}
					}
				}
			}
		}
		dataset.close();
	}
	else {
		return false;
	}
	//Calcular los indices
	if (edges.size() > 0) {
		int nodo = edges[0];
		this->indexs[nodo] = 0;
		for (unsigned int i = 0; i < edges.size(); i++) {
			if (edges[i] != nodo) {
				nodo = edges[i];
				this->indexs[nodo] = i;
			}
		}
		
		this->edges_cost = new float[costs.size()]; // costs.data();
		this->edges_tail = new int[tails.size()]; //tails.data();
		
		for(int i = 0; i < edges.size(); i++){
			this->edges_cost[i] = costs[i];
			this->edges_tail[i] = tails[i];
		}
		
		this->numberEdges = edges.size();
	}


	return true;
}

// for string delimiter
__host__
vector<string> Graph::split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

string Graph::removeWhiteSpaces(string s){
	s.erase(remove(s.begin(), s.end(), ' '), s.end());
	s.erase(remove(s.begin(), s.end(), '\n'), s.end());
	s.erase(remove(s.begin(), s.end(), '\t'), s.end());
	s.erase(remove(s.begin(), s.end(), '\v'), s.end());
	s.erase(remove(s.begin(), s.end(), '\r'), s.end());

	return s;
}

/**
	--FILE--
	Add all the edges from a dataset
*/
__host__
bool Graph::createFromFile(string filename, int directed, int sorted) {
	string line;
	
	//Iterators
	map<string, vector<string>> mp;
	vector<string> words;
	string delimiter = "\t";
	string nodei;
	string nodej;

	ifstream dataset;
	dataset.open(filename);

	//Read the datafile and create the nodes in the same time with the edges
	if (dataset.is_open()) {
		while (getline(dataset, line)) {
			if (line.at(0) != '#') {
				words = split(line, delimiter);

				nodei = words[0];
				nodej = words[1];

				nodei = this->removeWhiteSpaces(nodei);
				nodej = this->removeWhiteSpaces(nodej);

				if(mp.find(nodei) == mp.end()){ //The nodei not exist in the map then create it
					vector<string> edges;
					mp.insert({nodei, edges});
				}
				if(mp.find(nodej) == mp.end()){ //The nodej not exist in the map then create it
					vector<string> edges;
					mp.insert({nodej, edges});
				}

				mp.find(nodei)->second.push_back(nodej); //Add the edge nodei->nodej
				if(directed == 0)						 //If is undiredcted create the edge nodej->nodei
					mp.find(nodej)->second.push_back(nodei);
			}
		}
		dataset.close();
	}
	else {
		return false;
	}
	//-----------------File readed-------------------

	//-----------------------------------------------
	int index = 0;
	vector<string> edges;
	int i = 0;
	int *indexs_aux = new int[mp.size()]; 

	//Create the nodes with the name and the iteration in the map
	//then create a vector with all the edges
    if(sorted == 1){
    	//Preposesing to the parallel algorithms: Sort the nodes depending of his grade
		// Declaring the type of Predicate that accepts 2 pairs and return a bool
		typedef std::function<bool(std::pair<string, vector<string>>, 
								   std::pair<string, vector<string>>)> Comparator;

		// Defining a lambda function to compare two pairs. It will compare two pairs using second field
		Comparator compFunctor =
				[](std::pair<string, vector<string>> elem1,
				   std::pair<string, vector<string>> elem2)
				{
					return elem1.second.size() < elem2.second.size();
				};

		// Declaring a set that will store the pairs using above comparision logic
		std::multiset<std::pair<string, vector<string>>, Comparator> sortedNodes(
				mp.begin(), mp.end(), compFunctor);

	    for (auto itr : sortedNodes){
	    	indexs_aux[i] = index;
	    	this->nodes.insert({itr.first, i});
	        i++;
	        index = index + itr.second.size();
	        //cout << itr.second.size() << endl;
	        for(int j = 0;j < itr.second.size(); j++){
	        	edges.push_back(itr.second[j]);
	        }
	    }
	}
	else{
		for (auto itr = mp.begin(); itr != mp.end(); ++itr) { 
	        indexs_aux[i] = index;  	
	        this->nodes.insert({itr->first, i});
	        i++;
	        index = index + itr->second.size();
	        for(int j = 0;j < itr->second.size(); j++){
	        	edges.push_back(itr->second[j]);
	        }
    	}
	}


    this->numberEdges = edges.size();
    this->edges_cost = new float[edges.size()];
    this->edges_tail = new int[edges.size()];

    //Create the edges array tails and costs 
    for(int edge_i = 0; edge_i < edges.size(); edge_i++){
    	this->edges_cost[edge_i] = 1.0;
    	if(this->nodes.find(edges[edge_i]) == this->nodes.end()){//The node doesn't exist
    		this->nodes.insert({edges[edge_i], i});
        	this->edges_tail[edge_i] = i;
        	i++;
    	}
    	else{
    		this->edges_tail[edge_i] = this->nodes.find(edges[edge_i])->second;
    	}
    }

    //Create the indexs array
    this->indexs = new int[this->nodes.size()];
    this->centrality = new float[this->nodes.size()];
    this->numberNodes = this->nodes.size();
    for(int node_i = 0; node_i < this->nodes.size(); node_i++){
    	this->centrality[node_i] = 0.0;
    	if(node_i < mp.size()){
    		this->indexs[node_i] = indexs_aux[node_i];
    	}
    	else{
    		this->indexs[node_i] = this->numberEdges;
    	}
    }


    delete[] indexs_aux;
	return true;
}

/**
	--FILE--
	Add all the edges from a dataset
*/
__host__
bool Graph::createFromFileNET(string filename, int sorted) {
	string line;

	//Iterators
	map<string, map<string, string>> mp;
	vector<string> words;
	string delimiter = " ";
	string nodei;
	string nodej;
	string cost;
	string stringArcs = "*Arcs";
	string stringEdges = "*Edges";

	size_t found1, found2;
	ifstream dataset;
	dataset.open(filename);

	if (dataset.is_open()) {
		while (getline(dataset, line)){
			found1 = line.find(stringArcs);
			found2 = line.find(stringEdges);

			if(found1 != string::npos || found2 != string::npos){
				while (getline(dataset, line)) {
					//cout <<line <<endl;
					words = split(line, delimiter);
					nodei = words[0];
					nodej = words[1];
					cost = words[2];

					if(mp.find(nodei) == mp.end()){ //The nodei not exist
						map<string, string> edges;
						mp.insert({nodei, edges});
					}
					if(mp.find(nodej) == mp.end()){ //The nodej not exist
						map<string, string> edges;
						mp.insert({nodej, edges});
					}

					mp.find(nodei)->second.insert({nodej, cost});//push_back(nodej);
					mp.find(nodej)->second.insert({nodei, cost});//push_back(nodej);
				}
			}
		}


		dataset.close();
	}
	else {
		return false;
	}
	//-----------------File readed-------------------
	//-----------------------------------------------

	int index = 0;
	vector<string> edges;
	vector<string> costs;
	int i = 0;
	int *indexs_aux = new int[mp.size()]; 

    if(sorted == 1){
    		//Preposesing to the parallel algorithms: Sort the nodes depending of his grade
		// Declaring the type of Predicate that accepts 2 pairs and return a bool
		typedef std::function<bool(std::pair<string, map<string, string>>, 
								   std::pair<string, map<string, string>>)> Comparator;

		// Defining a lambda function to compare two pairs. It will compare two pairs using second field
		Comparator compFunctor =
				[](std::pair<string, map<string, string>> elem1,
				   std::pair<string, map<string, string>> elem2)
				{
					return elem1.second.size() < elem2.second.size();
				};

		// Declaring a set that will store the pairs using above comparision logic
		std::multiset<std::pair<string, map<string, string>>, Comparator> sortedNodes(
				mp.begin(), mp.end(), compFunctor);

		//cout<<sortedNodes.size()<<endl;
	    for (auto itr : sortedNodes){
	    	indexs_aux[i] = index;  	
	        this->nodes.insert({itr.first, i});
	        i++;
	        index = index + itr.second.size();
	        //cout<<itr.second.size()<<endl;
	        for (auto itedge = itr.second.begin(); itedge != itr.second.end(); ++itedge) { 
	        	edges.push_back(itedge->first);
	        	costs.push_back(itedge->second);
	        }
	    }
    }
    else{
    	for (auto itr = mp.begin(); itr != mp.end(); ++itr) { 
	        indexs_aux[i] = index;  	
	        this->nodes.insert({itr->first, i});
	        i++;
	        index = index + itr->second.size();
	        //cout<<itr->second.size()<<endl;
	        for (auto itedge = itr->second.begin(); itedge != itr->second.end(); ++itedge) { 
	        	edges.push_back(itedge->first);
	        	costs.push_back(itedge->second);
	        }
    	}
    }


    this->numberEdges = edges.size();
    this->edges_cost = new float[costs.size()];
    this->edges_tail = new int[edges.size()];

    for(int cost_i = 0; cost_i < costs.size(); cost_i++){
    	this->edges_cost[cost_i] = stof(costs[cost_i]);
    }


    //agregar nuevos 
    for(int edge_i = 0; edge_i < edges.size(); edge_i++){
    	if(this->nodes.find(edges[edge_i]) == this->nodes.end()){
    		this->nodes.insert({edges[edge_i], i});
        	this->edges_tail[edge_i] = i;
        	i++;
    	}
    	else{
    		this->edges_tail[edge_i] = this->nodes.find(edges[edge_i])->second;
    	}
    }

    // agregar lo nuevo nuevo
    this->indexs = new int[this->nodes.size()];
    this->centrality = new float[this->nodes.size()];
    this->numberNodes = this->nodes.size();
    for(int node_i = 0; node_i < this->nodes.size(); node_i++){
    	this->centrality[node_i] = 0.0;
    	if(node_i < mp.size()){
    		this->indexs[node_i] = indexs_aux[node_i];
    	}
    	else{
    		this->indexs[node_i] = this->numberEdges;
    	}
    }


    delete[] indexs_aux;
	return true;
}

__host__ 
string Graph::getName(int id){
	for (auto itr = this->nodes.begin(); itr != this->nodes.end(); ++itr) {
		if(itr->second == id){
			return itr->first;
		}
	}
	string ret = "null";

	return ret;
}

/**
	--FILE--
	Get the numbers of nodes of the data set
*/
__host__
int Graph::readNumberOfNodes(string filename) {
	int numbersOfNode = -1;
	string line, stringFind, number = "";
	ifstream dataset;
	size_t found;
	stringFind = "Nodes:";

	dataset.open(filename);
	if (dataset.is_open()) {
		while (getline(dataset, line)) {
			if (line.length() > 0) {
				if (line.at(0) == '#') {
					found = line.find(stringFind);
					if (found != string::npos) { //The word is inside of the line
						//Find the number
						for (unsigned int i = found + stringFind.length(); i < line.length(); i++) {
							if (isdigit(line[i])) {
								while (isdigit(line[i]) && i < line.length()) {
									number += line[i];
									i++;
								}
								break;
							}
						}
						//Convert the number to integer
						numbersOfNode = stoi(number);
						break;
					}
				}
			}
		}
		dataset.close();
	}
	else {
		cout << "Unable to open the data set";
		return -1;
	}

	return numbersOfNode;
}

/**
	Compute the centrality of all nodes in the Graph
*/
__host__
void Graph::computePartCentrality(int start, int end) {
    //Compute the new centrality
	for (int nodo_i = start; nodo_i < end; nodo_i++) {
		this->dijkstraDevice(nodo_i);
	}
}


/**
	Compute the centrality of all nodes in the Graph
*/
__host__
void Graph::computeCentrality() {
    //Compute the new centrality
	for (int nodo_i = 0; nodo_i < this->numberNodes; nodo_i++) {
		//cout << "Nodo:" << nodo_i << endl;
		this->dijkstra(nodo_i);
	}
}

/**
	Compute the Dijkstra algorithm to find the shortest path from the node source to
*/
__host__
void Graph::dijkstra(int source) {
	int cost = 0, totalcost = 0;
	int node = 0, endpoint = 0;
	int *endpoints, length_e; 
	float *costs;
	
	bool *visited = new bool[this->numberNodes]; 
	int *distance = new int[this->numberNodes];  
	List *parents = new List[this->numberNodes];
	
	for(int i = 0; i < this->numberNodes; i++){
		visited[i] = false;
		distance[i] = this->inf;
	}
	distance[source] = 0;
	
	
	//printf("Dijkstra: %d\n",source);
 	//Iterate in the node
	while ((node = this->getSmallDistance(visited, distance)) != -1) {
		endpoints = this->getEdgesEdpoints(node);
		costs = this->getEdgesCost(node);
		
		if(endpoints != nullptr && costs != nullptr){
			length_e = endpoints[0];// length_c = costs[0];
			
			//printf("source:%d, node:%d, length_e:%d\n", source, node, length_e);
			for (int i = 1; i < length_e + 1; i++) { //regresar el tamaño de los endpoints
				endpoint = endpoints[i];
				
				//printf("for[%d]: source:%d, node:%d\n",i , source, node);
				//printf("endpoint:%d\n", endpoint);
				if (endpoint != -1 && !visited[endpoint]) {
					
					cost = costs[i];
					totalcost = cost + distance[node];
					if (totalcost < distance[endpoint]) { //Add only one path
						distance[endpoint] = totalcost;
						//printf("[EnterIF] for[%d]: source:%d, node:%d\n",i , source, node);
						parents[endpoint].clear();
						parents[endpoint].push_back(node);
					}
					else if (totalcost == distance[endpoint]) { //Add other shortest path
						parents[endpoint].push_back(node);
					}
				}
			}
			delete[] endpoints;
			delete[] costs;
		}
		
		//printf("visited source:%d, node:%d\n", source, node);
		visited[node]= true;
	}
	
	
	float incremento;
	for (int nodo_j = 0; nodo_j < this->numberNodes; nodo_j++) {
		incremento = 1.0 / parents[nodo_j].size(); //Tal vez el incremento se pueda recalcular
		this->computeCentralityPath(source, nodo_j, incremento, parents);
	}
	
	delete[] visited;
	delete[] distance;
	delete[] parents;
}


/**
	Compute the Dijkstra algorithm to find the shortest path from the node source to
*/
__host__
void Graph::dijkstraDevice(int source) {
	int cost = 0, totalcost = 0;
	int node = 0, endpoint = 0;
	int *endpoints, length_e; 
	float *costs;
	
	bool *visited = new bool[this->numberNodes]; 
	int *distance = new int[this->numberNodes];  
	List *parents = new List[this->numberNodes];
	
	
	for(int i = 0; i < this->numberNodes; i++){
		visited[i] = false;
		distance[i] = this->inf;
	}
	distance[source] = 0;
	
 	//Iterate in the node
	while ((node = this->getSmallDistance(visited, distance)) != -1) {
		endpoints = this->getEdgesEdpoints(node);
		costs = this->getEdgesCost(node);
		
		if(endpoints != nullptr && costs != nullptr){
			length_e = endpoints[0];// length_c = costs[0];
			
			for (int i = 1; i < length_e + 1; i++) { //regresar el tamaño de los endpoints
				endpoint = endpoints[i];
				
				if (endpoint != -1 && !visited[endpoint]) {
					
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
			delete[] endpoints;
			delete[] costs;
		}
		
		visited[node]= true;
	}
	
	
	float incremento;
	for (int nodo_j = 0; nodo_j < this->numberNodes; nodo_j++) {
		incremento = 1.0 / parents[nodo_j].size(); //Tal vez el incremento se pueda recalcular
		this->computeCentralityPath(source, nodo_j, incremento, parents);
	}
	
	delete[] visited;
	delete[] distance;
	delete[] parents;
}


__host__
void Graph::floydWharshall(
				float* costs, 
				int* tails, 
				int* indexs, 
				float* centrality,
				const int nNodes,
				const int nEdges)
{
	int distance[nNodes * nNodes];
	List paths[nNodes * nNodes];
	int index, nextIndex, endpoint;
	int infinity = 1000000000;

	//Crear matrices
	//--------------------Paths------------------------
	int pos = 0;
	for(int i = 0; i < nNodes; i++){
		//Paths
		for(int j = 0; j < nNodes; j++){
			pos = (i * nNodes) + j;
			paths[pos].push_back(j);
			distance[pos] = (i == j)? 0 : infinity;
		}
		//Distancias
		index = indexs[i];
		nextIndex = getNextIndex(i);
		if(index != -1){
			for (int n_i = index; n_i < nextIndex ; n_i++) { //regresar el tamaño de los endpoints
				endpoint = tails[n_i];
				if (endpoint != -1) {
					pos = (i * nNodes) + endpoint;
					distance[pos] = costs[n_i];
				}
			}

		}
	}

	int nodei, nodej, nodek;
	int distAux;
	bool op1, op2;
	//Calcular Shortest paths
	//-------------------------------------------------
	for(int i = 0; i < nNodes; i++){
		for(int j = 0; j < nNodes; j++){
			if(i != j){
				nodei = ( nNodes * j ) + i; //se mueve por las columnas
				op1 = distance[nodei] != infinity;

				if(distance[nodei] != 0 && op1){
					for(int k = 0; k < nNodes; k++){
						nodej = ( nNodes * i ) + k;//se mueve por los renglones
						nodek = ( nNodes * j ) + k;
						
						op2 = distance[nodej] != infinity;					

						if( distance[nodej] != 0 && op2 ) 
						{
							/*cout<< " nodei[" << i << "]=" << distance[nodei]
							    << ", nodej[" << k << "]=" << distance[nodej]
							    << ", nodek[" << j << "]=" << distance[nodek];*/
							distAux = distance[nodei] + distance[nodej];

							if(distAux < distance[nodek]){
								//cout << " menor: " << distAux;
								distance[nodek] = distAux;
								paths[nodek].clear();
								paths[nodek].push_back(i); //Listas
								//cout << ", se inicializa:" << i;
							}
							else if(distAux == distance[nodek]){
								//cout << " igual: " << distAux;
								paths[nodek].push_back(i);
								//cout << ", se agrega:" << i;
							}

							//cout << endl;
						}
					}
				}
			}
		}
	}
	//cout<<"a calcular centralidad"<<endl;
	//------------------------Calcular paths-------------------------------
	for (int i = 0; i < nNodes; i++) {
		for(int j = 0; j < nNodes; j++){
			if(i != j){
				computeCentralityPathFloydWharshall(paths, centrality, i, j, nNodes);
			}
		}
	}

}

__host__
void Graph::computeCentralityPathFloydWharshall(List paths[], float* centrality, int source, int tail, int nNodes){
	int pos = (source * nNodes) + tail;
	float incremento = 1.0 / paths[pos].size();
	List queue;
	int i = 0, node;

	do{
		pos = (source * nNodes) + tail;
		for(int p_i = 0; p_i < paths[pos].size(); p_i++){
			node = paths[pos].at(p_i);
			if (node != tail) {
				queue.push_back(node);
				centrality[node] += incremento;
			}
		}
		source = queue.at(i);
		i++;
	}while(source != -1 && source != tail && source < nNodes);
}

__host__
void Graph::computeCentralityPathFloydWharshall_2(List paths[], float* centrality, int i, int j, int nNodes){
	int pos = (i * nNodes) + j;
	float incremento = 1.0 / paths[pos].size();
	int node;

	for(int p_i = 0; p_i < paths[pos].size(); p_i++){
		node = paths[pos].at(p_i);
		if(node != i && node != j)
			centrality[node] += incremento;
	}
}


/**
   Compute the centrality of a path from a source and a tail
*/
__host__ 
void Graph::computeCentralityPath(int source, int tail, float incremento, List parents[]) {
	int parent, i = 0;
	List *queue = new List();

	if (tail != source) {
		do{
			for(int parent_i = 0; parent_i < parents[tail].size(); parent_i++){
				parent = parents[tail].at(parent_i);
				if (parent != source) {
					queue->push_back(parent);
					this->centrality[parent] += incremento;
				}
			}
			tail = queue->at(i);
			i++;
		}while(tail != -1 && tail != source);
	}

	delete queue;
}

/**
	Get the node's small distance in the graph
*/
__host__ 
int Graph::getSmallDistance(bool *visited, int *distance) {
	int smallDistance = -1;
	int mini = this->inf;
	for (int node_i = 0; node_i < this->numberNodes; node_i++) {
		if (!visited[node_i] && distance[node_i] < mini) {
			mini = distance[node_i];
			smallDistance = node_i;
		}
	}

	return smallDistance;
}
/**
   Print the centrality of the nodes in the graph
*/
__host__
void Graph::printCentrality() {
	for (int node_i = 0; node_i < this->numberNodes; node_i++) {
		cout << "Node: " << node_i
			<< " Centrality:" << this->centrality[node_i] / 2 << endl;
	}
}

//------------------------------------------------------------------------------------------

//-----------------------Node---------------------------------------------------------------


/**
	reset the centrality
*/
__host__
void Graph::resetCentrality(int source) {
	this->centrality[source] = 0.0;
}
/**
	Increment the centrality 1 by 1
*/
__host__ 
void Graph::incrementCentrality(int source, float increment) {
    this->centrality[source] += increment;
}
/**
	Get the centrality of the node
*/
__host__
float Graph::getCentrality(int source) {
	return this->centrality[source];
}
/**
	Get the edges cost of the node
*/
__host__ 
float* Graph::getEdgesCost(int source) {
	int index = this->indexs[source]; //Get the node source
	int nextIndex = this->getNextIndex(source);
	float* cost = nullptr;
	int cont = 1;
	int lenght;
	
	if (index != -1) {
		lenght = nextIndex - index;
		cost = new float[lenght + 1];
		cost[0] = lenght;
		for (int i = index; i < nextIndex; i++) {
			cost[cont] = this->edges_cost[i];
			cont++;
		}
	}

	return cost;
}
/**
	Get the edges tail of the node
*/
__host__ 
int* Graph::getEdgesEdpoints(int source) {
	int index = this->indexs[source]; //Get the node source
	int nextIndex = this->getNextIndex(source);
	int *endpoints = nullptr;
	int cont = 1;
	int lenght;
	
	if (index != -1) {
		lenght = nextIndex - index;
		endpoints = new int[lenght + 1];
		endpoints[0] = lenght;
		for (int i = index; i < nextIndex; i++) {
			//printf("edges_tail[%d]=%d\n", i, this->edges_tail[i]);
			endpoints[cont] = this->edges_tail[i];
			cont++;
		}
	}

	return endpoints;
}
/**
	Get the next index to obtain the edges of a node
*/
__host__ 
int Graph::getNextIndex(int source) {
	int edges_size = this->numberEdges; 
	
	for (int i = source + 1; i < this->numberNodes; i++) {
		if (this->indexs[i] > this->indexs[source] &&
			this->indexs[i] < edges_size) {
			return this->indexs[i];
		}
	}

	return edges_size;
}

//---------------------------------------------------------------------------------