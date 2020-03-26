#include "Graph.h"
//-----------------------------Graph-----------------------------------------------
 
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
}

Graph::~Graph() {
}
/**
	Get the number of nodes
*/
 
int Graph::getNumberNodes() {
	return this->numberNodes;
}
/**
	Set the data's nodes 
*/
 
void Graph::setNodeData(float* costs, int* tails, int* indexs, int nEdges, int nNodes){    
	this->edges_cost = costs; 
    this->edges_tail = tails;
    this->indexs = indexs; 
    printf("Assignated data Graph\n");
}
/*
	Get the array of costs of the nodes
*/
 
float* Graph::getCosts(){
	return this->edges_cost;
}
/*
	Get the array of tails nodes
*/
 
int* Graph::getTails(){
	return this->edges_tail;
}
/*
	Get the indexs of the nodes
*/
 
int* Graph::getIndexs(){
	return this->indexs;
}

/*
	Get the array of graph's nodes
*/
  
int* Graph::getNodesArray(){
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
 
int Graph::getNumberEdges(){
	return this->numberEdges;
}

/**
	Own implementation to split a string
*/
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

/**
	Remove white spaces in a string
*/
string Graph::removeWhiteSpaces(string s){
	s.erase(remove(s.begin(), s.end(), ' '), s.end());
	s.erase(remove(s.begin(), s.end(), '\n'), s.end());
	s.erase(remove(s.begin(), s.end(), '\t'), s.end());
	s.erase(remove(s.begin(), s.end(), '\v'), s.end());
	s.erase(remove(s.begin(), s.end(), '\r'), s.end());

	return s;
}

/**
	Get the array of real labels or communities from a file 
*/  
int* Graph::getRealCommunities(string truedata){
	string line;
	
	//Iterators
	map<string, string> mp;
	vector<string> words;
	string delimiter = "\t";
	string node;
	string label;

	ifstream dataset;
	//cout << "True communities file "<< truedata << endl;
	dataset.open(truedata);

	//Read the datafile and create the nodes in the same time with the edges
	if (dataset.is_open()) {
		//cout << "opened" << endl;
		while (getline(dataset, line)) {
			//cout << "new line: " << line << endl;
			if (line.at(0) != '#') {
				words = split(line, delimiter);

				node = words[0];
				label = words[1];

				node = this->removeWhiteSpaces(node);
				label = this->removeWhiteSpaces(label);

				if(mp.find(node) == mp.end()){ //The node not exist in the map then create it
					mp.insert({node, label});
				}
			}
		}
		dataset.close();
	}
	else {
		return nullptr;
	}

	int *trueLabels = new int[this->getNumberNodes()];
	int id;

	for (auto itr = mp.begin(); itr != mp.end(); ++itr) { 
		id = this->getId(itr->first);
		if(id >= 0 && id < this->getNumberNodes()){
			trueLabels[id] = stoi(itr->second);
		}
	}

	return trueLabels;
}

/**
	Save the compute labels in a file with the send name
*/
bool Graph::saveCommunitiesinFile(string filename, int* labels)
{
	ofstream outdata;
	outdata.open(filename); //open the file

	if(!outdata){
		cerr << "Error: file could not be opened" << endl;
		return false;
	}

	int community  = 0;
	for (int i=0; i < this->getNumberNodes(); i++){
		community = labels[i]; 
		outdata << this->getName(i) << "\t" << community << endl;
	}

	outdata.close();

	return true;
}

/**
	--FILE--
	Read a file and create the graph in mode directed or indirected
*/
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
    this->numberNodes = this->nodes.size();
    for(int node_i = 0; node_i < this->nodes.size(); node_i++){
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
	Read a file with extension .net and create the graph in mode indirected
*/
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

					nodei = this->removeWhiteSpaces(nodei);
					nodej = this->removeWhiteSpaces(nodej);
					cost = this->removeWhiteSpaces(cost);

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

	    for (auto itr : sortedNodes){
	    	indexs_aux[i] = index;  	
	        this->nodes.insert({itr.first, i});
	        i++;
	        index = index + itr.second.size();
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
	        for (auto itedge = itr->second.begin(); itedge != itr->second.end(); ++itedge) { 
	        	edges.push_back(itedge->first);
	        	costs.push_back(itedge->second);
	        }
    	}
    }


    this->numberEdges = edges.size();
    this->edges_cost = new float[costs.size()];

    //Create Costs array (O(n))
    for(int cost_i = 0; cost_i < costs.size(); cost_i++){
    	this->edges_cost[cost_i] = stof(costs[cost_i]);
    }


    //Create Tails or ptr edges (O(m))
    this->edges_tail = new int[edges.size()];
    for(int edge_i = 0; edge_i < edges.size(); edge_i++){
    	if(this->nodes.find(edges[edge_i]) == this->nodes.end()){//Insert a new node when this doesnt exist
    		this->nodes.insert({edges[edge_i], i});
        	this->edges_tail[edge_i] = i;
        	i++;
    	}
    	else{//The node already exist
    		this->edges_tail[edge_i] = this->nodes.find(edges[edge_i])->second;
    	}
    }

    // Create Indexs array (O(n))
    this->indexs = new int[this->nodes.size()];
    this->numberNodes = this->nodes.size();
    for(int node_i = 0; node_i < this->nodes.size(); node_i++){
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

  
int Graph::getId(string name){
	for (auto itr = this->nodes.begin(); itr != this->nodes.end(); ++itr) {
		if(itr->first == name){
			return itr->second;
		}
	}

	return -1;
}

  
string Graph::getName(int id){
	for (auto itr = this->nodes.begin(); itr != this->nodes.end(); ++itr) {
		if(itr->second == id){
			return itr->first;
		}
	}
	string ret = "null";

	return ret;
}

//------------------------------------------------------------------------------------------

//-----------------------Node---------------------------------------------------------------

/**
	Get the edges cost of the node
*/
  
float* Graph::getEdgesCost(int source) {
	int index = this->indexs[source]; //Get the node source
	int nextIndex = (source + 1 < this->getNumberNodes())? this->indexs[source + 1]: this->getNumberEdges(); 
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
  
int* Graph::getEdgesEdpoints(int source) {
	int index = this->indexs[source]; //Get the node source
	int nextIndex = (source + 1 < this->getNumberNodes())? this->indexs[source + 1]: this->getNumberEdges(); 
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
    Return an array thar contains the total of communities and those don't repeat inside of array
*/
 
int* Graph::getCommunities(int *labels, int* numCommunities){
	int nNodes = this->getNumberNodes();
    std::set<int> s(labels, labels + nNodes);
    int* communities = new int[s.size()];

    *numCommunities = s.size();
    int i = 0;
    for(int community: s){
        communities[i] = community;
        i++;
    }

    return communities;
}

//---------------------------------------------------------------------------------