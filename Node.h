#ifndef NODE_H
#define NODE_H

#include <cstdio>

using namespace std;

class Node
{
	public:
		__host__ __device__
		Node()
		{
			this->value = 0;
			this->next = nullptr;
		}
		__host__ __device__
		~Node(){}
		
		__host__ __device__
		int getValue(){ return this->value; }
		__host__ __device__
		void setValue(int value){ this->value = value; }
		__host__ __device__
		Node* getNext() { return this->next; }
		__host__ __device__
		void setNext(Node* n) { this->next = n; } 

	private:
		int value;
		Node *next;
};

#endif
