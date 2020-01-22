#ifndef LIST_H
#define LIST_H

#include <cstdio>
#include "Node.h"

using namespace std;

class List
{
    public:
        __host__ __device__ 
        List(){
        	this->head = nullptr;
        	this->size_list = 0;
        }
        __host__ __device__
        ~List(){
        	this->deleteNodes();
        }
		
        __host__ __device__
        void clear(){
        	this->deleteNodes();
        	this->head = nullptr;
        	this->size_list = 0;
        }
        __host__ __device__
        void push_back(int val){
        	Node* node = new Node();
        	node->setValue(val);
        	
        	if(this->head == nullptr){
        		this->head = node;
        	}
        	else{
        		Node* aux = this->head;
        		while(aux->getNext() != nullptr){
        			aux = aux->getNext();
        		}
        		aux->setNext(node);
        	}
        	
        	this->size_list++;
        }
        __host__ __device__
        int pop_back(){
            int value = -1;
            
            Node *aux = this->head;
            Node *auxAnt = nullptr;

            if(aux != nullptr){
                while(aux->getNext() != nullptr){
                    auxAnt = aux;
                    aux = aux->getNext();
                }

                value = aux->getValue();
                if(auxAnt != nullptr){
                    auxAnt->setNext(nullptr);
                }
                else{
                    this->head = nullptr;
                }

                delete aux;
                this->size_list--;
            }
            

            return value;
        }
        __host__ __device__
        int pop_front(){
            int value = -1;
            
            Node *aux = this->head;

            if(aux != nullptr){

                value = aux->getValue();
                this->head = aux->getNext();

                delete aux;
                this->size_list--;
            }
            

            return value;
        }


        __host__ __device__
        int size(){
        	return this->size_list;
        }
        __host__ __device__
        int at(int i){
        	Node* aux = this->head;
        	int count = 0;
        	
        	while(aux != nullptr){
        		if(count == i){ return aux->getValue(); }
        		aux = aux->getNext();
        		count++;
        	}
        	
        	return -1;
        }

    private:
    	Node* head;
        int size_list;
        
        __host__ __device__
        void deleteNodes(){
        	Node* aux = this->head;
        	Node* aux2 = nullptr;
        	
        	while(aux != nullptr){
        		aux2 = aux->getNext();
        		delete aux;
        		aux = aux2;
        	}
        }
        
};

#endif 
