kernel: main.o Graph.o
		nvcc main.o Graph.o -o kernel
Graph.o: Graph.cu
		nvcc Graph.cu -dc -g -G
main.o: main.cu
		nvcc main.cu -dc -g -G

clean:
	rm kernel \
		Graph.o main.o
