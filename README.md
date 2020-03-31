# Label propagation - Secuencial
Community detection in large scale networks based in label propagation algorithm


#USE
Execute the command make to generate the executable

$make

this command will generate the file ./kernel

OPTIONS
$./kernel Option 
Options:
0: Execute a file of the datasets in the folder  
	./kernel 0 params	
	Params: 
	[path of dataset] 
	[type of file: 1-directed, 2-undirected, 3-NET file] 
	[0-unsorted, 1-sorted ]
	[path of communities in case of compute NMI]
	[algorithm: 0-synchronous, 1-asychronous, 2-semisyncronous, 3-asyncronous with primitives]
	
	Example
	./kernel 0 datasets/converted/karate.net 3 1 none 3

	Out:
	Dataset: datasets/converted/karate.net
			Nodes	Edges	Com	Mod	NMI	Time
	seq async 	34	156	3	0.42422	-1	7.045e-05
	par async2 	34	156	6	0.20085	-1	0.096486

	The result of the execution shoe both modes, sequential and parallel ways, each show the 	 number of nodes, the number of edges, the number of communities, the modulariy measure, 	the NMI measure, and the execution's time.
 
1: Execute the files with extension .NET
	./kernel 1 params
	
	Params:
	[algorithm: 
		0-synchronous, 
		1-asychronous, 
		2-semisyncronous, 
		3-asyncronous with primitives
		4-all algorithms
	]
	[0 unsorted, 1 sorted]

	Example 
	./kernel 1 1 0

2: Execute the files with information of NMI measure
	./kernel 2 params
	
	Params:
	[algorithm: 
		0-synchronous, 
		1-asychronous, 
		2-semisyncronous, 
		3-asyncronous with primitives
		4-all algorithms
	]
	[0 unsorted, 1 sorted]

	Example 
	./kernel 2 1 0

3: Execute the files with a medium size (nodes >= 300,000)
	./kernel 3 params
	Params:
	[algorithm: 
		0-synchronous, 
		1-asychronous, 
		2-semisyncronous, 
		3-asyncronous with primitives
		4-all algorithms
	]
	[0 unsorted, 1 sorted]
	
	Example 
	./kernel 3 1 0

4: Execute the files with a big size (nodes >= 1,000,000)
	./kernel 4 params
	Params:
	[algorithm: 
		0-synchronous, 
		1-asychronous, 
		2-semisyncronous, 
		3-asyncronous with primitives
		4-all algorithms
	]
	[0 unsorted, 1 sorted]
	
	Example 
	./kernel 4 1 0

5: Execute all datasets 
	./kernel 5 params
	Params:
	[algorithm: 
		0-synchronous, 
		1-asychronous, 
		2-semisyncronous, 
		3-asyncronous with primitives
		4-all algorithms
	]
	[0 unsorted, 1 sorted]

	Example 
	./kernel 5 1 0
 
