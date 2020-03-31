#include "tests.h"

using namespace std;
//-----------------------------------------------main--------------------------------------------------------------

int main(int argc, char **argv)
{
	int test = 1;
	int sorted = 0;
	int algorithm = 1; //0-synchronous, 1-asychronous, 2-semisynchronous(colors), 3-asynchronous primitives 4-all algorithms
	string filename = "datasets/karate_test.txt";
	int type = 2; //1-directed, 2-undirected, 3-NET extension
	string truedata = "";

	cout<<"argc:"<< argc << endl;

	if(argc > 1){//Add the filename of the datasets
		test = atoi(argv[1]);
	}

	switch(test){
		case 0:		
			/*if(argc == 3){//Add the filename of the datasets
				filename = argv[2];
			}
			if(argc == 4){//Add the type of the filename
				filename = argv[2];
				type = atoi(argv[3]);
			}
			if(argc == 5){//Add the type of the filename and a sorted way desc-asc
				filename = argv[2];
				type = atoi(argv[3]);
				sorted = atoi(argv[4]);
			}
			if(argc == 6){//Add the type of the filename and a sorted way desc-asc and a file with true data
				filename = argv[2];
				type = atoi(argv[3]);
				sorted = atoi(argv[4]);
				truedata = argv[5];
			}
			if(argc == 7){//Add the type of the filename and a sorted way desc-asc and a file with true data
				filename = argv[2];
				type = atoi(argv[3]);
				sorted = atoi(argv[4]);
				truedata = argv[5];
				algorithm = atoi(argv[6]);
			}*/
			if(argc > 2){//Add the filename of the datasets
				filename = argv[2];
			}
			if(argc > 3){//Add the type of the filename
				type = atoi(argv[3]);
			}
			if(argc > 4){//Add the type of the filename and a sorted way desc-asc
				sorted = atoi(argv[4]);
			}
			if(argc > 5){//Add the type of the filename and a sorted way desc-asc and a file with true data
				truedata = argv[5];
			}
			if(argc > 6){//Add the type of the filename and a sorted way desc-asc and a file with true data
				algorithm = atoi(argv[6]);
			}

			testGraph(filename, type, sorted, truedata, algorithm);
		break;
		case 1: //Net files
			if(argc > 2){//Add the filename of the datasets
				algorithm = atoi(argv[2]);
			}
			if(argc > 3){//Add the type of the filename
				sorted = atoi(argv[3]);
			}

			testNETFiles(algorithm, sorted);
		break;
		case 2://NMI Files
			if(argc > 2){//Add the filename of the datasets
				algorithm = atoi(argv[2]);
			}
			if(argc > 3){//Add the type of the filename
				sorted = atoi(argv[3]);
			}

			testNMIFiles(algorithm, sorted);
		break;
		case 3://Medium files
			if(argc > 2){//Add the filename of the datasets
				algorithm = atoi(argv[2]);
			}
			if(argc > 3){//Add the type of the filename
				sorted = atoi(argv[3]);
			}

			testMediumFiles(algorithm, sorted);
		break;
		case 4://Big files
			if(argc > 2){//Add the filename of the datasets
				algorithm = atoi(argv[2]);
			}
			if(argc > 3){//Add the type of the filename
				sorted = atoi(argv[3]);
			}

			testBigFiles(algorithm, sorted);
		case 5://All files
			if(argc > 2){//Add the filename of the datasets
				algorithm = atoi(argv[2]);
			}
			if(argc > 3){//Add the type of the filename
				sorted = atoi(argv[3]);
			}

			testAllFiles(algorithm, sorted);
		break;
	}
}


