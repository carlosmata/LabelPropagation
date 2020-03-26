#include "tests.h"

using namespace std;
//-----------------------------------------------main--------------------------------------------------------------

int main(int argc, char **argv)
{
	int test = 0;
	string filename = "datasets/karate_test.txt";
	int type = 2; //1-directed, 2-undirected, 3-NET extension
	int sorted = 0;
	string truedata = "";
	int mode = 0; //0-synchronous, 1-asychronous, 2-semisynchronous(colors), 3-asynchronous

	if(argc == 2){//Add the filename of the datasets
		test = atoi(argv[1]);
	}
	if(argc == 3){//Add the filename of the datasets
		test = atoi(argv[1]);
		filename = argv[2];
	}
	if(argc == 4){//Add the type of the filename
		test = atoi(argv[1]);
		filename = argv[2];
		type = atoi(argv[3]);
	}
	if(argc == 5){//Add the type of the filename and a sorted way desc-asc
		test = atoi(argv[1]);
		filename = argv[2];
		type = atoi(argv[3]);
		sorted = atoi(argv[4]);
	}
	if(argc == 6){//Add the type of the filename and a sorted way desc-asc and a file with true data
		test = atoi(argv[1]);
		filename = argv[2];
		type = atoi(argv[3]);
		sorted = atoi(argv[4]);
		truedata = argv[5];
	}
	if(argc == 7){//Add the type of the filename and a sorted way desc-asc and a file with true data
		test = atoi(argv[1]);
		filename = argv[2];
		type = atoi(argv[3]);
		sorted = atoi(argv[4]);
		truedata = argv[5];
		mode = atoi(argv[6]);
	}

	cout<<"argc:"<< argc << endl;

	switch(test){
		case 0:
			testGraph(filename, type, sorted, truedata, mode);
		break;
		case 1:
			testNETFiles();
		break;
		case 2:
			testTrueData();
		break;
		case 3:
			testMediumFiles();
		break;
		case 4:
			testBigFiles();
		break;
		case 5://test the last algorithm
			testModeAlgorithm(3);
		break;
	}
}


