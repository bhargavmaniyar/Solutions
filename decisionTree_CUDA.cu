#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <string>
#include <fstream>
#include <sstream>
#include "gputimer.h"

#include "math.h"
#include "limits.h"

#define MINIMUM -99
#define training_set "car.data.txt"
#define test_set "car.data.txt"
#define M 6
#define N 104
#define trainFileData(row,col) trainFileData[row*M+col]
#define testFileData(row,col) testFileData[row*M+col]

using namespace std;

vector <vector <int> > trainFile;
vector <vector <int> > testFile;

GpuTimer kernelTimer;
GpuTimer mallocTimer;
float kernelTime = 0;
float mallocTime = 0;

int *d_trainFileData, *d_cardinality;
float *initialize_info_gain;
__device__ float d_infoGainOfData;

dim3 blocks(M);
dim3 threads(N);


struct Node{
	int numOfChildren;
	int val;
	int branchVal;
	int attribute;
	struct Node *child[10];
};

typedef struct Node node;

// initialising tree node
node* create(){
	node* n = new node;
	n->numOfChildren = 0;
	n->attribute = -1;
	n->val = -1;
	n->branchVal = -1;
	return n;
}

// function to read data and store in fileContent & testFileContent vectors(2d)

void readCSV(string str)
{
													// input file stream (ifs) for reading data from file
	if(str.compare("training")==0){
		ifstream ifs(training_set);
		string line;

													// read from ifs into string 'line'
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <int> values;
													// collecting row data from file delimited by ','
			while(getline(lineStream,cell,',')){
				const char *cstr = cell.c_str();
				values.push_back(atoi(cstr));
			}
			trainFile.push_back(values);
		}
		ifs.close();
	}
	else if(str.compare("testing")==0){
		ifstream ifs(test_set);
		string line;
		
													// read from ifs into string 'line'
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <int> values;
													// collecting row data from file delimited by ','
			while(getline(lineStream,cell,',')){
				const char *cstr = cell.c_str();
				values.push_back(atoi(cstr));
			}
			testFile.push_back(values);
		}
		ifs.close();
	}
}

__global__ void getInfoGains(int *attr,int *data,int dataSize,float *infoGains,int *trainFileData,int *cardinality)
{
	if(attr[blockIdx.x]==0 && blockIdx.x!=0 && blockIdx.x!=M-1){
		int tid,bid,j;
		tid=threadIdx.x;
		bid=blockIdx.x;
		__shared__ int attrValCount[10];
		__shared__ int classAttrValCount[10][10];
		if(tid<10){
			attrValCount[tid]=0;
			for(j=0;j<10;j++){
				classAttrValCount[tid][j]=0;
			}
		}
		__syncthreads();
		int classVal = trainFileData(data[tid],M-1);
		int attrVal = trainFileData(data[tid],bid);
		atomicAdd(&attrValCount[attrVal],1);					// atomic operation is used to prevent RACE condition
		atomicAdd(&classAttrValCount[attrVal][classVal],1);
		__syncthreads();										// synchronize the threads
		if(tid==0){
			int i,j;
			float infoGain,betweens_gain;
			infoGain=0;
			for(i=1;i<=cardinality[bid];i++){
				betweens_gain=0;
				if(attrValCount[i]==0){
					continue;
				}
				for(j=1;j<=cardinality[M-1];j++){
					if(classAttrValCount[i][j]==0){
						continue;
					}
					betweens_gain+=(float(classAttrValCount[i][j])/(float)attrValCount[i])*(log((float)classAttrValCount[i][j]/(float)attrValCount[i])/log((float)2));
				}
				betweens_gain*=(float(attrValCount[i])/(float)dataSize);
				infoGain-=betweens_gain;
			}
			infoGains[bid]=infoGain;
		}
	}
}

__global__ void Info_gain_data(int *data,int dataSize,int *trainFileData,int *cardinality)
{
	__shared__ int classValCount[10];
	int classVal,i,tid;
	float infoGain;
	tid=threadIdx.x;
	if(tid<10){
		classValCount[tid]=0;
	}
	__syncthreads();
	classVal=trainFileData(data[threadIdx.x],M-1);
	atomicAdd(&classValCount[classVal],1);
	__syncthreads();
	if(tid==0){
		infoGain=0;
		for(i=1;i<=cardinality[M-1];i++){
			if(classValCount[i]==0){
				continue;
			}
			infoGain+=((float)classValCount[i]/(float)dataSize)*(log((float)classValCount[i]/(float)dataSize)/log((float)2));
		}
		d_infoGainOfData=-1*infoGain;
	}
}

int popularVote(int *data,int dataSize)
{
	int i,outputClass,ans,maxVal;
	map <int, int> dataCount;
	map <int, int>::iterator it;
	for(i=0;i<dataSize;i++){
		outputClass = trainFile[data[i]][M-1];
		if(dataCount.find(outputClass)==dataCount.end()){
			dataCount.insert(make_pair(outputClass,1));
		}
		else{
			dataCount[outputClass]++;
		}
	}
	maxVal = MINIMUM;
	for(it=dataCount.begin();it!=dataCount.end();it++){
		if(it->second > maxVal){
			ans = it->first;
		}
	}
	return ans;
}

void id3_decision_tree(int *h_attr,int *h_data, node *root,int h_dataSize)
{

	int flag,h_selectedAttribute,i;
	float maxGain;
	if(h_dataSize==0){
		return;
	}
	flag=1;
	for(i=1;i<h_dataSize;i++){
		if(trainFile[h_data[i]][M-1]!=trainFile[h_data[i-1]][M-1]){
			flag=0;
			break;
		}
	}
	if(flag==1){
		root->val=trainFile[h_data[0]][M-1];
		return;
	}

	int *d_attr, *d_data;
	float *d_infoGains;
	float h_infoGains[M];
	float h_infoGainOfData;

	mallocTimer.Start();
	cudaMalloc((void**)&d_attr,M*sizeof(int));
	cudaMalloc((void**)&d_data,h_dataSize*sizeof(int));
	cudaMalloc(&d_infoGains,M*sizeof(float));
	cudaMemcpy((void*)d_attr,(void*)h_attr,M*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_data,(void*)h_data,h_dataSize*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_infoGains, initialize_info_gain, M*sizeof(float),cudaMemcpyHostToDevice);
	mallocTimer.Stop();
	mallocTime+=mallocTimer.Elapsed();

	kernelTimer.Start();
	getInfoGains<<<blocks,h_dataSize>>>(d_attr,d_data,h_dataSize,d_infoGains,d_trainFileData,d_cardinality);
	kernelTimer.Stop();
	kernelTime+=kernelTimer.Elapsed();

	mallocTimer.Start();
	cudaMemcpy((void*)h_infoGains,(void*)d_infoGains,M*sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(d_attr);
	cudaFree(d_infoGains);
	mallocTimer.Stop();
	mallocTime+=mallocTimer.Elapsed();	

	kernelTimer.Start();
	Info_gain_data<<<1,h_dataSize>>>(d_data,h_dataSize,d_trainFileData,d_cardinality);
	kernelTimer.Stop();
	kernelTime+=kernelTimer.Elapsed();

	mallocTimer.Start();
	cudaMemcpyFromSymbol(&h_infoGainOfData,d_infoGainOfData,sizeof(float),0,cudaMemcpyDeviceToHost);

	cudaFree(d_data);
	mallocTimer.Stop();
	mallocTime+=mallocTimer.Elapsed();

	maxGain=MINIMUM;
	h_selectedAttribute=-1;

	// printf("infoGain of data: %f\n",h_infoGainOfData);
	// printf("attribute gains:\n");

	for(i=1;i<M-1;i++){
		if(h_attr[i]==0){
			h_infoGains[i]=h_infoGainOfData-h_infoGains[i];

			// printf("%d %f\n",i,h_infoGains[i]);

			if(h_infoGains[i]>maxGain){
				maxGain=h_infoGains[i];
				h_selectedAttribute=i;
			}
		}
	}

	root->attribute = h_selectedAttribute;
	h_attr[h_selectedAttribute]=1;

	if(h_selectedAttribute==-1){
		root->val = popularVote(h_data, h_dataSize);
		return;
	}

	map<int, vector <int> > dividedData;
	map<int, vector <int> >::iterator it;
	int attrVal;

	for(i=0;i<h_dataSize;i++){
		attrVal = trainFile[h_data[i]][h_selectedAttribute];
		if(dividedData.find(attrVal) == dividedData.end()){
			vector <int> x;
			x.push_back(h_data[i]);
			dividedData.insert(make_pair(attrVal,x));
		}
		else{
			dividedData[attrVal].push_back(h_data[i]);
		}
	}
	for(i=0,it=dividedData.begin();it!=dividedData.end();it++,i++){
		root->numOfChildren++;
		node* childNode;
		childNode = create();
		childNode->branchVal = it->first;
		root->child[i] = childNode;
		int* h_childData = &(it->second[0]);

		int new_attr[M];
		for(int z=0;z<M;z++){
			new_attr[z]=h_attr[z];
		}

		id3_decision_tree(new_attr, h_childData, childNode, it->second.size());
	}
}

__global__ void getCardinality(int *trainFileData,int *cardinality)
{
	__shared__ int x[10];
	int bid,tid,i;
	bid=blockIdx.x;
	tid=threadIdx.x;
	if(tid<10){
		x[tid]=0;
	}
	__syncthreads(); 					// to synchronize threads in device
	if(blockIdx.x!=0){
		x[trainFileData(tid,bid)]=1;
		__syncthreads();
		for(i=1;i<10;i*=2){
			int index = 2*i*tid;
			if(index+i<10){
				x[index]+=x[index+i];
			}
			__syncthreads();
		}
		if(tid==0){
			cardinality[bid]=x[0];
		}
	}
	__syncthreads();
}

// function for printing and debugging id3_decision_tree tree : bfs traversal
void printid3_decision_treeTree(node *root)
{
	printf("Printing id3_decision_tree tree:\n");
	queue <node> bfsQ;
	int x,j;
	node* nextNode;
	bfsQ.push(*root);
	cout << root->attribute << endl;
	// implementing bfs traversal of tree
	while(bfsQ.size()!=0){
		nextNode = &(bfsQ.front());
		bfsQ.pop();
		x = nextNode->numOfChildren;
		j=0;
		while(j<x){
			bfsQ.push(*(nextNode->child[j]));
			cout << nextNode->child[j]->attribute << " ";
			j++;
		}
		cout << endl;
	}
	return;
}

// function for testing id3_decision_tree tree
void test(node* root)
{
	int i,pos,neg,noResult,attr,attrVal,j,flag;
	node* temp;
	pos=0;
	neg=0;
	noResult=0;
	readCSV("testing");
	for(i=0;i<testFile.size();i++){
		temp=root;
		flag=0;
		//traverse id3_decision_tree tree
		while(temp->val==-1 && temp->attribute!=-1){
			attr = temp->attribute;
			attrVal=testFile[i][attr];
			for(j=0;j<temp->numOfChildren;j++){
				if(temp->child[j]->branchVal == attrVal){
					break;
				}
			}
			if(j==temp->numOfChildren){
				flag=1;
				break;
			}
			else{
				temp=temp->child[j];
			}
		}
		if(temp->val == testFile[i][M-1]){
			// predicted value = actual value
			pos++;
		}
		else{
			// predicted value != actual value
			neg++;
		}
		if(temp->val == -1 || flag==1){
			// no predicted value
			noResult++;
		}
	}
	cout << "Positive: " << pos << endl;
	cout << "Negative: " << neg << endl;
	cout << "No Result: " << noResult << endl;

	return;
}

int main()
{
	GpuTimer timer,timer_test;
	GpuTimer timer_total;

	timer_total.Start();
	int i;
	node* root;

	readCSV("training");

	int h_trainFileData[N*M];

	for(i=0;i<N*M;i++){
		h_trainFileData[i] = trainFile[i/M][i%M];
	}

	int h_data[N],h_attr[M];

	for(i=0;i<N;i++){
		h_data[i]=i;
	}

	for(i=0;i<M;i++){
		h_attr[i]=0;
	}

	mallocTimer.Start();
	cudaMalloc((void**)&d_trainFileData,N*M*sizeof(int));
	cudaMemcpy((void*)d_trainFileData,(void*)h_trainFileData,M*N*sizeof(int),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_cardinality,M*sizeof(int));
	cudaMemset(d_cardinality,0,M*sizeof(int));
	mallocTimer.Stop();
	mallocTime+=mallocTimer.Elapsed();

	kernelTimer.Start();
	getCardinality<<<blocks,threads>>>(d_trainFileData,d_cardinality);
	kernelTimer.Stop();
	kernelTime+=kernelTimer.Elapsed();

	root = create();

	initialize_info_gain = (float*)malloc(M*sizeof(float));
	for(i=0;i<M;i++){
		initialize_info_gain[i]=MINIMUM;
	}

	timer.Start();
	id3_decision_tree(h_attr,h_data,root,N);
 	timer.Stop();

	cudaFree(d_trainFileData);
	cudaFree(d_cardinality);

	//print id3_decision_tree tree
	//printid3_decision_treeTree(root);

	// test id3_decision_tree tree
	timer_test.Start();
	test(root);
	timer_test.Stop();
	timer_total.Stop();
	


	printf("id3_decision_tree Time taken: %gms\n",timer.Elapsed());
	printf("Test Time taken: %gms\n",timer_test.Elapsed());

	printf("Kernel Time: %gms\n",kernelTime);
	printf("Malloc Time: %gms\n",mallocTime);
	printf("Total Time: %gms\n",timer_total.Elapsed());


	return 0;
}
