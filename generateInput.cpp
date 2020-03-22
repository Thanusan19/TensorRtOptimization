#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


using namespace std;

int main(int argc, char** argv)
{
    cout<<"Convert"<<endl;
    const int inputH = 12; //mInputDims.d[1];
    const int inputW = 5*10000;//mInputDims.d[2];

    //Import and convert inout data from X.txt to a vector Xfloat//
    ifstream f("/home/localadmin/NNOptimization/TensorRtOptimization/X.txt");
    string Xstr;
    std::vector<float> Xfloatv;

    while(getline(f,Xstr,',')){
        float Xfloat= std::stof(Xstr);
        Xfloatv.push_back(Xfloat);
	//printf("%s ",Xstr);
    }
    f.close();

    //Create Input vector of shape =(2000,12,5)
    float X[2000][12][5];
    for(int i=0;i<2000;i++){
        for(int j=0;j<12;j++){
                for(int k=0;k<5;k++){
                        X[i][j][k]=Xfloatv[i*12*5 + 5*j + k];
                        //X[i][j][k]=Xfloatv[0];
                }
        }
    }

}
