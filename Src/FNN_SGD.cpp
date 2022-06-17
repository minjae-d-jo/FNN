#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <RandomNumber.hpp>
#include <cmath>


using namespace std;
using namespace Snu::Cnrc;

using Size = unsigned int;
using Time = unsigned int;

/*class Layer {
private:
    int nodeNum;
    vector<vector<double>> W;
public:
    Layer(int n, int l) : nodeNum(n),linkNum(l), nodeToNode(n) {}
 };
double evaluateDistance(vector<int>& v1, vector<double>& v2){
    double distance = 0;
    for(Size i=0; i<v1.size(); ++i) distance += pow((double)v1[i]-v2[i], 2);
    return sqrt(distance);
}*/



////////////////////////////////////////////////////////////////////////////////////////////
void getInputAndLabel(vector<vector<double>>& XOR_input, vector<vector<double>>& XOR_label) {
    std::ifstream infile("inputAndLabel.txt");
    string line;
    while(getline(infile, line)){
        istringstream iss(line);
        double a, b, c, d, e;
        while(iss >> a >> b >> c >> d >> e){
            vector<double> input = {a, b, c};
            vector<double> label = {d, e};
            // inputData[floor(a)] = b; // If $1 is double number.... float(a)
            XOR_input.emplace_back(input);
            XOR_label.emplace_back(label);
            if (!(iss >> a >> b >> c >> d >> e)) break; // error
        }
    }
}

void generateInputAndLabel(vector<vector<double>>& XOR_input, vector<vector<double>>& XOR_label,
    const Size& inputNeuronNum, const Size& trialNum) {
    RandomIntGenerator rndInt(0, 1);
    Size sum;
    vector<double> v(inputNeuronNum);
    vector<double> w;
    for(Size trainingNum=0; trainingNum<trialNum; ++trainingNum) {
        sum = 0;
        for(Size i=0; i<inputNeuronNum; ++i) {
            v[i] = rndInt();
            sum += v[i];
        }
        XOR_input.emplace_back(v);
        if( sum%2==0 ) {
            w = {1, 0};
            XOR_label.emplace_back(w);
        }
        else {
            w = {0, 1};
            XOR_label.emplace_back(w);
        }
    }
}

void pickInputForOneBatch(vector<vector<double>>& XOR_input, vector<vector<double>>& XOR_pickedInput,
    vector<vector<double>>& XOR_label, vector<vector<double>>& XOR_pickedLabel) {
    RandomIntGenerator rndInt(0, XOR_input.size()-1);
    Size selection;
    for(Size i=0; i<XOR_pickedInput.size(); ++i) {
        selection = rndInt();
        XOR_pickedInput[i] = XOR_input[selection];
        XOR_pickedLabel[i] = XOR_label[selection];
    }
}

Size isItCorrect(vector<double>& outputLayer, vector<double>& XOR_ith_label) {
    if( outputLayer[0]<outputLayer[1] ) {
        if( XOR_ith_label[1] == 1) return 1;
        else return 0;
    }
    else {
        if( XOR_ith_label[0] == 1) return 1;
        else return 0;
    }
}

double kroneckerDelta(Size l, Size i) {
    if(l==i) return 1;
    else return 0;
}

void initWB(vector<vector<vector<double>>>& w, vector<vector<double>>& b) {
    RandomGaussianGenerator rndNormal(0, 1);
    for(Size i=0; i<w.size(); ++i){
        for(Size j=0; j<w[i].size(); ++j){
            for(Size k=0; k<w[i][j].size(); ++k){
                w[i][j][k] = rndNormal();
            }
            b[i][j] = rndNormal();
        }
    }
}

void calculateError(vector<double>& outputLayer, vector<double>& XOR_ith_label, double& error) {
    for(Size j=0; j<XOR_ith_label.size(); j++){
        error -= XOR_ith_label[j]*log(outputLayer[j]); // negative log-likelihood
    }
}


double calculateError(vector<double>& outputLayer, vector<double>& XOR_ith_label) {
    double error=0;
    for(Size j=0; j<XOR_ith_label.size(); j++){
        error -= XOR_ith_label[j]*log(outputLayer[j]); // negative log-likelihood
    }
    return error;
}

void forwardPropagate(vector<double>& XOR_ith_input, vector<double>& hiddenLayer, vector<double>& outputLayer, 
    vector<vector<vector<double>>>& w, vector<vector<double>>& b){
    for(Size indice=1; indice<w.size()+1; ++indice) {
        double exponentSum = 0;
        if( indice==w.size() ) {
            for(Size i=0; i<outputLayer.size(); i++) {
                outputLayer[i] = b[indice-1][i];
                for(Size j=0; j<hiddenLayer.size(); j++) {
                    outputLayer[i] += w[indice-1][i][j] * hiddenLayer[j];
                }
                exponentSum += exp(outputLayer[i]);
                outputLayer[i] = exp(outputLayer[i]); // softmax fct. for output layer
            }
            outputLayer[0] /= exponentSum;
            outputLayer[1] /= exponentSum;
        }
        else {
            for(Size i=0; i<hiddenLayer.size(); i++) {
                hiddenLayer[i] = b[indice-1][i];
                for(Size j=0; j<XOR_ith_input.size(); j++) {
                    hiddenLayer[i] += w[indice-1][i][j] * XOR_ith_input[j];
                }
                hiddenLayer[i] = (1 / (1+exp(-hiddenLayer[i]))); // sigmoid fct. for hidden layer
            }
        }
    }
}

void backPropagate(vector<double>& hiddenLayer, vector<double>& outputLayer, vector<vector<vector<double>>>& w, vector<vector<double>>& b, vector<double>& XOR_ith_input, vector<double>& XOR_ith_label) {
    Size outputLayerNum = w.size();
    double learningRate = 0.01;
    vector<double> delta1(outputLayer.size(), 0); 
    vector<double> delta0(hiddenLayer.size(), 0); 

    for(Size indice=outputLayerNum; indice>0; --indice) {
        if(indice==outputLayerNum) {
            for(Size i=0; i<outputLayer.size(); ++i) {
                delta1[i] = (outputLayer[i] - XOR_ith_label[i]); // negative log-likelihood
            }
        }
        else {
            for(Size i=0; i<hiddenLayer.size(); ++i) {
                for(Size m=0; m<outputLayer.size(); ++m) {
                    delta0[i] += w[indice][m][i] * delta1[m];
                }
                delta0[i] *= hiddenLayer[i] * (1-hiddenLayer[i]);
            }
        }
        // }
    }
    for(Size i=0; i<outputLayer.size(); ++i) {
        for(Size j=0; j<hiddenLayer.size(); ++j) {
            w[1][i][j] -= learningRate * hiddenLayer[j] * delta1[i];
        }
        b[1][i] -= learningRate * delta1[i];
    }
    for(Size i=0; i<hiddenLayer.size(); ++i) {
        for(Size j=0; j<XOR_ith_input.size(); ++j) {
            w[0][i][j] -= learningRate * XOR_ith_input[j] * delta0[i];
        }
        b[0][i] -= learningRate * delta0[i];
    }
}

void TrainFNN(vector<double>& hiddenLayer, vector<double>& outputLayer, vector<vector<vector<double>>>& w, 
    vector<vector<double>>& b, vector<vector<double>>& XOR_pickedInput, vector<vector<double>>& XOR_pickedLabel, 
    vector<vector<double>>& XOR_input, vector<vector<double>>& XOR_label, 
    double targetError, const Size& trainingNum){
    
    double error;
    error = 0;
    for(Size trial=0; trial<trainingNum/XOR_pickedInput.size(); ++trial) {
        pickInputForOneBatch(XOR_input, XOR_pickedInput, XOR_label, XOR_pickedLabel);
        for(Size i=0; i<XOR_pickedInput.size(); ++i) { // train the number of the XOR_pickedInput
            forwardPropagate(XOR_pickedInput[i], hiddenLayer, outputLayer, w, b);
            error += calculateError(outputLayer, XOR_pickedLabel[i]);
            backPropagate(hiddenLayer, outputLayer, w, b, XOR_pickedInput[i], XOR_pickedLabel[i]);
        }
    }
    cout << "Stopping at error : " << error/trainingNum << endl;
}

void TestFNN(vector<double>& hiddenLayer, vector<double>& outputLayer, vector<vector<vector<double>>>& w, 
    vector<vector<double>>& b, vector<vector<double>>& XOR_input_test, vector<vector<double>>& XOR_label_test, 
    Size epoch, ofstream& os){
    Size correct = 0;
    for(Size i=0; i<XOR_input_test.size(); ++i) { 
        forwardPropagate(XOR_input_test[i], hiddenLayer, outputLayer, w, b);
        correct += isItCorrect(outputLayer, XOR_label_test[i]);
        // cout << error << endl;
    }
    os << epoch << '\t' << (double)correct/XOR_input_test.size() << endl;
}

int main(int, char** argv){
    const Size inputNeuronNum = stoul( argv[1] );
    const Size hiddenNeuronNum = stoul( argv[2] );
    const Size outputNeuronNum = stoul( argv[3] );
    const Size trainingNum = stoul( argv[4] );
    const Size testNum = stoul( argv[5] );
    vector<vector<double>> XOR_input;
    vector<vector<double>> XOR_label;
    vector<vector<double>> XOR_pickedInput(50);
    vector<vector<double>> XOR_pickedLabel(50);
    generateInputAndLabel(XOR_input, XOR_label, inputNeuronNum, trainingNum);
    
    vector<vector<double>> w1(hiddenNeuronNum, vector<double>(inputNeuronNum)); 
    vector<double> b1(hiddenNeuronNum); 
    vector<vector<double>> w2(outputNeuronNum, vector<double>(hiddenNeuronNum)); 
    vector<double> b2(outputNeuronNum); 
    vector<vector<vector<double>>> w;
    vector<vector<double>> b;
    w.emplace_back(w1);
    b.emplace_back(b1);
    w.emplace_back(w2);
    b.emplace_back(b2);
    double targetError = 0.1;

    vector<double> hiddenLayer(hiddenNeuronNum);
    vector<double> outputLayer(outputNeuronNum);
    initWB(w, b);
    ostringstream oss;
    oss << "FNN_" << inputNeuronNum << "_" << hiddenNeuronNum << "_" << outputNeuronNum << "_" << trainingNum << "_" << testNum << "_SGD.table";
    ofstream os("/pds/pds181/jmj/ML/FNN/Result/"+oss.str());
        
    for(Size epoch=0; epoch<1000; ++epoch) {
        TrainFNN(hiddenLayer, outputLayer, w, b, XOR_pickedInput, XOR_pickedLabel, XOR_input, XOR_label, targetError,trainingNum);

        vector<vector<double>> XOR_input_test;
        vector<vector<double>> XOR_label_test;
        generateInputAndLabel(XOR_input_test, XOR_label_test, inputNeuronNum, testNum);
        TestFNN(hiddenLayer, outputLayer, w, b, XOR_input_test, XOR_label_test, epoch, os);
    }
    return 0;
}
