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

class Layer {
private:
    Size previousNeuronNum;
    Size nextNeuronNum;
    Size batchSize;
    vector<vector<double>> w;
    vector<double> b;
    vector<vector<double>> y;
    vector<vector<double>> delta;

public:
    Layer(Size __previousNeuronNum__, Size __nextNeuronNum__, Size __batchSize__) : previousNeuronNum(__previousNeuronNum__), nextNeuronNum(__nextNeuronNum__), batchSize(__batchSize__), w(__nextNeuronNum__, vector<double>(__previousNeuronNum__)), b(__nextNeuronNum__), y(__batchSize__, vector<double>(__nextNeuronNum__)), delta(__batchSize__, vector<double>(__nextNeuronNum__, 0)){initialize();}
    
    vector<vector<double>>& getY() {return y;}
    vector<vector<double>>& getDelta() {return delta;}
    vector<vector<double>>& getW() {return w;}

    void initialize() {
        RandomGaussianGenerator rndNormal(0, 1);
        for(Size i=0; i<nextNeuronNum; ++i){
            for(Size j=0; j<previousNeuronNum; ++j){
                w[i][j] = rndNormal();
            }
            b[i] = rndNormal();
        }
    }    
    
    void sigmoid(Size batchNum) {
        for(Size i=0; i<y[batchNum].size(); i++) {
            y[batchNum][i] = 1/(1+exp(-y[batchNum][i]));
        }
    }

    void ReLU(Size batchNum) {
        for(Size i=0; i<y[batchNum].size(); i++) {
            if(y[batchNum][i]<0) y[batchNum][i] = 0;
        }
    }

    void softmax(Size batchNum) {
        double exponentSum = 0;
        for(Size i=0; i<y[batchNum].size(); ++i) {
            exponentSum += exp(y[batchNum][i]);
            y[batchNum][i] = exp(y[batchNum][i]); // softmax fct. for output layer
        }
        for(Size i=0; i<y[batchNum].size(); ++i) {
            y[batchNum][i] /= exponentSum;
        }
    }

    void sigmoid_backpropagate(Size batchNum) {
        for(Size i=0; i<nextNeuronNum; ++i) {
            delta[batchNum][i] *= y[batchNum][i] * (1-y[batchNum][i]);
        }
    }

    void ReLU_backpropagate(Size batchNum) {
        for(Size i=0; i<nextNeuronNum; ++i) {
            if(y[batchNum][i]<0) delta[batchNum][i] *= 0;
        }
    }

    void forwardPropagate(vector<vector<double>>& previousLayer, Size batchNum) {
        for(Size i=0; i<nextNeuronNum; ++i) {
            y[batchNum][i] = b[i];
            for(Size j=0; j<previousNeuronNum; ++j) {
                y[batchNum][i] += w[i][j] * previousLayer[batchNum][j];
            }
        }
    }
    
    void forwardPropagate_test(vector<double>& previousLayer) {
        for(Size i=0; i<nextNeuronNum; ++i) {
            y[0][i] = b[i];
            for(Size j=0; j<previousNeuronNum; ++j) {
                y[0][i] += w[i][j] * previousLayer[j];
            }
        }
    }

    void backPropagate_softmax_loglikelihood(Size batchNum, vector<vector<double>>& XOR_label) {
        for(Size i=0; i<y[batchNum].size(); ++i) {
            delta[batchNum][i] = y[batchNum][i] - XOR_label[batchNum][i]; // negative log-likelihood
            // LOG-LIKELIHOOD energy function and SOFTMAX activation function.
        }
    }

    void backPropagate_loglikelihood(Size batchNum, vector<vector<double>>& nextW ,vector<vector<double>>& nextDelta) {
        for(Size i=0; i<nextNeuronNum; ++i) {
            for(Size m=0; m<nextDelta[batchNum].size(); ++m) {
                delta[batchNum][i] += nextW[m][i] * nextDelta[batchNum][m];
            }
            // delta[batchNum][i] *= differentialOfSigmoid(y[i]);
            // delta[batchNum][i] *= y[i] * (1-y[i]);
            // LOG-LIKELIHOOD energy function and SIGMOID activation function.
        }
    }

    void updateWB(Size batchNum, vector<vector<double>>& previousLayer) {
        double learningRate = 1e-2;
        for(Size i=0; i<nextNeuronNum; ++i) {
            for(Size j=0; j<previousLayer[batchNum].size(); ++j) {
                w[i][j] -= learningRate * previousLayer[batchNum][j] * delta[batchNum][i];
            }
            b[i] -= learningRate * delta[batchNum][i];
            delta[batchNum][i] = 0;
        }
    }   
  
    double calculateError(vector<vector<double>>& XOR_label, Size batchNum) {
        double error=0;
        for(Size j=0; j<XOR_label[batchNum].size(); j++){
            // error += 1./4.*(XOR_ith_label[j]-outputLayer[j])*(XOR_ith_label[j]-outputLayer[j]); // MSE
            error -= XOR_label[batchNum][j]*log(y[batchNum][j]); // negative log-likelihood
        }
        return error;
    }
    
    double calculateError_test(vector<double>& XOR_ith_label) {
        double error=0;
        for(Size j=0; j<XOR_ith_label.size(); j++){
            // error += 1./4.*(XOR_ith_label[j]-outputLayer[j])*(XOR_ith_label[j]-outputLayer[j]); // MSE
            error -= XOR_ith_label[j]*log(y[0][j]); // negative log-likelihood
        }
        return error;
    }

    Size isItCorrect(vector<double>& XOR_ith_label) {
        if( y[0][0]<y[0][1] ) {
            if( XOR_ith_label[1] == 1) return 1;
            else return 0;
        }
        else {
            if( XOR_ith_label[0] == 1) return 1;
            else return 0;
        }
    }
};

namespace XOR{
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
}

namespace FNN{
    void TestFNN(Layer& layer1, Layer& layer2, 
        vector<vector<double>>& XOR_input_test, vector<vector<double>>& XOR_label_test, 
        Size epoch, ofstream& os){
        Size correct = 0;
        for(Size i=0; i<XOR_input_test.size(); ++i) { 
            layer1.forwardPropagate_test(XOR_input_test[i]); 
            layer1.sigmoid(0);
            // layer1.ReLU();
            layer2.forwardPropagate_test(layer1.getY()[0]); 
            layer2.softmax(0);
            correct += layer2.isItCorrect(XOR_label_test[i]);
        }
        os << epoch << '\t' << (double)correct/XOR_input_test.size() << endl;
    }
    
    
    void ValidateFNN(Layer& layer1, Layer& layer2, 
        vector<vector<double>>& XOR_input_validate, vector<vector<double>>& XOR_label_validate, 
        Size epoch, ofstream& os){
        double error=0;
        for(Size i=0; i<XOR_input_validate.size(); ++i) { 
            layer1.forwardPropagate_test(XOR_input_validate[i]); 
            layer1.sigmoid(0);
            // layer1.ReLU();
            layer2.forwardPropagate_test(layer1.getY()[0]); 
            layer2.softmax(0);
            error += layer2.calculateError_test(XOR_label_validate[i]);
        }
        os << error/XOR_input_validate.size() << endl;
    }
    
    
    void TrainFNN(Layer& layer1, Layer& layer2, 
        vector<vector<double>>& XOR_pickedInput, vector<vector<double>>& XOR_pickedLabel, 
        vector<vector<double>>& XOR_input, vector<vector<double>>& XOR_label, 
        const Size& trainingNum, const Size& batchSize) {
    
        double error=0;
        for(Size trial=0; trial<trainingNum/batchSize; ++trial) {
            XOR::pickInputForOneBatch(XOR_input, XOR_pickedInput, XOR_label, XOR_pickedLabel);
            for(Size i=0; i<batchSize; ++i) { // train the number of the XOR_pickedInput
                layer1.forwardPropagate(XOR_pickedInput, i);
                layer1.sigmoid(i);
                // layer1.ReLU();
                layer2.forwardPropagate(layer1.getY(), i);
                layer2.softmax(i);
                layer2.backPropagate_softmax_loglikelihood(i, XOR_pickedLabel);
                layer1.backPropagate_loglikelihood(i, layer2.getW() ,layer2.getDelta());
                layer1.sigmoid_backpropagate(i);
                // layer1.ReLU_backpropagate(i);
                error += layer2.calculateError(XOR_pickedLabel, i);
            }
            for(Size i=0; i<batchSize; ++i) {
                layer1.updateWB(i, XOR_pickedInput);
                layer2.updateWB(i, layer1.getY());
            }
        }
        cout << error/trainingNum << endl;
        // os << error/trainingNum << endl;
    }
}



int main(int, char** argv){
    const Size inputNeuronNum = stoul( argv[1] );
    const Size hiddenNeuronNum = stoul( argv[2] );
    const Size outputNeuronNum = stoul( argv[3] );
    const Size trainingNum = stoul( argv[4] );
    const Size testNum = stoul( argv[5] );
    Size batchSize = 50;
    vector<vector<double>> XOR_input;
    vector<vector<double>> XOR_label;
    vector<vector<double>> XOR_pickedInput(batchSize);
    vector<vector<double>> XOR_pickedLabel(batchSize);
    XOR::generateInputAndLabel(XOR_input, XOR_label, inputNeuronNum, trainingNum);
    
    Layer layer1(inputNeuronNum, hiddenNeuronNum, batchSize);
    Layer layer2(hiddenNeuronNum, outputNeuronNum, batchSize);

    ostringstream oss_test;
    oss_test << "FNN_" << inputNeuronNum << "_" << hiddenNeuronNum << "_" << outputNeuronNum << "_" << trainingNum << "_" << testNum << ".table";
    ofstream os_test("/pds/pds181/jmj/ML/FNN/Result/"+oss_test.str());
    for(Size epoch=0; epoch<2000; ++epoch) {
        FNN::TrainFNN(layer1, layer2, XOR_pickedInput, XOR_pickedLabel, XOR_input, XOR_label, trainingNum, batchSize);

        vector<vector<double>> XOR_input_test;
        vector<vector<double>> XOR_label_test;
        XOR::generateInputAndLabel(XOR_input_test, XOR_label_test, inputNeuronNum, testNum);
        FNN::TestFNN(layer1, layer2, XOR_input_test, XOR_label_test, epoch, os_test);
    }
    return 0;
}
