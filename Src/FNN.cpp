#include <iostream>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <RandomNumber.hpp>
#include <cmath>
#include <Eigen/Dense>


using namespace std;
using namespace Snu::Cnrc;
using namespace Eigen;

using Size = unsigned int;
using Time = unsigned int;

double log_function(double val) {
    if ( val == 0 ) return 0;
    else return log(val);
}

class Layer {
private:
    Size previousNeuronNum;
    Size nextNeuronNum;
    Size batchSize;
    MatrixXd W;
    VectorXd B;
    MatrixXd Y;
    MatrixXd Delta;

public:
    Layer(Size _previousNeuronNum_, Size _nextNeuronNum_, Size _batchSize_) 
    : previousNeuronNum(_previousNeuronNum_), nextNeuronNum(_nextNeuronNum_), batchSize(_batchSize_), W(_nextNeuronNum_, (_previousNeuronNum_)), B(_nextNeuronNum_), Y(_nextNeuronNum_, _batchSize_), Delta(_nextNeuronNum_, _batchSize_) {
        initialize();
    }
    
    MatrixXd& getY() {return Y;}
    MatrixXd& getDelta() {return Delta;}
    MatrixXd& getW() {return W;}

    void initialize() {
        RandomGaussianGenerator rndNormal(0, 1);
        W = W.unaryExpr([&](double elem) { // changed type of parameter
            return elem = rndNormal(); // return instead of assignment
        });
        B = B.unaryExpr([&](double elem) { // changed type of parameter
            return elem = rndNormal(); // return instead of assignment
        });
    }

    void sigmoid() {
        Y = (1+Y.array().exp().inverse()).inverse();
    }

    void ReLU() {
        Y = Y.cwiseMax(0);
    }

    void softmax() {
        VectorXd J = VectorXd::Ones(nextNeuronNum);
        Y = Y.array() - (J * Y.array().colwise().maxCoeff().matrix()).array();
        Y = Y.array().exp() / (J * Y.array().exp().colwise().sum().matrix()).array();
    }

    void sigmoid_backpropagate() {
        Delta.array() *= Y.array() * (1.-Y.array());
    }

    void ReLU_backpropagate() {
        Delta.array() *= Y.cwiseMax(0).cwiseSign().array();
    }

    void forwardPropagate(MatrixXd& X) {
        VectorXd J = VectorXd::Ones(batchSize);
        Y = W * X + B * J.transpose();
    }
    
    void backPropagate_softmax_loglikelihood(MatrixXd& label) {
        Delta = Y - label;
    }

    void backPropagate_softmax_crossEntropy(MatrixXd& label) {
        Delta = (Y - label) / Y.rows();
    }

    void backPropagate(MatrixXd& nextW, MatrixXd& nextDelta) {
        Delta = nextW.transpose() * nextDelta;
    }

    void update_WB(MatrixXd& X) {
        // double learningRate = 1e-2;
        double learningRate = 2e-4;
        VectorXd J = VectorXd::Ones(batchSize);
        W -= learningRate * Delta * X.transpose();
        B -= learningRate * Delta * J;
        Delta = MatrixXd::Zero(Delta.rows(), Delta.cols());
    }   
  
    double calculate_loss(MatrixXd& pickedLabel) {
        double error = -(pickedLabel.array()*Y.array().log()).sum(); // log-likelihood
        
        /*
        double error = -(pickedLabel.array()*Y.array().log()).sum(); 
        for(Size i=0; i<Y.rows(); ++i) {
            for(Size j=0; j<Y.cols(); ++j) {
                if(Y(i,j) != 1) {
                    error -= ((1.-pickedLabel(i, j)) * log(1.-Y(i, j)));
                }
            }
        }
        error /= Y.rows(); // cross entropy
        */

        // double error = -((1.-pickedLabel.array()) * (1.-Y.array()).log()).sum() / Y.rows(); // cross entropy
        // if(Y(0,0) == 1) {
            // cout << "Y "<<Y(0,0) << endl;
            // cout << "error "<< (1-pickedLabel(0,0))*((1-Y.array()).log())(0,0) << endl;
            // cout << "error2 "<< (1.-1.)*log_function(1.-1.) << endl;
        // }
        return error;
    }

    double is_it_correct(MatrixXd& label_test, MatrixXd& Y_test) {
        double correct = 0;
        ptrdiff_t pos;
        for(Size j=0; j<Y_test.cols(); ++j) {
            Y_test.col(j).maxCoeff(&pos);
            if( label_test(pos, j) == 1) ++correct;
        }
        return correct;
    }
};

namespace XOR {
    void get_image_and_label(MatrixXd& input, MatrixXd& label,
        const Size& inputNeuronNum, const Size& trainingNum) {
        RandomIntGenerator rndInt(0, 1);
        Size sum;
        VectorXd v(input.rows());
        VectorXd w(label.rows());
        for(Size trialNum=0; trialNum<trainingNum; ++trialNum) {
            sum = 0;
            for(Size i=0; i<inputNeuronNum; ++i) {
                v(i) = rndInt();
                sum += v(i);
            }
            input.col(trialNum) = v;
            if( sum%2==0 ) {
                w(0) = 1, w(1) = 0;
                label.col(trialNum) = w;
            }
            else {
                w(0) = 0, w(1) = 1;
                label.col(trialNum) = w;
            }
        }
    }
}

void pick_image_and_label(MatrixXd& input, MatrixXd& pickedInput,
    MatrixXd& label, MatrixXd& pickedLabel) {
    RandomIntGenerator rndInt(0, input.cols()-1);
    Size selection;
    for(Size i=0; i<pickedInput.cols(); ++i) {
        selection = rndInt();
        pickedInput.col(i) = input.col(selection);
        pickedLabel.col(i) = label.col(selection);
    }
}

namespace FNN {
    void test_FNN(Layer& layer1, Layer& layer2, 
        MatrixXd& input_test, MatrixXd& label_test, 
        const Size& testNum, const Size& batchSize, ofstream& os){
        
        double correct = 0;
        MatrixXd pickedInput(input_test.rows(), batchSize);
        MatrixXd pickedLabel(label_test.rows(), batchSize);
        for(Size trial=0; trial<testNum/batchSize; ++trial) {
            pick_image_and_label(input_test, pickedInput, label_test, pickedLabel);
            layer1.forwardPropagate(pickedInput); 
            // layer1.sigmoid();
            layer1.ReLU();
            layer2.forwardPropagate(layer1.getY()); 
            layer2.softmax();
            correct += layer2.is_it_correct(pickedLabel, layer2.getY());
        }
        os << correct/testNum << endl;
    }

    
    void validate_FNN(Layer& layer1, Layer& layer2, 
        MatrixXd& input_validate, MatrixXd& label_validate,
        const Size& testNum, const Size& batchSize,
        ofstream& os) {

        double error = 0;
        MatrixXd pickedInput(input_validate.rows(), batchSize);
        MatrixXd pickedLabel(label_validate.rows(), batchSize);
        for(Size trial=0; trial<testNum/batchSize; ++trial) {
            pick_image_and_label(input_validate, pickedInput, label_validate, pickedLabel);
            layer1.forwardPropagate(pickedInput); 
            // layer1.sigmoid();
            layer1.ReLU();
            layer2.forwardPropagate(layer1.getY()); 
            layer2.softmax();
            error += layer2.calculate_loss(pickedLabel);
        }
        os << error/testNum << endl;
    }
    
    
    void train_FNN(Layer& layer1, Layer& layer2, 
        MatrixXd& input, MatrixXd& label, 
        const Size& trainingNum, const Size& batchSize) {
    
        double error = 0;
        MatrixXd pickedInput(input.rows(), batchSize);
        MatrixXd pickedLabel(label.rows(), batchSize);
        for(Size trial=0; trial<trainingNum/batchSize; ++trial) {
            pick_image_and_label(input, pickedInput, label, pickedLabel);
            layer1.forwardPropagate(pickedInput);
            // layer1.sigmoid();
            layer1.ReLU();
            layer2.forwardPropagate(layer1.getY());
            layer2.softmax();
            layer2.backPropagate_softmax_loglikelihood(pickedLabel);
            // layer2.backPropagate_softmax_crossEntropy(pickedLabel);
            layer1.backPropagate(layer2.getW(), layer2.getDelta());
            layer1.ReLU_backpropagate();
            error += layer2.calculate_loss(pickedLabel);
            layer1.update_WB(pickedInput);
            layer2.update_WB(layer1.getY());
        }
        cout << error/trainingNum << endl;
    }
}


namespace MNIST {
    MatrixXd get_image(const unsigned int& image_number, const unsigned int& image_size
        , const string& filename, const unsigned int& offset) {
        MatrixXd images(image_size, image_number);
        ifstream ifs(filename, ios::binary);
        vector<char> buffer(image_number * image_size);
        ifs.seekg(offset);
        ifs.read(buffer.data(), buffer.size());
        for(unsigned int i = 0; i < image_number; ++i) {
            vector<double> res;
            for(auto j = buffer.begin() + image_size * i; j < buffer.begin() + image_size * (i + 1); ++j) {
                res.push_back(*reinterpret_cast<unsigned char *>(&(*j)));
            }
            images.col(i) = Map<VectorXd> (res.data(), res.size());
        }
        images /= 255.0;
        return images;
    }
    MatrixXd get_label(const unsigned int& label_number, const unsigned int& label_size
        , const string& filename, const unsigned int& offset) {
        MatrixXd labels = MatrixXd::Zero(label_size, label_number);
        vector<char> buffer(label_number * label_size);
        ifstream ifs(filename, ios::binary);
        ifs.seekg(offset);
        ifs.read(buffer.data(), buffer.size());
        for(unsigned int i = 0; i < label_number; ++i) {
            labels(*reinterpret_cast<unsigned char *>(&(buffer[i])), i) = 1.0;
        }
        return labels;
    }
}

int main(int, char** argv){
    const Size inputNeuronNum = stoul( argv[1] );
    const Size hiddenNeuronNum = stoul( argv[2] );
    const Size outputNeuronNum = stoul( argv[3] );
    const Size trainingNum = stoul( argv[4] );
    const Size testNum = stoul( argv[5] );
    Size batchSize = 100;
    MatrixXd input(inputNeuronNum, trainingNum);
    MatrixXd label(outputNeuronNum, trainingNum);
    MatrixXd input_test(inputNeuronNum, testNum);
    MatrixXd label_test(outputNeuronNum, testNum);
    
    input = MNIST::get_image(trainingNum, inputNeuronNum, "train-images-idx3-ubyte", 4 * 4);
    label = MNIST::get_label(trainingNum, outputNeuronNum, "train-labels-idx1-ubyte", 4 * 2);
    input_test = MNIST::get_image(testNum, inputNeuronNum, "t10k-images-idx3-ubyte", 4 * 4);
    label_test = MNIST::get_label(testNum, outputNeuronNum, "t10k-labels-idx1-ubyte", 4 * 2);

    Layer layer1(inputNeuronNum, hiddenNeuronNum, batchSize);
    Layer layer2(hiddenNeuronNum, outputNeuronNum, batchSize);

    ostringstream oss_test;
    oss_test << "FNN_Eigen_" << inputNeuronNum << "_" << hiddenNeuronNum << "_" << outputNeuronNum << "_" << trainingNum << "_" << testNum << ".table";
    ofstream os_test("/pds/pds181/jmj/ML/FNN/Result/"+oss_test.str());

    for(Size epoch=0; epoch<2000; ++epoch) {
        FNN::train_FNN(layer1, layer2, input, label, trainingNum, batchSize);
        // FNN::validate_FNN(layer1, layer2, input_test, label_test, testNum, batchSize, os_validate);
        FNN::test_FNN(layer1, layer2, input_test, label_test, testNum, batchSize, os_test);
    }
    return 0;
}
