#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <RandomNumber.hpp>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

// using Eigen::MatrixXd;
using namespace std;
using namespace Eigen;

using Size = unsigned int;

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
			// if(i==0) cout << images.col(i) << endl;
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
			labels(*reinterpret_cast<unsigned char *>(&(buffer[i])), i) = 1.0; // one-hot encoding
		}
		return labels;
	}
}

int main() {
	Size training_set_size = 54000;
	Size image_size = 28*28;
	Size label_size = 10;
	ostringstream oss;
    oss << "MNIST.table";
    ofstream os("/pds/pds181/jmj/ML/FNN/"+oss.str());
    
	MatrixXd training_set_patterns = MNIST::get_image(training_set_size, image_size, "train-images-idx3-ubyte", 4 * 4);
	MatrixXd training_set_keys = MNIST::get_label(training_set_size, label_size, "train-labels-idx1-ubyte", 4 * 2);
	// VectorXd a = Map<VectorXd> (training_set_patterns.col(0).data(), training_set_patterns.col(0).size());
	RowVectorXd a = Map<VectorXd> (training_set_patterns.col(0).data(), training_set_patterns.col(0).size());
	// MatrixXd b = Map<MatrixXd> (a.data(), a.size());
	// cout << a<< endl;
	// Map<Matrix<double, 100, 10>> a_eigen(a.data())
	// os << Map<Matrix<double, 28, 28>> (a.data()) << endl;
	cout << training_set_patterns.col(0) << endl;
	// cout << Map<Matrix<unsigned char, 28, 28>> (training_set_patterns.col(0).data(), 28) << endl;

	cout << training_set_keys.col(0) << endl;
	return 0;
}
