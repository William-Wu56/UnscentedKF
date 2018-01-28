#include <iostream>
#include "tools.h"
using namespace std;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd residual_sum(4), residual_mean(4), rmse(4);
    residual_sum << 0,0,0,0;
    
    if ((estimations.size() < 1) | (estimations.size() != ground_truth.size())) {
        cout << "Invalid estimation or ground_truth data";
        return residual_sum;
    }
    
    for(int i=0; i < estimations.size(); ++i){
        
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
        residual_sum += residual;
    }
    
    residual_mean = residual_sum/estimations.size();
    rmse = residual_mean.array().sqrt();
    
    return rmse;
}
