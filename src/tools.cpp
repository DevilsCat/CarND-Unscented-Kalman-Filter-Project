#include <iostream>
#include "tools.h"
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if(estimations.size() != ground_truth.size()
         || estimations.size() == 0) {
      std::cout << "Invalid estimation or ground_truth data" << std::endl;
      return rmse;
  }

  // Accumulate squared residuals
  for (auto it_estimate = estimations.begin(), it_truth = ground_truth.begin();
          it_estimate != estimations.end() && it_truth != ground_truth.end();
          ++it_estimate, ++it_truth) {
      VectorXd residual = *it_estimate - *it_truth;

      // Coefficient-wise multiplication
      residual = residual.array() * residual.array();
      rmse += residual;
  }

  // Calculate the mean
  rmse /= estimations.size();

  // Calculate the squared root
  return rmse.array().sqrt();
}

float Tools::normalizeAngle(const float& angle) {
  float ret = angle;

  while (ret > M_PI) ret -= 2. * M_PI;
  while (ret < -M_PI) ret += 2. * M_PI;

  return ret;
}