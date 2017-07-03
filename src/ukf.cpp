#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  initializeSigmaPointsWeights_(n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    // Initialize the state and convariance matrix
    x_ = VectorXd(5);

    float px, py;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates.
      float rho = meas_package.raw_measurements_[0];
      float theta = meas_package.raw_measurements_[1];
      px = rho * cos(theta);
      py = rho * sin(theta);
    } else {
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
    }
    x_ << px, py, 0, 0, 0;

    P_ = MatrixXd::Identity(5, 5);

    time_us_ = meas_package.timestamp_;

    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // Compute the time elapsed between the current and previous measurements
  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  // Use the sensor type to perform the update step.
  // UPdate the state and covariance matrices.
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location. Modify the state vector, x_. 
   * Predict sigma points, the state, and the state covariance matrix.
   */
  MatrixXd Xsig_aug;
  GenerateAugmentedSigmaPoints_(Xsig_aug);

  PredictSigmaPoints_(Xsig_aug, delta_t);

  PredictMeanAndCovirance_();
}

void UKF::initializeSigmaPointsWeights_(const int& n_aug) {
  weights_ = VectorXd(2 * n_aug + 1);
  // Set weights.
  weights_(0) = lambda_ / (lambda_ + n_aug);
  for (int i = 1; i < 2 * n_aug + 1; ++i) {
    double weight = 0.5 / (n_aug + lambda_);
    weights_(i) = weight;
  }
}

void UKF::GenerateAugmentedSigmaPoints_(MatrixXd& Xsig_aug_out) {
  int nu_a_idx = n_x_;
  int nu_yawdd_idx = n_x_ + 1;

  // Create augmented mean vetor.
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(nu_a_idx) = 0;
  x_aug(nu_yawdd_idx) = 0;

  // Create augmented state covariance.
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(nu_a_idx, nu_a_idx) = std_a_ * std_a_;
  P_aug(nu_yawdd_idx, nu_yawdd_idx) = std_yawdd_ * std_yawdd_;

  // Calculate square root of P.
  MatrixXd L = P_aug.llt().matrixL();

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Create augmented sigma points.
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  Xsig_aug_out = Xsig_aug;
}

void UKF::PredictSigmaPoints_(const MatrixXd& Xsig_aug, const float& delta_t) {
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Extract value for better readability.
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // Predicted state values.
    double px_p, py_p;

    // Avoid division by zero.
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yaw * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yaw * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yaw * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p += nu_a * delta_t;
    yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p += nu_yawdd * delta_t;

    // Write  predicted sigma point into right column.
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovirance_() {
  // Predict state mean.
  x_ = Xsig_pred_ * weights_;

  // Predict state covirance matrix.
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;  // state difference
    x_diff(3) = tools_.normalizeAngle(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Complete this function! Use radar data to update the belief about the object's
   * position. Modify the state vector, x_, and covariance, P_.
   * 
   * You'll also need to calculate the radar NIS.
   */
  VectorXd z_pred;
  MatrixXd S_pred;
  MatrixXd Zsig;
  PredictRadarMeasurement_(z_pred, S_pred, Zsig);

  UpdateState_(meas_package.raw_measurements_, z_pred, Zsig, S_pred);
}

void UKF::PredictRadarMeasurement_(
    VectorXd& z_out, MatrixXd& S_out, MatrixXd& Zsig_out) {
  int n_z = 3;
  double lambda = 3 - n_aug_;

  // Create matrix for sigma points in measurement space.
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space.
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Extract value for better readability.
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
  }

  // mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = tools_.normalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix.
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S += R;

  z_out = z_pred;
  S_out = S;
  Zsig_out = Zsig;
}

void UKF::UpdateState_(
    const VectorXd& z_measure, 
    const VectorXd& z_pred, 
    const MatrixXd& Zsig,
    const MatrixXd& S_pred) {
  int n_z = 3;

  // Create matrix for cross correlation.
  MatrixXd Tc = MatrixXd(n_x_,  n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = tools_.normalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools_.normalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //  Kalman gain K.
  MatrixXd K = Tc * S_pred.inverse();

  // residual
  VectorXd z_diff = z_measure - z_pred;
  z_diff(1) = tools_.normalizeAngle(z_diff(1));

  // Update state mean and convariance matrix.
  x_ += K * z_diff;
  P_ -= K * S_pred * K.transpose();
}