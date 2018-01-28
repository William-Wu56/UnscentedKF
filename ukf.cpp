#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#define PI 3.14159265

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  
  // Start UKF Instance and wait for measurement to set is_initialized to true
  is_initialized_ = false;
  time_us_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.75;        // Need TUNING -Assuming 1.5 m/s^2

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.75;    //0.75; // Need TUNING -Assuming 1.5 rad/s^2
    
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
    
  // State dimension [px,py,v,yaw,yawdot]
  n_x_ = 5;
    
  // Augmented state dimension [px,py,v,yaw,yawdot,noise_a,noise_yaw]
  n_aug_ = 7;
    
  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;
    
  // Weights Vector
  weights_ = VectorXd(2*n_aug_ + 1);
  weights_.fill(1/(2*(lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_+n_aug_);
    
  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  x_ = VectorXd::Zero(n_x_);
    
  // state covariance matrix --> nx . nx: 5x5
  P_ = MatrixXd::Identity(n_x_,n_x_);
    
  // predicted sigma points matrix --> nx . 2*n_aug + 1: 5x15
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2*n_aug_+1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    
    // FIRST MEASUREMENT
    if (!is_initialized_)
    {
        if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            x_.head(2) = meas_package.raw_measurements_; // [px,py,0,0,0]
            P_(0,0) = 0.15;
            P_(1,1) = 0.15;
            P_(2,2) = 50.0;
            P_(3,3) = 50.0;
            P_(4,4) = 50.0;
        }
        
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
//            double rhodot = meas_package.raw_measurements_(2);
            
            x_(0) = rho * cos(phi);
            x_(1) = rho * sin(phi);
            P_(0,0) = 0.15;
            P_(1,1) = 0.15;
            P_(2,2) = 50.0;
            P_(3,3) = 50.0;
            P_(4,4) = 50.0;
        }
        
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        
        return;
    }
    
    if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) return;
    if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) return;
    
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    
    Prediction(delta_t);
    
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        UpdateRadar(meas_package);
    }
    
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /*
   * Estimate the object's location.
   * Modify the state vector, x_.
   * Predict sigma points, the state, and the state covariance matrix.
   */
    
    // ALREADY INITIALIZED
    // Augmented State x_aug
    VectorXd x_aug(n_aug_);
    x_aug.fill(0.0);
    x_aug.head(n_x_) = x_;
    
    MatrixXd P_aug(n_aug_,n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_,n_x_) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;
    
    // Augmented Stata - Sigma Points
    MatrixXd Xsig_aug(n_aug_, 2*n_aug_+1);
    Xsig_aug.fill(0.0);
    Xsig_aug.col(0) = x_aug;
    
    MatrixXd sqrtP_aug = P_aug.llt().matrixL();
    
    for (int i=0; i < n_aug_; i++)
    {
        Xsig_aug.col(i+1) = x_aug + (sqrt(lambda_+n_aug_) * sqrtP_aug.col(i));
        Xsig_aug.col(n_aug_+i+1) = x_aug - (sqrt(lambda_+n_aug_) * sqrtP_aug.col(i));
    }
    
    // Predict Sigma Points (n_x . 2*n_aug+1)
    Xsig_pred_.fill(0.0);
    for (int i=0; i < Xsig_aug.cols(); i++)
    {
        double px = Xsig_aug.col(i)(0);
        double py = Xsig_aug.col(i)(1);
        double v = Xsig_aug.col(i)(2);
        double yaw = Xsig_aug.col(i)(3);
        double yawdot = Xsig_aug.col(i)(4);
        double nu_a = Xsig_aug.col(i)(5);
        double nu_yaw = Xsig_aug.col(i)(6);
        
        // Calculate the Prediction Integral
        VectorXd integral(n_x_);
        integral.fill(0.0);
        // Check for yawdot = 0; Means that the car is moving in a straight line. NOT TURNING-
        if (fabs(yawdot) < 0.001)
        {
            integral <<  v * cos(yaw) * delta_t,
                         v * sin(yaw) * delta_t,
                                   0,
                                   0,//yawdot * delta_t,
                                   0;
        }
        else
        {
            integral <<  (v* ( sin(yaw+(yawdot*delta_t)) - sin(yaw) ) )/yawdot,
                         (v* (-cos(yaw+(yawdot*delta_t)) + cos(yaw) ) )/yawdot,
                                                 0,
                                          yawdot * delta_t,
                                                 0;
        }
        
        // Calculate the Noise Influence on the Predicted State
        VectorXd influence(n_x_);
        influence.fill(0.0);
        influence <<    (pow(delta_t,2)*cos(yaw)*nu_a)/2,
                        (pow(delta_t,2)*sin(yaw)*nu_a)/2,
                        delta_t * nu_a,
                        (pow(delta_t,2)*nu_yaw)/2,
                        delta_t * nu_yaw;
        
        Xsig_pred_.col(i) = Xsig_aug.col(i).head(5) + integral + influence;
    }
    
    // Get Predicted Sigma Points Mean
    x_.fill(0.0);
    P_.fill(0.0);
    
    x_ = Xsig_pred_ * weights_;
    
    for (int i=0; i < Xsig_pred_.cols(); i++)
    {
        VectorXd diff = Xsig_pred_.col(i) - x_;
        while (diff(3) < -PI) diff(3) += 2*PI;
        while (diff(3) > PI) diff(3) -= 2*PI;
        
        P_ += weights_(i) * (diff * diff.transpose());
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /*
   * Use lidar data to update the belief about the object's position.
   * Modify the state vector, x_, and covariance, P_.
   * Calculate the lidar NIS.
   */
   
    int n_laser = 2;       // LIDAR measurements are in 2-Dimensions.
    // Transform Sigma Points Xsig_pred_ to Measurement(LIDAR) Space.
    MatrixXd Zsig(n_laser, 2*n_aug_ + 1);
    Zsig.fill(0.0);
    
    MatrixXd F(n_laser, n_x_);
    F.fill(0.0);
    F(0,0) = 1.0;
    F(1,1) = 1.0;
    
    Zsig = F * Xsig_pred_;
    
    // Calculate Predicted State Mean in Measurement Space
    VectorXd z_ = Zsig * weights_;
    
    // Calculate Predicted State Covariance in Measurement Space
    MatrixXd S(n_laser,n_laser);
    S.fill(0.0);
    
    for (int i=0; i < Zsig.cols(); i++)
    {
        VectorXd diff = Zsig.col(i) - z_;
        S += weights_(i) * (diff*diff.transpose());
    }
    
    // Measurement Noise Covariance LIDAR
    MatrixXd R(n_laser,n_laser);
    R.fill(0.0);
    R(0,0) = std_laspx_*std_laspx_;
    R(1,1) = std_laspy_*std_laspy_;
    
    S += R;
    
    // Calculate Cross-Correlation Matrix between Sigma Points in State and Measurement Spaces.
    MatrixXd T(n_x_,n_laser);
    T.fill(0.0);
    
    for (int i=0; i < Zsig.cols(); i++)
    {
        VectorXd Xdiff = Xsig_pred_.col(i) - x_;
        while(Xdiff(3) < -PI) Xdiff(3) += 2*PI;
        while(Xdiff(3) > PI) Xdiff(3) -= 2*PI;
        
        VectorXd Zdiff = Zsig.col(i) - z_;
        
        T += weights_(i) * (Xdiff*Zdiff.transpose());
    }
    
    // Calculate Kalman Gain
    MatrixXd K = T * S.inverse();
    
    // Calculate Error
    VectorXd y = meas_package.raw_measurements_ - z_;
    
    // UPDATE STATE
    x_ += K * y;
    P_ -= K * S * K.transpose();
    
    // CALCULATE NIS
    nis_lidar_ = y.transpose() * S.inverse() * y;
//    cout << nis_lidar_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /*
    * Use radar data to update the belief about the object's
      position.
    * Modify the state vector, x_, and covariance, P_.
    * Calculate the radar NIS.
  */
    int n_radar = 3;    // RADAR measurements are in 3-Dimensions.
    // Transform Sigma Points Xsig_pred_ to Measurement(RADAR) Space.
    MatrixXd Zsig(n_radar, 2*n_aug_+1);
    Zsig.fill(0.0);
    
    for (int i=0; i < Xsig_pred_.cols(); i++)
    {
        double px = Xsig_pred_.col(i)(0);
        double py = Xsig_pred_.col(i)(1);
        double v = Xsig_pred_.col(i)(2);
        double yaw = Xsig_pred_.col(i)(3);
        
        if (fabs(px) < 0.001) px = 0.001;
        
        Zsig.col(i) <<  sqrt(pow(px,2) + pow(py,2)),
                        atan2(py,px),
                        (px*cos(yaw)*v + py*sin(yaw)*v)/(sqrt(pow(px,2) + pow(py,2)));
    }
    
    // Predicted Mean State in Measurement Space
    VectorXd z_ = Zsig * weights_;
    // Predicted State Covariance in Measurement Space
    MatrixXd S(n_radar,n_radar);
    S.fill(0.0);
    
    for (int i=0; i < Zsig.cols(); i++)
    {
        VectorXd diff = Zsig.col(i) - z_;
        while(diff(1) < -PI) diff(1) += 2*PI;
        while(diff(1) > PI) diff(1) -= 2*PI;
        
        S += weights_(i) * (diff * diff.transpose());
    }
    
    // Measurement Noise Covariance Matrix
    MatrixXd R(n_radar, n_radar);
    R.fill(0.0);
    R(0,0) = std_radr_*std_radr_;
    R(1,1) = std_radphi_*std_radphi_;
    R(2,2) = std_radrd_*std_radrd_;
    
    S += R;
    // Cross-Correlation Matrix T between state and measurement spaces
    MatrixXd T(n_x_,n_radar);
    T.fill(0.0);
    
    for (int i=0; i < Zsig.cols(); i++)
    {
        VectorXd Xdiff = Xsig_pred_.col(i) - x_;
        while(Xdiff(3) < -PI) Xdiff(3) += 2*PI;
        while(Xdiff(3) > PI) Xdiff(3) -= 2*PI;
        
        VectorXd Zdiff = Zsig.col(i) - z_;
        while(Zdiff(1) < -PI) Zdiff(1) += 2*PI;
        while(Zdiff(1) > PI) Zdiff(1) -= 2*PI;
        
        T += weights_(i) * (Xdiff * Zdiff.transpose());
    }
    // Calculate Kalman Gain
    MatrixXd K = T * S.inverse();
    // Calculate Error
    VectorXd y = meas_package.raw_measurements_ - z_;
    while(y(1) < -PI) y(1) += 2*PI;
    while(y(1) > PI) y(1) -= 2*PI;
    
    // UPDATE STATE
    x_ += K * y;
    P_ -= K * S * K.transpose();
    // CALCULATE NIS
    nis_radar_ = y.transpose() * S.inverse() * y;
//    cout << nis_radar_ << endl;
}
