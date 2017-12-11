#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#define EPS 0.0001

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.6;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5;

    //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation  eposition2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

    // initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;

    // time when the state is true, in us
    time_us_ = 0.0;

    // state dimension
    n_x_ = x_.size();

    // Augmented state dimension
    n_aug_ = n_x_+2;

    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;

    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //create vector for weights
    weights_ = VectorXd(2 * n_aug_ + 1);

    // the current NIS for radar
    NIS_radar_ = 0.0;

    // the current NIS for laser
    NIS_laser_ = 0.0;

    //Initialise lidar measurement noise covariance matrix
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << std_laspx_*std_laspx_, 0,
                0, std_laspy_*std_laspy_;

    //Initialise radar measurement noise covariance matrix
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << std_radr_*std_radr_, 0,                                           0,
                 0,                   std_radphi_*std_radphi_,                     0,
                 0,                   0,                                           std_radrd_*std_radrd_;
}

UKF::~UKF() {}

void NormalizeAngle(double& phi)
{
    phi = atan2(sin(phi), cos(phi));
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    // CTRV Model, x_ is [px, py, vel, ang, ang_rate]
    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {

        // first measurement
        cout << "Initializing UKF" << endl;

        VectorXd xi(5); //Initial state

        // initialise timestamp
        time_us_ = meas_package.timestamp_;

        // Initialise state covariance matrix.
        P_ <<   MatrixXd::Identity(n_x_,n_x_);


        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

            //set the state with the initial location and zero velocity
            float p_x = meas_package.raw_measurements_[0];
            float p_y = meas_package.raw_measurements_[1];
            float v = 0;
            float si = 0;
            float si_dot = 0;
            xi << p_x, p_y, v, si, si_dot;
            x_ = xi;

        }
        else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */

            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */

            const float rho = meas_package.raw_measurements_[0];
            const float phi = meas_package.raw_measurements_[1];
            const float rho_dot = meas_package.raw_measurements_[2];

            const float p_x = rho * cos(phi);
            const float p_y = rho * sin(phi);
            const float v  = sqrt(rho_dot * cos(phi) * rho_dot * cos(phi) + rho_dot * sin(phi) * rho_dot * sin(phi));
            const float si = 0;
            const float si_dot = 0;

            xi << p_x, p_y, v, si, si_dot;
            x_ = xi;

        }


        // Edge case initial state
        if (fabs(x_(0)) < 0.0001 and fabs(x_(1)) < 0.0001){
            x_(0) = 0.0001;
            x_(1) = 0.0001;
        }

        // Initialize weights
        weights_(0) = lambda_ / (lambda_ + n_aug_);
        for (int i = 1; i < weights_.size(); i++) {
            weights_(i) = 0.5 / (n_aug_ + lambda_);
        }

        // done initializing, no need to predict or update
        is_initialized_ = true;

        return;
    }


    // Ignore RADAR and LASER measurements as per configuration
    if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) ||
        (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)) {

        /*****************************************************************************
        *  Prediction
        ****************************************************************************/
        //Time elapsed between the current and previous measurements
        float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
        time_us_ = meas_package.timestamp_;

        Prediction(dt);

        /*****************************************************************************
        *  Update
        ****************************************************************************/

        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            UpdateLidar(meas_package);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            UpdateRadar(meas_package);
        }
    }
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    /*****************************************************************************
    * Generate Augmented Sigma Points
    ****************************************************************************/
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
    AugmentedSigmaPoints(&Xsig_aug);

    /*****************************************************************************
    *  Predict Sigma Points
    ****************************************************************************/
    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        //extract values for better readability
        const double p_x      = Xsig_aug(0, i);
        const double p_y      = Xsig_aug(1, i);
        const double v        = Xsig_aug(2, i);
        const double yaw      = Xsig_aug(3, i);
        const double yawd     = Xsig_aug(4, i);
        const double nu_a     = Xsig_aug(5, i);
        const double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    /*****************************************************************************
    *  Convert Predicted Sigma Points to Mean/Covariance
    ****************************************************************************/

    //predicted state mean
    x_.fill(0.0);             //******* necessary? *********
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);             //******* necessary? *********
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        NormalizeAngle(x_diff(3));
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_. Calculate the lidar NIS.
    */

    //extract measurements
    VectorXd z = meas_package.raw_measurements_;

    //set measurement dimension
    // lidar mesaures p_x and p_y
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        // extract values for better readability
        const double p_x = Xsig_pred_(0, i);
        const double p_y = Xsig_pred_(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residuals
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R_laser_;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //calculate NIS
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_. Calculate the radar NIS.
    */

    //extract measurements
    VectorXd z = meas_package.raw_measurements_;

    //set measurement dimension
    //radar can measure r, phi, and r_dot
    int n_z = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readability
        const double p_x = Xsig_pred_(0, i);
        const double p_y = Xsig_pred_(1, i);
        const double v   = Xsig_pred_(2, i);
        const double yaw = Xsig_pred_(3, i);

        const double v1 = cos(yaw)*v;
        const double v2 = sin(yaw)*v;

        // measurement model
        const double rho = sqrt(p_x*p_x + p_y*p_y);                       //rho
        const double phi = atan2(p_y, max(EPS,p_x));                       //phi
        const double rho_dot = (p_x*v1 + p_y*v2) / max(EPS,rho);   //rho_dot

        Zsig(0, i) = rho;
        Zsig(1, i) = phi;
        Zsig(2, i) = rho_dot;

    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R_radar_;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);


    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    //calculate NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

}

/**
 * Generates Augmented Sigma Points.
 * @param {MatrixXd*} Xsig_aug_out
 */
void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_aug_out) {

    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance matrix
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0;
    x_aug(n_x_+1) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_,n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_*std_a_;
    P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;

    double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);
    VectorXd sqrt_lambda_n_aug_L;

    for (int i = 0; i< n_aug_; i++)
    {
        sqrt_lambda_n_aug_L = sqrt_lambda_n_aug * L.col(i);
        Xsig_aug.col(i+1)        = x_aug + sqrt_lambda_n_aug_L;
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt_lambda_n_aug_L;
    }

    *Xsig_aug_out = Xsig_aug;

}
