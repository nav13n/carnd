#include "FusionEKF.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    noise_ax = 9.0;
    noise_ay = 9.0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
        * Initialize the ekf state with first measurement.
        * Create the covariance matrix.
        * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        // first measurement
        cout << "Initializing Fusion EKF" << endl;

        VectorXd xi(4); //Initial state

        // Initialise state covariance matrix
        // Assign greater uncertainties to initial velocities
        MatrixXd P(4, 4);
        P <<    1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;

        // Initialize state transition matrix
        MatrixXd F(4, 4);
        F <<    1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;

        MatrixXd Q(4,4);

        previous_timestamp_ = measurement_pack.timestamp_;
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */

            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float rho_dot = measurement_pack.raw_measurements_[2];

            float x = rho * cos(phi);
            float y = rho * sin(phi);
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);



            // Initialize measurement matrix for laser
            Hj_ <<  1, 1, 0, 0,
                    1, 1, 0, 0,
                    1, 1, 1, 1;

            //Initialise EKF with first state vector
            ekf_.Init(xi, P, F, Hj_, R_radar_, Q);

        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            //set the state with the initial location and zero velocity
            float x = measurement_pack.raw_measurements_[0];
            float y = measurement_pack.raw_measurements_[1];
            float vx = 0;
            float vy = 0;
            xi << x, y, vx, vy;

            // Initialize measurement matrix for laser
            H_laser_ << 1, 0, 0, 0,
                        0, 1, 0, 0;

            //Initialise EKF with first state vector
            ekf_.Init(xi, P, F, H_laser_, R_laser_, Q);

        }

        // Edge case initial state
        if (fabs(ekf_.x_(0)) < 0.0001 and fabs(ekf_.x_(1)) < 0.0001){
            ekf_.x_(0) = 0.0001;
            ekf_.x_(1) = 0.0001;
        }

        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
     * Update the state transition matrix F according to the new elapsed time.
       - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    //compute the time elapsed between the current and previous measurements
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

    if(dt > 0.0) {
        float dt_2 = dt * dt;
        float dt_3 = dt_2 * dt;
        float dt_4 = dt_3 * dt;

        //Modify the F matrix so that the time is integrated
        ekf_.F_(0, 2) = dt;
        ekf_.F_(1, 3) = dt;

        //Update the process covariance matrix Q
        ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
                0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
                dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
                0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

        ekf_.Predict();

        /*****************************************************************************
         *  Update
         ****************************************************************************/

        /**
         * Use the sensor type to perform the update step.
         * Update the state and covariance matrices.
         */

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // Radar updates
            Hj_ = tools.CalculateJacobian(ekf_.x_);
            ekf_.H_ = Hj_;
            ekf_.R_ = R_radar_;
            ekf_.UpdateEKF(measurement_pack.raw_measurements_);
        } else {
            // Laser updates
            ekf_.H_ = H_laser_;
            ekf_.R_ = R_laser_;
            ekf_.Update(measurement_pack.raw_measurements_);
        }

        // print the output
        cout << "x_ = " << ekf_.x_ << endl;
        cout << "P_ = " << ekf_.P_ << endl;
    };
}
