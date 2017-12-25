/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Need to be careful with this, impact algorithm performance
	num_particles = 100;

    // Initializes a random generator engine for gaussian sampling
    default_random_engine gen;

     //Initialise gaussian noise for sensor parameters. Mean zero and standard deviation as specified
     normal_distribution<double> dist_x(0, std[0]);
     normal_distribution<double> dist_y(0, std[1]);
     normal_distribution<double> dist_theta(0, std[2]);


     //Initialise particles
     for (int i = 0; i < num_particles; i++) {
         Particle p;
         p.id = i;
         p.x = x;
         p.y = y;
         p.theta = theta;
         p.weight = 1.0;

         // add gaussian sensor noise
         p.x += dist_x(gen);
         p.y += dist_y(gen);
         p.theta += dist_theta(gen);
         particles.push_back(p);
     }

     is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Initializes a random generator engine for gaussian sampling
    default_random_engine gen;

	// Define gaussian distributions for sensor noise
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {

        // Predict new state based on bicycle motion model
        if (fabs(yaw_rate) < 0.00001) {
          particles[i].x += velocity * delta_t * cos(particles[i].theta);
          particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else {
          particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
          particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
          particles[i].theta += yaw_rate * delta_t;
        }

        // Add gaussian noise to new state
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	int observation_size = observations.size();
	int predictions_size = predicted.size();

	for (unsigned int i = 0; i < observation_size; i++) {

        // Get current observation
        LandmarkObs o = observations[i];

        // Initialise with max distance possible
        double min_dist = numeric_limits<double>::max();

        int min_j = -1;

        for (unsigned int j = 0; j < predictions_size; j++) {
          // Get current prediction
          LandmarkObs p = predicted[j];

          // Compute euclidean distance between current and predicted landmarks
          double cur_dist = dist(o.x, o.y, p.x, p.y);

          // Associate the predicted landmark nearest to the current observed landmark
          if (cur_dist < min_dist) {
            min_dist = cur_dist;
            min_j = j;
          }
        }
        observations[i].id = min_j;
    }

}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// STEPS:
    //  1 - For each particle, convert measurements to map coordinate
    //  2 - For each observation, find the closest landmark and compute error with respect to observation
    //  4 - Accumulate error for particle and assign as weight

    // Initialise the weight vector
    weights.clear();

    for (int i = 0; i < num_particles; i++) {

        // Get all possible landmarks in current particle sensor range
        vector<LandmarkObs> predictions;
        predictions = getMapLandmarksInParticleRange(particles[i], sensor_range, map_landmarks);

        // Convert each observation to map coordinates
        vector<LandmarkObs> transformed_observations;
        for (unsigned int j = 0; j < observations.size(); j++) {
          const LandmarkObs transformed_obs = toMapCoords(particles[i], observations[j]);
          transformed_observations.push_back(transformed_obs);
        }

        // Associate each observation with its closest landmark
        dataAssociation(predictions, transformed_observations);

        // Initialise particle weight
        double particle_weight = 1.0;

        // For each observation, find the closest landmark and compute error with respect to observation
        for (unsigned int j = 0; j < transformed_observations.size(); j++) {
           const LandmarkObs associated_prediction = predictions[transformed_observations[j].id];
           particle_weight *= calculateWeights(transformed_observations[j], associated_prediction, std_landmark);
        }

        //Update weights
        particles[i].weight = particle_weight;
        weights.push_back(particle_weight);
    }
}

// Filters landmarks in the sensor range of the particle from the list of all landmarks
std::vector<LandmarkObs> ParticleFilter::getMapLandmarksInParticleRange( Particle p, double sensor_range, const Map &map_landmarks) {

  vector<LandmarkObs> filtered_landmarks;

  for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        // get id and x,y coordinates
        float lm_x = map_landmarks.landmark_list[j].x_f;
        float lm_y = map_landmarks.landmark_list[j].y_f;
        int lm_id = map_landmarks.landmark_list[j].id_i;

        if (fabs(lm_x - p.x) <= sensor_range && fabs(lm_y - p.y) <= sensor_range) {

          // add prediction to vector
          filtered_landmarks.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
        }
  }
  return filtered_landmarks;
}

// Maps observation to map coordinate system from vehicle coordinate system
LandmarkObs ParticleFilter::toMapCoords(Particle p, LandmarkObs obs) {

  LandmarkObs transformed_obs;

  transformed_obs.id = obs.id;
  transformed_obs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
  transformed_obs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;

  return transformed_obs;
}

// Calculates weight using predicted position and the ground truth position as a multi variate gaussian
double ParticleFilter::calculateWeights(LandmarkObs obs, LandmarkObs pred, double std_landmark[]) {

  // storing for readability
  const double sigma_x = std_landmark[0];
  const double sigma_y = std_landmark[1];
  const double d_x = obs.x - pred.x;
  const double d_y = obs.y - pred.y;

  double e = (1/(2. * M_PI * sigma_x * sigma_y)) * exp(-((d_x * d_x / (2 * sigma_x * sigma_x)) + (d_y * d_y / (2 * sigma_y * sigma_y))));

  return e;
}


void ParticleFilter::resample() {
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> sampled_particles;
    int index;

    // Sample according to weight with replacements
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<int> weight_distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++) {
        index = weight_distribution(gen);
        sampled_particles.push_back(particles[index]);
    }
    particles = sampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
