/* weighted_select_levels.cpp --- an mixture model algorithm to automatically
 *   select the number of clusters for the given weighted data set.
 *
 * Copyright (C) 2015, 2016 Mingzhou Song
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

// Joe Song
// Created: May 21, 2016. Extracted from select_levels.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "Ckmeans.1d.dp.h"

#ifndef M_PI
const double M_PI = 3.14159265359;
#endif

void shifted_data_variance_weighted(
    const std::vector<double> & x,
    const std::vector<double> & y,
    const double total_weight,
    const size_t left,
    const size_t right,
    double & mean, double & variance)
{
  double sum = 0.0;
  double sumsq = 0.0;

  mean = 0.0;
  variance = 0.0;

  size_t n = right - left + 1;

  if(right >= left) {

    double median = x[(left + right) / 2];

    for (size_t i = left; i <= right; ++i) {
      sum += (x[i] - median) * y[i];
      sumsq += (x[i] - median) * (x[i] - median) * y[i];
    }
    mean = sum / total_weight + median;

    if (n > 1) {
      variance = (sumsq - sum * sum / total_weight) / (total_weight - 1);
    }
  }
}

// Choose an optimal number of levels between Kmin and Kmax
size_t select_levels_weighted(
    const std::vector<double> & x, const std::vector<double> & y,
    const std::vector< std::vector< size_t > > & J,
    size_t Kmin, size_t Kmax, double * BIC)
{
  const size_t N = x.size();

  if (Kmin > Kmax || N < 2) {
    return std::min(Kmin, Kmax);
  }

  /*
   if(BIC.size() != Kmax - Kmin + 1) {
   BIC.resize(Kmax - Kmin + 1);
   }
   */

  // double variance_min, variance_max;
  // range_of_variance(x, variance_min, variance_max);

  size_t Kopt = Kmin;

  long double maxBIC(0.0);

  std::vector<double> lambda(Kmax);
  std::vector<double> mu(Kmax);
  std::vector<double> sigma2(Kmax);
  std::vector<double> coeff(Kmax);
  std::vector<size_t> counts(Kmax);
  std::vector<double> weights(Kmax);

  for(size_t K = Kmin; K <= Kmax; ++K) {

    // std::vector< std::vector< size_t > > JK(J.begin(), J.begin()+K);

    // Backtrack the matrix to determine boundaries between the bins.
    backtrack_weighted(x, y, J, counts, weights, (int)K);

      
    // double totalweight = std::accumulate(weights.begin(), weights.begin() + K, 0, std::plus<double>());
          
    double totalweight;

    totalweight = 0;
    for(size_t k=0; k<K; k++) {
      totalweight += weights[k];
    }
    

    size_t indexLeft = 0;
    size_t indexRight;

    for (size_t k = 0; k < K; ++k) { // Estimate GMM parameters first

      lambda[k] = weights[k] / totalweight;

      indexRight = indexLeft + counts[k] - 1;

      shifted_data_variance_weighted(x, y, weights[k], indexLeft, indexRight, mu[k], sigma2[k]);

      if(sigma2[k] == 0 || counts[k] == 1) {

        double dmin;

        if(indexLeft > 0 && indexRight < N-1) {
          dmin = std::min(x[indexLeft] - x[indexLeft-1], x[indexRight+1] - x[indexRight]);
        } else if(indexLeft > 0) {
          dmin = x[indexLeft] - x[indexLeft-1];
        } else {
          dmin = x[indexRight+1] - x[indexRight];
        }

        // std::cout << "sigma2[k]=" << sigma2[k] << "==>";
        if(sigma2[k] == 0) sigma2[k] = dmin * dmin / 4.0 / 9.0 ;
        if(counts[k] == 1) sigma2[k] = dmin * dmin;
        // std::cout << sigma2[k] << std::endl;
      }

      /*
       if(sigma2[k] == 0) sigma2[k] = variance_min;
       if(size[k] == 1) sigma2[k] = variance_max;
       */

      coeff[k] = lambda[k] / std::sqrt(2.0 * M_PI * sigma2[k]);

      indexLeft = indexRight + 1;
    }

    long double loglikelihood = 0;

    for (size_t i=0; i<N; ++i) {
      long double L=0;
      for (size_t k = 0; k < K; ++k) {
        L += coeff[k] * std::exp(- (x[i] - mu[k]) * (x[i] - mu[k]) / (2.0 * sigma2[k]));
      }
      loglikelihood += y[i] * std::log(L);
    }

    // double & bic = BIC[K-Kmin];
    long double bic;

    // Compute the Bayesian information criterion
    bic = 2 * loglikelihood - (3 * K - 1) * std::log(totalweight);  //(K*3-1)

    // std::cout << "k=" << K << ": Loglh=" << loglikelihood << ", BIC=" << BIC << std::endl;

    if (K == Kmin) {
      maxBIC = bic;
      Kopt = Kmin;
    } else {
      if (bic > maxBIC) {
        maxBIC = bic;
        Kopt = K;
      }
    }

    BIC[K-Kmin] = (double)bic;
  }
  return Kopt;
}

// Choose an optimal number of levels between Kmin and Kmax
size_t select_levels_weighted_3_4_13(
    const std::vector<double> & x, const std::vector<double> & y,
    const std::vector< std::vector< size_t > > & J,
    size_t Kmin, size_t Kmax)
{
  if (Kmin == Kmax) {
    return Kmin;
  }

  double variance_min, variance_max;

  range_of_variance(x, variance_min, variance_max);

  size_t Kopt = Kmin;

  // const size_t N = x.size();

  double maxBIC(0.0);

  for(size_t K = Kmin; K <= Kmax; ++K) {

    // std::vector< std::vector< size_t > > JK(J.begin(), J.begin()+K);
    std::vector<size_t> counts(K);
    std::vector<double> weights(K);

    // Backtrack the matrix to determine boundaries between the bins.
    backtrack_weighted(x, y, J, counts, weights, (int)K);

    size_t indexLeft = 0;
    size_t indexRight;

    double loglikelihood = 0;

    double totalweight = std::accumulate(begin(weights), end(weights), 0, std::plus<double>());

    for (size_t k = 0; k < K; ++k) { // Compute the likelihood

      size_t numPointsInBin = counts[k];

      indexRight = indexLeft + numPointsInBin - 1;

      double mean = 0.0;
      double variance = 0.0;

      shifted_data_variance_weighted(
        x, y, weights[k], indexLeft, indexRight, mean, variance);

      if(variance == 0) variance = variance_min;
      if(numPointsInBin == 1) variance = variance_max;

      for (size_t i = indexLeft; i <= indexRight; ++i) {
        loglikelihood += - (x[i] - mean) * (x[i] - mean) * y[i]
        / (2.0 * variance);
      }

      loglikelihood += weights[k] *
        (std::log(weights[k] / totalweight) -
        0.5 * std::log ( 2.0 * M_PI * variance));

      indexLeft = indexRight + 1;
    }

    double BIC = 0.0;

    // Compute the Bayesian information criterion
    // BIC = 2 * loglikelihood - (3 * K - 1) * std::log((double)N);  //(K*3-1)
    BIC = 2 * loglikelihood - (3 * K - 1) * std::log(totalweight);  //(K*3-1)

    // cout << ", Loglh=" << loglikelihood << ", BIC=" << BIC << endl;

    if (K == Kmin) {
      maxBIC = BIC;
      Kopt = Kmin;
    } else {
      if (BIC > maxBIC) {
        maxBIC = BIC;
        Kopt = K;
      }
    }
  }
  return Kopt;
}

size_t select_levels_weighted_3_4_12(
    const std::vector<double> & x, const std::vector<double> & y,
    const std::vector< std::vector< size_t > > & J,
    size_t Kmin, size_t Kmax, double * bic)
  // Ckmeans.1d.dp version 3.4.6 or earlier
  // Choose an optimal number of levels between Kmin and Kmax
{
  /*
   if (Kmin == Kmax) {
   return Kmin;
   }
   */

  const std::string method = "normal"; // "uniform" or "normal"

  size_t Kopt = Kmin;

  const size_t base = 0;  // The position of first element in x: 1 or 0.
  const size_t N = x.size() - base;

  double maxBIC(0);

  for(size_t K = Kmin; K <= Kmax; ++K) {

    // std::vector< std::vector< size_t > > JK(J.begin(), J.begin()+K+base);
    std::vector<size_t> counts(K+base);
    std::vector<double> weights(K+base);

    // Backtrack the matrix to determine boundaries between the bins.
    backtrack_weighted(x, y, J, counts, weights, (int)K); // backtrack_weighted(x, y, JK, counts, weights);

    size_t indexLeft = base;
    size_t indexRight;

    double loglikelihood = 0;
    double binLeft, binRight;

    double totalweight = std::accumulate(begin(weights), end(weights), 0, std::plus<double>());

    for (size_t k = 0; k < K; ++k) { // Compute the likelihood

      size_t numPointsInBin = counts[k+base];

      indexRight = indexLeft + numPointsInBin - 1;

      /* Use mid point inbetween two clusters as boundary
       binLeft = ( indexLeft == base ) ?
       x[base] : (x[indexLeft-1] + x[indexLeft]) / 2;

       binRight = ( indexRight < N-1+base ) ?
       (x[indexRight] + x[indexRight+1]) / 2 : x[N-1+base];
       */

      if(x[indexLeft] < x[indexRight]) {
        binLeft = x[indexLeft];
        binRight = x[indexRight];
      } else if(x[indexLeft] == x[indexRight]) {
        binLeft = ( indexLeft == base ) ?
        x[base] : (x[indexLeft-1] + x[indexLeft]) / 2;
        binRight = ( indexRight < N-1+base ) ?
        (x[indexRight] + x[indexRight+1]) / 2 : x[N-1+base];
      } else {
        throw "ERROR: binLeft > binRight";
        // cout << "ERROR: binLeft > binRight" << endl;
      }

      double binWidth = binRight - binLeft;

      if(method == "uniform") {

        loglikelihood += weights[k] * std::log(weights[k] / binWidth / N);

      } else if(method == "normal") {

        double mean = 0.0;
        double variance = 0.0;

        shifted_data_variance_weighted(
          x, y, weights[k], indexLeft, indexRight, mean, variance);

        if (variance > 0) {
          for (size_t i = indexLeft; i <= indexRight; ++i) {
            loglikelihood += - (x[i] - mean) * (x[i] - mean) * y[i]
            / (2.0 * variance);
          }
          loglikelihood += weights[k]
          * (std::log(weights[k] / (double) totalweight)
               - 0.5 * std::log ( 2.0 * M_PI * variance));
        } else {
          loglikelihood += weights[k] * std::log(1.0 / binWidth / N);
        }

      } else {
        throw "ERROR: Wrong likelihood method!";
        // cout << "ERROR: Wrong likelihood method" << endl;
      }

      indexLeft = indexRight + 1;
    }

    double & BIC = bic[K-Kmin];

    // Compute the Bayesian information criterion
    if (method == "uniform") {
      BIC = 2 * loglikelihood - (3 * K - 1) * std::log((double)N);  // K-1
    } else if(method == "normal") {
      BIC = 2 * loglikelihood - (3 * K - 1) * std::log((double)N);  //(K*3-1)
    }

    // cout << ", Loglh=" << loglikelihood << ", BIC=" << BIC << endl;

    if (K == Kmin) {
      maxBIC = BIC;
      Kopt = Kmin;
    } else {
      if (BIC > maxBIC) {
        maxBIC = BIC;
        Kopt = K;
      }
    }
  }
  return Kopt;
}
