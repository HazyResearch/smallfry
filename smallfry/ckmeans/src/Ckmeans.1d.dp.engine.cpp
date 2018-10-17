/* Ckmeans_1d_dp.engine.cpp --- wrapper function for "kmeans_1d_dp()"
 *
 * Copyright (C) 2010-2016 Mingzhou Song and Haizhou Wang
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

/*
 Created: Oct 10, 2010

 Haizhou Wang
 Computer Science Department
 New Mexico State University
 hwang@cs.nmsu.edu

 Modified:
 March 20, 2014. Joe Song. Removed parameter int *k from the function.
 Added "int *Ks" and "int *nK" to provide a range of the number of clusters
 to search for. Made other changes.
 March 29, 2014. Haizhou Wang. Replaced "int *Ks" and "int *nK" by
 "int *minK" and "int *maxK".
 May 29, 2017. Joe Song. Change size from integer to double.
 Sep 2, 2018. Nimit Sohoni. Modified API to facilitate calling from Python.
 */

#include <string>
#include <iostream>

#include "Ckmeans.1d.dp.h"

// Wrapper function to call kmeans_1d_dp().
extern "C" {
  /*
   x: An array containing input data to be clustered.
   length: length of the one dimensional array.
   minK: Minimum number of clusters.
   maxK: Maximum number of clusters.
   cluster:  An array of cluster IDs for each point in x.
   centers:  An array of centers for each cluster.
   withinss: An array of within-cluster sum of squares for each cluster.
   size:     An array of (weighted) sizes of each cluster.
   */

  void kmeans_dp(const double* x, int length, int k, int* clusters,
                 double* centers, double* within_ss, double* cluster_sizes,
                 bool loglinear, bool do_L2) {
    double BIC = 0;
    // Call C++ one-dimensional clustering algorithm.
    kmeans_1d_dp(x, (size_t)length, 0, (size_t)k, (size_t)k, clusters,
                 centers, within_ss, cluster_sizes, &BIC, "BIC",
                 loglinear ? "loglinear" : "linear", do_L2 ? L2 : L1);
  }

  // For verification.
  int status() {
    return 200;
  }

} // End of extern "C"
