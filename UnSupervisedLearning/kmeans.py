import sys
import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import pairwise_distances


def assign_clusters(data, centroids):
    # Compute distances between each data point and the set of centroids:
    distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')

    # Compute cluster assignments for each data point:
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)
    return cluster_assignment


class Kmeans:
    """
    K-Means unsupervised clustering algorithm.
        Parameters
        ----------
        n_clusters : int, default=8
            integer describes how many number of clusters required
        max_iter : int, default=400
            No of max passes over the training set.
            Clustering process will be forced stop once current iteration reach max_iter.
        verbose : bool, default=False
            if True, print how many data points changed their cluster labels in each iteration.
        num_runs : int, default=10
            Number of times kmeans algorithm run to find optimum value.
        seed_list : list, default=None
            (optional) number of seed values equal to num_runs, to generate random initial centroids for run.

        Attributes
        ----------
        centroids : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers.
        cluster_assignment : ndarray of shape (n_samples,)
            assigned cluster of each sample.
        heterogeneity : dict, default={}
             to store the history of heterogeneity as function of num_runs.
    """

    def __init__(self, n_clusters=8, max_iter=400, verbose=False, num_runs=10, seed_list=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.num_runs = num_runs
        self.seed_list = seed_list
        self.centroids = None
        self.cluster_assignment = None
        self.heterogeneity = {}

    def fit(self, data):
        min_heterogeneity_achieved = float('inf')
        final_centroids = None
        final_cluster_assignment = None

        for i in range(self.num_runs):

            # Use UTC time if no seeds are provided
            if self.seed_list is not None:
                seed = self.seed_list[i]
                np.random.seed(seed)
            else:
                seed = int(time.time())
                np.random.seed(seed)

            # Use k-means++ initialization
            initial_centroids = self._smart_initialize(data, seed)

            # Run k-means
            centroids, cluster_assignment = self._fit_single_run(data, initial_centroids)

            # To save time, compute heterogeneity only once in the end
            self.heterogeneity[seed] = self._compute_heterogeneity(data, centroids, cluster_assignment)

            if self.verbose:
                print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, self.heterogeneity[seed]))
                sys.stdout.flush()

            # if current measurement of heterogeneity is lower than previously seen,
            # update the minimum record of heterogeneity.
            if self.heterogeneity[seed] < min_heterogeneity_achieved:
                min_heterogeneity_achieved = self.heterogeneity[seed]
                final_centroids = centroids
                final_cluster_assignment = cluster_assignment

        # Return the centroids and cluster assignments that minimize heterogeneity.
        self.cluster_assignment = final_cluster_assignment
        self.centroids = final_centroids

    def _fit_single_run(self, data, initial_centroids):
        """
        This function runs k-means on given data and initial set of centroids.
        """
        centroids = initial_centroids[:]
        prev_cluster_assignment = None

        for itr in range(self.max_iter):
            if self.verbose:
                print(itr)

            # 1. Make cluster assignments using the nearest centroids
            cluster_assignment = assign_clusters(data, centroids)

            # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
            centroids = self._revise_centroids(data, cluster_assignment)

            # Check for convergence: if none of the assignments changed, stop
            if prev_cluster_assignment is not None and \
                    (prev_cluster_assignment == cluster_assignment).all():
                break

            # Print number of new assignments
            if prev_cluster_assignment is not None:
                num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
                if self.verbose:
                    print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))

            prev_cluster_assignment = cluster_assignment[:]

        return centroids, prev_cluster_assignment

    def _compute_heterogeneity(self, data, centroids, clusters):
        heterogeneity = 0.0
        for i in range(self.n_clusters):
            # Select all data points that belong to cluster i. Fill in the blank (RHS only)
            member_data_points = data[clusters == i, :]

            if member_data_points.shape[0] > 0:  # check if i-th cluster is non-empty
                # Compute distances from centroid to data points
                distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
                squared_distances = distances ** 2
                heterogeneity += np.sum(squared_distances)

        return heterogeneity

    def _smart_initialize(self, data, seed=None):
        if seed is not None:  # useful for obtaining consistent results
            np.random.seed(seed)
        centroids = np.zeros((self.n_clusters, data.shape[1]))

        # Randomly choose the first centroid.
        # Since we have no prior knowledge, choose uniformly at random
        idx = np.random.randint(data.shape[0])
        centroids[0] = data[idx, :].toarray()
        # Compute distances from the first centroid chosen to all the other data points
        squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten() ** 2

        for i in range(1, self.n_clusters):
            # Choose the next centroid randomly, so that the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid.
            # Roughly speaking, a new centroid should be as far as from other centroids as possible.
            idx = np.random.choice(data.shape[0], 1, p=squared_distances / sum(squared_distances))
            centroids[i] = data[idx, :].toarray()
            # Now compute distances from the centroids to all data points
            squared_distances = np.min(pairwise_distances(data, centroids[0:i + 1], metric='euclidean') ** 2, axis=1)

        return centroids

    def _revise_centroids(self, data, cluster_assignment):
        new_centroids = []
        for i in range(self.n_clusters):
            # Select all data points that belong to cluster i. Fill in the blank (RHS only)
            member_data_points = data[cluster_assignment == i]
            # Compute the mean of the data points. Fill in the blank (RHS only)
            centroid = member_data_points.mean(axis=0)

            # Convert numpy.matrix type to numpy.ndarray type
            try:
                centroid = centroid.A1
            except AttributeError:
                print('centroid is array itself !')
            new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)

        return new_centroids
