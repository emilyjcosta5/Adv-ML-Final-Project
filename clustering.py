'''
# resources:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
'''

# Author: Emily Costa
# Created on: Apr 26, 2022
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class BayesianDistance:
    def __init__(self, data) -> None:
        '''
        Parameters
        ----------
        data: pandas.Dataframe
            Dataframe with contents being the rows of data point and
            index as the correct cluster and what the point is classified
        '''
        # TO-DO
        # this is where you can make sure the probabilities are tracked?
        pass

    def calculate_distance(self, point):
        '''
        Calculate distance to a point of all points in dataframe

        Parameters
        ----------
        point: array-like
            The given point; dimensions must match dataframe

        Returns
        -------
        distances: list
            The distances of all the points from a given point
        '''
        distances = []
        # TO-DO
        # use pandas apply with _calculate_distance?
        return distances

    def _calculate_distance(self, point1, point2):
        distance = 0
        # TO-DO
        return distance

class Clustering:
    def __init__(self, data, number_clusters, linkage="single", verbose=False) -> None:
        '''
        Parameters
        ----------
        data: pd.DataFrame
            Dataframe containing the data to be clustered
        number_clusters: int
            The number of clusters that the ML should identify
        verbose: boolean, optional
            For debugging and info on amount of info being collected.
        linkage: str, {'single', 'ward'} default: 'single'
            How to link points to clusters
            - Ward minimizes the sum of squared differences within all clusters. 
              It is a variance-minimizing approach and in this sense is similar 
              to the k-means objective function but tackled with an agglomerative 
              hierarchical approach.
            - Single linkage minimizes the distance between the closest 
              observations of pairs of clusters.
        '''
        self.data = data
        self.linkage = linkage
        self.n_clusters = number_clusters
        self.cluster_data = self.cluster_data()
    
    def get_clustered_data(self):
        return self.cluster_data

    def cluster_data(self):
        scaler = StandardScaler() 
        try:
            scaled = scaler.fit_transform(data)
        except ValueError:
            return None
        data = pd.DataFrame(scaled, index=data.index, columns=data.columns)
        X = data.copy()
        clusters = AgglomerativeClustering(affinity='euclidean', n_clusters=self.n_clusters).fit(X)
        data['Labels'] = clusters.labels_
        X = data.copy()
        data['Cluster Number'] = data.index
        data = data.reset_index(drop=True)
        return data

    def add_point(self, point):
        if self.linkage=='single':
            self._single_point(point)
        elif self.linkage=='ward':
            self._ward_point(point)

    def _single_point(self, point):
        # TO-DO
        pass

    def _ward_point(self, point):
        # TO-DO
        pass
     
