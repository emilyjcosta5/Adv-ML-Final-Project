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
from scipy.spatial import distance
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

class Clustering:
    def __init__(self, data, number_clusters, linkage="single", verbose=False) -> None:
        '''
        Parameters
        ----------
        data: pd.DataFrame
            Dataframe containing the data to be clustered; must provide cluster number
            in the index so we can test the classification
        number_clusters: int
            The number of clusters that the ML should identify
        verbose: boolean, optional
            For debugging and info on amount of info being collected.
        linkage: str, {'single', 'ward'} default: 'single'
            How to link points to clusters; we are implementing 2 methods
            - Ward minimizes the sum of squared differences within all clusters. 
              It is a variance-minimizing approach and in this sense is similar 
              to the k-means objective function but tackled with an agglomerative 
              hierarchical approach.
            - Single linkage minimizes the distance between the closest 
              observations of pairs of clusters.
        '''
        if not linkage in ['single', 'ward']:
            raise ValueError("Linkage must be 'single' or 'ward'")
        self.linkage = linkage
        self.n_clusters = number_clusters
        self.data = self.cluster_data(data)
    
    def get_clustered_data(self):
        return self.data

    def cluster_data(self, data):
        scaler = StandardScaler() 
        try:
            scaled = scaler.fit_transform(data)
        except ValueError:
            return None
        X = scaled.copy()
        clusters = AgglomerativeClustering(affinity='euclidean', linkage=self.linkage, n_clusters=self.n_clusters).fit(X)
        data['Point'] = data.apply(lambda d: list((d[p] for p in data.columns)), axis=1)
        data.drop(data.columns.difference(['Point']), 1, inplace=True)
        data['Label'] = clusters.labels_
        X = data.copy()
        data['Cluster Number'] = data.index
        data = data.reset_index(drop=True)
        return data

    def add_point(self, point):
        classsification = None
        if self.linkage=='single':
            classification = self._single_point(point)
        elif self.linkage=='ward':
            classification = self._ward_point(point)
        return classification

    def _single_point(self, point):
        data = self.data
        dist = data.apply(lambda d: distance.euclidean(d['Point'],point), axis=1) # this will be changed to Bayesian later on
        #dist = data.apply(lambda d: self._bayesian_distance(d['Point'],point), axis=1)
        classification = data['Label'].values[dist.idxmin()]
        return classification

    def _ward_point(self, point):
        X = self.data['Point'].tolist()
        X.append(point)
        y = pdist(X, metric='Euclidean')
        Z = ward(y)
        c = fcluster(Z, self.n_clusters, criterion='maxclust').tolist()
        i = c.index(c[-1])
        try:
            label = self.data['Label'].tolist()[i]
        except IndexError:
            return None
        return label

    def _bayesian_distance(self, point0, point1):
        '''
        Calculate distance to a point of all points in dataframe

        Parameters
        ----------
        point0-1: array-like
            A given point; dimensions must match dataframe

        Returns
        -------
        distance: list
            Bayesian distance
        '''
        distance = None
        # TO-DO
        # use pandas apply with _calculate_distance?
        return distance
     
if __name__=="__main__":
    # scenario 1: 3 equal sized clusters generally separate
    column_values = ["x","y"]
    mean = (0,0)
    cov = ((1,1),(1,1))
    size = 100
    cluster_number = 0
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)

    mean = (10,10)
    cov = ((1,1),(1,1))
    size = 100
    cluster_number = 1
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = df.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))

    mean = (20,20)
    cov = ((1,1),(1,1))
    size = 100
    cluster_number = 2
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = df.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
    
    clustering = Clustering(df, 3)
    print(clustering.get_clustered_data())
    print(clustering.add_point([0,0]))