from clustering import Clustering
import numpy as np
import pandas as pd

def linear_equally_spaced_clusters(n_points=100, proportion=(1,1), n_clusters=2):
    '''
    This serves as a base case and simple evaluation - how does our methodology perform
    in the case where clusters are equally space, same covariance, and linearly space? 
    The only variables we can modify are the cluster sizes and number of clusters. This
    methodology will test how the algorithm performs as clusters are close (and hence
    more intertwined) versus more separated.

    Parameters
    ----------
    n_points: int
        number of data points to generate cumulatively in all clusters
    proportion: tuple
        the size proportions of the clusters; must be the length of the number of clusters
    n_clusters: int
        the number of clusters
    
    '''
    euclidean_accuracies = []
    bayesian_accuracies = []
    column_values = ["x","y"]
    mean = (0,0)
    cov = ((1,1),(1,1))
    size = int(n_points*proportion[0]/sum(proportion))
    cluster_number = 0
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)
    distances = np.linspace(0., 10., num=20)
    for x in distances:
        tmp = df.copy()
        # generate clusters for this particular distance
        for i in range(1,n_clusters):
            mean = (i*x,i*x)
            cov = ((1,1),(1,1))
            size = int(n_points*proportion[i]/sum(proportion))
            index_values = [i for x in range(size)]
            cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
            tmp = tmp.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
        # do initial clustering and check accuracy of clustering methodology
        clustering = Clustering(tmp, n_clusters)
        tmp = clustering.get_clustered_data()
        euclidean_accuracy = np.abs(tmp['Label'].corr(tmp['Cluster Number'], method='spearman'))
        euclidean_accuracies.append(euclidean_accuracy)
        # TO-DO
        # add 100 new points per cluster and measure accuracy
        points = pd.DataFrame(columns=column_values + ['Label', 'Cluster Number'])
        for p in range(100):
            for i in range(0,n_clusters):
                mean = (i*x,i*x)
                cov = ((1,1),(1,1))
                point = np.random.multivariate_normal(mean=mean, cov=cov, size=1)
                classification = clustering.add_point(point)
                points.loc[len(points.index)] = [point[0], point[1], classification, i]
        bayesian_accuracy = np.abs(points['Label'].corr(points['Cluster Number'], method='spearman'))
        bayesian_accuracies.append(bayesian_accuracy)
    print(distances)
    print(euclidean_accuracies)
    print(bayesian_accuracies)
    return euclidean_accuracies, bayesian_accuracies

if __name__=="__main__":
    linear_equally_spaced_clusters()

    # scenario 1: 3 equal sized clusters generally separate
    # scenario 2: 3 equal size clusters that are generally mixed 
    # scenario 3: 2 unequal size (4:1) clusters that are generally separate 
    # scenario 4: 2 unequal size (4:1) clusters that are generally mixed