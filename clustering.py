'''
# experiment with
# linkage {‘ward’, ‘complete’, ‘average’, ‘single’}

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

def cluster_data(data, number_clusters, verbose=False):
    '''
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe containing the data to be clustered
    number_clusters: int
        The number of clusters that the ML should identify
    verbose: boolean, optional
        For debugging and info on amount of info being collected.

    Returns
    -------
    clusters: pd.DataFrame
        How clusters were formed
    '''
    scaler = StandardScaler() 
    try:
        scaled = scaler.fit_transform(data)
    except ValueError:
        return None
    data = pd.DataFrame(scaled, index=data.index, columns=data.columns)
    X = data.copy()
    clusters = AgglomerativeClustering(affinity='euclidean', n_clusters=number_clusters).fit(X)
    data['Euclidean Labels'] = clusters.labels_
    X = data.copy()
    #clusters = AgglomerativeClustering(affinity=bayesian, n_clusters=number_clusters).fit(X)
    #data['Bayesian Labels'] = clusters.labels_
    data['Cluster Number'] = data.index
    data = data.reset_index(drop=True)
    return data

# experiments
def vary_mean(proportion=(1,1)):
    euclidean_accuracies = []
    column_values = ["x","y"]
    mean = (0,0)
    cov = ((1,1),(1,1))
    size = int(100*proportion[0]/proportion[1])
    cluster_number = 0
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)
    distances = np.linspace(0., 10., num=20)
    for x in distances:
        mean = (x,x)
        cov = ((1,1),(1,1))
        size = int(100*proportion[0]/proportion[1])
        cluster_number = 1
        index_values = [cluster_number for x in range(size)]
        cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
        tmp = df.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
        tmp = cluster_data(tmp, 2)
        euclidean_accuracy = np.abs(tmp['Euclidean Labels'].corr(tmp['Cluster Number'], method='spearman'))
        euclidean_accuracies.append(euclidean_accuracy)
    print(euclidean_accuracies)

if __name__=="__main__":
    # scenario 1: 3 equal sized clusters generally separate
    '''
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
    print(cluster_data(df, 3))
    '''
    vary_mean()

    # scenario 2: 3 equal size clusters that are generally mixed 
    # scenario 3: 2 unequal size (4:1) clusters that are generally separate 
    # scenario 4: 2 unequal size (4:1) clusters that are generally mixed
     
