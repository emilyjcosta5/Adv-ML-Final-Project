from clustering import Clustering
import numpy as np
import pandas as pd

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
        clustering = Clustering(tmp, 2)
        tmp = clustering.get_clustered_data()
        euclidean_accuracy = np.abs(tmp['Euclidean Labels'].corr(tmp['Cluster Number'], method='spearman'))
        euclidean_accuracies.append(euclidean_accuracy)
        # now time to mess around with adding new points
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