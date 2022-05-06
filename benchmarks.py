from sre_parse import fix_flags
from turtle import color
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from clustering import Clustering
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import spearmanr
from matplotlib.lines import Line2D
from time import time

def linear_equally_spaced_clusters(n_points=100, proportion=(1,1), n_clusters=2, linkage='single'):
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
    times_recluster = []
    times_add = []
    column_values = ["x","y"]
    mean = (0,0)
    cov = ((1,1),(1,1))
    size = int(n_points*proportion[0]/sum(proportion))
    cluster_number = 0
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)
    distances = range(1,11)
    for x in distances:
        dist_add = []
        dist_recluster = []
        tmp = df.copy()
        # generate clusters for this particular distance
        for i in range(1,n_clusters):
            mean = (i*x,i*x)
            cov = ((1,1),(1,1))
            size = int(n_points*proportion[i]/sum(proportion))
            index_values = [i for x in range(size)]
            cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
            tmp = tmp.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
        clustering = tmp.copy()
        # do initial clustering and check accuracy of clustering methodology
        clustering = Clustering(clustering, n_clusters, linkage=linkage)
        # add new points every cluster
        for p in range(n_points):
            for i in range(0,n_clusters):
                mean = (i*x,i*x)
                cov = ((1,1),(1,1))
                point = np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0]
                index_values = [i]
                tmp = tmp.append(pd.DataFrame(data=point[0], index=index_values, columns=column_values))
                cc = tmp.copy()
                start = time()
                cc = Clustering(cc, n_clusters, linkage=linkage)
                end = time()
                t = (end - start)*1000
                dist_recluster.append(t)
                start = time()
                end = time()
                t = (end - start)*1000
                dist_add.append(t)
        times_recluster.append(np.sum(dist_recluster)/len(dist_recluster))
        times_add.append(np.sum(dist_add)/len(dist_add))
    return distances, times_recluster, times_add

def radial_equally_spaced_clusters(n_points=200, proportion=(1,1,1,1), n_clusters=4, linkage='single'):
    '''
    Like linear_equally_spaced_clusters, but all the clusters are evenly place                print(tmp)d in a circle
    around the origin
    Parameters
    ----------
    n_points: int
        number of data points to generate cumulatively in all clusters
    proportion: tuple
        the size proportions of the clusters; must be the length of the number of clusters
    n_clusters: int
        the number of clusters
    
    '''
    times_recluster = []
    times_add = []
    original_accuracies = []
    new_accuracies = []
    column_values = ["x","y"]
    pi = math.pi
    mean = (0,0)
    cov = ((1,0),(0,1))
    size = int(n_points*proportion[0]/sum(proportion))
    cluster_number = 0
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)
    distances = range(1,11)
    for x in distances:
        dist_add = []
        dist_recluster = []
        tmp = df.copy()
        # generate clusters for this particular distance
        for i in range(1,n_clusters):
            mean = (math.cos(i*pi/n_clusters)*x,math.sin(i*pi/n_clusters)*x)
            cov = ((1,0),(0,1))
            size = int(n_points*proportion[i]/sum(proportion))
            index_values = [i for x in range(size)]
            cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
            tmp = tmp.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
        # do initial clustering and check accuracy of clustering methodology
        clustering = Clustering(tmp, n_clusters, linkage=linkage)
        tmp = clustering.get_clustered_data()
        original_accuracy = np.abs(spearmanr(tmp['Label'],tmp['Cluster Number'])[0])
        original_accuracies.append(original_accuracy)
        # add 100 new points per cluster and measure accuracy
        points = pd.DataFrame(columns=column_values + ['Label', 'Cluster Number'])
        for p in range(n_points):
            for i in range(0,n_clusters):
                mean = (math.cos(i*pi/n_clusters)*x,math.sin(i*pi/n_clusters)*x)
                cov = ((1,0),(0,1))
                point = np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0]
                index_values = [i]
                tmp = tmp.append(pd.DataFrame(data=point[0], index=index_values, columns=column_values))
                cc = tmp.copy()
                start = time()
                cc = Clustering(cc, n_clusters, linkage=linkage)
                end = time()
                t = (end - start)*1000
                dist_recluster.append(t)
                start = time()
                end = time()
                t = (end - start)*1000
                dist_add.append(t)
        times_recluster.append(np.sum(dist_recluster)/len(dist_recluster))
        times_add.append(np.sum(dist_add)/len(dist_add))
    return distances, times_recluster, times_add

def two_equal_clusters_additional_points(n_points=100, linkage='single'):
    '''
    Like linear_equally_spaced_clusters, but all the clusters are evenly placed in a circle
    around the origin
    Parameters
    ----------
    n_points: int
        number of data points to generate cumulatively in all clusters
    proportion: tuple
        the size proportions of the clusters; must be the length of the number of clusters
    n_clusters: int
        the number of clusters
    
    '''
    times_recluster = []
    times_add = []
    original_accuracies = []
    new_accuracies = []
    column_values = ["x","y"]
    pi = math.pi
    mean = (0,0)
    cov = ((1,0),(0,1))
    size = 50
    cluster_number = 0
    n_clusters = 2
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)
    distances = range(1,11)
    data_holder = df.copy()
    for x in distances:
        dist_add = []
        dist_recluster = []
        tmp = df.copy()
        # generate clusters for this particular distance
        for i in range(1,n_clusters):
            mean = (0,i*5)
            cov = ((1,0),(0,1))
            size = 50
            index_values = [i for x in range(size*x,size*(x+1))]
            cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
            tmp = tmp.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
        # do initial clustering and check accuracy of clustering methodology
        clustering = Clustering(tmp, n_clusters, linkage='single')
        tmp = clustering.get_clustered_data()
        #original_accuracy = np.abs(tmp['Label'].corr(tmp['Cluster Number'], method='spearman'))
        try:
            label = tmp['Label'].tolist()
        except TypeError:
            continue
        cn = tmp['Cluster Number'].tolist()
        original_accuracy = spearmanr(label,cn)
        original_accuracy = original_accuracy[0]
        original_accuracy = np.abs(original_accuracy)
        original_accuracies.append(original_accuracy)
        # add 100 new points per cluster and measure accuracy
        points = pd.DataFrame(columns=column_values + ['Label', 'Cluster Number'])
        for p in range(n_points):
            for i in range(0,n_clusters):
                mean = (math.cos(i*pi/n_clusters)*x,math.sin(i*pi/n_clusters)*x)
                cov = ((1,0),(0,1))
                point = np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0]
                index_values = [i]
                tmp = tmp.append(pd.DataFrame(data=point[0], index=index_values, columns=column_values))
                cc = tmp.copy()
                start = time()
                cc = Clustering(cc, n_clusters, linkage=linkage)
                end = time()
                t = (end - start)*1000
                dist_recluster.append(t)
                start = time()
                end = time()
                t = (end - start)*1000
                dist_add.append(t)
        times_recluster.append(np.sum(dist_recluster)/len(dist_recluster))
        times_add.append(np.sum(dist_add)/len(dist_add))
    return distances, times_recluster, times_add

def two_equal_clusters_different_covariances(n_points=100, linkage='single'):
    '''
    Like linear_equally_spaced_clusters, but all the clusters are evenly placed in a circle
    around the origin
    Parameters
    ----------
    n_points: int
        number of data points to generate cumulatively in all clusters
    proportion: tuple
        the size proportions of the clusters; must be the length of the number of clusters
    n_clusters: int
        the number of clusters
    
    '''
    times_recluster = []
    times_add = []
    original_accuracies = []
    new_accuracies = []
    column_values = ["x","y"]
    pi = math.pi
    mean = (0,0)
    cov = ((1,0),(0,1))
    size = 50
    cluster_number = 0
    n_clusters = 2
    index_values = [cluster_number for x in range(size)]
    cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)
    distances = range(1,11)
    for x in distances:
        dist_add = []
        dist_recluster = []
        tmp = df.copy()
        # generate clusters for this particular distance
        for i in range(1,n_clusters):
            mean = (0,i*8)
            cov = ((x,0),(0,1))
            size = 50
            index_values = [i for x in range(size)]
            cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
            tmp = tmp.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
        # do initial clustering and check accuracy of clustering methodology
        clustering = Clustering(tmp, n_clusters, linkage=linkage)
        tmp = clustering.get_clustered_data()
        original_accuracy = np.abs(spearmanr(tmp['Label'],tmp['Cluster Number'])[0])
        original_accuracies.append(original_accuracy)
        # add 100 new points per cluster and measure accuracy
        points = pd.DataFrame(columns=column_values + ['Label', 'Cluster Number'])
        for p in range(n_points):
            for i in range(0,n_clusters):
                mean = (math.cos(i*pi/n_clusters)*x,math.sin(i*pi/n_clusters)*x)
                cov = ((1,0),(0,1))
                point = np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0]
                index_values = [i]
                tmp = tmp.append(pd.DataFrame(data=point[0], index=index_values, columns=column_values))
                cc = tmp.copy()
                start = time()
                cc = Clustering(cc, n_clusters, linkage=linkage)
                end = time()
                t = (end - start)*1000
                dist_recluster.append(t)
                start = time()
                end = time()
                t = (end - start)*1000
                dist_add.append(t)
        times_recluster.append(np.sum(dist_recluster)/len(dist_recluster))
        times_add.append(np.sum(dist_add)/len(dist_add))
    return distances, times_recluster, times_add

def plot_single(n_trials=100):
    fig, axes = plt.subplots(2,2,figsize=(8,6))
    linkage = 'single'
    palette = {
        'Original Cluster' : 'lightcoral', 
        'New Point' : 'red'
    }
    n_points = 10 # number of points to add to each cluster to test for new point accuracy in each trial
    
    # first experiment and plot
    columns = ['Distance between Cluster Means', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = linear_equally_spaced_clusters(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Distance between Cluster Means'
    y='Original Cluster'
    sns.lineplot(ax=axes[0,0], data=results, x=x, y=y, color=palette[y], ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[0,0], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[0,0].set_ylabel("")
    axes[0,0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0,0].set_axisbelow(True)
    #axes[0,0].set_ylim(0,1)
    axes[0,0].set_xlim(1,10)
    
    # second experiment and plot
    columns = ['Distance from Original', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = radial_equally_spaced_clusters(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Distance from Original'
    y='Original Cluster'
    sns.lineplot(ax=axes[0,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[0,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[0,1].set_ylabel("")
    axes[0,1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0,1].set_axisbelow(True)
    #axes[0,1].set_ylim(0,1)
    axes[0,1].set_xlim(1,10)
    
     # third experiment and plot
    columns = ['Distance between Cluster Means', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = two_equal_clusters_additional_points(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Distance between Cluster Means'
    y='Original Cluster'
    sns.lineplot(ax=axes[1,0], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[1,0], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[1,0].set_ylabel("")
    axes[1,0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1,0].set_axisbelow(True)
    #axes[1,0].set_ylim(0,1)
    axes[1,0].set_xlim(1,10)

     # fourth experiment and plot
    columns = ['X Value of Covariance [[X,0],[0,1]]', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = two_equal_clusters_different_covariances(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='X Value of Covariance [[X,0],[0,1]]'
    y='Original Cluster'
    sns.lineplot(ax=axes[1,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[1,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[1,1].set_ylabel("")
    axes[1,1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1,1].set_axisbelow(True)
    #axes[1,1].set_ylim(0,1)
    axes[1,1].set_xlim(1,10)

    fig.text(0.01, 0.42, 'Time (ms)', rotation=90)
    fig.text(0.15, 0.865, '(a) Linearly Spaced Equal Gaussians')
    fig.text(0.62, 0.865, '(b) Radially Spaced Equal Gaussians')
    fig.text(0.15, 0.415, '(c) Cirularly Spaced Equal Gaussians')
    fig.text(0.62, 0.415, '(d) Gaussians with Varying Covariance')
    fig.text(0.37, 0.965, 'Single Linkage Real-Time Clustering')
    lines = [Line2D([0],[0],color=palette['Original Cluster'], lw=2),
            Line2D([0],[0],color=palette['New Point'], lw=2, linestyle='--')]
    fig.legend(lines, ['Orignal Cluster', 'New Point'], ncol=2, loc=[0.35,0.9])
    fig.subplots_adjust(left=0.12, right=0.98, top=.85, bottom=0.10, wspace=0.2, hspace=0.5)
    plt.savefig('single_clustering_benchmark.jpg')

def plot_ward(n_trials=100):
    fig, axes = plt.subplots(2,2,figsize=(8,6))
    linkage = 'ward'
    palette = {
        'Original Cluster' : 'lightblue', 
        'New Point' : 'blue'
    }
    n_points = 10 # number of points to add to each cluster to test for new point accuracy in each trial
    
    # first experiment and plot
    columns = ['Distance between Cluster Means', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = linear_equally_spaced_clusters(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Distance between Cluster Means'
    y='Original Cluster'
    sns.lineplot(ax=axes[0,0], data=results, x=x, y=y, color=palette[y], ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[0,0], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[0,0].set_ylabel("")
    axes[0,0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0,0].set_axisbelow(True)
    #axes[0,0].set_ylim(0,1)
    axes[0,0].set_xlim(1,10)
    
    # second experiment and plot
    columns = ['Distance from Original', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = radial_equally_spaced_clusters(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Distance from Original'
    y='Original Cluster'
    sns.lineplot(ax=axes[0,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[0,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[0,1].set_ylabel("")
    axes[0,1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0,1].set_axisbelow(True)
    #axes[0,1].set_ylim(0,1)
    axes[0,1].set_xlim(1,10)
    
     # third experiment and plot
    columns = ['Distance between Cluster Means', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = two_equal_clusters_additional_points(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Distance between Cluster Means'
    y='Original Cluster'
    sns.lineplot(ax=axes[1,0], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[1,0], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[1,0].set_ylabel("")
    axes[1,0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1,0].set_axisbelow(True)
    #axes[1,0].set_ylim(0,1)
    axes[1,0].set_xlim(1,10)

     # fourth experiment and plot
    columns = ['X Value of Covariance [[X,0],[0,1]]', 'Original Cluster', 'New Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = two_equal_clusters_different_covariances(n_points=n_points, linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='X Value of Covariance [[X,0],[0,1]]'
    y='Original Cluster'
    sns.lineplot(ax=axes[1,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, lw=2)
    y='New Point'
    sns.lineplot(ax=axes[1,1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[1,1].set_ylabel("")
    axes[1,1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1,1].set_axisbelow(True)
    #axes[1,1].set_ylim(0,1)
    axes[1,1].set_xlim(1,10)

    fig.text(0.01, 0.42, 'Time (ms)', rotation=90)
    fig.text(0.15, 0.865, '(a) Linearly Spaced Equal Gaussians')
    fig.text(0.62, 0.865, '(b) Radially Spaced Equal Gaussians')
    fig.text(0.15, 0.415, '(c) Cirularly Spaced Equal Gaussians')
    fig.text(0.62, 0.415, '(d) Gaussians with Varying Covariance')
    fig.text(0.37, 0.965, 'Ward Linkage Real-Time Clustering')

    lines = [Line2D([0],[0],color=palette['Original Cluster'], lw=2),
            Line2D([0],[0],color=palette['New Point'], lw=2, linestyle='--')]
    fig.legend(lines, ['Orignal Cluster', 'New Point'], ncol=2, loc=[0.35,0.9])
    fig.subplots_adjust(left=0.12, right=0.98, top=.85, bottom=0.10, wspace=0.2, hspace=0.5)
    plt.savefig('ward_clustering_benchmark.jpg')

def linear_varying_size(proportion=(1,1), n_clusters=2, linkage='single'):
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
    n_points = np.arange(100,1100,100)
    times_recluster = []
    times_add = []
    column_values = ["x","y"]
    mean0 = (0,0)
    mean1 = (4,4)
    means = [mean0,mean1]
    cov = ((1,1),(1,1))
    for n_point in n_points:
        dist_add = []
        dist_recluster = []
        #first cluster
        size = int(n_point*proportion[0]/sum(proportion))
        cluster_number = 0
        index_values = [cluster_number for x in range(size)]
        cluster = np.random.multivariate_normal(mean=mean0, cov=cov, size=size)
        df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)
        # generate clusters for this particular distance
        size = int(n_point*proportion[1]/sum(proportion))
        cluster_number = 1
        index_values = [cluster_number for x in range(size)]
        cluster = np.random.multivariate_normal(mean=mean1, cov=cov, size=size)
        df = df.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))
        clustering = df.copy()
        # do initial clustering and check accuracy of clustering methodology
        clustering = Clustering(clustering, n_clusters, linkage=linkage)
        # add new points every cluster
        for p in range(10):
            for i in range(0,n_clusters):
                mean = means[i]
                cov = ((1,1),(1,1))
                point = np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0]
                index_values = [i]
                df = df.append(pd.DataFrame(data=point[0], index=index_values, columns=column_values))
                cc = df.copy()
                start = time()
                cc = Clustering(cc, n_clusters, linkage=linkage)
                end = time()
                t = (end - start)*1000
                dist_recluster.append(t)
                start = time()
                end = time()
                t = (end - start)*1000
                dist_add.append(t)
        times_recluster.append(np.sum(dist_recluster)/len(dist_recluster))
        times_add.append(np.sum(dist_add)/len(dist_add))
    return n_points, times_recluster, times_add

def linear_varying_size_plot(n_trials=100):
    fig, axes = plt.subplots(2,1,figsize=(8,6), sharex=True)
    linkage = 'single'
    palette = {
        'Recluster' : 'lightcoral', 
        'Add Point' : 'red'
    }

    columns = ['Cumulative Size of Clusters', 'Recluster', 'Add Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = linear_varying_size(linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Cumulative Size of Clusters'
    y='Recluster'
    sns.lineplot(ax=axes[0], data=results, x=x, y=y, color=palette[y], ci=None, lw=2)
    y='Add Point'
    sns.lineplot(ax=axes[0], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[0].set_ylabel("")
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[0].set_xlim(100,1000)

    linkage = 'ward'
    palette = {
        'Recluster' : 'lightblue', 
        'Add Point' : 'blue'
    }

    columns = ['Cumulative Size of Clusters', 'Recluster', 'Add Point']
    results = pd.DataFrame(columns=columns)
    for i in range(n_trials):
        d, o, n = linear_varying_size(linkage=linkage)
        tmp = pd.DataFrame(zip(d,o,n), columns=columns)
        results = results.append(tmp, ignore_index=True)
    x='Cumulative Size of Clusters'
    y='Recluster'
    sns.lineplot(ax=axes[1], data=results, x=x, y=y, color=palette[y], ci=None, lw=2)
    y='Add Point'
    sns.lineplot(ax=axes[1], data=results, x=x, y=y, color=palette[y], markers=True, ci=None, linestyle='--', lw=2)
    axes[1].set_ylabel("")
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    axes[1].set_xlim(100,1000)

    fig.text(0.09, 0.42, 'Time (ms)', rotation=90)
    fig.text(0.43, 0.915, '(a) Single Linkage')
    fig.text(0.43, 0.478, '(b) Ward Linkage')

    lines = [Line2D([0],[0],color=palette['Recluster'], lw=2),
            Line2D([0],[0],color=palette['Add Point'], lw=2, linestyle='--')]
    fig.legend(lines, ['Recluster', 'Add Point'], ncol=2, loc=[0.35,0.425])
    palette = {
        'Recluster' : 'lightcoral', 
        'Add Point' : 'red'
    }
    lines = [Line2D([0],[0],color=palette['Recluster'], lw=2),
            Line2D([0],[0],color=palette['Add Point'], lw=2, linestyle='--')]
    fig.legend(lines, ['Recluster', 'Add Point'], ncol=2, loc=[0.35,0.86])
    fig.subplots_adjust(left=0.15, right=0.85, top=.85, bottom=0.10, wspace=0.2, hspace=0.4)
    plt.savefig('cmp_recluster_benchmark.jpg')

if __name__=="__main__":
    #plot_single(n_trials=100)
    #plot_ward(n_trials=100)
    linear_varying_size_plot(n_trials=100)