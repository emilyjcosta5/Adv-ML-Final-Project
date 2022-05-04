from clustering import Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_values = ["x","y"]
mean = (0,0)
cov = ((1,0),(1,1))
size = 100
cluster_number = 0
index_values = [cluster_number for x in range(size)]
cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
df = pd.DataFrame(data=cluster, index=index_values, columns=column_values)

mean = (3,3)
cov = ((1,1),(2,1))
size = 100
cluster_number = 1
index_values = [cluster_number for x in range(size)]
cluster = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
df = df.append(pd.DataFrame(data=cluster, index=index_values, columns=column_values))

clustering = Clustering(df,2)
original_clusters = clustering.get_clustered_data()

fig, axes = plt.subplots(3,1,figsize=(5,5))

hues = {
    0 : 'red',
    1 : 'blue'
}
sns.scatterplot(ax=axes[0], data=original_clusters, x='x', y='y', hue='Cluster Number', palette=hues)

hues = {
    0 : '#FFCCCB',
    1 : '#ADD8E6'
}
point = np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0]
c = clustering.add_point(point)
sns.scatterplot(ax=axes[1], data=original_clusters, x='x', y='y', hue='Cluster Number', palette=hues)
axes[1].plot(point[0],point[1],'bo')
axes[1].get_legend().remove()

points = np.random.multivariate_normal(mean=mean, cov=cov, size=5).tolist()
points = pd.DataFrame(data=points, columns=column_values)
sns.scatterplot(ax=axes[2], data=original_clusters, x='x', y='y', hue='Cluster Number', palette=hues)
sns.scatterplot(ax=axes[2], data=points, x='x', y='y', color='blue')
axes[2].get_legend().remove()

axes[0].legend(ncol=2, loc='lower right')
fig.text(0.12, 0.94, "(a)")
fig.text(0.12, 0.615, "(b)")
fig.text(0.12, 0.285, "(c)")
plt.subplots_adjust(right=.98,left=0.1, bottom=0.05, top=0.98)
plt.savefig('images/realtime_sample,jpg')