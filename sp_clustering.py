'''
# resources:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://github.com/leoduan/BayesDistanceClustering/blob/main/hmc.ipynb
'''

# Authors: Emily Costa + Spenser Cheung 
# Created on: Apr 26, 2022
from ipaddress import summarize_address_range
from secrets import token_urlsafe
from xml.etree.ElementInclude import include
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.spatial import distance
from math import pi, exp, sqrt
import torch
import hamiltorch
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
import pymc3 as pm3
import arviz
import warnings #remove later -> using to make readability of output better


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
        # dist = data.apply(lambda x: pd.Series([self._bayesian_distance(x['Point'],point), x['Label'], x['Cluster Number']], index=['Dist', 'Label', 'Cluster Number']), axis = 1) #replace 'Point' column with 'Dist' column
        dist = data.apply(lambda x: pd.Series([distance.euclidean(x['Point'],point), x['Label'], x['Cluster Number']], index=['Dist', 'Label', 'Cluster Number']), axis = 1) 
        classification = data['Label'].values[dist['Dist'].idxmin()]
        
        # euclid_dist = data.apply(lambda d: distance.euclidean(d['Point'],point), axis=1) # this will be changed to Bayesian later on
        # old_dist = data.apply(lambda d: self._bayesian_distance(d['Point'],point), axis=1) #df is single column of just dist
        # old_classification = data['Label'].values[old_dist.idxmin()]
        return classification

    def _ward_point(self, point):
        # TO-DO
        pass

    # def prob_calculation(self, df): 
    #     probs = df.groupby('Cluster Number').size().div(len(df))
    #     return probs

    def one_hot(a, num_classes): 
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def closure(self, optimizer, theta): 
        optimizer.zero_grad()
        loss = -self.log_prob(theta) 
        loss.backward() 
        return loss

    def extractW2(i,n,K,theta):
        idx = 0
        idx1 = n*(K-1)

        t = 1E-1

        W0 = theta[ idx: idx1].reshape([n,(K-1)])
        W1 = torch.hstack([W0,torch.zeros(n,1)])
        W = torch.softmax((W1)/t,1)
        return W@W.T

    def extractC(self, i, n, K):
        theta = self.param_trace[i, ]
        idx = 0
        idx1 = n*(K-1)

        t = 1E-1

        W0 = theta[ idx: idx1].reshape([n,(K-1)])
        W1 = torch.hstack([W0,torch.zeros(n,1)])
        C = torch.softmax((W1)/t,1)@ np.arange(K)
        return C
    
    def _bayesian_distance(self, dist_matrix, theta): #point0, point1):
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


        #point1 = single point
        #point0 = all points in dataframe 
        # point0 = np.array(point0)
        # point1 = np.array(point1)   
        # bayes_dist = np.linalg.norm(point0 - point1)
        

        #calculations done on Gaussian "blobs" for clustering: 
            #blobs = datasets.make_blobs(n_samples=n_samples, random_stat=0, shuffle=False)
            #y = blobs[0]

        D = distance.squareform(distance.pdist(dist_matrix)) #input = 'blobs'
        logD = np.log(D)
        np.fill_diagonal(logD,0)
        D_tc = torch.tensor(D)
        logD_tc = torch.tensor(logD)
        svdD = np.linalg.svd(D) 
        p=2  
        trunc = p+2 
        n=y.shape[0] 
        K=32
        P=n*(K-1)+K+K  

        #log_prob method
        idx = 0
        idx1 = n*(K-1)
        
        W0 = theta[ idx: idx1].reshape([n,(K-1)])
        W1 = torch.hstack([W0,torch.zeros(n,1)])
        W = torch.softmax((W1)/t,1)
        
        prior_W0 =  - 1/9.0/2.0 * (W0**2).sum()
        
        
        idx = idx1
        idx1 += K
        sigma = torch.nn.functional.softplus(theta[ idx: idx1])
        
        
        idx = idx1
        idx1 += K
        alpha = torch.nn.functional.softplus(theta[ idx: idx1])
        
        prior_alpha =  (- alpha).sum()
        
        n_h = torch.diag(W.T@W)
        
        prior_sigma = -1000*(sigma**2).sum()
    
        loglik = torch.trace(W.T@logD_tc@W*(alpha-1)/(n_h)) - torch.trace(W.T@D_tc@W/(n_h*sigma)) +  \
            (- n_h* ( alpha* torch.log(sigma) + torch.lgamma(alpha))).sum() + prior_sigma + prior_W0 + prior_alpha
        
        #initialization using spectral clustering: 
        scFit = cluster.spectral_clustering(logD -D + 1E6, n_clusters=K) 

        W_init = torch(self.one_hot(scFit,3))
        W_init = torch.log((W_init+0.1))
        W_init -= torch.reshape(W_init[:,K-1],[n,1])

        theta = torch.randn(P)
        theta[:(n*(K-1))] = (W_init[:,:(K-1)]).flatten()
        theta = theta.requires_grad_() 
        optimizer = torch.optim.LBFGS([theta], lr=1E-1)

        for t in range(100): 
            optimizer.step(self.closure(optimizer,theta))
        
        idx = 0
        idx1 = n*(K-1)

        t = 1E-1

        W0 = theta[ idx: idx1].reshape([n,(K-1)])
        W1 = torch.hstack([W0,torch.zeros(n,1)])
        W = torch.softmax((W1)/t,1)

        idx = idx1
        idx1 += K
        sigma = self.softplus(theta[ idx: idx1])

        idx = idx1
        idx1 += K
        alpha = self.softplus(theta[ idx: idx1])

        n_h = torch.diag(W.T@W)

        W_np = W.detach().cpu().numpy() 
        
        
        #plots: 
        plt.plot((W_np@ np.diag(np.arrange(3))).sum(1))
        #plt.plot(blobs[1])
        plt.imshow((W@W.T).detach().numpy())

        H = torch.autograd.functional.hessian(self.log_prob, theta).diagonal()
        inv_mass = 1.0/H.abs()

        #HMC NUTS
        step_size = 5E-2
        hamiltorch.set_random_seed(123)
        params_init = theta

        num_samples = 3000 # For results in plot num_samples = 12000
        L = 100
        burn = 1000 # For results in plot burn = 2000

        params_hmc_nuts = hamiltorch.sample(log_prob_func=(self.log_prob),
                                            params_init=params_init, num_samples=num_samples,
                                            step_size=step_size, num_steps_per_sample=L,
                                            desired_accept_rate=0.6,
                                            sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn,
                                            inv_mass = inv_mass
                                        )
        
        #extractW2: 
        param_trace = torch.vstack(params_hmc_nuts)
        trace_np = param_trace.detach().cpu().numpy()
        plt.plot(trace_np[:,1]*1)

        theta = param_trace[1, ]
        idx = 0
        idx1 = n*(K-1)
        t = 1E-1
        W0 = theta[ idx: idx1].reshape([n,(K-1)])
        W1 = torch.hstack([W0,torch.zeros(n,1)])
        W = torch.softmax((W1)/t,1)
        W_np = W.detach().numpy()
        plt.plot((W_np@ np.diag(np.arange(3))).sum(1))
        
        theta = param_trace[99, ]

        
        idx = 0
        idx1 = n*(K-1)

        t = 1E-1

        W0 = theta[ idx: idx1].reshape([n,(K-1)])
        W1 = torch.hstack([W0,torch.zeros(n,1)])
        W = torch.softmax((W1)/t,1)
        W_np = W.detach().numpy()

        plt.plot((W_np@ np.diag(np.arange(3))).sum(1))
        

        W2 = torch.zeros([n,n])
        for i in range(param_trace.shape[0]):
            W2 += self.extractW2(i,n, K,theta)
        W2_var = (W2/param_trace.shape[0])
        plt.imshow(W2_var.detach().numpy(),vmin=0, vmax=1)

        C_trace = torch.stack([self.extractC(i,n,K) for i in range(2000)]).round()

        acf_mat = np.vstack([statsmodel.tsa.stattools.acf(param_trace[:,i], fft=False, nlags=40) for i in range(300)])
        acf_mat[np.isnan(acf_mat)]=0
        acf_mat[:,0]=1

        ess = np.stack([arviz.ess(param_trace[:,i].numpy()) for i in range(P)])

        #plots
        fig, ax = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 1] })
        fig.set_size_inches([8,6])

        ax[0,0].plot(param_trace[:,(299)*(K-1)])
        ax[0,0].set_title("Traceplot of $v_{300,1}$", y=-0.3)
        ax[0,1].plot(np.arange(2000),C_trace[:,299]+1)
        ax[0,1].set_title("Traceplot of $c_{300}$", y=-0.3)
        ax[0,1].set_ylim([0.5,3.5])
        ax[0,1].set_yticks([1,2,3])

        ax[1,0].boxplot(acf_mat[:,:40],  showfliers=False, )
        ax[1,0].set_xticks( np.arange(6)*5+1)
        ax[1,0].set_xticklabels(np.arange(6)*5)
        ax[1,0].set_title("ACF of all the parameters", y=-0.3)

        ax[1,1].boxplot(ess/2000,  showfliers=False, )
        ax[1,1].set_xticks( [1])
        ax[1,1].set_xticklabels([""])
        ax[1,1].set_title("ESS per HMC iteration of all the parameters", y=-0.3)
        fig.tight_layout(pad=1)
        # fig.savefig("benchmark_hmc.png")
       
        idx = 0
        idx1 = n*(K-1)
        W0 = theta[ idx: idx1].reshape([n,(K-1)])
        W1 = torch.hstack([W0,torch.zeros(n,1)])
        W = torch.softmax((W1)/t,1)

        prior_W0 =  - 1/9.0/2.0 * (W0**2).sum()
        idx = idx1
        idx1 += K
        sigma = self.softplus(theta[ idx: idx1])
        idx = idx1
        idx1 += K
        alpha = self.softplus(theta[ idx: idx1])
        prior_alpha =  (- alpha).sum()
        n_h = torch.diag(W.T@W)

        mapC = C_trace.mode(0)[0]
        uncertainty = ((C_trace == mapC)*1.0).mean(0)

        #cluster plots 
        fig, ax = plt.subplots(1,3, gridspec_kw={'width_ratios': [1, 1,1.2] })
        fig.set_size_inches([8,2.5])

        ax[0].scatter(blobs[0][:,0],blobs[0][:,1], c=blobs[1], alpha=0.5, s=20)
        ax[0].set_title("Three clusters", y= -0.4)


        ax[1].scatter(blobs[0][:,0],blobs[0][:,1], c=mapC.numpy(), alpha=0.5, s=20, cmap='jet')
        ax[1].set_title("Estimated  $\hat c_i$", y= -0.4)


        im1 = ax[2].scatter(blobs[0][:,0],blobs[0][:,1], c=1-uncertainty, alpha=0.5, s=20, cmap='jet', vmin=0,  vmax=0.7)
        fig.colorbar(im1 ,ax=ax[2])
        ax[2].set_title("Estimated uncertainty $pr(c_i \\neq \hat c_i)$", y= -0.4)


        fig.tight_layout(pad=1.5)
        fig.savefig("benchmark_hmc_plot.png")
        bayes_dist = None
        return bayes_dist

if __name__=="__main__":
    warnings.filterwarnings("ignore", category=FutureWarning) #filters FutureWarnings

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
    # print(clustering.get_clustered_data())
    print(clustering.add_point([0,0]))
