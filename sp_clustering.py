'''
# resources:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
'''

# Authors: Emily Costa + Spenser Cheung 
# Created on: Apr 26, 2022
from ipaddress import summarize_address_range
from xml.etree.ElementInclude import include
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.spatial import distance
from math import pi
from math import exp
from math import sqrt

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

########################################################################
#  Testing Native Bayes Algorithm 
########################################################################
    def df_mean(self, df): 
        return df.agg("mean")

    def df_stdev(self, df): 
        return df.agg("std")

    def dist_summary_dataset(self,data): 
        #Calculate mean, stdev and count for each Label(?)
        grouped = data.groupby('Label').agg(['mean', 'std', 'size'])
        return grouped

    def summarize_by_class(self, data):
        separated = data.groupby('Label')
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.dist_summary_dataset(rows)
        return summaries

    #Calculate Gaussian prob distribution function for x 
    def calculate_probability(x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
    
    def calculate_class_probabilities(self,summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, count = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities
    
    # Predict the class for a given row
    def predict(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
    
    # Naive Bayes Algorithm
    def naive_bayes(self,train, test):
        summarize = self.summarize_by_class(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return(predictions)
########################################################################
#  End of Naive Bayes Algorithm 
########################################################################

    def _single_point(self, point):
        data = self.data
        # dist = data.apply(lambda x: pd.Series([self._bayesian_distance(x['Point'],point), x['Label'], x['Cluster Number']], index=['Dist', 'Label', 'Cluster Number']), axis = 1) #replace 'Point' column with 'Dist' column
        dist = data.apply(lambda x: pd.Series([distance.euclil(x['Point'],point), x['Label'], x['Cluster Number']], index=['Dist', 'Label', 'Cluster Number']), axis = 1) 
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

    def _bayesian_distance(self, dist_matrix): #point0, point1):
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
        
        """pseudocode  from 2021 Journal 
        https://www.jmlr.org/papers/volume22/20-688/20-688.pdf


        for iteration = 1,2,... do 
            Sample v~N(0,M), set Beta*<-Beta and v*<-v; 
            for l = 1,...,L do 
                Update v* <- v* + (epsilon/2)*(partial gradient of log(Product(B*|D))/partial gradient B*);
                Update B* <- B* + epsilon*M^-1*v*;
                Update v* <- v* + (epsilon/2)*(partial gradient of log(Product(B*|D))/partial gradient B*);
                if (B* - B)^T* v* < 0 then 
                    Break; 
            Sample u ~ Uniform(0,1); 
            if u < min{1, exp[-H(B*,-v*) + H(B,v)]}. then 
                Set B< B*; 

        Algorithm 1: pseudocode of No-U-Turn Hamiltonian MC sampler for Bayesian distance clustering 

        """


        """ Lift-and-project HMC from 2019 revision 
        1) W_i initiallized and sample momentum Q with each q_ij ~ No(0, sig_q**2)
        2) Leap-frog algorithm in simplex inter for L steps using kinetic and potential functions; 
            K(Q) = 1/2(sig_q**2)* tr(Q^t * Q)
            U(W) = -tr{W^t(logD)W lambda (W^t W)^-1} + tr{W^t DW(sigma W^t W)^-1}
        3) . Compute vertex projection C∗i by setting the largest coordinate in Wi to 1 and others to 0 (corresponding to minimizing Hellinger distance between C∗i and Wi
        4) . Run Metropolis-Hastings, and accept proposal C∗ if u < U(C∗)K(Q∗)U(C)K(Q), where u ∼ Uniform(0, 1)
        5) . Sample σh ∼ Inverse-Gamma{(nh − 1)2 + 2, Pi,i0 d[h]i,i0 / 2nh + βσ}, if nh > 1; otherwise update from σh ∼ Inverse-Gamma(2, β
        6) Sample (π1, . . . , πh) ∼ Dir(α + n1, . . . , α + nh)
        7) Sample αh using random-walk Metropolis
        """
        
        bayes_dist = None

        # TO-DO
        # use pandas apply with _calculate_distance?      <-- since we're passing in a single point instead of the Dataframe, i don't think this is possible?
        
    
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
