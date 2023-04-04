import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import time

### converts csv file with embeddings to pandas dataframe ###
df = pd.read_csv('foo.csv')

### taking every entry expect the last; last contains the name ###
X = df.iloc[:, :510].values


### main function, takes different inputs ###
def cluster(amount, X, quantile, samples, bin_seeding, random_state, max_iter):
    start_Z = time.time()
    ### Subset of X
    Z = Z = X[:amount] 

    ### estimates the bandwidth for mean-shift
    bandwidth_Z = estimate_bandwidth(Z, quantile=quantile, n_samples=samples, random_state=random_state) 
    ms_Z = MeanShift(bandwidth=bandwidth_Z, bin_seeding=bin_seeding, max_iter=max_iter)
    ### fits the points to the means
    y_ms_Z = ms_Z.fit(Z)
    ###  labels the means
    labels_Z = ms_Z.labels_
    cluster_centers_Z = ms_Z.cluster_centers_
    labels_unique_Z = np.unique(labels_Z)
    ### length of labels is equal to the amount of clusters generated
    n_clusters_Z_ = len(labels_unique_Z)
    end_Z = time.time()
    total_time_Z = end_Z - start_Z
    print("---------------------------------------------------------------")
    print("total time Z: " + str(total_time_Z))
    print("number of estimated clusters for Z: %d" % n_clusters_Z_)
    print("bandwidth: " + str(bandwidth_Z))
    print("---------------------------------------------------------------")



cluster(2500, X, 0.3, 2500, False, None, 1)

"""
Examples of inputs to compute
cluster(10000, X, 0.3, 100, False, None, 300)
cluster(10000, X, 0.3, 750, False, None, 300)
cluster(10000, X, 0.3, 1250, False, None, 300)
cluster(10000, X, 0.3, 2500, False, None, 300)
cluster(10000, X, 0.3, 5000, False, None, 300)
cluster(10000, X, 0.3, 10000, True, None, 300)
cluster(10000, X, 0.3, 10000, False, 42, 300)
cluster(10000, X, 0.3, 10000, False, 0, 300)
cluster(10000, X, 0.3, 10000, False, 1000, 300)
cluster(10000, X, 0.3, 10000, False, 42, 600)
cluster(10000, X, 0.3, 10000, False, 42, 1200)
"""

