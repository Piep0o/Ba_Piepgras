import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import numpy as np
import time

df = pd.read_csv('foo.csv')

X = df.iloc[:, :510].values


def cluster_with_set_x(amount, X, quantile):
    start_Z = time.time()
    Z = Z = X[:amount]

    bandwidth_Z = estimate_bandwidth(Z, quantile=quantile)
    ms_Z = MeanShift(bandwidth=bandwidth_Z)
    y_ms_Z = ms_Z.fit(Z)
    labels_Z = ms_Z.labels_
    cluster_centers_Z = ms_Z.cluster_centers_
    labels_unique_Z = np.unique(labels_Z)
    n_clusters_Z_ = len(labels_unique_Z)
    end_Z = time.time()
    max_iter = ms_Z.n_features_in_
    total_time_Z = end_Z - start_Z
    print("---------------------------------------------------------------")
    print("total time Z: " + str(total_time_Z))
    print("number of estimated clusters for Z: %d" % n_clusters_Z_)
    print("Max features: " + str(max_iter))
    print("---------------------------------------------------------------")

cluster_with_set_x(10000, X, 0.3)

