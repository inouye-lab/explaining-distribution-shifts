import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
import ot
from torch import Tensor
from copy import deepcopy

class BaseTransport():
    def __init__(self, X, Y, fit=True, alg='EMD', max_iter=1e9, rng=None, **transport_kwargs):
       
        self.source = self._copy(self._to_numpy(X))
        self.target = self._copy(self._to_numpy(Y))
        self.rng = check_random_state(rng)
        
        if 'max_iter' not in transport_kwargs:
            transport_kwargs['max_iter'] = max_iter
        
        if alg.upper() == 'EMD':
            self.transport = ot.da.EMDTransport(**transport_kwargs)
        elif alg.upper() == 'SINKHORN':
            self.transport = ot.da.SinkhornTransport(**transport_kwargs)
        else:
            raise NotImplementedError('Please pick an alg from {emd, sinkhorn}')
        
        if fit:
            self.transport = self.transport.fit(X, Xt=Y)
           
            
    def forward(self, X, Y=None):
        if Y is None:
            Z = self.transport.transform(self._to_numpy(X), Xt=self.target)
        else:
            Z = self.transport.transform(self._to_numpy(X), Xt=self._to_numpy(Y))
        return Z
    
    def feature_forward(self, X, active_set, Y=None, unconstrained_Z=None):
        if unconstrained_Z is None:
            if Y is None:
                Y = self.target
            unconstrained_Z = self.transport.transform(X, Xt=Y)
        constrained_Z = self._copy(X)
        constrained_Z[:, active_set] = unconstrained_Z[:, active_set]
        return constrained_Z
    
    def cluster_forward(self, X, n_clusters, Y=None, unconstrained_Z=None,
                        labels=None, return_labels=False):
        if unconstrained_Z is None:
            if Y is None:
                Y = self.target
            unconstrained_Z = self.transport.transform(X, Xt=Y)
        if labels is None:
            labels = self._pair_clustering(X, unconstrained_Z, n_clusters, self.rng)
        Z_clusters = self._cluster_mean_transport(X, unconstrained_Z, labels)
        if return_labels:
            return Z_clusters, labels
        else:
            return Z_clusters    
    
    def inverse(self, Z, X=None):
        if X is None:
            source_back = self.transport.inverse_transform(self.source, Xt=Z)
        else:
            source_back = self.transport.inverse_transform(X, Z)
        return source_back
    
    def feature_inverse(self, Z, active_set, X=None):
        if X is None:
            X = self.source
        unconstrained_X_back = self.transport.inverse_transform(X, Xt=Z)
        constrained_X_back = self._copy(Z)
        constrained_X_back[:, active_set] += unconstrained_X_back[:, active_set] - Z[:, active_set]
        return constrained_X_back
    
    def fit(self, X, Y):
        X, Y = self._to_numpy(X), self._to_numpy(Y)
        # refits our mapping
        self.target = self._copy(Y)
        self.source = self._copy(X)
        self.transport.fit(X, Xt=Y)
        return self
        
    def fit_transform(self, X, Y, direction):
        X, Y = self._to_numpy(X), self._to_numpy(Y)
        self.fit(X, Y)
        if direction == 'forward':
            return self.forward(X)
        elif direction == 'inverse' or direction == 'backward':
            return self.inverse(Y)
        else:
            raise NotImplementedError(f'{direction} is not a valid direction. Pick forward or backward')
            
    def _cluster_mean_transport(self, X, Z, labels):
        Z_clusters = self._copy(X)  # the final output of the cluster mean shift transport
        for cluster_idx in np.unique(labels):
            X_cluster = X[labels == cluster_idx]
            Z_cluster = Z[labels == cluster_idx]
            # since we are doing mean shift cluster transport,
            # C_z = C_x + mean_shift  (mean_shift = C_z_mu - C_x_mu)
            X_cluster_pushed = X_cluster - X_cluster.mean(axis=0) + Z_cluster.mean(axis=0)
            Z_clusters[labels == cluster_idx] = X_cluster_pushed
        return Z_clusters
    
    @staticmethod
    def _pair_clustering(X, Z, n_clusters, rng=None):
        rng = check_random_state(rng)
        # Pairing X and Z
        XZ = np.concatenate((X, Z), 1)
        XZ_km = KMeans(n_clusters, init='k-means++', random_state=rng).fit(XZ)
        XZ_labels = XZ_km.predict(XZ)

        return XZ_labels
    
    def _to_numpy(self, X):
        # making sure the input is a numpy array
        if X.__class__ is Tensor:
            return X.numpy()
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)
            
    def _copy(self, X):
        # cloning X in its native function
        try:
            if X.__class__ is Tensor:
                return X.clone()
            else:
                return X.copy()
        except AttributeError:  # if X  does not have a copy/clone function
            return deepcopy(X)

class GaussianTransport():
    def __init__(self, X, Y, X_mean=None, X_cov=None, Y_mean=None, Y_cov=None):
        if X_mean is None or Y_mean is None:
            X_mean = X.mean(axis=0)
            Y_mean = Y.mean(axis=0)
        if X_cov is None or Y_cov is None:
            X_cov = np.cov(X, rowvar=False) + 1e-8 * np.eye(X.shape[1])
            Y_cov = np.cov(Y, rowvar=False) + 1e-8 * np.eye(Y.shape[1])
        self.source_mean = X_mean
        self.target_mean = Y_mean
        self.source_cov = X_cov
        self.target_cov = Y_cov
        self.A_forward = self.calculate_A(X_cov, Y_cov)
        self.A_backward = np.linalg.inv(self.A_forward)

    def forward(self, X):
        """Calculates Z, where Z = T(X)"""
        # double transpose since we are trying to apply A to each sample in x
        Z = self.target_mean + (self.A_forward @ (X - self.source_mean).T).T
        return Z
    
    def inverse(self, Z):
        """Finds Z's preimage (i.e. X), where X = T^{-1}(Z)"""
        # here source and target are actually flipped since we are taking the inverse!
        X = self.source_mean + (self.A_backward @ (Z - self.target_mean).T).T
        return X
    
    def inv(self, X):
        # alias for inverse
        return self.inverse(X)
    
    def calculate_A(self, source_cov, target_cov):
        source_root_inv = self._matrix_power(source_cov, -0.5)
        source_root = self._matrix_power(source_cov, 0.5)
        return source_root_inv @ self._matrix_power( source_root @ target_cov @ source_root, 0.5 ) @ source_root_inv

    def _matrix_power(self, matrix, power):
        Sigma, V = np.linalg.eig(matrix)  # performs SVD
        Sigma_p = np.power(Sigma, power)
        return V @ np.diag(Sigma_p) @ V.T

    
# def balanced_K_means_labeling(X, n_clusters, cluster_centers=None, rng=None):
#     """A rather inefficient and greedy method for performing balanced K means clustering, meaning that each cluster
#     has the same number of points in it. The balanced requirement is needed for emperical OT."""
#     rng = check_random_state(rng)
#     X = X.copy()
#     n_data_per_cluster = X.shape[0] // n_clusters
#     n_data = n_data_per_cluster * n_clusters
#     if cluster_centers is None:
#         cluster_centers = KMeans(n_clusters=n_clusters, random_state=rng).fit(X).cluster_centers_
#     distance_from_centers = np.zeros(shape=(X.shape[0], n_clusters))
#     selected_points_mask = np.zeros(shape=X.shape[0], dtype=np.bool)  # True if point has already been selected
#     labels = np.zeros(X.shape[0]) - 1
#     helper_index = list(range(X.shape[0]))
#     for center_idx in range(n_clusters):
#         distance_from_centers[:, center_idx] = np.linalg.norm(X - cluster_centers[center_idx], axis=-1)
#     # assigning points to clusters in a greedy fashion
#     for _ in range(n_data_per_cluster):
#         for cluster_idx in range(n_clusters):
#             argmin_point = distance_from_centers[~selected_points_mask, cluster_idx].argmin()
#             index_corrected_argmin_point = helper_index.pop(argmin_point)
#             labels[index_corrected_argmin_point] = cluster_idx
#             selected_points_mask[index_corrected_argmin_point] = True
#     return labels  # note, some labels are -1, which means they should be thrown away...
# #     return labels[labels != -1]  # makes sure we throw away any points that did not evenly fit into the clusters

# def pair_cluster_labels(X_labels, X_centers, Y_labels, Y_centers):
#     """Takes in two labeled clusters, and reindex's the labels such that the (ith, ith) labels of the X, Y clusters
#     are near each other. For example, the returned result will be labels such that the 0th cluster of X is close to the
#     0th cluster of Y, and same for the 1st, 2nd, etc."""
#     M = ot.utils.euclidean_distances(X_centers, Y_centers, squared=True)
#     a,b = ot.utils.unif(X_centers.shape[0]), ot.utils.unif(Y_centers.shape[0])
#     coupling_mat = ot.emd(a, b, M)
#     label_mapping = dict(zip(*np.nonzero(coupling_mat)))  # creates a mapping from X_label -> Y_label
#     Y_relabeled = Y_labels.copy()
#     # Relabeling Y such that it matches the mapping from X. Thus giving us dist(X_label==i, Y_label==j) is min when i=j
#     for X_label, original_Y_label in label_mapping.items():
#         Y_relabeled[Y_labels == original_Y_label] = X_label
#     return Y_relabeled


def calc_fidelity(Y, Z, M=None):
    # Calculates the W2 distance between Y and Z ( where Z=T(X) ) 
    if M is None:
        M = ot.utils.dist(Y, Z, metric='sqeuclidean')
    a, b = [], []  # sets a and b to be uniformly weighted
    W2 = ot.emd2(a, b, M)
    return W2

def calc_parsimony(X, Z, norm_type='fro'):
    distance = np.linalg.norm(Z - X, ord=norm_type)
    return distance**2

def calc_interpretability():
    # impossible?
    pass

def get_desiderata(X, Y, Z, inter=None, M=None):
    if M is None:
        fid = calc_fidelity(Y, Z)
    else:
        fid = calc_fidelity(Y, Z, M)
    par = calc_parsimony(X, Z)
    if inter is not None:
        print(f'Fid: {fid:.3f}, Par: {par:.3f}, Inter: {inter}')
    return fid, par, inter

def W2_dist(X, Y, squared=True):
    # Calculates and returns the [squared] emperical W2 distance between X and Y 
    a,b = ot.utils.unif(X.shape[0]), ot.utils.unif(Y.shape[0])
    M = ot.utils.euclidean_distances(X, Y, squared=squared)
    W2_sqaured = ot.emd2(a, b, M, numItermax=1e9)
    return W2_sqaured


def get_trajectories_for_plotting(X, Tx, points_to_show=None):
    """Takes x and Tx, and returns a np array such that calling plot on the array should plot
    the trajectories"""
    if points_to_show is not None:  # if we are only plotting a subset of X and Tx
        X, Tx = X[points_to_show], Tx[points_to_show]
    trajectories = np.hstack((X, Tx)).reshape(X.shape[0]*2, X.shape[1])
    # performing subtransposes to account for the way plt.plot(x1, y1, x2, y2, ...) works
    placeholder = trajectories[::2, 1].copy()
    trajectories[::2, 1] = trajectories[1::2, 0]
    trajectories[1::2, 0] = placeholder
    return trajectories