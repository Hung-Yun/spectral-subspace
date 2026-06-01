#%% imports

import numpy as np
import scipy
from numpy.random import default_rng
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score



class PCAResults:
    pass


class FAResults:

    def __init__(self):
        self.fa = None # The FactorAnalysis object from sklearn
        self.cv_scores = None # Cross-validation scores for each number of components
        self.cv_components = None # The numbers of components used for cross-validation
        self.n_components = None # The number of components selected (either from CV or provided directly)
        self.shared_var_per_unit = None # The proportion of shared variance for each unit
        self.explained_variance_ratio_ = None # The proportion of shared variance explained by each component
        self.d_shared = None # The number of shared dimensions (based on shared_var_thresh)
        self.subspace = None # The orthonormal basis for the subspace defined by the first d_shared dimensions
        self.first_d_components = None # The first d_shared components of the loading matrix L
        self.total_variance = None # The total covariance matrix of the data as modeled by FA (shared + private variance)
        self.private_variance = None # The private variance (noise variance) for each unit
        self.data = None # The original data matrix used for fitting FA
        self.transformed_latents = None # The transformed latent variables Z (posterior mean) from FA

    def __repr__(self):
        if self.fa is not None:
            return f'{self.subspace.shape[0]} components from {self.subspace.shape[1]} channels'
        else: 
            return None


    def fa_transform(self, orthonormalize=True):
        """
        Transform data X using a fitted Factor Analysis model fa.
        X is a (N x D) matrix of observations, where N is the number of samples and D is the number of features.
        fa is a fitted FactorAnalysis object.
        orthonormalize is a boolean indicating whether to orthonormalize the latent variables.
        
        Returns the transformed latent variables Z, and optionally the orthonormal basis L_tilde.
        If orthonormalize is True: returns Z_tilde and L_tilde, where:
        - Z_tilde is the transformed latent variables in the orthonormal basis.
        - L_tilde is the (DxK) orthonormal basis for the subspace defined by the first K dimensions.
        If orthonormalize is False: returns Z, the posterior mean of the latent variables.
        """
        # get posterior mean of latent variables
        L = self.all_components.T # (D x n_components)
        Ph = self.private_variance # (D,)
        P = (L.T @ np.linalg.inv(L @ L.T + np.diag(Ph))).T # (D x n_components)
        Z = np.dot(self.data - self.fa.mean_, P) # (N x n_components)
        if not orthonormalize:
            self.transformed_latents = Z
        else:
            # spikes arise from latents as:
            #   Y = Z @ L.T = (Z @ V @ S) @ U.T
            #       where U.T is an orthonormal basis
            #   so we will define Z_tilde = Z @ V @ S
            #   and L_tilde = U (orthonormal basis for subspace)
            [U,S,V] = np.linalg.svd(L, full_matrices=False)
            Z_tilde = Z @ V @ np.diag(S) # (N x n_components), orthonormalized latent variables
            self.transformed_latents = Z_tilde

    def fa_crossval_scores(self, verbose=True):
        """
        Compute cross-validation scores for Factor Analysis with varying numbers of components.
        X is a (N x D) matrix of observations, where N is the number of samples and D is the number of features.
        n_components is a list of numbers of components to use for cross-validation.
        Returns a numpy array of cross-validation scores for each number of components.
        If verbose is True, it will print the cross-validation scores for each number of components.
        """
        if self.n_components is not None:
            raise Exception('Cannot compute cross-validation scores if n_components is already set.')
        n_components = self.cv_components

        fa_scores = []
        for n in n_components:
            fa = FactorAnalysis()
            fa.n_components = n
            fa_scores.append(np.mean(cross_val_score(fa, self.data)))
            if verbose:
                print(f'CV score for {n} components: {fa_scores[-1]}')
        return np.array(fa_scores)

    def fa_fit(self, X, n_components=None, cv_components=None, shared_var_thresh=0.95, max_iter:int=int(1e6), svd_method='lapack', tol=1e-6, verbose=False):
        """
        X is a (N x D) matrix of observations, where N is the number of samples and D is the number of features.
        n_components is the number of latent dimensions to fit, or None to use all features.
        cv_components is a list of numbers of components to use for cross-validation, or None to skip CV.
        shared_var_thresh is the proportion of shared variance to use for determining d_shared.
        verbose is a boolean indicating whether to print progress messages.
        Returns a dictionary with the fitted FactorAnalysis object, cross-validation scores, number of components used,
        proportion of shared variance, and the orthonormal basis for the subspace defined by the first d_shared dimensions.
        If verbose is True, it will print the number of shared dimensions found.
        
        The params n_components and cv_components determine how many latent dimensions to fit.
        - If n_components is provided, FA will find that many components directly.
        - If cv_components is provided, the selected number of FA components is based on cross-validation scores.
        - If neither is provided, FA will use all features, and d_shared will be determined by shared_var_thresh.
        """


        # apply cross-validation if needed
        scores = None
        if cv_components is not None:
            # do cross-validation to select number of components
            if n_components is not None:
                raise Exception('Cannot provide both n_components and cv_components.')
            scores = self.fa_crossval_scores(cv_components)
            n_components = cv_components[np.argmax(scores)]
        elif n_components is None:
            n_components = X.shape[1]

        # Store parameters
        self.cv_components = cv_components
        self.n_components = n_components
        self.shared_var_thresh = shared_var_thresh

        # fit FA
        self.fa = FactorAnalysis(n_components=self.n_components, 
                                 svd_method=svd_method, tol=tol, max_iter=max_iter)
        self.fa.fit(X)
        L = self.fa.components_.T # (D x n_components)

        # get proportion of shared variance and d_shared
        [U,shared_var,V] = np.linalg.svd(L @ L.T)
        prop_shared_var = shared_var / shared_var.sum()
        if self.n_components is None or self.n_components == X.shape[1]:
            d_shared = np.where(np.cumsum(prop_shared_var) >= self.shared_var_thresh)[0][0] + 1
        else:
            d_shared = self.n_components
        if verbose:
            print(f'{d_shared=}')
        self.d_shared = d_shared


        # get shared variance per unit
        Lc = L[:,:d_shared] # (D x d_shared)
        Ph = self.fa.noise_variance_ # (D,)
        Cov = Lc @ Lc.T + np.diag(Ph) # (D x D)
        shared_var_per_unit = np.diag(Lc @ Lc.T) / np.diag(Cov) # (D,)

        # get orthonormal basis for subspace defined by first d_shared dims
        [Lrot,s,v] = np.linalg.svd(Lc, full_matrices=False)

        # Assign results to self
        self.cv_scores = scores
        self.shared_var_per_unit = shared_var_per_unit
        self.explained_variance_ratio_ = prop_shared_var
        self.subspace = Lrot.T
        self.all_components = L.T
        self.first_d_components = Lc.T
        self.total_variance = Cov
        self.private_variance = Ph
        self.data = X

# %%
