import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class UDM:
    """
    Implementation of Unified Distance Metric, introduced in:

    Zhang, Yiqun, and Yiu-Ming Cheung. 
    "A New Distance Metric Exploiting Heterogeneous Interattribute Relationship for Ordinal-and-Nominal-Attribute Data Clustering."
    IEEE Transactions on Cybernetics (2020).
    """
    def __init__(self, X: np.ndarray, ordinal: list): 
        """
        Initialise the metric by precomputing entropy and interdependence measures on a reference dataset X.

        Args:
            X:       Array containing N samples with d features.
            ordinal: Binary list of length d indicating whether each feature is ordinal (True) or nominal (False).

        Returns:
            None
        """
        # === Collect measurements from dataset.===
        N, self.d = X.shape # Number of samples and features.
        categories = [np.unique(X[:,r]) for r in range(self.d)] 
        n = [len(c) for c in categories] # Number of categories per feature.
        for r in range(self.d): assert (categories[r] == range(n[r])).all(), "Categories must be 0,1,...n[r] for each r."
        Z = N*(N-1)/2 # Used for normalisation.

        # === Initialise data structures that define the distance metric. ===
        self.R = np.zeros((self.d, self.d)) # Pairwise inderdependence measure between features.
        self.psi = [np.zeros((self.d, n[r], n[r])) for r in range(self.d)] # Entropy-based distance for each feature r w.r.t. each other feature s.
        self.phi = [np.zeros((n[r], n[r])) for r in range(self.d)] # Overall distance metric for each feature.

        for r in range(self.d): # Iterate through feature pairs.
            for s in range(self.d):
 
                # === Calculate interdependence measure R. ===
                C = np.zeros((n[r], n[s]), dtype=int)
                for i in range(N): C[X[i,r], X[i,s]] += 1 # Counts for feature combinations.
                C_eq = (C * np.maximum(C-1, 0)).sum() / 2 # Number of equal-concordant sample pairs. 
                C_diff = 0 # Net difference between positive- and negative-concordant sample pairs.
                if ordinal[r] and ordinal[s]: # If both r and s are ordinal.
                    for t in range(n[r]-1):
                        Cul = 0; Cuu = C[t+1:,1:].sum() # Sums of quadrants of C matrix below current t, g.             
                        for g in range(n[s]):
                            C_diff += C[t,g] * (Cuu - Cul)                            
                            if g < n[s]-1: Cul += C[t+1:,g].sum(); Cuu -= C[t+1:,g+1].sum()
                    C_diff = abs(C_diff) # Just need absolute value of net difference.  
                else: # If at least one of r and s is nominal.
                    for t in range(n[r]):
                        for h in range(t):
                            for g in range(n[s]):
                                for u in range(g): C_diff += abs((C[t,g] * C[h,u]) - (C[t,u] * C[h,g]))                
                self.R[r,s] = (C_eq + C_diff) / Z # Final calculation to get R.     

                # === Calculate entropy-based distance psi. ===
                P = C / N # Joint probabilities found by dividing C by N.
                S_A_s = np.log2(n[s]) # Maximum-entropy normalisation term.
                if ordinal[r]: # If r is ordinal.
                    for t in range(1,n[r]): # Consider adjacent categories only.
                        P_sum = P[t] + P[t-1]
                        self.psi[r][s,t,t-1] = sum([-p * np.log2(p) for p in P_sum if p > 0]) / S_A_s # Normalised entropy of summed joint distributions.                    
                    for t in range(1,n[r]): # Fill in remaining by summation to preserve monotonicity.
                        for h in range(t-1):
                            for g in range(h, t): self.psi[r][s,t,h] += self.psi[r][s,g+1,g]
                    self.psi[r][s] += self.psi[r][s].T # Symmetric.        
                else: # If r is nominal.
                    for t in range(n[r]): # Consider all pairs of categories.
                        for h in range(t): 
                            P_sum = P[t] + P[h]
                            self.psi[r][s,t,h] = self.psi[r][s,h,t] = sum([-p * np.log2(p) for p in P_sum if p > 0]) / S_A_s 

                # === Add to overall per-feature distance phi. ===
                self.phi[r] += self.R[r,s] * self.psi[r][s] / self.d

    def __call__(self, XA: np.ndarray, XB: np.ndarray=None, mask: np.ndarray=None, placeholder=np.nan): 
        """
        Pairwise distance between sample sets XA, XB according to the Unified distance Metric.

        Args: 
            XA:     Array containing NA samples with d features.
            XB:     Optional second array containing NB samples with d features.
                        NOTE: If XB is None, implicitly use XB = XA, NB = NA.
            mask:   Optional NA x NB array indicating whether to complete (True) or skip (False) each pair.
                        NOTE: If XB is None, calculation is done for pair i,j if mask[i,j] = True *OR* mask[i,j] = True.

        Returns:
            dist:   NA x NB array of distances.
        """
        if len(XA.shape) == 1: XA = XA.reshape(1,-1)
        NA, dA = XA.shape; indicesA = np.array(range(NA)).reshape(-1,1)
        if XB is None: 
            NB, dB = NA, dA
            indices = (indicesA,)
            func = lambda *args: squareform(pdist(*args))
        else:       
            if len(XB.shape) == 1: XB = XB.reshape(1,-1) 
            NB, dB = XB.shape
            indices = (indicesA, np.array(range(NB)).reshape(-1,1))
            func = cdist
        assert dA == dB == self.d
        if mask is None: mask = np.ones((NA, NB), dtype=bool)
        else: assert mask.dtype == bool and mask.shape == (NA, NB)
        if XB is None: XB = XA; _mask = np.logical_or(mask, mask.T) # Apply or operation to mask if XB is None.
        else: _mask = mask
        dist = func(*indices, # Using indices instead of samples allows masking.
                    lambda i, j: 
                        np.linalg.norm([self.phi[r][xir,xjr] for r, (xir, xjr) in enumerate(zip(XA[i[0]], XB[j[0]]))])
                    if _mask[i,j] else placeholder)
        dist[~mask] = placeholder # Reapply mask to pairs whose mirror has been computed.
        return np.squeeze(dist)