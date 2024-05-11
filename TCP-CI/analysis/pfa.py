import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

class PFA(object):
    def __init__(self, diff_n_features = 2, q=None, explained_var = 0.95):
        self.q = q
        self.diff_n_features = diff_n_features
        self.explained_var = explained_var

    def min_components(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA().fit(X)
        
        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
            for i,j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break

        return q

    def fit(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X = X.transpose()

        pca = PCA().fit(X)
        
        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
            for i,j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break
        else:
            q = self.q
                    
        A_q = pca.components_.T[:,:q]
        
        clusternumber = min([q + self.diff_n_features, X.shape[1]])
        
        kmeans = KMeans(n_clusters= clusternumber).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        
    def fit_transform(self,X):    
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X):
        return X[:, self.indices_]


if __name__ == '__main__':
    pfa = PFA(explained_var=0.95, q=8)
    df = pd.read_csv('C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv')
    SID = df["SID"]
    df_ = df[["Avg. TCFR / Build (%)","# Observations","Best APFDc"]]
    
    # melted_df = df.melt(id_vars=['SID'], var_name='Feature', value_name='Value')
    # pivoted_df = melted_df.pivot(index='Feature', columns='SID', values='Value').reset_index()
    # pivoted_df.columns.name = None
    
    # df.drop(['Build', 'Test'], axis=1, inplace=True)

    pfa.fit(df_.values)
    print(len(pfa.indices_))
    selected = list(SID[pfa.indices_])
    print(selected)

    print(df[df["SID"].isin(selected)])
