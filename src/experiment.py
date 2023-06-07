import numpy as np
import pandas as pd

from umap import UMAP
from sklearn.manifold import TSNE
from trimap import TRIMAP
from pacmap import PaCMAP

from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from utils.local_score import LocalMetric

class Experiment:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.X, self.y = dataset[:, :-1], dataset[:, -1]
        self.name = name
    
    def run_tSNE(self):
        tsne = TSNE()
        embedding = tsne.fit_transform(self.X)
        embedded_data = np.column_stack((embedding, self.y))

        return embedded_data

    def run_UMAP(self):
        umap = UMAP() 
        embedding = umap.fit_transform(self.X)
        embedded_data = np.column_stack((embedding, self.y))

        return embedded_data

    def run_PaCMAP(self):
        pacmap = PaCMAP() 
        embedding = pacmap.fit_transform(self.X)
        embedded_data = np.column_stack((embedding, self.y))

        return embedded_data

    def run_TriMAP(self):
        trimap = TRIMAP() 
        embedding = trimap.fit_transform(self.X)
        embedded_data = np.column_stack((embedding, self.y))

        return embedded_data

    def run_IVHD(self):
        pass

    def run_evaluation(self):
        tsne_embedded = self.run_tSNE()
        umap_embedded = self.run_UMAP()
        pacmap_embedded = self.run_PaCMAP()
        trimap_embedded = self.run_TriMAP()

        # SILHOUETTE
        original_silhouette = silhouette_score(self.X, self.y)

        tsne_silhouette = silhouette_score(tsne_embedded[:,:-1], self.y)
        umap_silhouette = silhouette_score(umap_embedded[:,:-1], self.y)
        pacmap_silhouette = silhouette_score(pacmap_embedded[:,:-1], self.y)
        trimap_silhouette = silhouette_score(trimap_embedded[:,:-1], self.y)

        # TRUSTWORTHINESS
        original_trustworthiness = trustworthiness(self.X, self.X)

        tsne_trustworthiness = trustworthiness(self.X, tsne_embedded[:,:-1])
        umap_trustworthiness = trustworthiness(self.X, umap_embedded[:,:-1])
        pacmap_trustworthiness = trustworthiness(self.X, pacmap_embedded[:,:-1])
        trimap_trustworthiness = trustworthiness(self.X, trimap_embedded[:,:-1])

        evaluation_df = pd.DataFrame(
            {
                "method": ["Original", "tSNE", "UMAP", "PaCMAP", "TriMAP"],
                "silhouette_score": [original_silhouette, tsne_silhouette, umap_silhouette, pacmap_silhouette, trimap_silhouette],
                "trustworthiness": [original_trustworthiness, tsne_trustworthiness, umap_trustworthiness, pacmap_trustworthiness, trimap_trustworthiness]
            }
        )
        evaluation_df.to_csv(f"{self.name}-n_points-{self.X.shape[0]}-n_dims-{self.X.shape[1]}.csv")

        # KNN GAIN and DR QUALITY
        lm = LocalMetric()

        lm.calculate_knn_gain_and_dr_quality(
            X_lds=tsne_embedded[:,:-1], 
            X_hds=self.X,
            labels=self.y,
            method_name='T-SNE'
        )

        lm.calculate_knn_gain_and_dr_quality(
            X_lds=umap_embedded[:,:-1], 
            X_hds=self.X,
            labels=self.y,
            method_name='UMAP'
        )

        lm.calculate_knn_gain_and_dr_quality(
            X_lds=pacmap_embedded[:,:-1], 
            X_hds=self.X,
            labels=self.y,
            method_name='PaCMAP'
        )

        lm.calculate_knn_gain_and_dr_quality(
            X_lds=trimap_embedded[:,:-1], 
            X_hds=self.X,
            labels=self.y,
            method_name='TriMAP'
        )

        lm.visualize()

if __name__ == "__main__":
    data = np.load("./triple_hyperspheres-n_points-3000-n_dims-4.npy")
    
    experiment = Experiment(dataset=data, name="Test")
    experiment.run_evaluation()