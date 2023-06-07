import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP
from trimap import TRIMAP
from pacmap import PaCMAP
from sklearn.manifold import TSNE
from data_generations import SyntheticDataGenerator

N_DIMS = list(range(3,21))
N_POINTS = [100, 500, 1000, 2500, 5000, 10_000, 20_000, 30_000]

def run_umap_experiments():
    for n_dims in N_DIMS:
        for n_points in N_POINTS:
            print(f"Running UMAP experiments for data with n_dims = {n_dims} and n_points = {n_points}...")
            data_generator = SyntheticDataGenerator(n_dims=n_dims, n_points=n_points)
            X, y = data_generator.make_triple_hypersphere_dataset()

            umap = UMAP(n_components=2) 

            embedding = umap.fit_transform(X)

            embedded_data = np.column_stack((embedding, y))

            np.save(f"./../data/TripleHyperSpheres-n_dims-{n_dims}-n_points-{n_points}", embedded_data)


def run_trimap_experiments():
    for n_dims in N_DIMS:
        for n_points in N_POINTS:
            print(f"Running TriMAP experiments for data with n_dims = {n_dims} and n_points = {n_points}...")
            data_generator = SyntheticDataGenerator(n_dims=n_dims, n_points=n_points)
            X, y = data_generator.make_triple_hypersphere_dataset()

            trimap = TRIMAP()

            embedding = trimap.fit_transform(X)

            embedded_data = np.column_stack((embedding, y))

            np.save(f"./../data/TripleHyperSpheres-TriMAP-n_dims-{n_dims}-n_points-{n_points}", embedded_data)


def run_pacmap_experiments():
    for n_dims in N_DIMS:
        for n_points in N_POINTS:
            print(f"Running PaCMAP experiments for data with n_dims = {n_dims} and n_points = {n_points}...")
            data_generator = SyntheticDataGenerator(n_dims=n_dims, n_points=n_points)
            X, y = data_generator.make_triple_hypersphere_dataset()

            pacmap = PaCMAP()

            embedding = pacmap.fit_transform(X)

            embedded_data = np.column_stack((embedding, y))

            np.save(f"./../data/TripleHyperSpheres-PaCMAP-n_dims-{n_dims}-n_points-{n_points}", embedded_data)


def run_tsne_experiments():
    for n_dims in N_DIMS:
        for n_points in N_POINTS:
            print(f"Running t-SNE experiments for data with n_dims = {n_dims} and n_points = {n_points}...")
            data_generator = SyntheticDataGenerator(n_dims=n_dims, n_points=n_points)
            X, y = data_generator.make_triple_hypersphere_dataset()

            tsne = TSNE()

            embedding = tsne.fit_transform(X)

            embedded_data = np.column_stack((embedding, y))

            np.save(f"./../data/TripleHyperSpheres-tSNE-n_dims-{n_dims}-n_points-{n_points}", embedded_data)


if __name__ == "__main__":
    #run_umap_experiments()
    #run_trimap_experiments()
    #run_pacmap_experiments()
    run_tsne_experiments()