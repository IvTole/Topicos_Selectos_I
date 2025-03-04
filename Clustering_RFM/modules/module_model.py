import pandas as pd
import os
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import matplotlib.pyplot as plt

from module_path import plots_data_path



class ModelClustering:
    """
    Supports the evaluation of clustering models, collecting the results.
    """

    def __init__(self, X: pd.DataFrame):
        """
        :param X: the inputs
        :param random_state: the random seed
        """

        self.X = X

    def evaluate_clustering(self, model):
        """
        :param model: the model to evaluate
        :return: silhouett score
        """

        model.fit(self.X)
        cluster_labels = model.labels_

        # silhouette score
        sil_score = silhouette_score(self.X, cluster_labels)
        print(f'For n_clusters={pd.Series(cluster_labels).nunique()}, the silhouette score is {sil_score}')

        return sil_score
    
    def kelbow_plot(self, model, k: tuple):
        visualizer = KElbowVisualizer(estimator=model, k=k)
        visualizer.fit(self.X)
        visualizer.show(outpath=os.path.join(plots_data_path(), 'KElbow.pdf'))
        plt.close()
        print(f'Saving KElbow plot in {plots_data_path()}')

    