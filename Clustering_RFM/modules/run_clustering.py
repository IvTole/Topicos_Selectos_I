from datetime import datetime
import pandas as pd

# scikit-learn
from sklearn.cluster import KMeans

# External modules
from module_data import Dataset
from module_model import ModelClustering



# Main function
def main():
    # get current date and time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    dataset = Dataset()
    df_rfm = dataset.load_data_frame_rfm_minmaxscale()

    # evaluate models
    ev = ModelClustering(X=df_rfm)
    model = KMeans(n_clusters=4, max_iter=50, init='k-means++', tol=0.001,
                   random_state=42, algorithm='lloyd')
    ev.evaluate_clustering(model=model)
    ev.kelbow_plot(model=model, k=(2,7))
    #ev.silhouette_plot(model=model)


if __name__ == '__main__':
    main()