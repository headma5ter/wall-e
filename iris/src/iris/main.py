from matplotlib import pyplot as plt
import matplotlib.lines as lines
from statistics import mode, StatisticsError
from csv import QUOTE_ALL
import pandas as pd
import pathlib
import json

from iris import logger
from iris import config
from iris import classifier
from iris.helpers.utils import log_function  # TODO: change to ceres

COLUMN_NAMES = ["w", "x", "y", "z"]


@log_function
def read_data(
    csv_path: pathlib.Path = None, serial_path: pathlib.Path = None
) -> (pd.DataFrame, dict):
    """
    Read in either raw CSV data or pickled data.
    :param csv_path: path to CSV data
    :param serial_path: path to pickled/JSON data
    :return: pd.DataFrame
    """
    if serial_path is not None:
        ext = serial_path.suffix
        if ext == ".pkl":
            # Read in centroids serialized file
            return pd.read_pickle(serial_path)
        elif ext == ".json":
            # Read in mapping serialized file
            with open(serial_path) as f:
                return {int(k): v for k, v in json.load(f).items()}
        else:
            msg = f"Unknown file extension ({serial_path})"
            logger.error(msg)
            raise ValueError(msg)

    # Read in default CSV file
    return pd.read_csv(
        csv_path, low_memory=True, header=None, names=COLUMN_NAMES + ["classification"],
    )


@log_function
def classify_clusters(
    df: pd.DataFrame, initial_centroids: pd.DataFrame = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    Send raw data to classifier.py to be run through the
    k-means algorithm in order to cluster the data points.
    :param df: raw data
    :param initial_centroids: centroids from training data (when applicable)
    :return: (finalized data, centroids for testing data)
    """

    initial_centroids = (
        initial_centroids.to_numpy() if initial_centroids is not None else None
    )

    # Initiate algo class
    all_clusters = classifier.LloydsAlgorithm(
        number_of_clusters=config.clusters,
        data_set=df[df.columns[:-1]].to_numpy(),
        initial_centroids=initial_centroids,
    )

    while (
        not all_clusters.is_optimized()
        and all_clusters.iteration <= all_clusters.max_iterations
    ):
        # Update centroids with their new center of mass
        all_clusters.update_centroids()

        # Assign data points to their closest centroid
        all_clusters.assign_clusters()

        # Increase the increment counter by one
        all_clusters.increment_iteration()

    merged_df = df.join(
        pd.DataFrame(all_clusters.clusters, columns=["cluster"], dtype=int)
    )
    centroids_df = pd.DataFrame(all_clusters.centroids, columns=COLUMN_NAMES)

    return merged_df, centroids_df


@log_function
def serialize_data(data_obj: (pd.DataFrame, dict), file_path: pathlib.Path) -> None:
    """
    Pickle training data.
    :param data_obj: Dataframe (post-algorithm) or dict (post-mapping)
    :param file_path: where to write serialized data
    """
    if not file_path.parent.is_dir():
        msg = f"The indicated path cannot be found; perhaps a parent folder is missing? ({file_path})"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if isinstance(data_obj, dict):
        with open(file_path, "w") as f:
            json.dump(data_obj, f)
    else:
        data_obj.to_pickle(file_path)


@log_function
def map_cluster_to_species(df: pd.DataFrame) -> dict:
    """
    Finds the most common species linked to each cluster
    value (for plotting, as well as sanity-checking).
    :param df: data (after running algorithm)
    :return: dict
    """
    cluster_map = {
        row["cluster"]: list(df[df["cluster"] == row["cluster"]]["classification"])
        for _, row in df.iterrows()
    }
    try:
        cluster_map = {int(k): mode(v) for k, v in cluster_map.items()}
    except StatisticsError as e:
        msg = f"Error finding unique mappings for clusters ({e})"
        logger.error(msg)
        raise ValueError(msg)

    if set(cluster_map.values()) != set(df["classification"].unique()):
        logger.warn("Not all classifications are mapped")
        cluster_map.update(
            {
                cluster: "UNMAPPED"
                for cluster in range(config.clusters)
                if cluster not in cluster_map
            }
        )

    return cluster_map


@log_function
def write_to_csv(df: pd.DataFrame, file_path: pathlib.Path) -> None:
    """
    Write final testing dataframe to file.
    :param df: testing df (post-algorithm)
    :param file_path: file path
    """
    if not file_path.parent.is_dir():
        msg = f"The indicated path cannot be found; perhaps a parent folder is missing? ({file_path})"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df.to_csv(file_path, index=False, quoting=QUOTE_ALL)


@log_function
def plot_clusters(df: pd.DataFrame, cluster_map: dict) -> None:
    """
    Create a plot containing (NxM - k) plots, showing
    the relationship between each parameter and the
    clusters contained in each sub-data set.
    :param df: the data after being classified
    :param cluster_map: mapping to convert cluster to Iris species
    """
    variants = [
        col
        for col in df.columns.tolist()
        if col not in ("cluster", "classification", "color", "model_classification")
    ]

    fig, ax = plt.subplots(
        nrows=len(variants), ncols=len(variants), figsize=[12.0, 12.0], squeeze=True
    )

    color_map = {
        k: plt.get_cmap("Dark2")((k + 1) / config.clusters)
        for k in range(config.clusters)
    }
    df["color"] = df["cluster"].apply(lambda x: color_map[x])

    for row_idx, _ in enumerate(ax):
        for col_idx, _ in enumerate(_):
            x_var = variants[col_idx]
            y_var = variants[row_idx]
            curr_plot = ax[row_idx][col_idx]

            if row_idx == col_idx:
                curr_plot.text(
                    0.5,
                    0.5,
                    f"{x_var.upper()}",
                    ha="center",
                    va="center",
                    fontsize="xx-large",
                    label="",
                )
                curr_plot.get_xaxis().set_visible(False)
                curr_plot.get_yaxis().set_visible(False)
            else:
                curr_plot.scatter(df[x_var], df[y_var], c=df["color"])

    fig.suptitle(f"Iris Classification ({config.stage} data)", fontsize="xx-large")
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    handles = list()
    labels = list()
    for classification, color in {
        cluster_map[cluster]: color for cluster, color in color_map.items()
    }.items():
        handles.append(
            lines.Line2D(list(), list(), marker="o", color=color, linestyle="none")
        )
        labels.append(classification)

    plt.legend(handles=handles, labels=labels)

    if config.save:
        plt.savefig(config.plot_path)

    plt.show()


@log_function
def map_species_onto_data(df: pd.DataFrame, cluster_map: dict) -> pd.DataFrame:
    """
    Add species names to the final dataframe.
    :param df: dataframe (after running through algorithm)
    :param cluster_map: mapping data to go from cluster to species
    :return: pd.DataFrame
    """
    df["model_classification"] = df["cluster"].map(cluster_map)
    return df


@log_function
def calculate_statistics(df: pd.DataFrame, cluster_map: dict) -> None:
    """
    Calculates accuracy of model.
    :param df: dataframe (after running through algorithm)
    :param cluster_map: mapping data to go from cluster to species
    """
    num_correct = len(df[df["classification"] == df["cluster"].map(cluster_map)])
    total_num = len(df)
    logger.info(f"Accuracy: {'{:.1%}'.format(num_correct / total_num)} (N={total_num})")


if __name__ == "__main__":
    # Get relevant paths
    data_path = getattr(config, f"{config.stage}_data_path")
    centroid_path = config.centroid_serial_path
    mapping_path = config.mapping_serial_path

    centroids = None
    mapping = dict()
    if config.stage == "testing":
        if not centroid_path.is_file() or not mapping_path.is_file():
            logger.warn(
                "No training data to be read -- could result in poor model performance"
            )
        else:
            # Get centroids and species mapping from training data
            centroids = read_data(serial_path=centroid_path)
            mapping = read_data(serial_path=mapping_path)

    # Classify data set
    data = read_data(csv_path=data_path)
    data, centroids = classify_clusters(data, initial_centroids=centroids)

    if config.stage == "training":
        # Map species to cluster
        mapping = map_cluster_to_species(data)

        if config.serialize:
            # Save data
            serialize_data(centroids, config.centroid_serial_path)
            serialize_data(mapping, config.mapping_serial_path)

    # Add the model's species classification to data
    data = map_species_onto_data(data, mapping)

    if config.save:
        # Save testing results to files
        write_to_csv(data, config.results_path)

    if config.visualize:
        # Plot data
        plot_clusters(data, mapping)

    calculate_statistics(data, mapping)

    logger.info(f"Process complete\n\t{config.summary}")
