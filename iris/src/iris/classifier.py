import numpy as np

from iris import config
from iris.helpers.utils import log_function  # TODO: change to ceres


class LloydsAlgorithm:
    def __init__(
        self,
        number_of_clusters: int,
        data_set: np.array,
        initial_centroids: np.array = None,
        max_iterations: int = 100,
        min_notable_shift: float = 0.05,
        decimal_places: int = 2,
    ):
        self._number_of_clusters = number_of_clusters
        self._data_set = data_set
        self._initial_centroids = initial_centroids
        self._max_iterations = max_iterations
        self._min_notable_shift = min_notable_shift
        self._decimal_places = decimal_places
        self._exponent = 2 if config.norm == "L2" else 1

        self._old_centroids = None
        self._clusters = None
        self._centroids = None
        self._iteration = 0

        self._find_starting_centroid()
        self.assign_clusters()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def clusters(self) -> np.array:
        return self._clusters

    @property
    def centroids(self) -> np.array:
        return self._centroids

    def increment_iteration(self) -> None:
        self._iteration += 1

    @log_function
    def _find_starting_centroid(self) -> None:
        """
        Initiates the centroids with an educated guess around
        the most populated areas of each distribution.
        :return: np.array
        """
        if self._initial_centroids is not None:
            # If centroids were given via training data
            self._centroids = self._initial_centroids
            return

        # Init centroids array
        self._centroids = np.zeros(
            shape=(self._number_of_clusters, self._data_set.shape[1])
        )

        # Make sure # of bins > # of clusters
        num_bins = 10 + (self._number_of_clusters * (self._number_of_clusters >= 10))
        for col_idx, column in enumerate(self._data_set.T):
            # Create a histogram for each variable
            hist, bins = np.histogram(column, bins=num_bins)
            hist_hash = {bin_val: hist_num for (hist_num, bin_val) in zip(hist, bins)}

            # Take the locations centering the k highest histogram values
            self._centroids[:, col_idx] = [
                c
                for c, _ in sorted(
                    hist_hash.items(), key=lambda item: item[1], reverse=True
                )
            ][: self._number_of_clusters]

    @log_function
    def update_centroids(self) -> None:
        """
        Shift the centroids to be in the center of
        mass of their respective (updated) clusters.
        """
        # Update previous centroids (for later comparison)
        self._old_centroids = self._centroids.copy()

        # Sets the centroids to the center of mass of each cluster
        self._centroids = np.empty(shape=self._centroids.shape)
        for idx, centroid in enumerate(self._centroids):
            cluster_data = self._data_set[np.where(self._clusters == idx)]

            if not cluster_data.size:
                continue
            self._centroids[idx, :] = np.average(cluster_data, axis=0)

    @log_function
    def assign_clusters(self) -> None:
        """
        Assigns clusters based on distance from each point
        to all centroids.
        """
        # Init cluster array
        self._clusters = np.zeros(shape=(self._data_set.shape[0]))
        for idx, row in enumerate(self._data_set):
            # Assign the centroid that minimizes the distance
            self._clusters[idx] = np.argmin(
                np.power(
                    np.sum(abs(row - self._centroids) ** self._exponent, axis=1),
                    1 / self._exponent,
                )
            )

    def is_optimized(self) -> bool:
        """
        Checks if the last shift was small enough to
        call the centroids optimized.
        :return: bool
        """
        if self._old_centroids is None:
            return False

        shift = np.ravel(self._old_centroids - self._centroids, order="K")
        return np.dot(shift, shift) <= self._min_notable_shift
