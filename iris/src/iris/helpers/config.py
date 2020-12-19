from time import perf_counter
import configparser
import pathlib

from iris.helpers.logger import LogHandler

DEFAULT_CONFIG = pathlib.Path(__file__).parents[1] / "config.ini"


class Storage:
    def __init__(self, logger: LogHandler, file_path: pathlib.Path = DEFAULT_CONFIG):
        self._logger = logger
        self._config_path = file_path
        self._config = self._open_config()
        self._init_time = perf_counter()

        # Paths
        self._training_data_path = self._pathify(
            self._read_config_entry("paths", "training")
        )
        self._testing_data_path = self._pathify(
            self._read_config_entry("paths", "testing")
        )
        self._centroid_serial_path = self._pathify(
            self._read_config_entry("paths", "centroids"), required=False
        )
        self._mapping_serial_path = self._pathify(
            self._read_config_entry("paths", "mapping"), required=False
        )
        self._results_path = self._pathify(
            self._read_config_entry("paths", "results"), required=False
        )
        self._plot_path = self._pathify(
            self._read_config_entry("paths", "plot"), required=False
        )

        # Settings
        self._stage = self._read_config_entry(
            "settings", "stage", value_options=("training", "testing")
        )
        self._clusters = self._read_config_entry("settings", "clusters")
        self._norm = self._read_config_entry(
            "settings", "norm", value_options=("L1", "L2")
        )
        self._visualize = self._boolify(
            self._read_config_entry("settings", "visualize")
        )
        self._serialize = self._boolify(
            self._read_config_entry("settings", "serialize")
        )
        self._save = self._boolify(self._read_config_entry("settings", "save"))

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def clusters(self) -> int:
        return int(self._clusters)

    @property
    def norm(self) -> str:
        return self._norm

    @property
    def visualize(self) -> bool:
        return self._visualize

    @property
    def serialize(self) -> bool:
        return self._serialize

    @property
    def save(self) -> bool:
        return self._save

    @property
    def training_data_path(self) -> pathlib.Path:
        return self._training_data_path

    @property
    def testing_data_path(self) -> pathlib.Path:
        return self._testing_data_path

    @property
    def centroid_serial_path(self) -> pathlib.Path:
        return self._centroid_serial_path

    @property
    def mapping_serial_path(self) -> pathlib.Path:
        return self._mapping_serial_path

    @property
    def results_path(self) -> pathlib.Path:
        return self._results_path

    @property
    def plot_path(self) -> pathlib.Path:
        return self._plot_path

    @property
    def summary(self) -> str:
        """
        Summary to be logged.
        """
        summary = {
            "Stage": self._stage,
            "Training data location": self._training_data_path if self._stage == "testing" else None,
            "Show plots": self._visualize,
            "Save plots & results": self._save,
            "Plot location": self._plot_path if self._save else None,
            "Results location": self._results_path if self._save else None,
            "Serialize": self._serialize,
            "Centroids location": self._centroid_serial_path if self._serialize else None,
            "Mapping location": self._mapping_serial_path if self._serialize else None,
            "Number of clusters": self._clusters,
            "Norm": self._norm,
            "Time (s)": perf_counter() - self._init_time,
        }
        return "\n\t".join(f"{k}: {v}" for k, v in summary.items() if v is not None)

    def _open_config(self) -> configparser.ConfigParser:
        """
        Open config file and save as ConfigParser object.
        """
        if not self._config_path.is_file():
            msg = f"Config file not found ({self._config_path})"
            self._logger.error(msg, extra={"stage": "config"})
            raise FileNotFoundError(msg)

        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        config.read(self._config_path)
        return config

    def _read_config_entry(
        self, section: str, key: str, required: bool = True, value_options: tuple = ()
    ) -> (str, None):
        """
        Read entry from config file.
        """
        if not required and (
            section not in self._config or key not in self._config[section]
        ):
            return

        try:
            value = self._config[section][key]
        except KeyError as e:
            msg = f"A required config section/key not found ({e})"
            self._logger.error(msg, extra={"stage": "config"})
            raise ValueError(msg)

        if value_options and value not in value_options:
            msg = f"The config value ({value}) must be one of the following: {value_options}"
            self._logger.error(msg, extra={"stage": "config"})
            raise ValueError(msg)

        return value

    def _pathify(self, path: str, required: bool = True):
        """
        Create pathlib.Path object from string.
        """
        path = pathlib.Path(path)
        if not path.is_absolute():
            path = pathlib.Path(__file__).parents[1] / path

        if required and not path.is_file() and not path.is_dir():
            msg = f"Required file not found ({path})"
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        return path.resolve()

    @staticmethod
    def _boolify(value: str) -> bool:
        """
        Convert string to boolean.
        """
        return value.lower() in ("t", "true", "y", "yes")
