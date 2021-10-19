from typing import Any, Dict, List

import pandas as pd

from carla.data.load_catalog import load_catalog

from ..api import Data
from .load_data import load_dataset


class DataCatalog(Data):
    def __init__(self, data_name: str):
        """
        Constructor for catalog datasets.

        Parameters
        ----------
        data_name : String
            Used to get the correct dataset from online repository
        """
        self.name = data_name

        catalog_content = ["continous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load_catalog(
            "data_catalog.yaml", data_name, catalog_content
        )

        for key in ["continous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        self._raw: pd.DataFrame = load_dataset(data_name)

    @property
    def categoricals(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continous(self) -> List[str]:
        return self.catalog["continous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()
