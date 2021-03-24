import pandas as pd

from mlos.DataSets.DataSetInterface import DataSetInterface
from mlos.Spaces import Hypergrid


class SimpleDataSet(DataSetInterface):
    """Maintains a dataframe and a corresponding Hypergrid in memory.

    """

    def __init__(self, hypergrid: Hypergrid, df: pd.DataFrame):
        ...
