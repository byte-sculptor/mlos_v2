from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from mlos.Spaces import Point
class ExperimentDesignerBase(ABC):
    """Base class for all experiment designers."""
    @abstractmethod
    def suggest(self, context_values_dataframe: Optional[pd.DataFrame]) -> Point:
        raise NotImplementedError

    def add_pending_suggestion(self, suggestion: Point) -> None:
        """Adds a pending suggestion to an internal data structure.

        The experimenter has committed to testing a given suggestion so we can add it and its predictions to our tentative
        pareto frontier. This is a resource management problem. If the experimenter never gives us back the result, we will
        have effectively leaked this pending suggestion and actually prevented the optimizer from exploring its neighborhoods
        in the future. So the experimenter must remember to either register a relevant observation, or drop a pending suggestion.

        """
        pass

    def remove_pending_suggestion(self, suggestion: Point) -> None:
        """Removes a previously added pending suggestion."""
        pass

    def remove_pending_suggestions(self, suggestions_df: pd.DataFrame) -> None:
        """Removes multiple previously pending suggestions."""
        pass
