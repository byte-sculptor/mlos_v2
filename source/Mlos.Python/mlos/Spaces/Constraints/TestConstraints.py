

import pytest

from mlos.Spaces.Constraints.Constraint import Constraint
from mlos.Spaces.SimpleHypergrid import SimpleHypergrid


class TestConstraints:
    """
    
    We'll need a lot of test cases:
        1. Not nested constraints.
        2. Nested constraints.
        3. Constraints on SimpleHypergrids
        4. Constraints on Adapters
    """

    def test_valid_constraints(self):
        constraint = Constraint()
        