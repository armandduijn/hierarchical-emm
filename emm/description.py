from __future__ import annotations
from emm import config
from math import floor
from typing import List, Union
import pandas as pd


def refine(data, targets: List[str], seed: Description):
    supported_types = ['object', 'bool', 'int32', 'int64', 'float32', 'float64']

    candidates = {seed}  # Relies on the behavior of a set to remove similar descriptions
    descriptors = data.columns[~data.columns.isin(targets)]

    for column in descriptors:
        dtype = data[column].dtype

        if dtype not in supported_types:
            raise NotImplementedError("Column '%s' must be of dtype: '%s'; dtype '%s' given." %
                                      (column, ', '.join(supported_types), dtype))

        if dtype == 'bool':
            candidates.add(seed.refine(Condition(f'`{column}` == 0')))
            candidates.add(seed.refine(Condition(f'`{column}` == 1')))

        if dtype in ('int32', 'int64', 'float32', 'float64'):
            # Only use values from subgroup for binning (dynamic discretization)
            subgroup = seed.subgroup()

            n_rows = len(subgroup)
            n_bins = config.NUM_BINS

            if n_rows > 0:
                indices = [floor(j * 1.0 * n_rows / n_bins) for j in range(1, n_bins)]  # Loops over [1, b - 1]

                # The equal-width binning creates b - 1 splits in the data. If the number of bins
                # is large or the frequency of a certain value is high, some splits may have the
                # same boundary. Normally, we would have to filter these to prevent duplicate
                # descriptions. But because the variable `descriptions` is a set, duplicates are
                # automatically pruned.
                split_values = subgroup[column].sort_values(ascending=True).iloc[indices]

                for value in split_values:
                    candidates.add(seed.refine(Condition(f'`{column}` <= {value}')))
                    candidates.add(seed.refine(Condition(f'`{column}` >= {value}')))

        if dtype == 'object':
            for value in data[column].unique():
                # Use double quotes to enclose values (e.g. Côte d'Ivoire)
                candidates.add(seed.refine(Condition(f'`{column}` == "{value}"')))
                candidates.add(seed.refine(Condition(f'`{column}` != "{value}"')))

    return candidates


def description_factory(conditions: Union[Condition, List[Condition]], data: Union[pd.DataFrame, None] = None):
    if isinstance(conditions, Condition):
        conditions = [conditions]

    if data is None:
        data = pd.DataFrame({})  # Empty dataframe

    return Description(conditions=conditions, evaluator=Evaluator(data=data))


class Evaluator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def evaluate(self, query: str) -> pd.DataFrame:
        if query == '':
            return self.data

        return self.data.query(query)


class Condition(str):
    def __init__(self, query):
        super().__init__()


class Description:
    def __init__(self, conditions: List[Condition], evaluator: Evaluator):
        self.conditions = conditions
        self.evaluator = evaluator

        self.__matches = None

    @property
    def matches(self) -> pd.NumericIndex:
        if self.__matches is None:  # Lazy-load matches
            self.__matches = self.subgroup().index

        return self.__matches

    def subgroup(self) -> pd.DataFrame:
        return self.evaluator.evaluate(self.to_querystring())

    def refine(self, condition: Condition) -> Description:
        refined = self.__class__(conditions=self.conditions + [condition], evaluator=self.evaluator)

        # Check if the added condition makes the description stronger
        if refined.is_stronger(self):
            return refined

        # Return a new instance to prevent unintended mutations due to referencing
        return self.__class__(conditions=self.conditions, evaluator=self.evaluator)

    def is_stronger(self, other: Description) -> bool:
        """Returns if the instance is stronger than the provided description.

        A description is weaker (i.e. more general) if it's a proper superset
        of another description Conversely, a description is stronger (i.e. more
        specialized) if it's a proper subset of another description.
        """
        return self.matches.isin(other.matches).all() and self != other

    def to_querystring(self) -> str:
        """Returns the description as a pandas compatible query string."""
        return str(self)

    def __eq__(self, other: Description) -> bool:
        """Returns if the instance covers the same rows as the provided description.

        Implemented magic method instead of custom method so that the object
        can be added to a set.
        """
        return self.matches.equals(other.matches)

    def __lt__(self, other: Description) -> bool:
        """Returns if the instance is weaker than the provided description.

        Operator is used when inserting description in a priority queue. By
        defining 'less than' to as weaker, we prefer stronger (i.e smaller)
        descriptions. The weaker descriptions are evicted first from the
        priority queue.

        NOTE: Returns False even if the description are incomparable (i.e.
        non-overlapping). As a consequence, the following implication doesn't
        hold: if description is not weaker and not equal, it must be stronger.
        """
        return other.matches.isin(self.matches).all() and self != other

    def __hash__(self) -> int:
        return hash(tuple(self.matches))

    def __str__(self) -> str:
        return ' & '.join(map(str, self.conditions))

    def __len__(self) -> int:
        return len(self.conditions)
