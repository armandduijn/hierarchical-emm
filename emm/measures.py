import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from anytree import Node, find_by_attr, Walker
from scipy.stats import pearsonr, norm
from typing import NamedTuple, List, Callable
from emm.description import Description


class Result(NamedTuple):
    coverage: int
    quality: float


class DetailedResult(NamedTuple):
    coverage: int
    quality: float
    size_component: float
    distance_component: float
    gap_component: float


class QualityMeasure(ABC):
    data = None

    def set_data(self, data) -> None:
        """"Sets the population data.

        Moving the setter out of the constructor allows the quality measure
        to apply necessary transformations or pre-compute values.

        Args:
            data: pandas DataFrame
        """
        self.data = data

    def get_data(self):
        """Returns the population data.

        Returns:
            The population data.

        Raises:
            ValueError if the `data` property isn't set.
        """
        if self.data is None:
            raise ValueError("The property `data` isn't set. Call `set_data()` before using the quality measure.")

        return self.data

    @abstractmethod
    def calculate(self, description: Description) -> Result:
        """Calculates the quality of a subgroup.

        Args:
            description: Description of the subgroup

        Returns:
            The subgroup's coverage and quality. Depending on the measure,
            the quality may be normalized to a range or it may increase/
            decrease unbounded.
        """
        pass


class LabelDistribution(QualityMeasure):
    def __init__(self, target: str, tree: Node, distance_func: Callable = None, gap_func: Callable = None):
        self.target = target
        self.tree = tree

        self.population_ranking = None
        self.max_ranking_distance = None
        self.population_gaps = None
        self.labels = None

        if distance_func is None:
            warnings.warn('No distance function provided. Using AKM Distance.')

            self.distance_func = lambda x, y: akm_distance(x, y, self.tree)
        else:
            self.distance_func = distance_func

        if gap_func is None:
            warnings.warn('No gap function provided. Using the absolute difference of the standard deviations.')

            self.gap_func = gap_distance_using_std
        else:
            self.gap_func = gap_func

    def set_data(self, data) -> None:
        self.data = data

        # Labels of each instance in the population
        values = data[self.target]

        # Unique labels in the population. Used to derive complete rankings for
        # subgroups that may not contain all labels.
        self.labels = values.unique().tolist()

        # Pre-compute ranking used to assess each subgroup
        self.population_ranking = derive_label_ranking(X=values, labels=self.labels)

        # Maximum ranking distance is the inverse of the population ranking
        inverted_ranking = invert_ranking(self.population_ranking)

        self.max_ranking_distance = self.ranking_distance(self.population_ranking, inverted_ranking)

        # Pre-compute gaps
        self.population_gaps = derive_label_gaps(X=values, labels=self.labels)

    def calculate(self, description: Description) -> Result:
        result = self.calculate_components(description)

        return Result(coverage=result.coverage, quality=result.quality)  # Disregard individual components

    def calculate_components(self, description: Description) -> DetailedResult:
        subgroup = description.subgroup()

        # The quality measure consists of three components:
        # (1) Size: The size of the subgroup
        # (2) Distance: How different the subgroup is compared to the population
        # (3) Gaps: How different the instances are distributed across the labels

        size = self.__calculate_size_component(subgroup)
        distance = self.__calculate_distance_component(subgroup)
        gaps = self.__calculate_gap_component(subgroup)

        # Combine the three components to compute the subgroup's quality
        quality = size * distance * gaps

        return DetailedResult(coverage=len(subgroup), quality=quality,
                              size_component=size,
                              distance_component=distance,
                              gap_component=gaps)

    def ranking_distance(self, first: pd.Series, second: pd.Series) -> float:
        return extended_spearman(first, second, distance_func=self.distance_func)

    def __calculate_size_component(self, subgroup) -> float:
        n_population = len(self.get_data())

        return np.sqrt(1.0 * len(subgroup) / n_population)

    def __calculate_distance_component(self, subgroup) -> float:
        ranking = derive_label_ranking(X=subgroup[self.target], labels=self.labels)
        distance = self.ranking_distance(self.population_ranking, ranking)

        return 1.0 * distance / self.max_ranking_distance

    def __calculate_gap_component(self, subgroup) -> float:
        gaps = derive_label_gaps(X=subgroup[self.target], labels=self.labels)

        return self.gap_func(self.population_gaps, gaps)


def invert_ranking(ranking: pd.Series) -> pd.Series:
    # Sort the ranking by rank from high to low. Use the reordered index to
    # derive the inverted ranking.
    labels = ranking.sort_values(ascending=False).index
    ranks = np.arange(1, len(ranking) + 1)  # Ranking is from 1 to n

    return pd.Series(ranks, index=labels)


def derive_label_ranking(X: pd.Series, labels: List[str]) -> pd.Series:
    """Derives a ranking from a list of labels

    Args:
        X: A list of repeated labels.
        labels: The labels that should be included in the rank. It may include
            labels not part of the list.

    Returns:
        A ranking from 1 to n, indexed by label, where n is the length of `labels`.
    """
    frequencies = X.value_counts(sort=True, ascending=False).reindex(labels, fill_value=0)

    # Solve ties by using the label's position in the frequency list
    return frequencies.rank(axis=0, method='first', ascending=False)


def derive_label_gaps(X: pd.Series, labels: List[str]) -> List[float]:
    missing_labels = [label for label in labels if label not in X.unique()]  # Use list comprehension instead of sets to preserve order
    missing_frequencies = pd.Series(0, index=missing_labels)

    frequencies = X.value_counts(sort=True, ascending=False)
    frequencies = frequencies.append(missing_frequencies)

    gaps = frequencies.diff().dropna().abs()

    return list(gaps / gaps.sum())  # Normalize gaps


def gap_distance_using_std(first: List[float], second: List[float]) -> float:
    return 1 + np.abs(np.std(first) - np.std(second))


def akm_distance(name_1: str, name_2: str, tree: Node):
    message = 'Unknown name "%s". The name must be associated to a node in the tree.'

    start = find_by_attr(tree, name_1)
    end = find_by_attr(tree, name_2)

    if start is None:
        raise ValueError(message % name_1)
    if end is None:
        raise ValueError(message % name_2)

    return sum([calculate_weight(edge) for edge in calculate_path(start, end)])


class Edge(NamedTuple):
    parent: Node
    child: Node


def calculate_path(start: Node, end: Node) -> List[Edge]:
    upwards, common, downwards = Walker().walk(start, end)
    edges = []

    if len(upwards) > 0:
        for i in range(0, len(upwards) - 1):
            edges.append(Edge(parent=upwards[i + 1], child=upwards[i]))

        edges.append(Edge(parent=common, child=upwards[-1]))

    if len(downwards) > 0:
        edges.append(Edge(parent=common, child=downwards[0]))

        for i in range(0, len(downwards) - 1):
            edges.append(Edge(parent=downwards[i], child=downwards[i + 1]))

    return edges


def calculate_weight(edge: Edge, tau=2.55) -> float:
    node = edge.parent  # Edge weight calculated using the parent
    n_children = len(node.children)

    if node.is_root:
        return 1 / (np.log10(n_children) + 1)

    upper_edge = Edge(parent=node.parent, child=node)

    return (1 / tau) * calculate_weight(upper_edge) / (np.log10(n_children) + 1)


def ranked_higher(ranking: pd.Series, label: str) -> pd.Series:
    """Returns the items ranked higher."""
    return ranking[ranking <= ranking[label]]


def extended_spearman(first: pd.Series, second: pd.Series, distance_func: Callable[[str, str], float]) -> float:
    """Returns the distance between two rankings using Spearman's footrule

    Args:
        first: A ranking
        second: A ranking
        distance_func: A distance function
    """
    if len(first) != len(second):
        raise ValueError(f'The rankings must have the same length. Given ({len(first)},) and ({len(second)},).')

    total_distance = 0.0

    for label in first.index:
        position_first = sum(ranked_higher(first, label).index.map(lambda x: distance_func(x, label)))
        position_second = sum(ranked_higher(second, label).index.map(lambda x: distance_func(x, label)))

        total_distance += abs(position_first - position_second)

    return total_distance


def correlation(X, Y):
    if X.shape[1] != 2 or Y.shape[1] != 2:
        raise ValueError(f'The arrays must have the shape (,2). Given (,{X.shape[1]}) and (,{Y.shape[1]}).')

    # Calculate Pearson's coefficient, discard the p-value
    rho_x, _ = pearsonr(X[X.columns[0]], X[X.columns[1]])
    rho_y, _ = pearsonr(Y[Y.columns[0]], Y[Y.columns[1]])

    # Apply Fisher's transformation, see https://en.wikipedia.org/wiki/Fisher_transformation#Definition
    z_x = np.arctanh(rho_x)
    z_y = np.arctanh(rho_y)

    pooled_se = np.sqrt((1 / (len(X) - 3)) + (1 / (len(Y) - 3)))
    z = (z_x - z_y) / pooled_se

    return norm.sf(abs(z)) * 2  # Two-tailed


class PearsonCorrelation(QualityMeasure):
    def __init__(self, targets: List):
        self.targets = targets

    def calculate(self, description: Description) -> Result:
        data = self.get_data()

        subgroup = description.subgroup()
        coverage = len(subgroup)

        # Compare to the population, alternatively use the subgroup's complement
        complement = data

        if coverage <= 3 or len(complement) <= 3:
            # Prevent division by zero error when calculating the pooled standard error
            return Result(coverage, 0)

        # Quality is equal to: 1 - p-value
        quality = 1 - correlation(subgroup[self.targets], complement[self.targets])

        return Result(coverage, quality)


class RelativeAverageRankingLoss(QualityMeasure):
    population_loss = None

    def __init__(self, base_column: str, confidence_column: str):
        self.b = base_column
        self.r = confidence_column

    def set_data(self, source) -> None:
        self.data = self.sort_values(source)

        # Pre-compute the population's ARL to speed-up RASL computations in calculate()
        self.population_loss = self.average_ranking_loss(self.data)

    def calculate(self, description: Description) -> Result:
        subgroup = self.sort_values(description.subgroup())
        rasl = self.average_ranking_loss(subgroup) - self.population_loss

        coverage = len(subgroup)
        quality = 1 - rasl  # Minimize RASL, alternatively use ABS to find the subgroup with largest difference

        return Result(coverage, quality)

    def sort_values(self, df):
        return df.sort_values(by=self.r, ascending=True)

    def average_ranking_loss(self, subgroup) -> float:
        positives = subgroup.query(f'`{self.b}` == 1')

        if len(positives) == 0:
            return 0  # Subgroup only contains False base values; no loss

        penalty = 0
        for index in positives.index:
            penalty += self.calculate_penalty(subgroup, index)

        return 1.0 * penalty / len(positives)

    def calculate_penalty(self, subgroup, index) -> int:
        i = subgroup.index.get_loc(index)  # Integer location of the current row

        if i >= (len(subgroup) - 1):
            return 0  # Last row has no penalty by definition

        current_confidence = subgroup.loc[index][self.r]
        successor_rows = subgroup.iloc[(i + 1):].query(f'`{self.b}` == 0')

        # Successor rows with a False base value, but the same confidence value
        ties = (successor_rows[self.r] == current_confidence)

        return np.where(ties, 0.5, 1).sum()
