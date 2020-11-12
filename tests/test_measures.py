import unittest
import anytree as tree
import pandas as pd
from emm.measures import LabelDistribution, akm_distance, calculate_path, calculate_weight, Edge, extended_spearman, derive_label_ranking, derive_label_gaps, invert_ranking
from emm.description import Condition, description_factory


class TestLabelDistribution(unittest.TestCase):
    def test_calculate(self):
        data = pd.DataFrame({'foo': ['a', 'a', 'b', 'c']})
        description = description_factory(Condition("`foo` == 'b'"), data)

        # Label ranking of population: [a: 1, b: 2, c: 3]
        # Label ranking of subgroup: [a: 2, b: 1, c: 3]

        t = tree.Node('a', children=[
            tree.Node('b'),
            tree.Node('c')
        ])

        qm = LabelDistribution(target='foo', tree=t, gap_func=lambda x, y: 1)  # Gap component is 1 so all subgroups are exceptional
        qm.set_data(data)

        coverage, quality = qm.calculate(description)

        self.assertEqual(1, coverage)
        self.assertNotEqual(0, quality)  # Subgroup has a different label ranking so the quality should be non-zero

    def test_akm_distance_unknown_labels(self):
        with self.assertRaises(ValueError):
            akm_distance('foo', 'bar', tree=tree.Node(name=''))

    def test_akm_distance_child(self):
        label_1 = 'root'
        label_2 = 'leaf'

        t = tree.Node(label_1, children=[tree.Node(label_2)])

        self.assertEqual(1, akm_distance(label_1, label_2, tree=t))

    def test_calculate_path_self_loop(self):
        root = tree.Node('root')
        path = calculate_path(root, root)

        self.assertEqual(0, len(path))

    def test_calculate_path_different_branch(self):
        root = tree.Node('root')
        left_inner = tree.Node('left_inner', parent=root)
        left_leaf = tree.Node('left_leaf', parent=left_inner)
        right_inner = tree.Node('right_inner', parent=root)
        right_leaf = tree.Node('right_child', parent=right_inner)

        path = calculate_path(left_leaf, right_leaf)

        self.assertEqual(4, len(path))
        self.assertEqual(left_leaf, path[0].child)
        self.assertEqual(right_leaf, path[-1].child)

    def test_calculate_weight(self):
        root = tree.Node('root')
        inner = tree.Node('inner', parent=root)
        leaf = tree.Node('leaf', parent=inner)

        path = [Edge(parent=root, child=leaf), Edge(parent=inner, child=leaf)]
        weight = sum([calculate_weight(edge, tau=2.0) for edge in path])

        # Answer is: w(root, inner) + w(inner, leaf) = 1.5
        #   w(root, inner) = 1 / (log(|child(root)|) + 1 = 1
        #   w(inner, leaf) = 1/2 * w(root, inner) / (log|child(x)| + 1) = 0.5
        self.assertEqual(1.5, weight)

    def test_derive_label_ranking(self):
        X = pd.Series(['a', 'a', 'a', 'b', 'c', 'c'])
        labels = ['a', 'b', 'c', 'd']  # Includes label not part of `X`

        ranking = derive_label_ranking(X, labels)

        self.assertCountEqual(labels, ranking.index.tolist())
        self.assertEqual(1, ranking['a'])
        self.assertEqual(2, ranking['c'])
        self.assertEqual(3, ranking['b'])
        self.assertEqual(4, ranking['d'])

    def test_derive_label_gaps(self):
        X = pd.Series(['a', 'a', 'a', 'a', 'b', 'b', 'c'])
        labels = ['a', 'b', 'c', 'd', 'e']  # Includes labels not part of `X`

        gaps = derive_label_gaps(X, labels)

        self.assertListEqual([0.5, 0.25, 0.25, 0], gaps)

    def test_extended_spearman_unit_distance(self):
        def distance_func(x, y):
            return 1  # All items are equality similar/different

        first = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        second = pd.Series([3, 2, 1], index=['a', 'b', 'c'])

        difference = extended_spearman(first, second, distance_func=distance_func)

        self.assertEqual(4, difference)

    def test_extended_spearman_partial_ranking(self):
        with self.assertRaises(ValueError):
            first = pd.Series([1, 2, 3])
            second = pd.Series([1])

            extended_spearman(first, second, distance_func=lambda x, y: 1)

    def test_invert_ranking(self):
        ranking = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

        inverted = invert_ranking(ranking)

        self.assertEqual(3, inverted['a'])
        self.assertEqual(2, inverted['b'])
        self.assertEqual(1, inverted['c'])
