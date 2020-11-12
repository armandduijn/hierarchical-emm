from unittest import TestCase
from unittest.mock import patch
from tests import test_config
import pandas as pd
from datetime import datetime
from emm.description import Condition, Description, refine, Evaluator, description_factory


@patch('emm.description.config', new=test_config)
class TestDescription(TestCase):
    def test_evaluator_empty_query(self):
        dataset = pd.DataFrame({'A': [1, 2, 3]})
        evaluator = Evaluator(data=dataset)

        result = evaluator.evaluate('')

        self.assertEqual(len(result), len(dataset), 'Returns all rows if the query is empty')

    def test_evaluator_single_condition(self):
        dataset = pd.DataFrame({'A': [1, 2, 3]})
        evaluator = Evaluator(data=dataset)

        result = evaluator.evaluate('A <= 2')

        self.assertEqual(len(result), 2, 'Matches two rows')

    def test_evaluator_multiple_conditions(self):
        dataset = pd.DataFrame({'A': [1, 2, 3]})
        evaluator = Evaluator(data=dataset)

        result = evaluator.evaluate('A > 1 & A < 3')

        self.assertEqual(len(result), 1, 'Evaluates a description with multiple conditions')

    def test_description_to_querystring(self):
        evaluator = Evaluator(data=pd.DataFrame())
        conditions = [Condition('A == 1'), Condition('B == 1')]
        description = Description(conditions=conditions, evaluator=evaluator)

        self.assertEqual(description.to_querystring(), 'A == 1 & B == 1', 'Combines conditions with a conjunction')

    def test_description_incomparable(self):
        data = pd.DataFrame({'A': [1, 2, 3]})

        evaluator = Evaluator(data=data)
        description1 = Description(conditions=[Condition('A <= 2')], evaluator=evaluator)
        description2 = Description(conditions=[Condition('A == 3')], evaluator=evaluator)

        self.assertFalse(description1 < description2, 'Descriptions incomparable')
        self.assertFalse(description2 > description1, 'Descriptions incomparable')

    def test_description_add_specialization_stronger(self):
        evaluator = Evaluator(data=pd.DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3]}))
        condition = Condition('A > 0')  # Very broad, matches all rows
        description = Description(conditions=[condition], evaluator=evaluator)

        refined = description.refine(condition=Condition('B == 2'))  # Add specific condition, matches one row

        self.assertIsNot(refined, description, 'Returns a new instance of Description')
        self.assertEqual(len(refined), 2, 'Description contains stronger condition')

    def test_description_add_specialization_weaker(self):
        evaluator = Evaluator(data=pd.DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3]}))
        condition = Condition('A <= 1')  # Specific condition, matches one row
        description = Description(conditions=[condition], evaluator=evaluator)

        refined = description.refine(condition=Condition('B <= 2'))  # Add condition that broadens the description

        self.assertIsNot(refined, description, 'Returns a new instance of Description')
        self.assertEqual(1, len(refined), 'Refined description does not contain weaker condition')

    def test_refine_boolean(self):
        dataset = pd.DataFrame({'A': [True, False]})

        descriptions = refine(dataset, [], description_factory([], dataset))
        queries = [d.to_querystring() for d in descriptions]

        self.assertEqual(len(descriptions), 3, 'Added 2 conditions')
        self.assertEqual('`A` == 1' in queries, True, 'Added condition equal to True')
        self.assertEqual('`A` == 0' in queries, True, 'Added condition equal to False')

    def test_refine_nominal(self):
        dataset = pd.DataFrame({'A': ['foo', 'bar', 'lex']})

        descriptions = refine(dataset, [], description_factory([], dataset))
        queries = [d.to_querystring() for d in descriptions]

        self.assertEqual(7, len(descriptions), 'Added 4 conditions (2g)')
        self.assertIn("`A` == 'foo'", queries, 'Added condition equal to g(1)')
        self.assertIn("`A` != 'foo'", queries, 'Added condition not equal to g(1)')
        self.assertIn("`A` == 'bar'", queries, 'Added condition equal to g(2))')
        self.assertIn("`A` != 'bar'", queries, 'Added condition not equal to g(2)')
        self.assertIn("`A` == 'lex'", queries, 'Added condition equal to g(3))')
        self.assertIn("`A` != 'lex'", queries, 'Added condition not equal to g(3)')

    def test_refine_numeric(self):
        dataset = pd.DataFrame({'A': [1, 2, 3, 4]})

        descriptions = refine(dataset, [], description_factory([], dataset))
        queries = [d.to_querystring() for d in descriptions]

        self.assertEqual(5, len(descriptions), 'Added 4 conditions (2 * (num_buckets - 1))')
        self.assertIn('`A` <= 2', queries)
        self.assertIn('`A` >= 2', queries)
        self.assertIn('`A` <= 3', queries)
        self.assertIn('`A` >= 3', queries)

    def test_refine_duplicate_splits(self):
        dataset = pd.DataFrame({'A': [0, 1, 1, 2]})

        # With 3 equal-width bins, the splits will be duplicated. Splits at position 1 and 2
        descriptions = refine(dataset, [], description_factory([], dataset))

        # Shouldn't contain `A >= 1` and `A <= 1` twice
        self.assertEqual(3, len(descriptions), 'Only contains unique inequalities')

    def test_refine_unsupported_type(self):
        dataset = pd.DataFrame({'A': [datetime.now()]})  # No refinement implemented for dates

        with self.assertRaises(NotImplementedError):
            refine(dataset, [], description_factory([], dataset))

    def test_refine_empty_subgroup(self):
        dataset = pd.DataFrame({'A': [1]})

        condition = Condition('`A` == 2')  # Condition matches 0 rows
        seed = description_factory(condition, dataset)

        descriptions = refine(dataset, [], seed)

        self.assertEqual(1, len(descriptions), 'No refinements added')
