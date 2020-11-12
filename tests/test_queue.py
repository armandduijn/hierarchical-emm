from unittest import TestCase
import pandas as pd
from emm.queue import MinPriorityQueue, Result
from emm.description import description_factory, Condition


class TestQueue(TestCase):
    def test_priority_queue_contains(self):
        queue = MinPriorityQueue(max_size=1)

        data = pd.DataFrame({'A': [1, 2], 'B': [1, 2]})  # Pass data because descriptions need to be evaluated

        description1 = description_factory(Condition('A == 1'), data)
        description2 = description_factory(Condition('A == 1'), data)  # Different object BUT same contents
        description3 = description_factory(Condition('B == 2'), data)  # Different object AND different contents

        item1 = Result(quality=1, description=description1)
        item2 = Result(quality=1, description=description2)
        item3 = Result(quality=1, description=description3)

        queue.put(item1)

        self.assertTrue(queue.contains(item2), 'Compares hashed value instead of references')
        self.assertFalse(queue.contains(item3), 'Hashes quality as well as description')

    def test_priority_queue_equal_quality(self):
        queue = MinPriorityQueue(max_size=2)

        data = pd.DataFrame({'A': [1, 2], 'B': [1, 2]})  # Pass data because descriptions need to be evaluated

        description1 = description_factory(Condition('A == 1'), data)
        description2 = description_factory(Condition('B >= 1'), data)  # Weaker description (larger coverage)

        item1 = Result(quality=1, description=description1)
        item2 = Result(quality=1, description=description2)  # Same quality, different description

        queue.put(item1)
        queue.put(item2)

        result = list(queue)

        self.assertEqual(item2, result[0], "The description 'B >= 1' is weaker")
        self.assertEqual(item1, result[1], "The description 'A == 1' is stronger")

    def test_priority_queue_overflowing(self):
        queue = MinPriorityQueue(max_size=2)

        description1 = description_factory(Condition('L'))
        description2 = description_factory(Condition('M'))
        description3 = description_factory(Condition('H'))

        item1 = Result(quality=1, description=description1)
        item2 = Result(quality=2, description=description2)
        item3 = Result(quality=3, description=description3)

        queue.put(item1)
        queue.put(item2)
        queue.put(item3)

        self.assertFalse(queue.contains(item1), 'Removed low quality item')
        self.assertTrue(queue.contains(item2), 'Contains medium priority item')
        self.assertTrue(queue.contains(item3), 'Contains high priority item')

    def test_priority_queue_list(self):
        queue = MinPriorityQueue(max_size=3)

        description1 = description_factory(Condition('C'))
        description2 = description_factory(Condition('B'))
        description3 = description_factory(Condition('A'))

        # Insert items on purpose in non-ascending order to force rebuilding the heap
        queue.put(Result(quality=3, description=description1))
        queue.put(Result(quality=2, description=description2))
        queue.put(Result(quality=1, description=description3))

        result = list(queue)

        self.assertEqual(len(result), 3)

        # Make sure the list of correctly ordered (low to high), i.e. not a dump of the heap
        self.assertEqual('A', result[0].description.to_querystring())
        self.assertEqual('B', result[1].description.to_querystring())
        self.assertEqual('C', result[2].description.to_querystring())


