from dataclasses import dataclass
from queue import PriorityQueue
from emm.description import Description


@dataclass(order=True)
class Result:
    quality: float
    description: Description


class MinPriorityQueue:
    """Wrapper for Python's PriorityQueue

    The queue holds items with a high priority. When the queue is full, it discards items
    with a low priority (sorted by ascending order) to free-up space.
    """

    # More information about queues on p. 72-75 of https://books.google.nl/books?id=-kZsDwAAQBAJ
    # and https://docs.python.org/3/library/queue.html#queue.PriorityQueue

    def __init__(self, max_size=1):
        self.max_size = max_size

        # Setting a maximum size on the native priority queue will cause it
        # to block/error when it's full. Instead, we want it to remove the
        # lowest priority item and insert the new item.
        self.q = PriorityQueue(maxsize=0)  # Allow queue to grow, bookkeeping done in put() function

    def put(self, item):
        self.q.put(item, block=False)  # Prevent wait if the queue is full, bookkeeping done manually

        # Check if the queue is too large
        if self.q.qsize() > self.max_size:
            self.q.get()  # Removes the lowest priority item

    def get(self):
        return self.q.get(block=False)  # Prevent wait if the queue is empty

    def empty(self):
        return self.q.empty()

    def contains(self, item):
        return item in self.q.queue

    def size(self):
        return self.q.qsize()

    def __iter__(self):
        while not self.empty():
            yield self.get()
