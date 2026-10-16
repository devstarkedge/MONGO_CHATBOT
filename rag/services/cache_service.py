"""Small thread-safe TTL + LRU cache (used for query embeddings and answers)."""
import threading
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, maxsize: int = 500, ttl: float = 600):
        self.maxsize, self.ttl = maxsize, ttl
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] < time.monotonic():
                self._data.pop(key, None)
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key, value):
        if self.ttl <= 0 or self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        return {"entries": len(self._data), "hits": self.hits, "misses": self.misses}
