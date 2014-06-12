import datetime
import collections
import threading

# Expires things that were added after a timeout
class TimedCounter:
  def __init__(self, timeout=1):
    self.lock = threading.Lock()
    self.timeout = timeout
    self.items = collections.Counter()

  def add(self, item):
    with self.lock:
      self.items.subtract({item: -1})
      threading.Timer(self.timeout, self.expire, [item]).start()

  def expire(self, item):
    with self.lock:
      # Only expire an item if it is there (sometimes list was cleared before item had been expired--still buggy)
      # should instead keep a list of active timers, clearing them when expired and canceling all of them when clear is called
      if self.items[item] > 0:
        self.items.subtract({item: 1})
  
  def clear(self):
    with self.lock:
      self.items.clear()

  def most_common(self):
    if len(self) > 0:
      with self.lock:
        return self.items.most_common()
    else:
      return None

  def __len__(self):
    with self.lock:
      return len(list(self.items.elements()))
