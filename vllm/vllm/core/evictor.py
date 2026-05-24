# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
import time
import statistics
import numpy as np
from collections import defaultdict, deque
from sortedcontainers import SortedDict
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

def probability_of_future_arrival(prob_has_next, exp_scale, elapsed_time, debug=False):
    if prob_has_next == 0 or exp_scale == 0:
        return 0.0
    prob_not_accessed_till_now = np.exp(-elapsed_time / exp_scale)
    return (prob_has_next * prob_not_accessed_till_now) / (
            prob_has_next * prob_not_accessed_till_now + (1 - prob_has_next)
    )

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass

class CacheStat:
    def __init__(self):
        self.stat = defaultdict(list)
        self.average = {}
        self.last_log_time = 0
        self.log_interval = 10000  # Log every 5 seconds
    
    def get_average(self, key):
        if key in self.average:
            return self.average[key]
        else:
            return 1
    
    def append(self, key, value):
        if key not in self.average:
            self.average[key] = 0
        self.average[key] = (self.average[key] * len(self.stat[key]) + value) / (len(self.stat[key]) + 1)
        self.stat[key].append(value)
    
    def summary(self):
        current_time = time.time()
        if current_time - self.last_log_time < self.log_interval:
            return 0
        self.last_log_time = current_time
        has_data = 0

        for key in self.stat.keys():
            data = self.stat[key]
            if not data:
                continue
            
            mean = statistics.mean(data)
            std_dev = statistics.stdev(data) if len(data) > 1 else 0
            print(f"Summary for {key}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")
            has_data = 1
        return has_data
        

class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, cache_hint: dict = None, score: float = 0):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.cache_hint = cache_hint
        self.score = score


class ShadowPolicyCache:
    RRIP_MAX_RRPV = 3
    RRIP_INSERT_RRPV = RRIP_MAX_RRPV - 1
    RRIP_HIT_RRPV = 0

    def __init__(self, policy: str, capacity: int):
        self.policy = policy
        self.capacity = capacity
        self.table = {}
        self.events = deque()

    def set_capacity(self, capacity: int):
        self.capacity = capacity
        while len(self.table) > self.capacity:
            self._evict()

    def hit_rate(self) -> float:
        if not self.events:
            return 0.0
        return sum(hit for _, hit in self.events) / len(self.events)

    def num_events(self) -> int:
        return len(self.events)

    def access(self, content_hash: int, now: float,
               cache_hint: Optional[dict] = None):
        hit = content_hash in self.table
        self.events.append((now, 1 if hit else 0))

        if hit:
            entry = self.table[content_hash]
            entry["last_accessed"] = now
            entry["cache_hint"] = cache_hint
            if self.policy == "rrip":
                entry["rrpv"] = self.RRIP_HIT_RRPV
            elif self.policy == "ml":
                entry["score"] = self._ml_score(entry, cache_hint, now)
            return

        if self.capacity <= 0:
            return
        while len(self.table) >= self.capacity:
            self._evict()
        self.table[content_hash] = {
            "first_accessed": now,
            "last_accessed": now,
            "rrpv": self.RRIP_INSERT_RRPV,
            "score": 0.0,
            "cache_hint": cache_hint,
        }
        if self.policy == "ml":
            self.table[content_hash]["score"] = self._ml_score(
                self.table[content_hash], cache_hint, now)

    def _ml_score(self, entry: dict, cache_hint: Optional[dict],
                  now: float) -> float:
        if cache_hint and "prob_has_next" in cache_hint:
            return probability_of_future_arrival(
                cache_hint["prob_has_next"], cache_hint.get("exp_scale", 1),
                now - entry["last_accessed"])
        return entry["last_accessed"]

    def _evict(self):
        if not self.table:
            return
        if self.policy == "fifo":
            victim = min(self.table,
                         key=lambda h: (self.table[h]["first_accessed"], h))
        elif self.policy == "rrip":
            victim = self._rrip_victim()
        elif self.policy == "ml":
            now = time.time()
            victim = min(
                self.table,
                key=lambda h: (
                    self._ml_score(self.table[h],
                                   self.table[h].get("cache_hint"), now),
                    self.table[h]["last_accessed"],
                    h,
                ))
        else:
            victim = min(self.table,
                         key=lambda h: (self.table[h]["last_accessed"], h))
        del self.table[victim]

    def _rrip_victim(self):
        while True:
            candidates = [
                h for h, entry in self.table.items()
                if entry["rrpv"] >= self.RRIP_MAX_RRPV
            ]
            if candidates:
                return min(candidates,
                           key=lambda h: (self.table[h]["last_accessed"], h))
            for entry in self.table.values():
                entry["rrpv"] = min(self.RRIP_MAX_RRPV, entry["rrpv"] + 1)


class EvictionPolicyScheduler:
    POLICIES = ("ml", "lru", "rrip", "fifo")
    SHADOW_POLICIES = ("lru", "rrip", "fifo")

    def __init__(self, config: dict):
        self.enabled = self._as_bool(config.get("enable_scheduler", "0"))
        self.warmup_s = float(config.get("scheduler_warmup", 20))
        self.min_events = int(config.get("scheduler_min_events", 1000))
        self.small_threshold = float(
            config.get("scheduler_small_threshold", 0.10))
        self.large_threshold = float(
            config.get("scheduler_large_threshold", 0.05))
        self.model_size_b = float(config.get("model_size_b", 7))
        self.current_policy = config.get("scheduler_initial_policy", "ml")
        self.start_time = time.time()
        self.finalized = False
        self.capacity = int(config.get("scheduler_capacity", 0))
        self.ml_events = deque()
        self.shadow_caches = {
            policy: ShadowPolicyCache(policy, self.capacity)
            for policy in self.SHADOW_POLICIES
        }

    def _as_bool(self, value) -> bool:
        return str(value).lower() in ("1", "true", "yes", "on")

    def set_capacity(self, capacity: int):
        self.capacity = capacity
        for cache in self.shadow_caches.values():
            cache.set_capacity(capacity)

    def observe(self, content_hash: int, cache_hint: Optional[dict] = None,
                real_hit: Optional[bool] = None):
        if not self.enabled:
            return
        now = time.time()
        if self.finalized:
            return
        if real_hit is not None:
            self.ml_events.append((now, 1 if real_hit else 0))
        for cache in self.shadow_caches.values():
            cache.access(content_hash, now, cache_hint)
        if now - self.start_time >= self.warmup_s:
            self._finalize_policy(now)

    def should_predict(self) -> bool:
        if not self.enabled:
            return True
        return (not self.finalized) or self.current_policy == "ml"

    def _ml_hit_rate(self) -> float:
        if not self.ml_events:
            return 0.0
        return sum(hit for _, hit in self.ml_events) / len(self.ml_events)

    def _finalize_policy(self, now: float):
        min_events = min([len(self.ml_events)] + [
            cache.num_events() for cache in self.shadow_caches.values()
        ])
        if min_events == 0:
            print("scheduler finalize: no warmup events, keep ml")
            self.current_policy = "ml"
            self.finalized = True
            return
        if min_events < self.min_events:
            print("scheduler finalize: insufficient warmup events",
                  min_events, "<", self.min_events)

        hit_rates = {
            "ml": self._ml_hit_rate(),
            **{
                policy: cache.hit_rate()
                for policy, cache in self.shadow_caches.items()
            }
        }
        best_shadow_policy = max(self.SHADOW_POLICIES,
                                 key=lambda p: hit_rates[p])
        best_other = hit_rates[best_shadow_policy]
        threshold = (self.small_threshold if self.model_size_b <= 14 else
                     self.large_threshold)
        if hit_rates["ml"] >= best_other + threshold:
            best_policy = "ml"
        else:
            best_policy = best_shadow_policy

        print("scheduler finalize:",
              self.current_policy, "->", best_policy, hit_rates)
        self.current_policy = best_policy
        self.finalized = True

class LRUMLEvictor(Evictor):
    RRIP_MAX_RRPV = 3
    RRIP_INSERT_RRPV = RRIP_MAX_RRPV - 1
    RRIP_HIT_RRPV = 0

    def __init__(self, config):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.sorted_dict = SortedDict()
        self.id_to_last_access = {}
        self.id_to_first_access = {}
        self.rrip_values = {}
        self.to_delete_blocks = []
        self.config = self.parse_str_to_dict(config)
        self.stat = CacheStat()
        self.last_refresh_time = time.time()
        self.INSPECT_INTERVAL = 5
        self.scheduler = EvictionPolicyScheduler(self.config)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def parse_str_to_dict(self, s: str) -> dict:
        if len(s) == 0:
            return {}
        return {key: value for key, value in (pair.split("=", 1) for pair in s.split(","))}

    def get_policy(self, cache_hint: dict) -> str:
        if cache_hint is None:
            return 'lru'
        if cache_hint.get('use_rrip'):
            return 'rrip'
        if cache_hint.get('use_lru'):
            return 'lru'
        if cache_hint.get('use_fifo'):
            return 'fifo'
        if cache_hint.get('scheduler_policy'):
            return cache_hint['scheduler_policy']
        if self.scheduler.enabled:
            return self.scheduler.current_policy
        if 'next_timestamp' in cache_hint:
            return 'belady'
        return 'ml'

    def set_capacity(self, capacity: int):
        self.scheduler.set_capacity(capacity)

    def observe_cache_access(self, content_hash: int,
                             cache_hint: Optional[dict] = None,
                             real_hit: Optional[bool] = None):
        self.scheduler.observe(content_hash, cache_hint, real_hit)

    def should_predict_cache_hint(self) -> bool:
        return self.scheduler.should_predict()

    def _bind_scheduler_policy(self, cache_hint: dict) -> dict:
        if not self.scheduler.enabled:
            return cache_hint
        if (cache_hint.get('use_lru') or cache_hint.get('use_rrip') or
                cache_hint.get('use_fifo') or cache_hint.get('scheduler_policy')):
            return cache_hint
        cache_hint = dict(cache_hint)
        cache_hint['scheduler_policy'] = self.scheduler.current_policy
        return cache_hint

    def calc_score(self, block_id, last_accessed, cache_hint):
        policy = self.get_policy(cache_hint)
        if policy == 'rrip':
            rrpv = self.rrip_values.get(block_id, self.RRIP_INSERT_RRPV)
            return -rrpv
        if policy == 'lru':
            return last_accessed
        if policy == 'fifo':
            return self.id_to_first_access[block_id]
        if policy == 'belady':
            return -cache_hint['next_timestamp']
        if 'prob_has_next' in cache_hint:
            prob = probability_of_future_arrival(
                cache_hint['prob_has_next'], cache_hint['exp_scale'], time.time() - last_accessed)
            return prob
        return last_accessed

    def _rebuild_sorted_dict(self):
        new_sorted_dict = SortedDict()
        for block_id, block in self.free_table.items():
            score = self.calc_score(block_id, block.last_accessed,
                                    block.cache_hint)
            block.score = score
            new_sorted_dict[(score, block.last_accessed, block_id)] = (
                block_id, block.content_hash)
        self.sorted_dict = new_sorted_dict

    def _age_rrip_blocks(self):
        for block_id, block in self.free_table.items():
            if self.get_policy(block.cache_hint) != 'rrip':
                continue
            rrpv = self.rrip_values.get(block_id, self.RRIP_INSERT_RRPV)
            self.rrip_values[block_id] = min(self.RRIP_MAX_RRPV, rrpv + 1)
        self._rebuild_sorted_dict()

    def _peek_valid_candidate(self):
        while self.sorted_dict:
            key, (block_id, content_hash) = self.sorted_dict.peekitem(0)
            if (block_id in self.free_table and
                    self.free_table[block_id].score == key[0] and
                    self.free_table[block_id].last_accessed == key[1]):
                return block_id, content_hash
            self.sorted_dict.popitem(0)
        return None

    def _remove_from_sorted_dict(self, block_id: int):
        block = self.free_table[block_id]
        entry = (block.score, block.last_accessed, block_id)
        if entry in self.sorted_dict:
            del self.sorted_dict[entry]
        else:
            raise ValueError("the score is not found in sorted_dict")

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        while True:
            if len(self.to_delete_blocks) > 0:
                (block_id, content_hash, last_accessed) = self.to_delete_blocks.pop()
                if block_id not in self.free_table or self.free_table[block_id].last_accessed != last_accessed:
                    continue
                break
            else:
                candidate = self._peek_valid_candidate()
                if candidate is None:
                    raise ValueError("No usable cache memory left")
                block_id, content_hash = candidate
                if self.get_policy(self.free_table[block_id].cache_hint) == 'rrip':
                    rrpv = self.rrip_values.get(block_id,
                                                self.RRIP_INSERT_RRPV)
                    if rrpv < self.RRIP_MAX_RRPV:
                        self._age_rrip_blocks()
                        continue
                self._remove_from_sorted_dict(block_id)
                break
        if block_id in self.free_table:
            survival_time = time.time() - self.free_table[block_id].last_accessed
            self.stat.append("survival_times", survival_time)
            if self.get_policy(self.free_table[block_id].cache_hint) == 'rrip':
                self.rrip_values.pop(block_id, None)
            del self.free_table[block_id]
            self.id_to_first_access.pop(block_id, None)
            return block_id, content_hash
        else:
            # print('block is not in the sorted_dict')
            raise ValueError("block is not in the sorted_dict")
    
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        cache_hint = self._bind_scheduler_policy(cache_hint)
        if self.get_policy(cache_hint) == 'rrip' and block_id not in self.rrip_values:
            self.rrip_values[block_id] = self.RRIP_INSERT_RRPV
        if block_id not in self.id_to_first_access:
            self.id_to_first_access[block_id] = last_accessed
        score = self.calc_score(block_id, last_accessed, cache_hint)
        # print("add: ", block_id, cache_hint)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint,
                                                  score)
        self.sorted_dict[(score, last_accessed, block_id)] = (block_id, content_hash)
        self.id_to_last_access[cache_hint['id']] = last_accessed
        if time.time() - self.last_refresh_time > self.INSPECT_INTERVAL:
            self._refresh()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if block_id not in self.free_table:
            raise ValueError("Attempting to update block that's not in the evictor")
        cache_hint = self._bind_scheduler_policy(cache_hint)
        print("update: ", block_id, cache_hint['turns'])
        self._remove_from_sorted_dict(block_id)
        
        if self.get_policy(cache_hint) == 'rrip':
            self.rrip_values[block_id] = self.RRIP_HIT_RRPV
        score = self.calc_score(block_id, last_accessed, cache_hint)
        self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint
        self.free_table[block_id].score = score
        
        self.sorted_dict[(score, last_accessed, block_id)] = (block_id, self.free_table[block_id].content_hash)
        self.id_to_last_access[cache_hint['id']] = last_accessed

    # remove is only called by 'hit', the blocks will be added back later after decoding
    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        
        # print("remove: ", block_id)
        if self.get_policy(self.free_table[block_id].cache_hint) == 'rrip':
            self.rrip_values[block_id] = self.RRIP_HIT_RRPV
        self._remove_from_sorted_dict(block_id)
        del self.free_table[block_id]

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def _refresh(self):
        self._rebuild_sorted_dict()

        print('num blocks: ', self.num_blocks)

        self.last_refresh_time = time.time()
        stat_ = CacheStat()
        # Create a list of (block_id, survival_time) tuples
        survival_list = [
            (block_id, self.free_table[block_id].cache_hint['id'], self.free_table[block_id].last_accessed)
            for block_id in self.free_table
        ]
        # Sort by survival_time in descending order
        survival_list.sort(key=lambda x: (x[2], x[1]))

        # mark outdated blocks 
        to_delete_cnt = 0
        for block_id, _, _ in survival_list:
            block = self.free_table[block_id]
            if self.get_policy(block.cache_hint) == 'rrip':
                continue
            id = block.cache_hint['id']
            if self.id_to_last_access[id] == block.last_accessed:
                continue
            if self.id_to_last_access[id] != block.last_accessed:
                self.to_delete_blocks.append((block_id, block.content_hash, block.last_accessed))
                to_delete_cnt += 1
        print("mark outdated blocks cnt: ", to_delete_cnt)

        # Print top 10
        cnt = 0
        temp = 0
        print('Top 10 blocks with oldest last_accessed time:')
        for block_id, _, _ in survival_list:
            block = self.free_table[block_id]
            if block.last_accessed == temp:
                continue
            temp = block.last_accessed
            cnt += 1
            if cnt > 10:
                break
            print(block.cache_hint['true_tta'], block.score, block.last_accessed)

        for block_id in self.free_table:
            survival_time = time.time() - self.free_table[block_id].last_accessed
            stat_.append("survival_times", survival_time)
        print('For all blocks in the cache:')
        stat_.summary()

class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.stat = CacheStat()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                survival_time = time.time() - self.free_table[block_id].last_accessed
                self.stat.append("survival_times", survival_time)
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if 'use_fifo' not in cache_hint or cache_hint['use_fifo'] == 0:
            self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

def make_evictor(eviction_algorithm: str, config: str) -> Evictor:
    if eviction_algorithm == 'lru':
        return LRUEvictor()
    else:
        return LRUMLEvictor(config)
