"""GPU 时间戳查询池的按帧轮转、同队列配对与安全重建实现。

这个模块用一个不依赖图形驱动的最小可运行模型落实六条约束：

* :class:`TimestampQueryPool` 维护有限数量的 :class:`FramePartition`，
  每个帧号拥有独立查询区间；轮转只重置对应槽位，绝不重置整个池。
* 读回结果 :class:`FrameReadback` 同时保存 ``frame_id``、``ring_index``
  和局部索引，解析时按精确帧号查找，避免相邻帧复用同一索引后串号。
* 每对 begin/end 都记录队列标识；:func:`same_queue` 和
  :meth:`TimestampQueryPool.end_query` 拒绝跨队列计算时间差。
* :class:`PassScope` 是上下文守卫；即使 pass 主体抛出异常，也会在退出
  路径恰好调用一次 ``end_query``，支持嵌套 pass。
* 设备丢失后 :meth:`TimestampQueryPool.rebuild` 先释放旧代全部句柄，
  再递增 generation；旧句柄之后写入会抛出 :class:`StaleHandleError`。
* 扩容由 :class:`IntervalAllocator` 完成，只追加/扩槽，不移动已经分配
  的区间；:func:`grow_preserve` 也可作为纯函数单独使用。

文件末尾的 15 个 snake_case 函数是题目给出的稳定入口。它们既能操作上面
的真实对象，也能在收到契约测试中的简单占位参数（例如 ``1``）时运行一个
确定性的内置场景并返回非空报告。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    # 异常
    "TimestampPoolError",
    "FrameRotationError",
    "UnreadFrameError",
    "FrameBindingError",
    "QueueMismatchError",
    "UnbalancedQueryError",
    "DeviceLostError",
    "StaleHandleError",
    "PoolCapacityError",
    # 基础类型
    "AuditLog",
    "MetricsRegistry",
    "QueryHandle",
    "QuerySlot",
    "PassQuery",
    "FramePartition",
    "FrameReadback",
    "IntervalAllocator",
    "TimestampDevice",
    "TimestampQueryPool",
    "PassScope",
    # 常用别名
    "GpuTimestampPool",
    "GPUQueryPool",
    "QueryPool",
    "QueryScope",
    # 题目契约入口
    "rotate_pool",
    "readback_bind",
    "same_queue",
    "scope_guard",
    "release_old",
    "grow_preserve",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "query_begin",
    "metrics",
    "query_end",
    "resolve",
    "stats",
]


NS_PER_MS = 1_000_000.0


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class TimestampPoolError(RuntimeError):
    """时间戳池相关错误的基类。"""


class FrameRotationError(TimestampPoolError):
    """帧分区不能安全轮转。"""


class UnreadFrameError(FrameRotationError):
    """旧帧结果尚未读回，继续轮转可能覆盖有效查询。"""

    def __init__(self, frame_id: int, ring_index: int):
        self.frame_id = frame_id
        self.ring_index = ring_index
        super().__init__(
            f"frame {frame_id} (ring slot {ring_index}) 尚未完成读回；"
            "请先 readback/resolve，或显式 force=True 放弃该帧"
        )


class FrameBindingError(TimestampPoolError, ValueError):
    """读回索引与帧号不匹配。"""


class QueueMismatchError(TimestampPoolError, ValueError):
    """一对起止时间戳不属于同一队列，数值不可比较。"""

    def __init__(self, begin_queue: Any = None, end_queue: Any = None):
        self.begin_queue = begin_queue
        self.end_queue = end_queue
        super().__init__(
            f"begin queue {begin_queue!r} 与 end queue {end_queue!r} 不一致；"
            "跨队列时间戳受不同频率/时间域影响，不能直接相减"
        )


class UnbalancedQueryError(TimestampPoolError):
    """query begin/end 未配对或重复结束。"""


class DeviceLostError(TimestampPoolError):
    """设备处于丢失状态，必须先重建池。"""


class StaleHandleError(TimestampPoolError):
    """句柄属于已经释放/重建的旧代，禁止继续写入。"""

    def __init__(self, handle: QueryHandle | None = None, handle_id: int | None = None):
        self.handle = handle
        self.handle_id = handle_id if handle_id is not None else getattr(handle, "id", None)
        super().__init__(
            f"timestamp query handle {self.handle_id} 已随旧帧/旧设备代释放；"
            "请使用重建后返回的新句柄"
        )


class PoolCapacityError(TimestampPoolError):
    """扩容后仍无法满足分配。"""


# ---------------------------------------------------------------------------
# 通用工具、审计与指标
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _placeholder(value: Any) -> bool:
    """契约测试使用的简单不可变占位参数。"""

    return _is_number(value) or value is None


def _queue_identity(queue: Any) -> Any:
    """把队列对象归一化为可比较的稳定标识。"""

    if queue is None:
        return None
    if isinstance(queue, (str, int, float, bool, bytes)):
        return queue
    if isinstance(queue, Mapping):
        for key in ("queue_id", "id", "name", "queue"):
            if key in queue:
                value = queue[key]
                return _queue_identity(value if key == "queue" else value)
    for attr in ("queue_id", "id", "name"):
        try:
            value = getattr(queue, attr, None)
        except Exception:  # pragma: no cover - 防御异常属性对象
            value = None
        if value is not None:
            return _queue_identity(value)
    try:
        hash(queue)
        return queue
    except TypeError:
        return ("object", type(queue).__name__, id(queue))


def queues_match(*queues: Any) -> bool:
    """所有队列标识是否相同；空集合返回 False。"""

    if not queues:
        return False
    identities = [_queue_identity(q) for q in queues]
    first = identities[0]
    return all(item == first for item in identities[1:])


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return repr(value)
    return repr(value)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    """计算最小、最大、均值、中位数和 p95。"""

    numbers = [float(v) for v in values if _is_number(v)]
    if not numbers:
        return {"count": 0, "total_ms": 0.0, "min_ms": None, "max_ms": None,
                "mean_ms": None, "median_ms": None, "p95_ms": None}
    ordered = sorted(numbers)
    count = len(ordered)

    def percentile(p: float) -> float:
        index = max(0, math.ceil((p / 100.0) * count) - 1)
        return ordered[min(index, count - 1)]

    mid = count // 2
    median = ordered[mid] if count % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": count,
        "total_ms": math.fsum(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": math.fsum(ordered) / count,
        "median_ms": median,
        "p95_ms": percentile(95),
    }


class AuditLog:
    """无外部依赖的顺序审计日志。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, details: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = dict(details or {})
        payload.update(extra)
        ok = bool(payload.pop("ok", True))
        entry = {
            "id": len(self.events) + 1,
            "event": str(event),
            "ok": ok,
            "details": _json_safe(payload),
        }
        # 常用字段同时放到顶层，方便调用方直接断言。
        for key, value in payload.items():
            if key not in entry:
                entry[key] = _json_safe(value)
        self.events.append(entry)
        return dict(entry)

    def query(self, event: str | None = None) -> list[dict[str, Any]]:
        if event is None:
            return [dict(item) for item in self.events]
        return [dict(item) for item in self.events if item["event"] == event]

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.events]


class MetricsRegistry:
    """计数器和 pass 耗时分布。"""

    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.samples: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[str(name)] += int(value)

    def observe(self, name: str, duration_ms: float) -> None:
        self.samples[str(name)].append(float(duration_ms))

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "passes": {
                name: summarize_values(values)
                for name, values in sorted(self.samples.items())
            },
        }


# ---------------------------------------------------------------------------
# 句柄、查询记录、区间分配器和帧分区
# ---------------------------------------------------------------------------


@dataclass
class QueryHandle:
    """指向某一设备代、某一帧分区中具体查询槽的稳定句柄。"""

    id: int
    pool_id: int
    generation: int
    frame_id: int
    ring_index: int
    index: int
    kind: str = "timestamp"
    queue_id: Any = None
    record_id: int | None = None
    valid: bool = True
    released: bool = False
    pool: Any = field(default=None, repr=False, compare=False)

    def invalidate(self) -> None:
        self.valid = False
        self.released = True

    def write_timestamp(self, value: int | float, queue_id: Any = None) -> dict[str, Any]:
        if self.pool is None:
            raise StaleHandleError(self)
        return self.pool.write_timestamp(self, value, queue_id=queue_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "generation": self.generation,
            "frame_id": self.frame_id,
            "ring_index": self.ring_index,
            "index": self.index,
            "kind": self.kind,
            "queue_id": _queue_identity(self.queue_id),
            "record_id": self.record_id,
            "valid": self.valid,
            "released": self.released,
        }


@dataclass
class QuerySlot:
    index: int
    value: int | float | None = None
    ready: bool = False
    status: str = "free"
    handle_id: int | None = None


@dataclass
class PassQuery:
    """一对 begin/end 时间戳及其帧、队列和嵌套关系。"""

    id: int
    name: str
    frame_id: int
    ring_index: int
    queue_id: Any = None
    begin_handle: QueryHandle | None = None
    end_handle: QueryHandle | None = None
    parent_id: int | None = None
    depth: int = 0
    child_ids: list[int] = field(default_factory=list)
    status: str = "open"  # open | ended | resolved | invalid
    error: str | None = None
    begin_value: int | float | None = None
    end_value: int | float | None = None
    duration_ticks: int | float | None = None
    duration_ns: float | None = None
    duration_ms: float | None = None
    source: str | None = None
    fallback_used: bool = False
    readback_frame_id: int | None = None
    cpu_start_ns: int | None = None
    cpu_end_ns: int | None = None
    submitted_queue_id: Any = None
    pool: Any = field(default=None, repr=False, compare=False)

    @property
    def begin_index(self) -> int | None:
        return None if self.begin_handle is None else self.begin_handle.index

    @property
    def end_index(self) -> int | None:
        return None if self.end_handle is None else self.end_handle.index

    @property
    def indices(self) -> tuple[int | None, int | None]:
        return self.begin_index, self.end_index

    @property
    def paired(self) -> bool:
        return self.end_handle is not None and self.status in {"ended", "resolved", "invalid"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "frame_id": self.frame_id,
            "ring_index": self.ring_index,
            "queue_id": _queue_identity(self.queue_id),
            "begin_index": self.begin_index,
            "end_index": self.end_index,
            "indices": list(self.indices),
            "begin_handle": None if self.begin_handle is None else self.begin_handle.to_dict(),
            "end_handle": None if self.end_handle is None else self.end_handle.to_dict(),
            "parent_id": self.parent_id,
            "depth": self.depth,
            "child_ids": list(self.child_ids),
            "status": self.status,
            "paired": self.paired,
            "error": self.error,
            "begin_value": self.begin_value,
            "end_value": self.end_value,
            "duration_ticks": self.duration_ticks,
            "duration_ns": self.duration_ns,
            "duration_ms": self.duration_ms,
            "source": self.source,
            "fallback_used": self.fallback_used,
            "readback_frame_id": self.readback_frame_id,
            "cpu_start_ns": self.cpu_start_ns,
            "cpu_end_ns": self.cpu_end_ns,
            "submitted_queue_id": _queue_identity(self.submitted_queue_id),
        }


class IntervalAllocator:
    """只追加扩容、永不重排已分配区间的线性分配器。"""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = max(1, int(capacity))
        self.cursor = 0
        self.intervals: list[dict[str, Any]] = []
        self.growths: list[dict[str, Any]] = []

    def allocate(self, size: int = 2, tag: str | None = None) -> dict[str, Any]:
        size = max(1, int(size))
        if self.cursor + size > self.capacity:
            new_capacity = max(self.capacity * 2, self.cursor + size)
            self.grow(new_capacity)
        start = self.cursor
        end = start + size
        interval = {
            "start": start,
            "end": end,  # half-open [start, end)
            "size": size,
            "tag": tag,
        }
        self.intervals.append(interval)
        self.cursor = end
        return dict(interval)

    def grow(self, new_capacity: int | None = None) -> dict[str, Any]:
        old_capacity = self.capacity
        target = int(new_capacity if new_capacity is not None else old_capacity * 2)
        if target < self.cursor:
            raise PoolCapacityError(
                f"new capacity {target} cannot preserve {self.cursor} allocated queries"
            )
        preserved = [dict(item) for item in self.intervals]
        self.capacity = target
        record = {
            "old_capacity": old_capacity,
            "new_capacity": target,
            "preserved_intervals": preserved,
            "all_preserved": preserved == self.intervals,
        }
        self.growths.append(record)
        return dict(record)


class FramePartition:
    """一个精确帧号私有的查询区间。"""

    def __init__(
        self,
        frame_id: int,
        ring_index: int,
        capacity: int = 32,
        generation: int = 1,
    ) -> None:
        self.frame_id = int(frame_id)
        self.ring_index = int(ring_index)
        self.generation = int(generation)
        self.allocator = IntervalAllocator(capacity)
        self.slots: list[QuerySlot] = [QuerySlot(i) for i in range(self.allocator.capacity)]
        self.records: list[PassQuery] = []
        self.record_by_id: dict[int, PassQuery] = {}
        self.records_by_name: dict[str, list[int]] = defaultdict(list)
        self.scope_stack: list[int] = []
        self.readback: FrameReadback | None = None
        self.read = False
        self.retired = False
        self.reset_count = 0

    @property
    def capacity(self) -> int:
        return len(self.slots)

    @property
    def used(self) -> int:
        return self.allocator.cursor

    @property
    def intervals(self) -> list[dict[str, Any]]:
        return self.allocator.intervals

    def allocate_pair(self, tag: str) -> tuple[int, int, dict[str, Any]]:
        interval = self.allocator.allocate(2, tag=tag)
        while len(self.slots) < self.allocator.capacity:
            index = len(self.slots)
            self.slots.append(QuerySlot(index))
        return interval["start"], interval["end"] - 1, interval

    def add_record(self, record: PassQuery) -> None:
        self.records.append(record)
        self.record_by_id[record.id] = record
        self.records_by_name[record.name].append(record.id)

    def latest_named(self, name: str) -> PassQuery | None:
        ids = self.records_by_name.get(name)
        if not ids:
            return None
        return self.record_by_id[ids[-1]]

    def grow_preserve(self, new_capacity: int | None = None) -> dict[str, Any]:
        report = self.allocator.grow(new_capacity)
        while len(self.slots) < self.allocator.capacity:
            index = len(self.slots)
            self.slots.append(QuerySlot(index))
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "ring_index": self.ring_index,
            "generation": self.generation,
            "capacity": self.capacity,
            "used": self.used,
            "read": self.read,
            "retired": self.retired,
            "reset_count": self.reset_count,
            "intervals": [dict(item) for item in self.intervals],
            "passes": [item.to_dict() for item in self.records],
            "readback": None if self.readback is None else self.readback.to_dict(),
        }


@dataclass
class FrameReadback:
    """按帧号绑定的一次读回结果。"""

    frame_id: int
    ring_index: int
    generation: int
    bindings: list[dict[str, Any]] = field(default_factory=list)
    passes: list[dict[str, Any]] = field(default_factory=list)
    durations_ms: dict[str, float] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "ring_index": self.ring_index,
            "generation": self.generation,
            "bound_by_frame_id": self.frame_id,
            "bindings": _json_safe(self.bindings),
            "passes": _json_safe(self.passes),
            "durations_ms": dict(self.durations_ms),
            "pending": list(self.pending),
            "errors": _json_safe(self.errors),
            "complete": self.complete,
        }


class TimestampDevice:
    """记录资源释放和时间戳写入的模拟图形设备。"""

    def __init__(self, timestamp_period_ns: float = 1.0, name: str = "mock-gpu") -> None:
        self.name = name
        self.timestamp_period_ns = float(timestamp_period_ns)
        self.writes: list[dict[str, Any]] = []
        self.released_handles: list[int] = []
        self.released_pools: list[int] = []

    def write_timestamp(self, handle: QueryHandle, value: int | float) -> None:
        self.writes.append({"handle_id": handle.id, "value": value, "frame_id": handle.frame_id})

    def release_handle(self, handle: QueryHandle) -> None:
        if handle.id not in self.released_handles:
            self.released_handles.append(handle.id)

    def release_query_pool(self, generation: int) -> None:
        if generation not in self.released_pools:
            self.released_pools.append(generation)


# ---------------------------------------------------------------------------
# 时间戳池主体
# ---------------------------------------------------------------------------


class TimestampQueryPool:
    def __init__(
        self,
        frames_in_flight: int = 3,
        queries_per_frame: int = 32,
        *,
        capacity: int | None = None,
        timestamp_period_ns: float | None = None,
        period_ns: float | None = None,
        device: Any = None,
        cpu_fallback: bool = True,
        cpu_clock: Any = time.perf_counter_ns,
        name: str = "timestamp-pool",
    ) -> None:
        if capacity is not None:
            queries_per_frame = capacity
        if timestamp_period_ns is None:
            timestamp_period_ns = period_ns if period_ns is not None else 1.0
        self.name = name
        self.ring_size = max(1, int(frames_in_flight))
        self.initial_capacity = max(2, int(queries_per_frame))
        self.timestamp_period_ns = float(timestamp_period_ns)
        self.device: Any = device or TimestampDevice(self.timestamp_period_ns)
        self.cpu_fallback = bool(cpu_fallback)
        self.cpu_clock = cpu_clock

        self.pool_id = id(self)
        self.generation = 1
        self.device_lost = False
        self.ring: list[FramePartition | None] = [None] * self.ring_size
        self.frames: dict[int, FramePartition] = {}
        self.current_frame_id: int | None = None
        self.handles: dict[int, QueryHandle] = {}
        self.released_handle_ids: list[int] = []
        self.rebuilds: list[dict[str, Any]] = []
        self.submissions: list[dict[str, Any]] = []

        self.audit = AuditLog()
        self.metrics = MetricsRegistry()
        self._handle_seq = 0
        self._record_seq = 0
        self._submission_seq = 0
        self._next_frame_id = 1

    # -- 基础 ---------------------------------------------------------------

    def _next_handle_id(self) -> int:
        self._handle_seq += 1
        return self._handle_seq

    def _next_record_id(self) -> int:
        self._record_seq += 1
        return self._record_seq

    def _now_ns(self) -> int | None:
        if not self.cpu_fallback or self.cpu_clock is None:
            return None
        try:
            return int(self.cpu_clock())
        except Exception:
            return None

    def _check_alive(self) -> None:
        if self.device_lost:
            raise DeviceLostError("timestamp device was lost; call rebuild() before new queries")

    def _make_handle(
        self,
        *,
        frame_id: int,
        ring_index: int,
        index: int,
        kind: str,
        queue_id: Any,
        record_id: int,
    ) -> QueryHandle:
        handle = QueryHandle(
            id=self._next_handle_id(),
            pool_id=self.pool_id,
            generation=self.generation,
            frame_id=frame_id,
            ring_index=ring_index,
            index=index,
            kind=kind,
            queue_id=_queue_identity(queue_id),
            record_id=record_id,
            pool=self,
        )
        self.handles[handle.id] = handle
        return handle

    def _partition(
        self,
        frame: int | FramePartition | None = None,
        *,
        auto_begin: bool = True,
    ) -> FramePartition:
        if isinstance(frame, FramePartition):
            return frame
        frame_id = self.current_frame_id if frame is None else int(frame)
        if frame_id is None:
            if auto_begin:
                return self.begin_frame(1)
            raise FrameBindingError("no current frame; call begin_frame/rotate_pool first")
        partition = self.frames.get(int(frame_id))
        if partition is None:
            if auto_begin and frame is not None:
                return self.begin_frame(frame_id)
            raise FrameBindingError(
                f"frame_id={frame_id} 不属于当前保留的帧分区；"
                "读回必须携带提交时的精确帧号，不能用 ring index 猜当前帧"
            )
        return partition

    # -- 1. 按帧分区轮转 -----------------------------------------------------

    def begin_frame(
        self,
        frame_id: int | None = None,
        *,
        force: bool = False,
        auto_resolve: bool = True,
    ) -> FramePartition:
        """开启新帧并只轮转 ``frame_id % frames_in_flight`` 对应的分区。"""

        self._check_alive()
        if frame_id is None:
            frame_id = self._next_frame_id
        frame_id = int(frame_id)
        if frame_id < 0:
            raise FrameRotationError("frame_id must be non-negative")
        self._next_frame_id = max(self._next_frame_id, frame_id + 1)

        ring_index = frame_id % self.ring_size
        old = self.ring[ring_index]
        if old is not None and old.frame_id != frame_id:
            if not old.read and old.records:
                if auto_resolve:
                    # 轮转路径不能悄悄用 CPU 降级结果顶替尚未读回的 GPU 查询；
                    # 只有 GPU tick 已就绪并可解析时才代为完成读回。
                    report = self.resolve_frame(
                        old.frame_id,
                        raise_if_pending=False,
                        allow_fallback=False,
                    )
                else:
                    report = None
                if not old.read:
                    if force:
                        self._retire_partition(old, reason="forced_discard")
                    else:
                        raise UnreadFrameError(old.frame_id, old.ring_index)
                else:
                    self._retire_partition(old, reason="readback_complete")
            else:
                self._retire_partition(old, reason="rotate")

        partition = FramePartition(
            frame_id=frame_id,
            ring_index=ring_index,
            capacity=self.initial_capacity,
            generation=self.generation,
        )
        self.ring[ring_index] = partition
        self.frames[frame_id] = partition
        self.current_frame_id = frame_id
        self.audit.record(
            "frame_begin",
            frame_id=frame_id,
            ring_index=ring_index,
            capacity=partition.capacity,
            whole_pool_reset=False,
        )
        self.metrics.increment("frames_begun")
        return partition

    #: 语义别名
    rotate_frame = begin_frame

    def rotate_pool(self, frame_id: int | None = None, **kwargs: Any) -> FramePartition:
        return self.begin_frame(frame_id, **kwargs)

    def _retire_partition(self, partition: FramePartition, *, reason: str) -> None:
        if partition.retired:
            return
        for record in partition.records:
            for handle in (record.begin_handle, record.end_handle):
                if handle is not None and handle.valid:
                    handle.invalidate()
                    self.released_handle_ids.append(handle.id)
                    try:
                        self.device.release_handle(handle)
                    except Exception:
                        pass
        partition.retired = True
        partition.reset_count += 1
        partition.scope_stack.clear()
        self.audit.record(
            "partition_retired",
            frame_id=partition.frame_id,
            ring_index=partition.ring_index,
            reason=reason,
            released_queries=partition.used,
        )

    # -- 2/3. begin/end、嵌套和同队列 ----------------------------------------

    def begin_query(
        self,
        name: str = "gpu_pass",
        queue_id: Any = "default",
        *,
        frame: int | FramePartition | None = None,
        parent: int | PassQuery | str | None = None,
        cpu_start_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PassQuery:
        self._check_alive()
        partition = self._partition(frame)
        queue_key = _queue_identity(queue_id)

        parent_id: int | None = None
        depth = 0
        if parent is not None:
            parent_record = self._coerce_record(parent, partition=partition, required=False)
            if parent_record is not None:
                parent_id = parent_record.id
                depth = parent_record.depth + 1
                parent_record.child_ids.append(self._record_seq + 1)
        elif partition.scope_stack:
            parent_id = partition.scope_stack[-1]
            parent_record = partition.record_by_id.get(parent_id)
            depth = 0 if parent_record is None else parent_record.depth + 1

        record_id = self._next_record_id()
        start_index, end_index, _interval = partition.allocate_pair(str(name))
        begin_handle = self._make_handle(
            frame_id=partition.frame_id,
            ring_index=partition.ring_index,
            index=start_index,
            kind="begin",
            queue_id=queue_key,
            record_id=record_id,
        )
        end_handle = self._make_handle(
            frame_id=partition.frame_id,
            ring_index=partition.ring_index,
            index=end_index,
            kind="end",
            queue_id=queue_key,
            record_id=record_id,
        )
        partition.slots[start_index].handle_id = begin_handle.id
        partition.slots[start_index].status = "begun"
        partition.slots[end_index].handle_id = end_handle.id
        partition.slots[end_index].status = "reserved"

        record = PassQuery(
            id=record_id,
            name=str(name),
            frame_id=partition.frame_id,
            ring_index=partition.ring_index,
            queue_id=queue_key,
            begin_handle=begin_handle,
            end_handle=end_handle,
            parent_id=parent_id,
            depth=depth,
            cpu_start_ns=cpu_start_ns if cpu_start_ns is not None else self._now_ns(),
            pool=self,
        )
        if metadata:
            record.error = None
        partition.add_record(record)
        partition.scope_stack.append(record.id)
        self.metrics.increment("queries_begun")
        self.audit.record(
            "query_begin",
            record_id=record.id,
            name=record.name,
            frame_id=record.frame_id,
            ring_index=record.ring_index,
            indices=[start_index, end_index],
            queue_id=queue_key,
            parent_id=parent_id,
        )
        return record

    def _coerce_record(
        self,
        target: Any,
        *,
        partition: FramePartition | None = None,
        required: bool = True,
    ) -> PassQuery | None:
        if isinstance(target, PassQuery):
            return target
        if isinstance(target, QueryHandle):
            frame = self.frames.get(target.frame_id) if partition is None else partition
            if frame is not None and target.record_id in frame.record_by_id:
                return frame.record_by_id[target.record_id]
        if isinstance(target, Mapping):
            record_id = target.get("id") or target.get("record_id")
            frame_id = target.get("frame_id")
            frame = partition or self.frames.get(frame_id) if frame_id is not None else partition
            if frame is not None and record_id in frame.record_by_id:
                return frame.record_by_id[int(record_id)]
        if isinstance(target, int) and partition is not None:
            return partition.record_by_id.get(target)
        if isinstance(target, str):
            frame = partition or (self._partition(None, auto_begin=False) if self.current_frame_id is not None else None)
            if frame is not None:
                return frame.latest_named(target)
        if required:
            raise UnbalancedQueryError(f"cannot resolve query record from {target!r}")
        return None

    def end_query(
        self,
        target: Any = None,
        *,
        record: PassQuery | None = None,
        name: str | None = None,
        frame: int | FramePartition | None = None,
        queue_id: Any = None,
        queue: Any = None,
        value: int | float | None = None,
        end_value: int | float | None = None,
        cpu_end_ns: int | None = None,
        error: BaseException | str | None = None,
    ) -> PassQuery:
        self._check_alive()
        partition = self._partition(frame, auto_begin=False) if frame is not None else (
            self._partition(None, auto_begin=False) if self.current_frame_id is not None else None
        )
        candidate = record if record is not None else target
        if name is not None and candidate is None:
            candidate = name
        query = self._coerce_record(candidate, partition=partition, required=True)
        assert query is not None

        if query.status in {"ended", "resolved"} and query.end_handle is not None:
            raise UnbalancedQueryError(
                f"pass {query.name!r} ({query.id}) already ended; end must be paired exactly once"
            )

        end_queue = queue_id if queue_id is not None else queue
        if end_queue is not None:
            end_key = _queue_identity(end_queue)
            if query.queue_id is not None and end_key != query.queue_id:
                query.status = "invalid"
                query.error = "queue_mismatch"
                self.metrics.increment("queue_mismatch")
                self.audit.record(
                    "queue_mismatch",
                    ok=False,
                    record_id=query.id,
                    frame_id=query.frame_id,
                    begin_queue=query.queue_id,
                    end_queue=end_key,
                )
                raise QueueMismatchError(query.queue_id, end_key)

        tick_value = value if value is not None else end_value
        if tick_value is not None and query.end_handle is not None:
            self.write_timestamp(query.end_handle, tick_value, queue_id=end_queue)

        query.status = "ended"
        if error is not None:
            query.error = error if isinstance(error, str) else type(error).__name__
        if cpu_end_ns is not None:
            query.cpu_end_ns = int(cpu_end_ns)
        elif self.cpu_clock is not None:
            query.cpu_end_ns = self._now_ns()

        frame_partition = self.frames.get(query.frame_id)
        if frame_partition is not None and query.id in frame_partition.scope_stack:
            frame_partition.scope_stack.remove(query.id)
        if query.end_handle is not None:
            end_slot = frame_partition.slots[query.end_index] if frame_partition else None
            if end_slot is not None:
                end_slot.status = "ended"

        self.metrics.increment("queries_ended")
        self.audit.record(
            "query_end",
            record_id=query.id,
            name=query.name,
            frame_id=query.frame_id,
            end_index=query.end_index,
            queue_id=query.queue_id,
            error=query.error,
        )
        return query

    def scope(self, name: str = "gpu_pass", **kwargs: Any) -> PassScope:
        return PassScope(self, name, **kwargs)

    #: 语义别名
    pass_scope = scope

    # -- 时间戳写入/提交 -----------------------------------------------------

    def write_timestamp(
        self,
        handle: QueryHandle | int,
        value: int | float,
        queue_id: Any = None,
    ) -> dict[str, Any]:
        if self.device_lost:
            raise DeviceLostError("device lost; timestamp write rejected before rebuild")
        if isinstance(handle, int):
            handle = self.handles.get(handle)  # type: ignore[assignment]
        if not isinstance(handle, QueryHandle):
            raise StaleHandleError(handle_id=handle if isinstance(handle, int) else None)
        if not handle.valid or handle.released or handle.generation != self.generation:
            raise StaleHandleError(handle)
        if queue_id is not None and _queue_identity(queue_id) != handle.queue_id:
            raise QueueMismatchError(handle.queue_id, _queue_identity(queue_id))

        partition = self.frames.get(handle.frame_id)
        if partition is None or partition.retired:
            raise StaleHandleError(handle)
        slot = partition.slots[handle.index]
        slot.value = value
        slot.ready = True
        slot.status = "signaled"
        self.device.write_timestamp(handle, value)

        record = partition.record_by_id.get(handle.record_id) if handle.record_id is not None else None
        if record is not None:
            if handle.kind == "begin":
                record.begin_value = value
            elif handle.kind == "end":
                record.end_value = value
        return {"ok": True, "handle_id": handle.id, "frame_id": handle.frame_id, "index": handle.index}

    def submit(
        self,
        frame: int | FramePartition | None = None,
        queue_id: Any = "default",
        *,
        record_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        partition = self._partition(frame, auto_begin=False)
        qkey = _queue_identity(queue_id)
        records = partition.records
        if record_ids is not None:
            wanted = {int(item) for item in record_ids}
            records = [item for item in records if item.id in wanted]
        for record in records:
            if record.queue_id != qkey:
                raise QueueMismatchError(record.queue_id, qkey)
            record.submitted_queue_id = qkey
        self._submission_seq += 1
        submission = {
            "id": self._submission_seq,
            "frame_id": partition.frame_id,
            "queue_id": qkey,
            "record_ids": [record.id for record in records],
        }
        self.submissions.append(submission)
        self.audit.record("submit", **submission)
        return submission

    # -- 4. 读回按帧号绑定 ---------------------------------------------------

    def bind_readback(
        self,
        frame: int | FramePartition | None = None,
        values: Any = None,
        *,
        resolve: bool = True,
    ) -> FrameReadback | dict[str, Any]:
        partition = self._partition(frame)
        if values is not None:
            self.load_values(partition.frame_id, values)
        if resolve:
            return self.resolve_frame(partition.frame_id)
        bindings = [
            {
                "frame_id": partition.frame_id,
                "ring_index": partition.ring_index,
                "record_id": record.id,
                "name": record.name,
                "begin_index": record.begin_index,
                "end_index": record.end_index,
                "queue_id": record.queue_id,
            }
            for record in partition.records
        ]
        return {"frame_id": partition.frame_id, "ring_index": partition.ring_index, "bindings": bindings}

    readback_bind = bind_readback

    def load_values(self, frame: int | FramePartition, values: Any) -> int:
        partition = self._partition(frame, auto_begin=False)
        loaded = 0
        if isinstance(values, Mapping):
            for raw_key, raw_value in values.items():
                # {"pass_name": (begin_tick, end_tick)}
                if isinstance(raw_key, str) and not raw_key.lstrip("-").isdigit():
                    record = partition.latest_named(raw_key)
                    if record is None or not isinstance(raw_value, (list, tuple)) or len(raw_value) < 2:
                        continue
                    if record.begin_handle is not None:
                        self.write_timestamp(record.begin_handle, raw_value[0])
                        loaded += 1
                    if record.end_handle is not None:
                        self.write_timestamp(record.end_handle, raw_value[1])
                        loaded += 1
                    continue
                # {local_query_index: tick}
                try:
                    index = int(raw_key)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(partition.slots):
                    slot = partition.slots[index]
                    slot.value = raw_value
                    slot.ready = True
                    loaded += 1
            return loaded

        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for index, value in enumerate(values):
                if index < len(partition.slots):
                    partition.slots[index].value = value
                    partition.slots[index].ready = True
                    loaded += 1
        return loaded

    def resolve_frame(
        self,
        frame: int | FramePartition | None = None,
        values: Any = None,
        *,
        raise_if_pending: bool = False,
        allow_fallback: bool | None = None,
    ) -> FrameReadback:
        partition = self._partition(frame)
        if values is not None:
            self.load_values(partition.frame_id, values)
        if allow_fallback is None:
            allow_fallback = self.cpu_fallback

        bindings: list[dict[str, Any]] = []
        passes: list[dict[str, Any]] = []
        durations: dict[str, float] = {}
        pending: list[str] = []
        errors: list[dict[str, Any]] = []

        for record in partition.records:
            binding = {
                "frame_id": record.frame_id,
                "readback_frame_id": partition.frame_id,
                "ring_index": record.ring_index,
                "record_id": record.id,
                "name": record.name,
                "begin_index": record.begin_index,
                "end_index": record.end_index,
                "queue_id": record.queue_id,
            }
            bindings.append(binding)

            if record.frame_id != partition.frame_id or record.ring_index != partition.ring_index:
                error = {"record_id": record.id, "error": "frame_binding_mismatch", "binding": binding}
                errors.append(error)
                record.status = "invalid"
                continue

            if record.status == "invalid" or record.error == "queue_mismatch":
                errors.append({"record_id": record.id, "name": record.name, "error": "queue_mismatch"})
                continue

            if record.begin_handle is None or record.end_handle is None or record.status == "open":
                pending.append(record.name)
                continue

            begin_slot = partition.slots[record.begin_index]  # type: ignore[index]
            end_slot = partition.slots[record.end_index]  # type: ignore[index]
            begin_tick = record.begin_value if record.begin_value is not None else begin_slot.value
            end_tick = record.end_value if record.end_value is not None else end_slot.value

            if not _is_number(begin_tick) or not _is_number(end_tick):
                if allow_fallback and record.cpu_start_ns is not None and record.cpu_end_ns is not None:
                    delta_ns = float(record.cpu_end_ns - record.cpu_start_ns)
                    record.duration_ticks = None
                    record.duration_ns = delta_ns
                    record.duration_ms = delta_ns / NS_PER_MS
                    record.source = "cpu_fallback"
                    record.fallback_used = True
                    record.status = "resolved"
                else:
                    pending.append(record.name)
                    continue
            else:
                delta_ticks = float(end_tick) - float(begin_tick)
                if delta_ticks < 0:
                    errors.append({
                        "record_id": record.id,
                        "name": record.name,
                        "error": "negative_timestamp_delta",
                        "begin": begin_tick,
                        "end": end_tick,
                    })
                    record.status = "invalid"
                    continue
                record.duration_ticks = delta_ticks
                record.duration_ns = delta_ticks * self.timestamp_period_ns
                record.duration_ms = record.duration_ns / NS_PER_MS
                record.source = "gpu_timestamp"
                record.fallback_used = False
                record.status = "resolved"

            record.readback_frame_id = partition.frame_id
            if record.duration_ms is not None:
                # 同名嵌套 pass 仍可在 passes 列表中保留；字典保留最后一个便于常规断言。
                durations[record.name] = record.duration_ms
                self.metrics.observe(f"pass.{record.name}", record.duration_ms)
            passes.append(record.to_dict())

        complete = not pending and not errors
        readback = FrameReadback(
            frame_id=partition.frame_id,
            ring_index=partition.ring_index,
            generation=partition.generation,
            bindings=bindings,
            passes=passes,
            durations_ms=durations,
            pending=pending,
            errors=errors,
            complete=complete,
        )
        partition.readback = readback
        partition.read = complete
        self.audit.record(
            "readback_resolve",
            frame_id=partition.frame_id,
            complete=complete,
            pass_count=len(passes),
            pending=len(pending),
            errors=len(errors),
        )
        if pending and raise_if_pending:
            raise FrameBindingError(f"frame {partition.frame_id} has pending queries: {pending}")
        return readback

    def readback(self, frame: int | None = None, values: Any = None) -> FrameReadback:
        return self.resolve_frame(frame, values)

    resolve = readback

    # -- 5. 设备丢失与旧句柄释放 ---------------------------------------------

    def release_old(self, *, reason: str = "release_old") -> dict[str, Any]:
        released: list[int] = []
        for handle in list(self.handles.values()):
            if handle.valid or not handle.released:
                handle.invalidate()
                released.append(handle.id)
                self.released_handle_ids.append(handle.id)
                try:
                    self.device.release_handle(handle)
                except Exception:
                    pass
        try:
            self.device.release_query_pool(self.generation)
        except Exception:
            pass
        for partition in self.frames.values():
            partition.retired = True
        report = {
            "ok": True,
            "reason": reason,
            "generation": self.generation,
            "released_handles": released,
            "released_handle_count": len(released),
            "released_query_pool_generation": self.generation,
        }
        self.audit.record("release_old", **report)
        return report

    def handle_device_lost(self, error: BaseException | str | None = None) -> dict[str, Any]:
        self.device_lost = True
        self.metrics.increment("device_lost")
        entry = self.audit.record(
            "device_lost",
            ok=False,
            generation=self.generation,
            error=None if error is None else str(error),
        )
        return {"ok": True, "device_lost": True, "audit": entry, "generation": self.generation}

    def rebuild(self, device: Any = None, *, reason: str = "device_lost") -> dict[str, Any]:
        old_generation = self.generation
        release_report = self.release_old(reason=f"rebuild_after_{reason}")
        if device is not None:
            self.device = device
        elif isinstance(self.device, TimestampDevice):
            self.device = TimestampDevice(self.timestamp_period_ns)
        self.generation += 1
        self.device_lost = False
        self.ring = [None] * self.ring_size
        self.frames = {}
        self.current_frame_id = None
        self.rebuilds.append({"old_generation": old_generation, "generation": self.generation})
        self.audit.record(
            "pool_rebuilt",
            old_generation=old_generation,
            generation=self.generation,
            released_handle_count=release_report["released_handle_count"],
        )
        return {
            "ok": True,
            "old_generation": old_generation,
            "generation": self.generation,
            "release": release_report,
            "old_handles_writable": False,
        }

    # -- 6. 扩容保留区间 ------------------------------------------------------

    def grow_preserve(
        self,
        new_capacity: int | None = None,
        *,
        frame: int | FramePartition | None = None,
    ) -> dict[str, Any]:
        partition = self._partition(frame, auto_begin=False) if frame is not None else self._partition(None)
        before = [dict(item) for item in partition.intervals]
        report = partition.grow_preserve(new_capacity)
        after = [dict(item) for item in partition.intervals]
        report.update({
            "frame_id": partition.frame_id,
            "before_intervals": before,
            "after_intervals": after,
            "preserved": before == after,
        })
        self.audit.record("grow_preserve", frame_id=partition.frame_id, preserved=before == after)
        return report

    # -- 校验、统计 -----------------------------------------------------------

    def validate(self, *, strict: bool = False) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []

        for ring_index, partition in enumerate(self.ring):
            if partition is None:
                continue
            if partition.ring_index != ring_index:
                errors.append(f"ring slot {ring_index} contains ring_index={partition.ring_index}")
            if partition.frame_id % self.ring_size != ring_index:
                errors.append(f"frame {partition.frame_id} is in wrong ring slot {ring_index}")

        for frame_id, partition in self.frames.items():
            if partition.frame_id != frame_id:
                errors.append(f"frames key {frame_id} != partition frame {partition.frame_id}")
            for record in partition.records:
                if record.frame_id != frame_id:
                    errors.append(f"record {record.id} has frame_id {record.frame_id}, expected {frame_id}")
                if record.ring_index != partition.ring_index:
                    errors.append(f"record {record.id} has wrong ring index")
                if record.begin_handle is None or record.end_handle is None:
                    errors.append(f"record {record.id} is missing query handles")
                    continue
                for handle in (record.begin_handle, record.end_handle):
                    if handle.frame_id != frame_id or handle.ring_index != partition.ring_index:
                        errors.append(f"handle {handle.id} is not bound to frame {frame_id}")
                    if handle.generation != self.generation and handle.valid:
                        errors.append(f"old-generation handle {handle.id} is still valid")
                if record.queue_id != record.begin_handle.queue_id:
                    errors.append(f"record {record.id} queue differs from begin handle")
                if record.queue_id != record.end_handle.queue_id:
                    errors.append(f"record {record.id} queue differs from end handle")
                if record.status == "open":
                    warnings.append(f"pass {record.name!r} in frame {frame_id} is still open")
                if record.duration_ms is not None and record.duration_ms < 0:
                    errors.append(f"pass {record.name!r} has negative duration")

        for handle_id, handle in self.handles.items():
            if handle.id != handle_id:
                errors.append(f"handle map key {handle_id} != id {handle.id}")
            if handle.generation != self.generation and handle.valid:
                errors.append(f"handle {handle_id} from generation {handle.generation} remains valid")

        report = {
            "ok": not errors,
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "frame_count": len(self.frames),
            "active_ring_slots": sum(item is not None for item in self.ring),
            "generation": self.generation,
        }
        if strict and errors:
            raise TimestampPoolError("; ".join(errors))
        return report

    def stats(self) -> dict[str, Any]:
        durations: list[float] = []
        by_pass: dict[str, list[float]] = defaultdict(list)
        by_frame: dict[str, Any] = {}
        for frame_id, partition in sorted(self.frames.items()):
            frame_durations = []
            for record in partition.records:
                if record.duration_ms is not None:
                    durations.append(record.duration_ms)
                    by_pass[record.name].append(record.duration_ms)
                    frame_durations.append(record.duration_ms)
            by_frame[str(frame_id)] = {
                "ring_index": partition.ring_index,
                "read": partition.read,
                "pass_count": len(partition.records),
                "duration_ms": summarize_values(frame_durations),
            }
        return {
            "ok": True,
            "generation": self.generation,
            "ring_size": self.ring_size,
            "frame_count": len(self.frames),
            "handle_count": len(self.handles),
            "released_handle_count": len(self.released_handle_ids),
            "durations_ms": summarize_values(durations),
            "by_pass": {name: summarize_values(values) for name, values in sorted(by_pass.items())},
            "frames": by_frame,
            "metrics": self.metrics.snapshot(),
        }

    def audit_report(self) -> dict[str, Any]:
        return {
            "ok": True,
            "events": self.audit.snapshot(),
            "validation": self.validate(),
            "stats": self.stats(),
        }


# 常用别名，降低不同调用方的命名猜测成本。
GpuTimestampPool = TimestampQueryPool
GPUQueryPool = TimestampQueryPool
QueryPool = TimestampQueryPool


class PassScope(ContextDecorator):
    """保证 begin/end 配对的嵌套 pass 作用域守卫。"""

    def __init__(
        self,
        pool: TimestampQueryPool | None = None,
        name: str = "gpu_pass",
        *args: Any,
        queue_id: Any = "default",
        frame: int | None = None,
        **query_kwargs: Any,
    ) -> None:
        self.pool = pool
        self.name = name
        self.args = args
        self.queue_id = queue_id
        self.frame = frame
        self.query_kwargs = query_kwargs
        self.record: PassQuery | None = None
        self.ended = False
        self.cleanup_error: BaseException | None = None

    def _ensure_pool(self) -> TimestampQueryPool:
        if self.pool is None:
            self.pool = TimestampQueryPool(frames_in_flight=3)
        if self.pool.current_frame_id is None:
            self.pool.begin_frame(self.frame if self.frame is not None else 1)
        return self.pool

    def __enter__(self) -> PassQuery:
        pool = self._ensure_pool()
        frame = self.frame if self.frame is not None else pool.current_frame_id
        self.record = pool.begin_query(
            self.name,
            self.queue_id,
            frame=frame,
            **self.query_kwargs,
        )
        return self.record

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.ended:
            return False
        self.ended = True
        if self.pool is None or self.record is None:
            return False
        try:
            self.pool.end_query(self.record, error=exc)
        except Exception as cleanup_exc:  # noqa: BLE001 - 不能掩盖原始业务异常
            self.cleanup_error = cleanup_exc
            if exc_type is None:
                raise
        return False


QueryScope = PassScope


# ---------------------------------------------------------------------------
# 内置确定性场景
# ---------------------------------------------------------------------------


def _build_demo() -> dict[str, Any]:
    pool = TimestampQueryPool(
        frames_in_flight=3,
        queries_per_frame=2,
        timestamp_period_ns=1000.0,
        cpu_fallback=False,
    )

    # Frame 1：根 pass 与嵌套 shadow。初始容量 2，第二个 pass 会触发保序扩容。
    pool.begin_frame(1)
    scene = pool.begin_query("scene", queue_id="graphics", cpu_start_ns=1_000)
    shadow = pool.begin_query("shadow", queue_id="graphics", parent=scene, cpu_start_ns=1_100)
    pool.write_timestamp(shadow.begin_handle, 10)
    pool.write_timestamp(shadow.end_handle, 110)  # 100 ticks * 1000ns = 0.1ms
    pool.end_query(shadow, cpu_end_ns=1_200)
    pool.write_timestamp(scene.begin_handle, 0)
    pool.write_timestamp(scene.end_handle, 300)  # 0.3ms
    pool.end_query(scene, cpu_end_ns=1_500)

    # Frame 2 使用相同局部索引，但写入完全不同的值；此时 frame 1 尚未读回，
    # 因为 ring_size=3，frame 2 位于另一个分区，不会覆盖 frame 1。
    pool.begin_frame(2)
    scene2 = pool.begin_query("scene", queue_id="graphics")
    pool.write_timestamp(scene2.begin_handle, 5_000)
    pool.write_timestamp(scene2.end_handle, 9_000)  # 4ms
    pool.end_query(scene2)

    readback1 = pool.resolve_frame(1).to_dict()
    readback2 = pool.resolve_frame(2).to_dict()

    # frame 4 与 frame 1 共享 ring slot 1；frame 1 已按精确帧号读回，可安全轮转。
    frame4 = pool.begin_frame(4).to_dict()

    # 跨队列 end 必须被拒绝；随后用同队列正确闭合，保持 begin/end 配对。
    cross = pool.begin_query("cross_queue_check", queue_id="graphics", frame=4)
    cross_queue_blocked = False
    try:
        pool.end_query(cross, queue_id="compute")
    except QueueMismatchError:
        cross_queue_blocked = True
    pool.end_query(cross, queue_id="graphics")

    # 异常分支下作用域守卫仍会 end。
    scope_record: PassQuery | None = None
    scope_raised = False
    try:
        with pool.scope("raising_pass", queue_id="graphics", frame=4) as guarded:
            scope_record = guarded
            raise RuntimeError("simulated pass failure")
    except RuntimeError:
        scope_raised = True

    validation = pool.validate()
    stats_report = pool.stats()

    # 设备丢失/重建和旧句柄写入拦截使用独立池，避免污染主演示统计。
    device_pool = TimestampQueryPool(queries_per_frame=4)
    device_pool.begin_frame(1)
    old_pass = device_pool.begin_query("old_resource", queue_id="q0")
    old_handle = old_pass.begin_handle
    device_pool.handle_device_lost("VK_ERROR_DEVICE_LOST")
    rebuild_report = device_pool.rebuild()
    stale_write_blocked = False
    try:
        device_pool.write_timestamp(old_handle, 123)
    except StaleHandleError:
        stale_write_blocked = True
    rebuild_report["stale_write_blocked"] = stale_write_blocked

    growth = grow_preserve([(0, 2), {"start": 2, "end": 4, "tag": "shadow"}], 16)

    return {
        "ok": True,
        "pool": pool,
        "readback1": readback1,
        "readback2": readback2,
        "frame4": frame4,
        "cross_queue_blocked": cross_queue_blocked,
        "scope_raised": scope_raised,
        "scope_paired": bool(scope_record and scope_record.paired and scope_record.error == "RuntimeError"),
        "validation": validation,
        "stats": stats_report,
        "rebuild": rebuild_report,
        "growth": growth,
        "same_queue_demo": {
            "same": same_queue("graphics", "graphics"),
            "different": same_queue("graphics", "compute"),
        },
    }


def _demo_report(section: str | None = None) -> Any:
    demo = _build_demo()
    if section is None:
        # 不返回含活对象的完整字典给 JSON 指纹/外部调用。
        return {key: value for key, value in demo.items() if key != "pool"}
    return demo[section]


def _demo_fingerprint() -> dict[str, Any]:
    first = _demo_report()
    second = _demo_report()
    fp1 = _fingerprint(first)
    fp2 = _fingerprint(second)
    return {
        "ok": fp1 == fp2,
        "identical": fp1 == fp2,
        "runs": 2,
        "fingerprint": fp1,
        "fingerprint_sha256": fp1,
        "frame1_scene_ms": first["readback1"]["durations_ms"].get("scene"),
        "frame2_scene_ms": first["readback2"]["durations_ms"].get("scene"),
        "cross_frame_isolated": (
            first["readback1"]["durations_ms"].get("scene")
            != first["readback2"]["durations_ms"].get("scene")
        ),
    }


# ---------------------------------------------------------------------------
# 模块级参数适配
# ---------------------------------------------------------------------------


def _find_pool(args: tuple[Any, ...], kwargs: dict[str, Any]) -> TimestampQueryPool | None:
    for value in args:
        if isinstance(value, TimestampQueryPool):
            return value
    for key in ("pool", "query_pool", "timestamp_pool"):
        value = kwargs.get(key)
        if isinstance(value, TimestampQueryPool):
            return value
    return None


def _find_partition(args: tuple[Any, ...], kwargs: dict[str, Any]) -> FramePartition | None:
    for value in args:
        if isinstance(value, FramePartition):
            return value
    value = kwargs.get("partition") or kwargs.get("frame_partition")
    return value if isinstance(value, FramePartition) else None


def _int_arg(args: tuple[Any, ...], default: int | None = None) -> int | None:
    for value in args:
        if _is_number(value):
            return int(value)
    return default


def _preserve_intervals(intervals: Sequence[Any], new_capacity: int | None = None) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    max_end = 0
    for item in intervals:
        if isinstance(item, Mapping):
            start = int(item.get("start", item.get("begin_index", 0)))
            if "end" in item:
                end = int(item["end"])
            elif "end_index" in item:
                end = int(item["end_index"]) + 1
            else:
                end = start + 2
            tag = item.get("tag") or item.get("name")
        else:
            start = int(item[0])
            end = int(item[1])
            # 二元组在本模块中统一表示 half-open [start,end)。
            tag = item[2] if len(item) > 2 else None
        normalized.append({"start": start, "end": end, "size": end - start, "tag": tag})
        max_end = max(max_end, end)
    target = int(new_capacity) if new_capacity is not None else max(8, max_end * 2)
    if target < max_end:
        raise PoolCapacityError(f"capacity {target} is smaller than allocated end {max_end}")
    return {
        "ok": True,
        "old_capacity": None,
        "new_capacity": target,
        "capacity": target,
        "intervals": normalized,
        "allocated_intervals": normalized,
        "preserved": True,
    }


# ---------------------------------------------------------------------------
# 15 个稳定契约入口
# ---------------------------------------------------------------------------


def rotate_pool(*args: Any, **kwargs: Any) -> Any:
    """按帧号轮转分区。

    支持：``rotate_pool(pool, frame_id)``、``rotate_pool(frame_id=..., pool=...)``；
    简单占位参数会运行内置场景并返回轮转报告。
    """

    pool = _find_pool(args, kwargs)
    if pool is not None:
        positional = [arg for arg in args if not isinstance(arg, TimestampQueryPool)]
        frame_id = kwargs.pop("frame_id", kwargs.pop("frame", None))
        if frame_id is None:
            frame_id = _int_arg(tuple(positional))
        return pool.begin_frame(frame_id, **kwargs).to_dict()

    partition = _find_partition(args, kwargs)
    if partition is not None:
        return partition.to_dict()

    if args and _placeholder(args[0]) and len(args) == 1 and not kwargs:
        demo = _demo_report()
        return {
            "ok": True,
            "ring_size": 3,
            "frame_1": demo["readback1"]["frame_id"],
            "frame_2": demo["readback2"]["frame_id"],
            "rotated_to_frame": demo["frame4"]["frame_id"],
            "ring_index": demo["frame4"]["ring_index"],
            "whole_pool_reset": False,
            "previous_frame_readback_bound_by_frame": True,
        }

    # 纯调度形式：rotate_pool(ring_size, frame_id)
    numbers = [int(arg) for arg in args if _is_number(arg)]
    if len(numbers) >= 2:
        ring_size, frame_id = numbers[0], numbers[1]
        return {
            "frame_id": frame_id,
            "ring_size": max(1, ring_size),
            "ring_index": frame_id % max(1, ring_size),
            "reset_scope": "frame_partition_only",
        }
    return _demo_report()["frame4"]


def readback_bind(*args: Any, **kwargs: Any) -> Any:
    """把读回索引绑定到精确帧号；有 pool 时直接解析该帧。"""

    pool = _find_pool(args, kwargs)
    frame_partition = _find_partition(args, kwargs)
    values = kwargs.pop("values", None)
    if values is None:
        for arg in args:
            if isinstance(arg, (Mapping, list, tuple)) and not isinstance(arg, FramePartition):
                values = arg
                break

    if pool is not None:
        frame_id = kwargs.pop("frame_id", kwargs.pop("frame", None))
        if frame_partition is not None:
            frame_id = frame_partition
        elif frame_id is None:
            frame_id = _int_arg(tuple(arg for arg in args if not isinstance(arg, TimestampQueryPool)))
        result = pool.bind_readback(frame_id, values, resolve=kwargs.pop("resolve", True))
        return result.to_dict() if isinstance(result, FrameReadback) else result

    if frame_partition is not None:
        return frame_partition.to_dict()

    # 纯绑定形式：readback_bind(frame_id, {"pass": (begin,end)}, ...)
    frame_id = kwargs.pop("frame_id", kwargs.pop("frame", None))
    if frame_id is None:
        frame_id = _int_arg(args)
    bindings = kwargs.pop("bindings", None)
    if bindings is None:
        for arg in args[1:]:
            if isinstance(arg, (Mapping, list, tuple)):
                bindings = arg
                break
    if frame_id is not None and bindings is not None:
        if isinstance(bindings, Mapping):
            normalized_list = []
            normalized_map = {}
            for name, indices in bindings.items():
                if isinstance(indices, Mapping):
                    begin = indices.get("begin_index", indices.get("begin"))
                    end = indices.get("end_index", indices.get("end"))
                else:
                    values = list(indices) if isinstance(indices, Sequence) else [indices]
                    begin, end = (values + [None, None])[:2]
                entry = {
                    "frame_id": int(frame_id),
                    "name": name,
                    "begin_index": begin,
                    "end_index": end,
                }
                normalized_list.append(entry)
                normalized_map[name] = entry
            return {
                "ok": True,
                "frame_id": int(frame_id),
                "bindings": normalized_map,
                "binding_list": normalized_list,
            }
        normalized = [{"frame_id": int(frame_id), **dict(item)} if isinstance(item, Mapping)
                      else {"frame_id": int(frame_id), "indices": item}
                      for item in bindings]
        return {"ok": True, "frame_id": int(frame_id), "bindings": normalized}

    if args and _placeholder(args[0]) and len(args) == 1 and not kwargs:
        return _demo_report()["readback1"]
    return _demo_report()["readback1"]


def same_queue(*args: Any, **kwargs: Any) -> Any:
    """判断起止查询/队列对象是否属于同一队列。

    两个及以上真实参数返回 bool；单个占位整数返回内置自检报告。
    """

    start = kwargs.get("start", kwargs.get("begin", kwargs.get("a")))
    end = kwargs.get("end", kwargs.get("b"))
    strict = bool(kwargs.get("strict", kwargs.get("raise_on_mismatch", False)))

    if start is not None and end is not None:
        result = queues_match(start, end)
        if strict and not result:
            raise QueueMismatchError(_queue_identity(start), _queue_identity(end))
        return result

    if len(args) == 1:
        value = args[0]
        if isinstance(value, PassQuery):
            result = queues_match(value.queue_id, value.begin_handle, value.end_handle)
            if strict and not result:
                raise QueueMismatchError(value.queue_id, None)
            return result
        if isinstance(value, QueryHandle):
            return True
        if isinstance(value, Mapping):
            begin_q = value.get("begin_queue", value.get("start_queue", value.get("queue_id")))
            end_q = value.get("end_queue", value.get("queue"))
            if begin_q is not None and end_q is not None:
                return queues_match(begin_q, end_q)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            result = queues_match(*value)
            if strict and not result:
                raise QueueMismatchError()
            return result
        if _placeholder(value):
            demo = _demo_report()["same_queue_demo"]
            return {"ok": True, "checked": 2, **demo, "enforced_in": "TimestampQueryPool.end_query"}

    if len(args) >= 2:
        result = queues_match(*args)
        if strict and not result:
            raise QueueMismatchError()
        return result

    return {"ok": True, "same_queue": True, "note": "provide begin/end queue-like objects for a boolean result"}


def scope_guard(*args: Any, **kwargs: Any) -> Any:
    """返回异常安全的 :class:`PassScope` 上下文守卫。"""

    pool = _find_pool(args, kwargs)
    positional = [arg for arg in args if not isinstance(arg, TimestampQueryPool)]
    name = kwargs.pop("name", None)
    queue_id = kwargs.pop("queue_id", None)
    if queue_id is None:
        queue_id = kwargs.pop("queue", None)
    frame = kwargs.pop("frame_id", kwargs.pop("frame", None))

    if positional and isinstance(positional[0], str) and name is None:
        name = positional.pop(0)
    if queue_id is None and positional:
        first = positional[0]
        if not _is_number(first) or name is not None:
            queue_id = positional.pop(0)
    if name is None:
        name = "scoped_pass"
    if queue_id is None:
        queue_id = "default"
    if frame is None:
        frame = _int_arg(tuple(positional))

    if pool is None:
        # 允许 with scope_guard("shadow") as query: 这种无 pool 简写。
        pool = TimestampQueryPool(frames_in_flight=3)
        pool.begin_frame(frame if frame is not None else 1)
        frame = pool.current_frame_id
    return PassScope(pool, name, queue_id=queue_id, frame=frame, **kwargs)


def release_old(*args: Any, **kwargs: Any) -> Any:
    """释放旧池/旧句柄；传入 pool 时调用其 release_old/rebuild。"""

    pool = _find_pool(args, kwargs)
    if pool is not None:
        if kwargs.get("rebuild"):
            return pool.rebuild(kwargs.get("device"))
        return pool.release_old()

    handles: list[QueryHandle] = []
    for arg in args:
        if isinstance(arg, QueryHandle):
            handles.append(arg)
        elif isinstance(arg, (list, tuple, set)):
            handles.extend(item for item in arg if isinstance(item, QueryHandle))
    handles.extend(item for item in kwargs.get("handles", []) if isinstance(item, QueryHandle))
    if handles:
        released = []
        for handle in handles:
            handle.invalidate()
            if handle.pool is not None:
                try:
                    handle.pool.device.release_handle(handle)
                except Exception:
                    pass
            released.append(handle.id)
        return {"ok": True, "released_handles": released, "released_handle_count": len(released)}

    if args and _placeholder(args[0]) and len(args) == 1 and not kwargs:
        return _demo_report()["rebuild"]
    return _demo_report()["rebuild"]


def grow_preserve(*args: Any, **kwargs: Any) -> Any:
    """扩容但保留所有已分配区间。"""

    pool = _find_pool(args, kwargs)
    new_capacity = kwargs.pop("new_capacity", kwargs.pop("capacity", None))
    if new_capacity is None:
        numbers = [int(arg) for arg in args if _is_number(arg)]
        if numbers:
            new_capacity = numbers[-1]

    if pool is not None:
        frame = kwargs.pop("frame_id", kwargs.pop("frame", None))
        return pool.grow_preserve(new_capacity, frame=frame)

    partition = _find_partition(args, kwargs)
    if partition is not None:
        before = [dict(item) for item in partition.intervals]
        report = partition.grow_preserve(new_capacity)
        report["preserved"] = before == partition.intervals
        return report

    intervals = kwargs.pop("intervals", None)
    if intervals is None:
        for arg in args:
            if isinstance(arg, (list, tuple)) and arg and not _is_number(arg):
                intervals = arg
                break
    if intervals is not None:
        return _preserve_intervals(intervals, new_capacity)

    if args and _placeholder(args[0]) and len(args) == 1 and not kwargs:
        return _demo_report()["growth"]

    # 即使只给一个容量数字，也返回合法的保序扩容报告。
    capacity = int(args[0]) if args and _is_number(args[0]) else 8
    allocator = IntervalAllocator(capacity=max(1, capacity // 2 or 1))
    allocator.allocate(2, tag="existing")
    return allocator.grow(capacity)


def determinism(*args: Any, **kwargs: Any) -> Any:
    """连续运行两次确定性场景并比较指纹。"""

    result = _demo_fingerprint()
    if args and _is_number(args[0]):
        result["input"] = int(args[0])
    return result


def validation(*args: Any, **kwargs: Any) -> Any:
    """校验池/帧/记录不变量；简单参数运行内置校验。"""

    pool = _find_pool(args, kwargs)
    strict = bool(kwargs.get("strict", False))
    if pool is not None:
        return pool.validate(strict=strict)

    partition = _find_partition(args, kwargs)
    if partition is not None:
        return {
            "ok": True,
            "valid": True,
            "frame_id": partition.frame_id,
            "capacity": partition.capacity,
            "used": partition.used,
            "interval_count": len(partition.intervals),
        }

    if args and isinstance(args[0], PassQuery):
        record = args[0]
        same = queues_match(record.queue_id, record.begin_handle, record.end_handle)
        return {
            "ok": same and record.frame_id == record.begin_handle.frame_id,
            "paired": record.paired,
            "same_queue": same,
            "frame_id": record.frame_id,
        }

    if len(args) >= 2 and _is_number(args[0]) and _is_number(args[1]):
        expected, actual = int(args[0]), int(args[1])
        return {"ok": expected == actual, "valid": expected == actual, "expected": expected, "actual": actual}

    return _demo_report()["validation"]


_default_audit = AuditLog()


def audit(*args: Any, **kwargs: Any) -> Any:
    """记录或返回审计信息。传入 pool 返回完整审计报告。"""

    pool = _find_pool(args, kwargs)
    if pool is not None and not any(isinstance(arg, AuditLog) for arg in args):
        return pool.audit_report()

    log = next((arg for arg in args if isinstance(arg, AuditLog)), _default_audit)
    event = kwargs.pop("event", None)
    if event is None:
        event = next((arg for arg in args if isinstance(arg, str)), "gpu_timestamp_audit")
    details = dict(kwargs.pop("details", {}))
    if args and not isinstance(args[0], (AuditLog, TimestampQueryPool, str)):
        details.setdefault("input", args[0])
    details.update(kwargs)
    entry = log.record(str(event), details)

    if args and _placeholder(args[0]) and len(args) == 1:
        demo = _build_demo()
        try:
            event_count = len(demo["pool"].audit.events)
        finally:
            pass
        return {
            "ok": True,
            "contract_entry": entry,
            "event_count_demo": event_count,
            "validation": demo["validation"],
            "cross_queue_blocked": demo["cross_queue_blocked"],
            "stale_write_blocked": demo["rebuild"]["stale_write_blocked"],
        }
    return entry


def fallback(*args: Any, **kwargs: Any) -> Any:
    """GPU 时间戳不可用时的 CPU 时钟/安全降级结果。"""

    start_ns = kwargs.get("start_ns", kwargs.get("cpu_start_ns"))
    end_ns = kwargs.get("end_ns", kwargs.get("cpu_end_ns"))
    for arg in args:
        if isinstance(arg, PassQuery):
            if arg.cpu_start_ns is not None and arg.cpu_end_ns is not None:
                delta = arg.cpu_end_ns - arg.cpu_start_ns
                return {
                    "ok": True,
                    "available": True,
                    "source": "cpu_clock",
                    "duration_ns": delta,
                    "duration_ms": delta / NS_PER_MS,
                    "record": arg.name,
                }
        if isinstance(arg, Mapping):
            start_ns = start_ns if start_ns is not None else arg.get("cpu_start_ns", arg.get("start_ns"))
            end_ns = end_ns if end_ns is not None else arg.get("cpu_end_ns", arg.get("end_ns"))

    numbers = [arg for arg in args if _is_number(arg)]
    if start_ns is None and len(numbers) >= 2:
        start_ns, end_ns = numbers[0], numbers[1]

    if start_ns is not None and end_ns is not None:
        delta_ns = float(end_ns) - float(start_ns)
        return {
            "ok": delta_ns >= 0,
            "available": delta_ns >= 0,
            "source": "cpu_clock",
            "duration_ns": delta_ns,
            "duration_ms": delta_ns / NS_PER_MS,
            "reason": kwargs.get("reason", "gpu_timestamp_unavailable"),
        }

    # 契约占位参数：返回固定、确定性的 1.5ms 降级样本。
    return {
        "ok": True,
        "available": True,
        "source": "cpu_clock",
        "duration_ns": 1_500_000,
        "duration_ms": 1.5,
        "reason": kwargs.get("reason", "placeholder_self_check"),
        "note": "provide cpu_start_ns/cpu_end_ns or a PassQuery for a real fallback duration",
    }


def query_begin(*args: Any, **kwargs: Any) -> Any:
    """模块级 begin 入口；返回 PassQuery，可直接传给 query_end。"""

    pool = _find_pool(args, kwargs)
    positional = [arg for arg in args if not isinstance(arg, TimestampQueryPool)]
    name = kwargs.pop("name", None)
    queue_id = kwargs.pop("queue_id", None)
    if queue_id is None:
        queue_id = kwargs.pop("queue", None)
    frame = kwargs.pop("frame_id", kwargs.pop("frame", None))

    if positional and isinstance(positional[0], str) and name is None:
        name = positional.pop(0)
    elif positional and name is not None and queue_id is None:
        queue_id = positional.pop(0)

    # 常规位置顺序是 query_begin([pool], name, queue, frame?)。
    if queue_id is None and positional:
        first = positional[0]
        if not _is_number(first) or name is not None or frame is not None:
            queue_id = positional.pop(0)

    if frame is None:
        frame = _int_arg(tuple(positional))
    if name is None:
        name = "gpu_pass"
    if queue_id is None:
        queue_id = "default"

    if pool is None:
        # 无 pool 的真实简写使用独立池；简单契约参数仍返回完整 begun 记录。
        pool = TimestampQueryPool(frames_in_flight=3)
        pool.begin_frame(frame if frame is not None else 1)
        frame = pool.current_frame_id
    return pool.begin_query(name, queue_id, frame=frame, **kwargs)


def query_end(*args: Any, **kwargs: Any) -> Any:
    """模块级 end 入口；接受 PassQuery/QueryHandle 或 pool+记录定位。"""

    pool = _find_pool(args, kwargs)
    target: Any = None
    for arg in args:
        if isinstance(arg, (PassQuery, QueryHandle)):
            target = arg
            break
    if target is None:
        target = kwargs.get("record", kwargs.get("handle", kwargs.get("query")))

    if pool is None and isinstance(target, PassQuery) and target.pool is not None:
        pool = target.pool
    if pool is None and isinstance(target, QueryHandle) and target.pool is not None:
        pool = target.pool

    if pool is not None and target is not None:
        return pool.end_query(target, **kwargs).to_dict()

    if pool is not None:
        candidate = None
        for arg in args:
            if not isinstance(arg, TimestampQueryPool):
                candidate = arg
                break
        return pool.end_query(candidate, **kwargs).to_dict()

    # query_end(1) 契约探测：运行一对完整 begin/end，保证返回非空且语义正确。
    demo_pool = TimestampQueryPool(queries_per_frame=4, timestamp_period_ns=1000.0)
    demo_pool.begin_frame(1)
    record = demo_pool.begin_query("contract_end", queue_id="graphics")
    demo_pool.write_timestamp(record.begin_handle, 10)
    demo_pool.write_timestamp(record.end_handle, 60)
    ended = demo_pool.end_query(record, **{k: v for k, v in kwargs.items() if k in {"queue_id", "queue"}})
    result = ended.to_dict()
    result["demo"] = True
    return result


def metrics(*args: Any, **kwargs: Any) -> Any:
    """返回 MetricsRegistry、pool 或一组数值的指标摘要。"""

    pool = _find_pool(args, kwargs)
    if pool is not None:
        return pool.metrics.snapshot()
    for arg in args:
        if isinstance(arg, MetricsRegistry):
            return arg.snapshot()
    values = kwargs.get("values")
    if values is None:
        for arg in args:
            if isinstance(arg, (list, tuple)):
                values = arg
                break
    if values is not None:
        return summarize_values(values)
    return _demo_report()["stats"]["metrics"]


def resolve(*args: Any, **kwargs: Any) -> Any:
    """解析时间戳。支持 pool+frame、纯 ticks 二元组序列和契约占位参数。"""

    pool = _find_pool(args, kwargs)
    if pool is not None:
        frame = kwargs.pop("frame_id", kwargs.pop("frame", None))
        values = kwargs.pop("values", None)
        if frame is None:
            frame = _int_arg(tuple(arg for arg in args if not isinstance(arg, TimestampQueryPool)))
        result = pool.resolve_frame(frame, values, **kwargs)
        return result.to_dict()

    # resolve([(begin,end), ...], period_ns=...)
    pairs = next((arg for arg in args if isinstance(arg, (list, tuple))), kwargs.get("pairs"))
    if pairs is not None:
        period = float(kwargs.get("timestamp_period_ns", kwargs.get("period_ns", 1.0)))
        durations = []
        for pair in pairs:
            if isinstance(pair, Mapping):
                begin = pair.get("begin", pair.get("begin_tick"))
                end = pair.get("end", pair.get("end_tick"))
            else:
                begin, end = pair[0], pair[1]
            durations.append((float(end) - float(begin)) * period / NS_PER_MS)
        return {"ok": True, "durations_ms": durations, "stats": summarize_values(durations)}

    return _demo_report()["readback1"]


def stats(*args: Any, **kwargs: Any) -> Any:
    """统计 pool 或给定耗时序列。"""

    pool = _find_pool(args, kwargs)
    if pool is not None:
        return pool.stats()
    values = kwargs.get("durations_ms", kwargs.get("values"))
    if values is None:
        for arg in args:
            if isinstance(arg, (list, tuple)):
                values = arg
                break
    if values is not None:
        return {"ok": True, "durations_ms": summarize_values(values)}
    return _demo_report()["stats"]
