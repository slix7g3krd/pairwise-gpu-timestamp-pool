# pairwise-gpu-timestamp-pool

## 领域

游戏、图形与多媒体

## 主题

GPU 时间戳查询池

## 说明

按帧轮转、同队列配对与句柄释放。

初始状态下 `graphics/gpu_timestamp_pool.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
