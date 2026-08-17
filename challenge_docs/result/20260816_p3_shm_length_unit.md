# P3 SHM framing 单测结果

- worktree/commit：`exp/p3-shm-length` / `ae707303`
- 命令：`pytest -q tests/distributed/omni_connectors/test_shm_connector.py`
- 原始日志：[p3_shm_length_unit_20260816_115600.log](../run_log/p3_shm_length_unit_20260816_115600.log)
- 断言：16 passed，15 warnings
- 进程退出：134（`corrupted size vs. prev_size`，与该仓库既有 NPU/扩展清理问题一致）

所有 SHM connector 断言在进程异常收尾前通过；因此只作为静态/协议门禁记录，不能替代
Chapter 10 A/B/A、Seed-TTS 和 full-duplex 长测。
