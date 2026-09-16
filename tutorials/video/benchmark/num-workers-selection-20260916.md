# 视频流水线 `num_workers` 选择依据（2026-09-16）

## 结论

当前 4 GPU、96 Ray CPU 环境推荐：

| Stage | `num_workers` |
| --- | ---: |
| `video_reader` | **12** |
| `video_scene_split` | **10** |
| `clip_transcoding` | **24** |
| `uniform_3_frame_sampling` | **6** |

完整配置为 `../run-op-6-alternative-throughput.yaml`，所有 stage 合计预留 89 CPU。

主要瓶颈是 **transcode**，不是 split。判断依据是 worker 饱和度、队列积压和端到端耗时，
不能只看 stage 的瞬时 CPU 百分比。

## 对照结果

两次运行使用相同的 4965 个视频和相同顺序，manifest SHA256：

```text
282f651d8bfed20404547f933533e9ba035f49b0fff922d5f37dcf624fa832ff
```

除 worker 分配外，模型、batch、阈值和编码参数保持一致；两边均预留 89 CPU。

| 配置 | Reader / Split / Transcode / Sampling | 端到端时间 | 吞吐 | GPU 活跃期均值 |
| --- | ---: | ---: | ---: | ---: |
| throughput | **12 / 10 / 24 / 6** | **338.0 s** | **14.69 视频/s** | **27.9%** |
| copy | 8 / 20 / 18 / 6 | 380.9 s | 13.03 视频/s | 21.8% |

throughput 配置：

- 端到端耗时降低 **11.3%**；
- 吞吐提高 **12.7%**；
- GPU 活跃期平均利用率提高 6.1 个百分点；
- 两边均正常完成，无 OOM、timeout 或 traceback，最终通过数均为 1118。

因此，把 split 从 10 增到 20、同时把 transcode 从 24 降到 18，会降低整体吞吐。

## 瓶颈判断

copy 配置的一个稳定运行快照：

| Stage | Actor | Running | Idle | 输出队列 |
| --- | ---: | ---: | ---: | ---: |
| `video_scene_split` | 20 | 2 | **18** | 38 |
| `clip_transcoding` | 18 | **18** | 0 | 2 |
| `uniform_3_frame_sampling` | 6 | 0 | 6 | 7 |

这说明：

- split 虽然约占 1900% CPU，但 20 个 worker 中有 18 个空闲，容量已经过量；
- split 后已有任务积压，而 18 个 transcode worker 全部忙碌；
- transcode 才是限制上游流量进入 GPU stages 的关键路径。

CPU 百分比不能直接决定 worker 数量，因为 split、FFmpeg 和视频解码会使用子进程或原生线程；
`cpus` 也是调度预留，不是操作系统线程上限。

## 选择准则

调整 `num_workers` 时按以下顺序判断：

1. 固定 manifest、输入顺序、模型、batch、输出位置和 GPU。
2. 确保 `sum(num_workers × cpus)` 小于实际可调度 CPU，并留出系统余量。
3. 优先增加长期 `Running == Actor Count` 且输入持续积压的 stage。
4. 减少长期大量 Idle、且输出队列已有数据的 stage。
5. 最终以相同输入的端到端时间和吞吐裁决，而不是以 CPU 峰值裁决。
6. 同时检查 OOM、GPU 利用率、输出数量和错误，避免用错误退出换取表面加速。

快速判断：

| 观测 | 判断 |
| --- | --- |
| 全部 worker 忙，上游任务持续积压 | worker 可能不足 |
| 大量 worker 空闲，下游仍有输入 | worker 过量 |
| GPU 间歇运行，紧邻上游全忙 | 增加或优化该上游 stage |
| 增加 worker 后总时间不降 | 已超过有效并行度，应回退 |

`batch_size` 与 `num_workers` 含义不同。当前 CPU stages 主要逐 Task 处理；增大 batch 不会自动在
单个 worker 内产生多路并行，batch 调优应单独测试。

## 结果与复现

结果文件：

- `../logs/compare-throughput-vs-copy-20260916/paired-throughput.json`
- `../logs/compare-throughput-vs-copy-20260916/paired-copy.json`
- `../logs/compare-throughput-vs-copy-20260916/paired-throughput-gpu.json`
- `../logs/compare-throughput-vs-copy-20260916/paired-copy-gpu.json`

复现命令：

```bash
source /team/xyq/data_juicer/envs/nemo-1/bin/activate

python tutorials/video/benchmark/pipeline_probe.py prepare \
  --root tutorials/video/logs/compare-workers-new \
  --config tutorials/video/run-op-6-alternative-throughput.yaml \
  --count 100000

python tutorials/video/benchmark/pipeline_probe.py run \
  --root tutorials/video/logs/compare-workers-new \
  --config tutorials/video/run-op-6-alternative-throughput.yaml \
  --label paired-throughput

python tutorials/video/benchmark/pipeline_probe.py run \
  --root tutorials/video/logs/compare-workers-new \
  --config tutorials/video/run-op-6-alternative-throughput-copy.yaml \
  --label paired-copy
```

本结论基于当前数据分布、存储和机器配置；输入视频长度或编码分布明显变化后应重新测试。
