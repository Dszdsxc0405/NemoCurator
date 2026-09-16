# 视频流水线 CPU / GPU 供给实验（2026-09-16）

## 范围与方法

- 环境：`/team/xyq/data_juicer/envs/nemo-1`；GPU 4–7；96 个 Ray CPU。
- 原始 `run-op-6.yaml` 未修改，SHA256 为
  `72b9a273d93bcb83f4b1381ac6a9cb3672ce02e632ca44d97a0e3428d430e7e8`。
- 从现有 manifest 中剔除 123 个不存在的路径，固定种子 20260916 抽取 320 个视频。
  实验 manifest SHA256：`9e362e68a741d4dbe8eed47351a4dea86b012de23627df1cb70fb11b6ba6f7b9`。
- 结果与每秒 GPU 采样保存在 `../logs/explore-20260916/`。
- 端到端时间包括 Ray 启动、模型加载、运行、正常退出；不包含额外 `ray stop` 清理。
  单次小样本对照不是长时间稳态吞吐基准。GPU 利用率均值的窗口仅由显存阈值估计，不能解释为纯推理时间。

## 已完成的对照

| 方案 | 320 视频端到端时间 | 结果 |
| --- | ---: | --- |
| 2 writer，共享盘 | 107.1 s | 完成 |
| 8 writer，共享盘，每 worker 0.5 CPU | 94.0 s | 完成 |
| 2 writer，本地 `/tmp` | 114.7 s | 完成 |
| 8 writer，本地 `/tmp`，每 worker 0.5 CPU | 90.5 s | 完成 |
| 2 writer，共享盘，快速元数据 | 86.9 s | 完成 |
| 8 writer，共享盘，split 复用 | 85.9 s | 完成 |

以上均为 320 个视频统计、350 个片段：65 通过，27 aesthetics 过滤，125 flow 过滤，133 OCR 过滤。
未发现 OOM 或 timeout。增加 writer 并发方向较一致；换存储位置的收益不一致，不能据此认定共享盘是主因。
8 writer 同时改变了 CPU 预留，所以上表严格说是两项资源配置一起调整的对照。

快速元数据相对相同 2 writer 的基线减少约 18.8% 总耗时；350 个片段 ID、
源视频、时间范围、宽高、帧率、帧数、编码名、字节数、过滤状态和错误分类全部一致。
split 复用相对相同 8 writer 的共享盘基线减少约 8.6% 总耗时。
这两项是分开验证，不能直接相加得到组合收益；也不能将组件约 12 倍、约 100 倍的速度变化当成整条流水线收益。

最初把 8 writer 都预留 1 CPU 时，总需求 94 CPU，超过 Xenna 当次可用的 91 CPU，启动失败。
后改为 8 × 0.5 CPU，总需求 90 CPU。CPU 预留是调度声明，不是操作系统硬线程上限。

## 组件实验

1. **split 子进程复用**：相同 12 个视频，逐视频 spawn 耗时合计 22.324 s，
   预热后复用进程 1.850 s，场景边界完全相同。直接读取源路径 1.819 s；
   读源字节仅 0.021 s、写临时文件仅 0.019 s。因此该样本中应优先消除重复解释器启动和导入，
   而不是先做输入缓存架构。这个约 12 倍是预热后的组件测试，不是流水线加速倍数。
2. **writer 元数据读取**：350 个实际转码片段，逐片段 ffprobe 合计 131.049 s，
   内存 PyAV 读取 1.278 s；宽高、帧率、帧数、编码名全部一致。
   需要保留 ffprobe 六位小数 duration 的计算语义，否则部分片段帧数会差 1。
3. **抽帧改为内存顺序解码**：24 个片段，当前 seek 抽 3 帧合计 1.159 s，
   内存顺序解码 2.972 s。像素相同但更慢，不采用这个替换。

## 实现与验证

- `VideoSceneSplitStage.reuse_detector_process`：默认关闭；每 worker 一个可复用子进程。
  每个视频仍创建独立 detector；超时杀掉子进程，下个视频重建。
- `ClipWriterStage.use_pyav_metadata`：默认关闭；从已有 bytes 中读取容器元数据，
  缺失字段或异常回退原 ffprobe 路径。
- 两个对应测试文件共 45 项通过，覆盖真实多场景视频、坏文件后的恢复、超时杀进程、
  下次重建、元数据一致性和 fallback。修改的生产文件及测试通过 Ruff 检查。

## 不能混淆的概念

- 当前 manifest reader 是每个视频一个 Task，不是把全数据塞进单个 block。
- Xenna 的 stage `batch_size` 是一次 actor 调用的 Task 数，不是 Ray Data 的 block 数。
- split / 抽帧没有向量化 `process_batch`，默认逐 Task 顺序处理。
  20 workers × batch 1 已经可以并行 20 个视频；改成 batch 32 不会在每个 worker 内多出 32 路并行，
  反而可能让下游等整批、增加尾部延迟和在途内存。是否能摊薄调度成本需要另做 batch sweep。
- writer 也有 CPU 工作：每片段 ffprobe 会启动外部进程。GPU 空闲可能是下游反压，
  不能仅凭低利用率认定 split 不够快。监控的 Output Size 不是该 stage 的 Input Size。

## 960 视频组合验证

扩大样本使用相同种子的前 960 个视频，manifest SHA256：
`5239be827110955b489d51b78c2f311d4c4ddce4f623becbff306b116869961c`。
组合方案为 split 复用 + 2 writer 快速元数据；结果位于 `../logs/explore-20260916-validation/`。

运行中第一份非启动阶段的监控显示：split 20 worker 仅 1 个 Running，Output Size 39；
转码 12 worker 全部 Running、24 个槽全部占用；writer 2 worker 都空闲，下游 GPU 大多等待。
说明在消除前述开销后，瓶颈已经转移到转码，不应继续增加 split 数量。
随后对照重新分配 CPU：split 20 → 12、转码 12 → 24、uniform-3 抽帧 11 → 8；
其余参数保持不变，总 CPU 预留由 88 变为 89，低于 91 的预算。

- 组合方案完整完成：171.722 s，960 个视频统计、1039 个片段，215 通过、86 aesthetics 过滤、
  338 flow 过滤、400 OCR 过滤；未发现 OOM / timeout。GPU 4–7 峰值分别为
  34771 / 28000 / 57378 / 22024 MiB。这不是全量 5k 视频安全性保证。
- CPU 重新分配方案在 104.260 s 被保护停止：检测到新的外部 SGLang GPU 进程。
  此时仅完成 832/960 个视频统计，不能报告完整耗时、最终加速比或完整安全性验证。
  其中同一监控时点转码累计完成 636 个任务，对照方案为 443 个，是积极信号而非最终结论。
  停止后核对已写出的 894 个片段：ID 全部属于组合方案输出，源视频、时间范围、宽高、帧率、
  帧数、编码名、字节数、过滤状态及错误分类均无差异。该核对不覆盖未完成的数据，
  也不比较随机生成的 caption，不能代替完整运行。
- 原配置及已有 alternative 均不改动。新增 `../run-op-6-alternative-optimized.yaml` 只开启两项
  已验证优化，保留 split 20 / 转码 12 / 抽帧 11 / writer 2。
- 新增 `../run-op-6-alternative-candidate.yaml` 在上述基础上改为 split 12 / 转码 24 / 抽帧 8，
  **尚未完整验证**。应先协调 GPU 4–7 的使用，再继续对照；不终止外部服务。

CPU 优化已经确认有效，但没有证明四张 GPU 能持续满载。采样中大量片段在 flow/OCR 阶段被过滤，
后续 caption/camera 的有效工作量显著减少；优化后的转码仍是值得继续处理的 CPU 瓶颈。

复现实验（在仓库根目录，使用已有 nemo-1 环境）：

```bash
python tutorials/video/benchmark/pipeline_probe.py prepare \
  --root tutorials/video/logs/new-controlled-probe \
  --config tutorials/video/run-op-6-alternative.yaml --count 960
python tutorials/video/benchmark/pipeline_probe.py run \
  --root tutorials/video/logs/new-controlled-probe --workers 2 --storage shared \
  --label combined --reuse-detector --fast-metadata
python tutorials/video/benchmark/pipeline_probe.py run \
  --root tutorials/video/logs/new-controlled-probe --workers 2 --storage shared \
  --label rebalanced --reuse-detector --fast-metadata --rebalance-cpu
```

## 无效 / 受干扰实验

- `writer8-shared`：CPU 资源需求不满足，不是性能样本。
- `writer8-reuse` 和 `writer2-fastmetadata`：GPU 6、7 被其他环境的 SGLang 服务占用约 73 GB/卡，
  BLIP 加载 OOM / 显存保护停止，不能用于判断 split 或 metadata 优化是否安全、有效。
  不能把整卡显存占用归因于 Curator。
- 已为后续实验加入外部 GPU 进程检测：启动前拒绝占用卡，运行中出现外部任务则终止本次探测。
  不终止其他用户的服务。

## GPU 释放后的继续验证

- `rebalanced-clean-resumed`：8 reader / 12 split / 24 转码 / 8 uniform-3 抽帧。
  收尾阶段，NVML 将已退出的本任务 BLIP worker 显示为 `[No data]`，监控器误判为外部任务。
  162.139 s 时中止，952 个视频统计，不计为完整成功或可比较的总耗时。
- 已修复监控器：记录 Ray worker 的 PID 和 `/proc` 启动标识，允许已知 worker 的退出过渡状态，
  不放行未知进程或复用 PID 的新进程；启动前拒绝任何已有 GPU 计算任务。
- 该轮监控显示 8 reader 全忙，而部分转码 worker 无任务，故进一步调整为
  **12 reader / 10 split / 24 转码 / 6 uniform-3 抽帧**，其余 worker、batch、阈值、模型与编码参数不变。
  总 CPU 预留仍为 89。
- `reader12-balanced-clean`：完整 960 视频，136.896 s；输出 1039 个片段，
  215 通过、86 aesthetics 过滤、338 flow 过滤、400 OCR 过滤，与原组合方案一致。
  片段 ID、时间范围、技术元数据及过滤状态逐一比较无差异；未比较随机 caption 文本。
  无 OOM / timeout，GPU 4–7 峰值为 41097 / 46640 / 30726 / 21452 MiB。
- 同环境复跑 `optimized-reference-resumed`：完整 960 视频，176.493 s，输出数量同上，
  无 OOM / timeout。相比它，新方案总耗时减少 22.4%，吞吐提高 28.9%。
  这是固定样本单次配对对照，不是多轮统计置信区间。
- 新增 `../run-op-6-alternative-throughput.yaml`，保留原始数据路径和所有模型参数，
  使用 12 reader / 10 split / 24 转码 / 6 uniform-3 抽帧，CPU 预留 89。

## 新方案大样本验证与结论

`../logs/explore-20260916-full-validation/throughput-full.json` 记录完整结果。
原清单有 4965 个视频路径，其中 123 个不存在；本轮覆盖全部 **4842 个现存、互不重复的输入**，
固定种子重排后的实验 manifest SHA256 为
`5e38140be754d1cb43035bc1ca3ed91d44d770fce4f3ef9874a1f92e60f82f25`。
没有修改原清单，也没有将缺失文件算作处理成功。

- 总时间 **381.108 s**，退出码 0，无 traceback、OOM 或 timeout。
- 输入路径集合与输出 video 元数据集合逐一相等：4842 / 4842，没有缺失或额外视频。
- 输出 5210 个片段：1077 个通过、4133 个过滤；其中 aesthetics 过滤 418、flow 过滤 1749、
  OCR 过滤 1966。MP4 文件数与片段元数据数一致。
- 对照样本的 1039 个片段也全部出现在大样本输出中；技术元数据、时间范围、过滤状态及错误分类
  逐一对比无差异。随机生成的 caption 不要求逐字一致。
- GPU 4–7 的峰值分别为 **40559 / 69962 / 61788 / 44716 MiB**，最大约 **68.3 GiB**。
  这是当前数据集的观测峰值，不是对任意新视频或并发外部任务的显存保证。
- 帧数不足仍有明确的数据质量记录：7 个片段因不足两帧被 flow 过滤；18 个片段不足 camera 所需
  的 16 帧，按原实现保留 `static` 回退和错误标记。未改变这些业务语义。
- 小样本与大样本仍可观察到 GPU 利用率波动，第四张卡尤为明显，不能宣称四卡已持续满载。
  已确认的是相同 960 视频端到端耗时下降 22.4%、吞吐提高 28.9%，且大样本无 OOM。
- 收尾复跑两个对应测试文件：45 passed；Ray/探测进程均已退出，GPU 4–7 显存归零。

推荐使用 `../run-op-6-alternative-throughput.yaml`。相比原 optimized YAML，仅改变以下 worker 数，
CPU batch 仍为 1，GPU batch、模型、阈值、编码参数均不变：

| stage | 原 optimized | 新 throughput |
| --- | ---: | ---: |
| video_reader | 8 | 12 |
| video_scene_split | 20 | 10 |
| clip_transcoding | 12 | 24 |
| uniform_3_frame_sampling | 11 | 6 |
| 合计 CPU 预留 | 88 | 89 |

原始 YAML、已有 alternative 和 optimized YAML 的 SHA256 在本轮前后保持一致。
新 YAML 保留用户原始 manifest 路径，因此正式运行仍会遇到那 123 个缺失路径；
需要补齐数据或由用户选择另用清理后的 manifest，不应把性能调优当作数据修复。

```bash
bash tutorials/video/run.sh tutorials/video/run-op-6-alternative-throughput.yaml
```
