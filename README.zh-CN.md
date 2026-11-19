# RoboHarness-Zero

本提交准备分支包含历史 Harness **2.0 / Xiaomi** 的 RoboCasa365 提交所需源码与材料。原始科学提交为 `c8911c64f24393e8665bd9ecc34f819173d9c3ba`，内部和包版本保持 **1.2.3.5**。项目源于 [RPent](https://github.com/RLinf/RPent)。

本分支已按提交范围裁剪其他集成、演示、部署工具和前端资源。保留的科学运行文件、模型提示与能力参考保持原始字节；原始来源和明确排除项记录于 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) 与[公开文件清单](publication/source-manifest.json)。成功反馈、验证和 RGB-D 协议保持原样。

## 既有结果

| 分组 | 成功 / 局数 | 成功率 |
| --- | ---: | ---: |
| Atomic-Seen | 62 / 72 | 86.111111% |
| Composite-Seen | 45 / 64 | 70.3125% |
| Composite-Unseen | 34 / 64 | 53.125% |
| 总计 | 141 / 200 | 70.5% |

这是历史作者评测 selected200 r002，每个任务四个既有 episode。原始 attempt、followup03 排除及十项 600 秒嵌套规划覆盖保持不变，详见[结果说明](results/2.0-selected200/REPORT.md)。不声称完成官方 2,500 局评测；原协议差异继续在[提交材料](evaluations/robocasa365/2.0/README.md)中披露。

## 离线检查与安装

在 Mac 或 Linux 的仓库根目录运行：

```bash
python3 tools/release.py verify
python3 tools/verify_results.py
python3 tools/setup_robocasa_submission.py --check-locks
python3 tools/configure_robocasa_planner.py
```

以上检查不需要 API、模型权重或模拟器，也不执行 episode。[安装说明](docs/policy-setup.md)提供 Linux x86_64 上固定的 coordinator、simulator、policy 三套环境、上游提交和模型版本：

```bash
python3 tools/setup_robocasa_submission.py \
  --prefix /path/to/new/harness-2.0-runtime \
  --download-assets --download-checkpoint
```

默认只预览；在具备 CUDA/EGL 条件的目标主机添加 `--apply` 才安装并检查导入及静态元数据。既有目录会被拒绝，失败记录会保留。当前整理未执行全新 Linux/GPU 安装或新评测。

## 评测与 Planner

[评测入口和配置](evaluations/robocasa365/2.0/README.md)区分历史结果与未来评审运行，提供原始选择、参数覆盖、核验及小型证据归档。评测入口默认只输出计划，实际执行需显式 `--execute`。原始状态与 RNG 记录在外部，历史轨迹回放不受支持。

[Planner API 说明](evaluations/robocasa365/2.0/PLANNER.md)提供私有 API 的配置方法，并保留历史 `gpt-6-astra` / `xhigh` 设置。API 地址和密钥由维护者在仓库外提供，通过 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 配置；源码包不要求部署服务。HTTP 超时为 330 秒，每次嵌套规划为 180 秒，整个 solve 的外层超时保持未设置。

[英文说明](README.md)、[文档索引](docs/README.md)、[源码与检查](docs/open-source-preparation.md)、[验证记录](publication/validation.json)和[发布清单](docs/release-checklist.md)提供其余提交所需说明。

项目源码（包括 RoboHarness-Zero 新增代码）采用 **[Apache-2.0](LICENSE)**。根目录 [NOTICE](NOTICE) 和[第三方声明](THIRD_PARTY_NOTICES.md)说明上游来源；RPent 原始许可证和既有归属保留。单独下载的模型权重、依赖实现及仿真素材继续遵循各自上游条款。

## 项目信息

| 项目 | 信息 |
| --- | --- |
| 团队 / 提交者 | Polyadic Research Group |
| 联系方式 | [123090492@link.cuhk.edu.cn](mailto:123090492@link.cuhk.edu.cn) |
| 论文标题 | RoboHarness-Zero: Any Frozen VLA, Zero Per-Task Preparation |
| 作者 | Yihao Shen、Yuxuan Chen、Xuantong Liu、Jun Gao、Weijian Luo、Tianyang Hu |
| 公开仓库 | [polyadic-research/roboharness0](https://github.com/polyadic-research/roboharness0) |

[BibTeX 引用](CITATION.bib) · [发布说明](docs/open-source-preparation.md)
