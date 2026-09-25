# 面向任意领域的知识采集与 RAG 入库工作流

这是一个可复用的自动化知识入库方案：从文献/资料发现开始，经过原件保存、MinerU 解析、带证据的质量审查、分库路由、RAGFlow 入库与检索评测，形成可追溯、可重试、可迭代的闭环。

> **当前代码范围：**本仓库的工作流设计面向任意知识领域；随仓库发布的可运行参考实现来自电磁仿真项目，因此其检索评分、参数字段和回答策略仍有领域专用逻辑。要用于其他领域，需要提供对应的领域配置/抽取器，不能只改标题或搜索词。这个边界在 [领域适配说明](docs/domain-adapters.md) 中列出。

中文说明 · [English overview](README.en.md) · [架构与数据流](docs/architecture.md) · [安全边界](SECURITY.md)

## 工作流一览

```text
可配置来源（Zotero / 开放文献 API / 本地资料）
  → 候选去重与准入排序
  → 保存原始文件并计算哈希
  → MinerU 全页解析 PDF
  → 抽取结构化信息、逐字段关联原文证据
  → 解析质量 + 来源质量 + 领域审查三道门
      ├─ 不通过：隔离、记录原因，不进入检索库
      └─ 通过：按知识域写入目标 RAGFlow 数据集
  → 确认解析完成且分块非空
  → 检索基准测试、错误分析与分块策略迭代
```

## 设计原则

- **原件优先、可追溯：**保留来源链接、文档哈希、页码/段落证据、解析状态与 RAGFlow 文档 ID；摘要和抽取字段不能取代原始 PDF。
- **证据门禁，不让模型自证：**模型只提出候选结论；确定性检查会核对字段证据是否能在解析文本中定位，缺证据、字段冲突或解析异常就隔离。
- **知识域分库：**用数据集映射分开存储不同类型的知识；每个库单独配置解析器、向量模型和检索评测集，避免跨域噪声。
- **可恢复且幂等：**SQLite 索引记录来源标识、哈希、处理阶段和远端 ID；重试前先核对状态，避免重复上传。
- **先测再替换：**解析/分块的新版本先与基准比较，未通过门槛时保留已有可用版本。
- **最小权限：**Zotero 走只读访问；开放文献来源只下载可合法获取的文件；密钥仅从环境变量读取；默认关闭远端写入。

## 运行参考实现

随仓库附带的命令行参考实现位于 `examples/electromagnetics/ingestion/`。它需要本机 Zotero（或本地 Zotero 数据库）、MinerU、RAGFlow，以及可选的审查模型。启动任何远端写入前，先复制并填写 `examples/electromagnetics/ingestion/config.example.json`，设置所需的环境变量，并执行只读预检。

```powershell
Copy-Item examples/electromagnetics/ingestion/config.example.json examples/electromagnetics/ingestion/config.json
$env:MODEL_API_KEY = "<本机环境变量中的模型密钥>"
$env:RAGFLOW_API_KEY = "<本机环境变量中的 RAGFlow 密钥>"
python examples/electromagnetics/ingestion/sync_library.py --help
python examples/electromagnetics/ingestion/sync_library.py --config examples/electromagnetics/ingestion/config.json preflight
```

不同版本的参考脚本 CLI 参数可能不同；先看 `--help`。不要将真实 `config.json`、`.env`、本地数据库、PDF 或解析产物提交到 Git。

## 评估方式

质量不是单一 LLM 分数。入库前分别检查来源可追溯性、PDF 解析完整度、字段证据覆盖/冲突、领域相关性和分块结构；入库后用带标注的查询集检查命中、证据覆盖和引用定位。调整分块参数时保留基线结果，并比较新旧版本。离线结构测试通过不代表语义准确率达到 100%；每个领域都需要自己的基准集。

## 仓库内容与排除项

- `examples/electromagnetics/`：来自电磁仿真项目的参考代码，不包含用户文献库或运行配置。
- `tests/`：可离线运行的单元测试；不会自动连接 Zotero、模型服务或 RAGFlow。
- `docs/`：系统边界、数据流和其他领域的适配点。
- 不发布任何论文 PDF、Zotero 数据库、RAGFlow 数据集/文档 ID、API Key、真实服务地址、个人磁盘路径、运行数据库或用户的评测结果。

## 许可

本仓库代码采用 MIT License。第三方服务、模型、文献和软件的授权仍由各自提供方的条款决定；本仓库不转授权外部内容。
