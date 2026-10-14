# Knowledge Ingestion Harness

面向专业领域的知识采集、入库与审查工作台。通过自然语言或网页任务组织公开文献与本地资料，将来源发现、资料获取、并行解析、分块核验及审查反馈连接为可追踪的工作流。

[English](README.en.md) · [部署说明](docs/deployment.md) · [AGH接入](docs/agnes-integration.md) · [架构](docs/architecture.md) · [领域适配](docs/domain-adapters.md)

## 功能

- **真实资产可视化**：知识库分布图、文档处理统计、文档筛选与分页、实际分块及表格查看。
- **自动与并行入库**：OpenAlex/arXiv来源发现、公开PDF获取、浏览器选取本地文件或目录、逐项处理进度。
- **可检查的审查流程**：来源、重复、文件与解析检查，读取实际分块后给出规则核验结果；AGH可进一步开展专业审查。
- **历史与恢复**：任务、检查结果及文档标识持续保存；重启后继续检查已上传文档，避免恢复时重复上传。
- **现有库保护**：已有知识库用于读取；工作台的写入目标限定为本实例创建并登记的新库，名称标记本身不能授予写权限。
- **AGH工具接入**：16项业务工具支持项目读取、资产盘点、来源发现、入库执行、状态读取、证据核验、恢复与知识包输出。

## 快速启动

需要Python 3.10+及一个可访问的RAGFlow实例，RAGFlow须配置可用embedding模型。网页及Python标准库HTTP后端无需Node或前端构建步骤。

```powershell
$env:RAGFLOW_BASE_URL = "http://127.0.0.1:9380"
$env:RAGFLOW_API_KEY = "<你的 RAGFlow API Key>"
python examples/knowledge-workbench-ui/server.py --port 4189
```

打开 http://127.0.0.1:4189/ 。选择实际解析与向量模板，创建新目标库，再从专业主题或本地文件开始任务。MinerU不可用时，新库可使用DeepDOC；现有库的配置保持原状。

自然语言使用方法见[AGH接入](docs/agnes-integration.md)。AGH是可选智能体入口，网页工作台可独立运行。

## 专业领域与原有CLI

电磁仿真是当前参考领域，包含教材、材料参数、论文方法、软件手册、案例与代码的组织思路。新领域通过类别、字段、来源、审查规则及评测集适配。入库规则核验与专业结论验证是不同环节，领域字段需要回指原文证据。

原有`examples/electromagnetics/ingestion/`保留Zotero、MinerU、质量门禁、分库路由和检索评测入口；可选问答示例位于`examples/electromagnetics/assistant/`。CLI与网页有各自的配置与执行路径，见[部署说明](docs/deployment.md)。

## 目录

| 目录 | 用途 |
|---|---|
| examples/knowledge-workbench-ui | 网页、标准库后端、持久任务与独立测试 |
| plugins/knowledge-ingestion | AGH业务工具插件 |
| examples/electromagnetics | 专业CLI及可选检索助手参考 |
| docs | 部署、架构与领域适配 |
| tests | 原有离线单元测试 |

## 配置与数据

API Key通过环境变量提供。任务历史、新库登记和知识包属于本地运行数据，Git默认忽略；每个部署实例使用自己的服务、账户及资料。开源代码遵循MIT许可；AGH、外部服务、模型和文献遵循各自许可。
