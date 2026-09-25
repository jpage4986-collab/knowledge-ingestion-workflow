# 领域适配：从参考实现到任意知识域

## 哪些层可以直接复用

文件哈希与去重、只读来源访问、MinerU PDF 解析、SQLite 生命周期索引、RAGFlow API 上传/状态轮询、失败队列、分块结构检查和评测运行器都属于可复用的工作流能力。密钥和服务地址由运行环境提供，不写在代码或示例配置中。

## 哪些内容必须按领域定义

- **检索词与来源：**例如开放文献库、标准/规范站点、软件手册、内部文档目录；只使用合法可访问的来源。
- **数据集路由：**把知识类别映射到独立的目标库，并指定每个库的 parser、embedding 与元数据字段。
- **抽取 schema：**字段名、单位、适用条件、允许的缺失值，以及每个字段需要的页级证据。
- **相关性与质量策略：**按来源类型、解析质量和任务用途制定阈值；不把某个领域的关键词列表直接沿用到另一领域。
- **验收集：**为每个领域准备覆盖常见问题、边界情况和拒答样本的固定基准，报告指标与错误案例。

## 当前参考实现的边界

`examples/electromagnetics/ingestion/` 是原电磁仿真项目的运行参考，包含 FDTD/COMSOL 相关性和仿真参数逻辑；`examples/electromagnetics/assistant/` 是电磁问答的可选下游示例。它们不是通用插件自动发现机制。

要让新领域达到“配置后即可运行”，下一步应将当前脚本中的领域关键词、摘要字段、分类规则和问答提示迁移到显式 `DomainProfile` 接口，并为至少两个不同领域各写一套离线基准。完成并通过该验证之前，本仓库定位为**可复用流程参考 + 领域示例**，而不是声称已验证的通用产品。

建议的领域配置契约：

```yaml
profile_id: example-domain
sources: []                 # enabled connectors and query templates
collections: {}             # logical category -> RAGFlow dataset settings
schema:                     # field -> type, unit, evidence requirements
  facts: []
review:
  rubric: []
  thresholds: {}
evaluation:
  benchmark_file: ./benchmarks/example-domain.json
```

这是适配契约示意，不是当前电磁参考脚本已实现的配置格式。
