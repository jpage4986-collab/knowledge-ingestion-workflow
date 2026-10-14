# Agnes Harness接入

业务插件位于plugins/knowledge-ingestion，包名@knowledge-ingestion/workbench。它提供16项工具：项目入口、资产盘点、文档、分块、公开来源、本地来源、任务列表、服务状态、实际模型配置、新目标库、本地入库、在线入库、任务状态、任务证据、恢复审查和知识包保存。

## 环境

先启动网页后端，然后在启动AGH的同一个终端设置：

```powershell
$env:KNOWLEDGE_WORKSPACE = (Get-Location).Path
$env:KNOWLEDGE_WORKBENCH_URL = "http://127.0.0.1:4189"
```

从仓库根目录启动官方AGH实例，并在插件管理中安装file:./plugins/knowledge-ingestion。检查版本、integrity和capabilityHash；按实际预览值审核、信任并启用。在CLI中安装命令为`node <AGH运行目录>/agnes.mjs install file:./plugins/knowledge-ingestion`，后续trust/enable/status步骤使用该版本官方指南及安装返回值。修改插件后重新核对摘要。

本地后端要求localhost/127.0.0.1的HTTP地址。项目入口默认为插件所在仓库；显式KNOWLEDGE_WORKSPACE可指定另一工作区。模型账户在AGH设置保存；读取能力无需写授权，创建/上传/恢复/输出通过AGH写工具审批。

## 使用示例

> 为有限差分时域研究建立专业知识库。先查看实际模型与库配置，从我指定的本地目录或公开网络寻找资料，创建新目标库，用3路并行处理，跟踪解析并读取实际分块。审查来源、参数单位、条件及代码版本，生成回指证据的知识包。

工具只从指定工作区读取本地资料；新知识包保存在data/knowledge-packets。全自动入库按选定并行度运行，AGH工具审批仍按实例的审核策略执行。工作台已有库只读；写入要求目标同时具备Harness标记和本实例登记。
