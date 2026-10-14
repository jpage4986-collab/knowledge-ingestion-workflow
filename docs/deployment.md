# 部署与运行

## 网页工作台

Python 3.10+；Windows启动脚本使用PowerShell 7。HTML/CSS/JavaScript直接由Python服务提供，无需npm install。

准备RAGFlow API地址及Key，确认服务中至少有一个配置可用embedding模型的模板数据集。启动命令见根README。默认绑定本机，界面与API在同一端口4189；`--help`可查看其他端口选项。

RAGFlow可部署在本机或远程。新部署按RAGFlow官方版本的Docker Compose安装配套数据库、对象存储、缓存和检索服务，避免混用不同版本组件；这里只通过RAGFlow HTTP API访问它们。选择能处理资料类型的解析器，MinerU不可用时新目标库使用DeepDOC。

## 可选AGH

单独安装官方Agnes Harness及其要求的Node版本，按官方构建指南准备运行目录。设置模型账户，并从本仓库根目录安装业务插件。详见agnes-integration.md。AGH模型与RAGFlow embedding分别配置，Key不互相替代。

## 原有CLI扩展

Zotero来源需要Zotero桌面客户端、本地只读API或只读数据库。MinerU解析需要MinerU及相应模型/计算环境；代理依赖可用`pip install -e ".[mineru-proxy]"`安装。可选检索助手依赖`pip install -e ".[assistant]"`。这些扩展不是网页DeepDOC入库的必需组件。

CLI复制examples/electromagnetics/ingestion/config.example.json为本机config.json，填写来源、解析、数据集映射及可选审查模型；先执行sync_library.py --help及只读preflight。原有CLI的数据发布路径独立于网页登记机制，使用新目标数据集和自己的配置。

## 本地状态

工作台目录的job-history.json与sandbox-registry.json保存任务和新库所有权。备份和恢复应同时保存二者；不要把其他实例的登记文件当作本实例授权。服务重启后活动任务会成为interrupted，可继续结果核验；缺失尚未上传的本地字节需要重新选择资料。

网上发现和模型推理需要相应网络服务。网页解析完成后的内容核验不是电磁仿真或材料实验；专业结论仍需领域验证。
