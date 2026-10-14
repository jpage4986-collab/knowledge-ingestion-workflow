# 知识入库工作台

Python标准库后端与HTML/CSS/JavaScript界面。设置RAGFLOW_BASE_URL、RAGFLOW_API_KEY后，从仓库根目录运行`python examples/knowledge-workbench-ui/server.py --port 4189`。

界面包含来源输入、自动/并行任务、分布图、文档表、实际分块、检查流程和恢复历史。选择解析/向量模板后创建新目标库再导入。分块表格只渲染允许的表格元素，过滤可执行内容与任意属性。

从此目录可运行`python -m unittest -v tests.test_actual_recovery`；单元测试使用临时索引和mock服务。详见根目录README和docs/deployment.md。
