# projects —— 旗舰项目 DataPilot

> 定位：面向企业数据中台的智能分析 Agent 平台（详见 `../PLAN.md` §5）

```
datapilot/
├── app/
│   ├── api/            # FastAPI 路由：会话、流式、审批
│   ├── agents/         # SQL / RAG / Analysis / Report Agent
│   ├── graph/          # LangGraph 编排与状态定义
│   ├── tools/          # 工具注册表 + 权限分级
│   ├── mcp/            # 自研 MCP Server / Client
│   ├── rag/            # 混合检索 + RRF + Rerank
│   ├── memory/         # 滑窗 + 摘要 + 向量长期记忆
│   ├── guardrails/     # SQL AST 校验 / 注入防御 / 输出校验
│   ├── router/         # 模型路由 + 熔断 + 降级 + 缓存
│   ├── observability/  # Langfuse Trace 与 Token 归因
│   └── eval/           # 评测集、LLM-as-Judge、回归门禁
├── web/                # React 前端：对话 / 工具时间线 / 审批中心 / 图表
├── data/               # 数据集与指标口径文档
├── docs/               # 架构图、ADR、评测报告
├── tests/
├── docker-compose.yml
└── README.md
```

**分周交付节奏**：W5 端到端跑通 → W6 全功能 + 评测报告 → W7 部署 + 演示视频 + 压测数据
