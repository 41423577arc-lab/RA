# 下一步工作记录

记录时间：2026-07-22（Asia/Shanghai）

## 当前已完成

- 首屏已改为大模型文字对话窗口。
- 新增 `/api/v1/intake/chat`，用于追问并整理后续分析需要的信息。
- 对话接口返回 `assistant_reply`、`analysis_input`、`ready_to_analyze` 和 `missing_information`。
- 发送对话消息不会创建研究任务，也不会调用 Tavily、MCP 或报告 Pipeline。
- 用户点击“立即分析”后，前端才使用 `analysis_input` 调用原有 `/api/v1/tasks/text` 并启动 Celery Pipeline。
- 信息采集提示词位于 `backend/prompts/intake_chat_v1.txt`。
- 后端测试 37 项通过，前端 TypeScript 检查和 Next.js 生产构建通过。
- backend、worker、mcp-server 和 frontend 已部署到本地 Docker。
- 实际验收通过：对话前后任务数保持 29，点击“立即分析”后新增任务 `362eeef2-fe77-4286-b932-7320365cfb20`。

## 当前交互边界

- 本次实现的是文字对话采集。
- 原 `/api/v1/tasks/audio` 语音分析接口仍保留。
- 原首屏语音入口暂未放入新的聊天窗口，因为它会绕过“先对话、再立即分析”的交互门槛。

## 后续可选工作

1. 如需恢复语音入口，新增“只转写、不创建任务”的接口，将转写文字作为一条用户消息发送到 `/api/v1/intake/chat`。
2. 根据真实使用反馈细化 `backend/prompts/intake_chat_v1.txt` 的追问顺序和信息齐全判断。
3. 决定是否把当前 WIP 合并到 `main`。
