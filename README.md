# 资源推动 Agent Demo

资源调查 Demo：规则提取与确定性降级、`MiniMax-M3` Chat Completions + Pydantic 结构化理解、Tavily Search/Extract、MCP 内部实体与项目查询、证据核验、关联分析和 Jinja2 报告。

## 启动

1. 在 `.env` 中填写 `TAVILY_API_KEY`、`OPENAI_API_KEY` 和自行生成的随机 `LLM_SAFETY_SALT`。模型网关为 `https://vftllmapi.vf-tech.cn`，主模型与复核模型均为 `MiniMax-M3`，推理强度为 `xhigh`。
2. 执行：

```powershell
docker compose up --build
```

3. 打开 `http://localhost:3000`。

首次启动只下载本地 Whisper 模型。内部项目向量由 HashingVectorizer 即时生成，不下载嵌入模型。页面固定支持最新版桌面端 Chrome。

未配置 `OPENAI_API_KEY`、模型请求超时、输出格式错误或网关不可用时，七个大模型节点会标记为降级，任务继续使用规则、Tavily、MCP 和 Jinja2 生成基础报告。

固定文本测试：

```text
老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。
```

歧义确认测试：

```text
华星的李总明天参加会议
```

## 自动测试

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements-test.txt
.\.venv\Scripts\python -m pytest backend\tests -q
cd frontend
npm install
npm run build
```

## 服务

- Web：`http://localhost:3000`
- API：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- MCP：`http://localhost:8001/mcp`
