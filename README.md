# 资源推动 Agent Demo

资源调查 Demo：规则提取与确定性降级、`MiniMax-M3` Chat Completions + Pydantic 结构化输出、intake 阶段受控的 Tavily 关键人身份补全、MCP 内部实体与项目查询、关联分析和 Jinja2 报告。

业务操作、历史会话和管理后台说明见 [资源推动 Agent 使用手册](docs/资源推动Agent使用手册.md)。

## 启动

1. 在 `.env` 中填写 `TAVILY_API_KEY`、`OPENAI_API_KEY` 和自行生成的随机 `LLM_SAFETY_SALT`。模型网关为 `https://vftllmapi.vf-tech.cn`，主模型与复核模型均为 `MiniMax-M3`，推理强度为 `xhigh`。
2. 执行：

```powershell
docker compose up --build
```

3. 打开 `http://localhost:3000`。

首次启动只下载本地 Whisper 模型。内部项目向量由 HashingVectorizer 即时生成，不下载嵌入模型。页面固定支持最新版桌面端 Chrome。

当前共有 9 个结构化 LLM 提示词节点：5 个 intake 节点、3 个研究节点和 1 个分析问答节点。未配置 `OPENAI_API_KEY`、模型请求超时、输出格式错误或网关不可用时，对应路径会记录降级事件，并在可降级的节点继续使用规则、Tavily、MCP 和 Jinja2 生成结果。

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

## 架构与流程

以下内容以当前 `main` 分支代码为准。当前未实现 Authentication、Nginx/API Gateway、独立 Report API、对象存储和集中可观测性；报告直接作为任务响应字段返回。

当前 LLM 节点按职责分为：

- Intake：`intake_chat`、`intake_followup`、`intake_identity_normalize`、`intake_readiness`、`intake_final_confirmation`
- Research：`evidence_verify`、`final_synthesis`
- Analysis chat：`analysis_chat`

### 图 1：系统总览

这张图只展示运行时主干，细节在后续小图中展开。

```mermaid
flowchart LR
    User["业务用户"] --> Browser["浏览器"]
    Browser --> Frontend["Next.js 前端"]
    Browser -->|"REST / SSE"| API["FastAPI"]
    API --> AppDB[("PostgreSQL<br/>会话与任务状态")]
    API -->|"入队"| Redis["Redis"]
    API -->|"Intake / 分析问答"| LLM["LLM Gateway"]
    API -->|"Intake 身份补全"| Tavily["Tavily"]
    API -->|"Intake 内部候选"| MCP["MCP Server"]
    Redis --> Worker["Celery Worker"]
    Worker --> AppDB
    Worker --> LLM
    Worker --> Tavily
    Worker --> MCP
    MCP -->|"只读"| ProjectDB[("PostgreSQL<br/>内部项目数据")]
```

前端页面由 Next.js 提供；浏览器使用 `NEXT_PUBLIC_API_BASE_URL` 直接访问 FastAPI。FastAPI 负责会话、任务和事件读取，并同步执行 Intake 与分析问答；耗时的音频转写与研究流程由 Celery Worker 执行。

### 图 2：用户业务流程

```mermaid
flowchart LR
    Input["输入会面信息"] --> Intake["补全人物、企业、活动和关注方向"]
    Intake --> Identity["确认标准身份"]
    Identity --> Summary["确认最终摘要"]
    Summary --> Start["开始分析"]
    Start --> Research["公开研究与内部项目检索"]
    Research --> Report["详细报告与行动简报"]
    Report --> Chat["围绕当前结果继续问答"]
```

身份不完整时流程留在 intake 阶段继续追问或让用户选择候选。只有最终摘要确认后，`POST /api/v1/intake/{session_id}/start-analysis` 才会创建研究任务。

### 图 3：信息采集

#### 图 3.1：文字采集与身份补全

```mermaid
flowchart TD
    Message["POST /api/v1/intake/chat"] --> Runner["IntakeRunner.run_chat"]
    Runner --> Parse["IntakeAgent 提取结构化上下文"]
    Parse --> Complete{"必要字段完整？"}
    Complete -->|"否"| Followup["生成追问"]
    Followup --> Message
    Complete -->|"是"| Internal["MCP find_entity_candidates"]
    Internal --> Resolved{"身份可自动确定？"}
    Resolved -->|"否"| Web["Tavily 身份搜索与正文提取"]
    Web --> Evidence["校验姓名、组织和原文证据"]
    Evidence --> Candidate{"唯一高置信候选？"}
    Candidate -->|"否"| UserConfirm["用户选择或手工填写身份"]
    UserConfirm --> ConfirmEntity["POST /confirm"]
    ConfirmEntity --> FinalSummary
    Candidate -->|"是"| FinalSummary["生成最终确认摘要"]
    Resolved -->|"是"| FinalSummary
    FinalSummary --> Confirm["POST confirm-summary"]
    Confirm --> Ready["IntakeSession = READY"]
```

内部实体查询始终先执行；Web 查询只补全仍未解决的身份。外部候选只有在来源页面正文能够支持标准姓名和关系信息时才会被接受。

#### 图 3.2：音频采集

```mermaid
flowchart LR
    Upload["上传 audio/webm"] --> Job["创建 IntakeAudioJob"]
    Job --> Queue["Celery 入队"]
    Queue --> Whisper["LocalWhisperTranscriber"]
    Whisper --> Review["NEEDS_REVIEW"]
    Review --> Correct["用户校对转写文本"]
    Correct --> Chat["作为消息进入 IntakeRunner"]
    Whisper -->|"失败"| Failed["FAILED，可重试"]
    Failed --> Queue
```

录音文件写入 `audio_data` 命名卷。前端轮询音频任务，转写成功后必须由用户确认文本，不能直接触发研究。

### 图 4：后端模块

#### 图 4.1：API 与状态读取

```mermaid
flowchart TD
    Main["app.main:app"] --> IntakeAPI["app.api.intake.router"]
    Main --> TaskAPI["app.api.tasks.router"]
    IntakeAPI --> IntakeRunner["IntakeRunner / IntakeAgent"]
    IntakeAPI --> IntakeRepo["IntakeSessionRepository"]
    TaskAPI --> TaskRepo["TaskRepository"]
    TaskAPI --> AnalysisChat["AnalysisChatAgent"]
    IntakeRepo --> ORM["SQLAlchemy ORM"]
    TaskRepo --> ORM
    ORM --> DB[("PostgreSQL")]
    TaskAPI --> SSE["execution_stream"]
    SSE -->|"读取 ExecutionEvent"| DB
```

Intake Router 提供对话、活动查询、音频上传/重试、身份确认、摘要确认和开始分析；Task Router 提供兼容的文字/音频任务入口，以及任务查询、执行日志、SSE、确认、取消、清空和分析问答。`GET /api/v1/tasks/{task_id}` 同时返回任务状态和报告字段，当前没有独立 Report Router。`GET /api/v1/tasks/{task_id}/events` 从 `execution_events` 读取事件并输出 SSE；前端另外轮询 intake activity 和音频任务状态。

#### 图 4.2：异步任务与研究服务

```mermaid
flowchart TD
    Celery["celery_app"] --> AudioTask["run_intake_audio_transcription"]
    Celery --> PipelineTask["run_research_pipeline"]
    AudioTask --> Whisper["LocalWhisperTranscriber"]
    PipelineTask --> Pipeline["ResearchPipeline"]
    Pipeline --> Tools["ResearchToolExecutor"]
    Pipeline --> Ranker["ProjectRanker"]
    Pipeline --> Association["ResourceAssociationBuilder"]
    Pipeline --> Synthesis["final_synthesis"]
    Pipeline --> Renderer["ReportRenderer"]
```

`ResearchPipeline` 按固定顺序调用公开检索、内部项目查询、排序、关联、综合和渲染；工具参数仍由规则代码生成。

### 图 5：研究流水线

#### 图 5.1：任务启动与上下文准备

```mermaid
flowchart TD
    Task["加载 ResearchTask"] --> Cancelled{"已取消？"}
    Cancelled -->|"是"| Stop["停止"]
    Cancelled -->|"否"| Context{"已有 confirmed_context？"}
    Context -->|"是"| Restore["从 Intake 快照恢复上下文"]
    Context -->|"否"| Extract["RuleExtractor.extract"]
    Extract --> Understand["fallback_understanding"]
    Understand --> Resolve["EntityResolver.resolve"]
    Resolve --> NeedConfirm{"存在身份歧义？"}
    NeedConfirm -->|"是"| Wait["NEEDS_CONFIRMATION<br/>保存后退出任务"]
    NeedConfirm -->|"否"| Persist["保存 confirmed_context"]
    Restore --> Research["进入固定研究流水线"]
    Persist --> Research
```

标准 intake 路径通常已经携带 `confirmed_context`。`/api/v1/tasks/text` 和 `/api/v1/tasks/audio` 仍保留兼容入口，因此 Pipeline 仍包含规则提取和任务级身份确认分支。用户确认完成后进入固定研究流水线。

#### 图 5.2：固定研究编排

```mermaid
flowchart TD
    Context["confirmed_context"] --> WebPlan["规则生成 Tavily 查询"]
    WebPlan --> WebTool["search → extract"]
    WebTool --> Evidence["证据规则分流"]
    Evidence -->|"歧义候选"| Verify["evidence_verify"]
    Evidence --> ProjectPlan["规则生成 MCP 参数"]
    Verify --> ProjectPlan
    ProjectPlan --> ProjectTool["search_projects"]
    ProjectTool --> Rank["ProjectRanker"]
    Rank --> Associate["ResourceAssociationBuilder"]
    Associate --> Synthesis["final_synthesis"]
    Synthesis --> Render["ReportRenderer"]
```

规则先把公开证据分成接受、拒绝和歧义三类，只有歧义候选交给 `evidence_verify`。Tavily 或 MCP 单路失败时记录降级状态并继续后续步骤。

#### 图 5.3：排序、关联与报告

```mermaid
flowchart LR
    Inputs["公开证据 + 内部项目"] --> Rank["ProjectRanker<br/>确定性评分"]
    Rank --> Associate["ResourceAssociationBuilder<br/>资源、缺口与风险"]
    Associate --> Fallback["生成规则版内容"]
    Associate --> LLM["final_synthesis<br/>白名单材料综合"]
    LLM --> Validate["校验 LLM 输出"]
    Fallback --> Merge["补全并再次校验"]
    Validate --> Merge
    Merge --> Render["Jinja2 渲染"]
    Render --> Detailed["详细报告"]
    Render --> Brief["行动简报"]
```

规则版内容始终先生成。`final_synthesis` 不可用或输出校验失败时使用规则版内容；成功时也会与规则版合并并再次校验，最后保存 `COMPLETED`。取消任务会在各阶段检查点停止；未处理异常将任务置为 `FAILED`。

### 图 6：外部调用边界

#### 图 6.1：公开信息

```mermaid
flowchart LR
    Intake["IntakeEntityCandidateService"] --> Client["TavilyClient"]
    Research["ResearchToolExecutor"] --> Client
    Client --> API["api.tavily.com"]
    API --> Results["搜索结果与网页正文"]
    Results --> Identity["身份原文校验"]
    Results --> Evidence["研究证据分流"]
```

Intake Web 查询只用于未解决的身份补全；研究 Web 查询用于公开事实收集。LLM 不直接访问 Tavily，也不自行执行网络请求。

#### 图 6.2：内部资源

```mermaid
flowchart LR
    Identity["身份补全"] --> MCPClient["ProjectMcpClient"]
    Research["项目检索"] --> MCPClient
    MCPClient --> MCP["FastMCP Server"]
    MCP --> Find["find_entity_candidates"]
    MCP --> Search["search_projects"]
    MCP --> Detail["get_project_details"]
    MCP --> Portfolio["get_sales_portfolio"]
    Find --> Repo["ProjectRepository"]
    Search --> Repo
    Detail --> Repo
    Portfolio --> Repo
    Repo -->|"resource_reader 只读"| DB[("PostgreSQL 内部业务表")]
```

内部业务查询必须经过 MCP Server。LLM 只接收经过裁剪的上下文并输出结构化决策，不能直接访问应用数据库或内部业务表。

### 图 7：当前数据模型

#### 图 7.1：应用状态表

```mermaid
erDiagram
    intake_sessions {
        varchar id PK
        varchar status
        jsonb messages
        jsonb structured_context
        jsonb confirmation_request
        varchar research_task_id UK
    }

    intake_audio_jobs {
        varchar id PK
        varchar session_id
        varchar status
        text audio_path
        text transcript
        text corrected_transcript
    }

    research_tasks {
        varchar id PK
        varchar intake_session_id UK
        varchar status
        jsonb input_snapshot
        json confirmed_context
        json public_claims
        json internal_results
        json association_analysis
        text detailed_report_markdown
        text action_brief_markdown
    }

    llm_call_logs {
        varchar id PK
        varchar task_id
        varchar node_name
        varchar status
        integer latency_ms
    }

    execution_events {
        bigint id PK
        varchar scope_id
        varchar event_type
        varchar node_name
        varchar status
        jsonb payload
    }

    intake_sessions ||--o{ intake_audio_jobs : "session_id 逻辑关联"
    intake_sessions ||--o| research_tasks : "intake_session_id 逻辑关联"
    research_tasks ||--o{ llm_call_logs : "task_id 逻辑关联"
    research_tasks ||--o{ execution_events : "scope_id 可指向任务"
    intake_sessions ||--o{ execution_events : "scope_id 也可指向会话"
```

这些表由 `backend/app/models/database.py` 定义，并由 `init_database()` 在 FastAPI 启动时创建或补充迁移。图中的关系均为逻辑关联：ORM 当前没有为这些字段声明数据库外键。会话消息、实体、证据、项目匹配和报告主要保存在 JSON/Text 字段中。

#### 图 7.2：内部项目表

```mermaid
erDiagram
    customers {
        varchar customer_id PK
        varchar customer_name UK
        varchar industry
        varchar region_name
        varchar account_tier
    }

    customer_contacts {
        varchar contact_id PK
        varchar customer_id FK
        varchar contact_name
        varchar job_title
    }

    sales_managers {
        varchar manager_id PK
        varchar manager_name
        varchar region_name
    }

    sales_representatives {
        varchar sales_rep_id PK
        varchar manager_id FK
        varchar sales_rep_name
        varchar territory
    }

    internal_projects {
        varchar project_id PK
        varchar customer_id FK
        varchar customer_contact_id FK
        varchar sales_rep_id FK
        varchar project_name
        varchar status
        varchar project_stage
        vector project_embedding
    }

    project_status_history {
        bigint history_id PK
        varchar project_id FK
        varchar status
        varchar project_stage
        varchar health_status
        timestamptz changed_at
    }

    customers ||--o{ customer_contacts : "拥有联系人"
    customers ||--o{ internal_projects : "拥有项目"
    customer_contacts ||--o{ internal_projects : "关联项目"
    sales_managers ||--o{ sales_representatives : "管理"
    sales_representatives ||--o{ internal_projects : "负责项目"
    internal_projects ||--o{ project_status_history : "记录状态变化"
```

内部项目表和查询视图由 `seed/init.sql` 创建，`mcp_server/project_repository.py` 使用 `resource_reader` 只读账号查询。项目检索按人物/企业精确匹配、文本匹配和 HashingVectorizer 向量匹配组合返回结果。

建议但尚未实现的规范化表包括 User、ConversationMessage、CanonicalEntity、Evidence、ProjectMatch 和 Report；它们不属于当前物理模型。

### 图 8：Docker Compose 部署

#### 图 8.1：启动依赖

```mermaid
flowchart LR
    Postgres["postgres"] --> Seed["seed<br/>一次性初始化数据"]
    Seed --> MCP["mcp-server"]
    MCP --> Backend["backend"]
    Backend --> Worker["worker"]
    Backend --> Frontend["frontend"]
    ModelInit["model-init<br/>一次性下载 Whisper"] --> Worker
    Redis["redis"] --> Backend
    Redis --> Worker
```

#### 图 8.2：运行时连接与卷

```mermaid
flowchart TD
    Browser["Browser"] -->|":3000"| Frontend["Next.js"]
    Browser -->|"REST / SSE :8000"| Backend["FastAPI"]
    Backend --> Postgres[("PostgreSQL + pgvector")]
    Backend --> Redis["Redis"]
    Redis --> Worker["Celery --pool=solo"]
    Worker --> Postgres
    Worker --> MCP["FastMCP :8001"]
    MCP -->|"只读"| Postgres
    Worker --> Tavily["Tavily API"]
    Worker --> LLM["LLM Gateway"]
    Backend --> Audio[("audio_data")]
    Worker --> Audio
    Worker --> Models[("model_cache 只读")]
    Postgres --> Data[("postgres_data")]
```

当前 Compose 暴露 `3000`、`8000` 和 `8001`，使用 `postgres_data`、`audio_data`、`model_cache` 三个命名卷。生产网关、TLS、认证授权、托管 Redis/PostgreSQL、对象存储和日志指标追踪仍属于未来规划。
