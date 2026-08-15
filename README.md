# 资源推动 Agent Demo

资源调查 Demo：规则提取与确定性降级、`MiniMax-M3` Chat Completions + Pydantic 结构化理解、intake 阶段受控的 Tavily 关键人身份补全、MCP 内部实体与项目查询、关联分析和 Jinja2 报告。

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

## 架构图

以下以当前代码为准。已有 `2026-07-21` 验收文档中的“无生成式大模型”等描述已经过期；当前实现已包含 `StructuredLLM`、Intake Agent 和七个研究节点。

当前边界：未实现 Authentication、Nginx/API Gateway、独立 Report API，也没有独立的 User、Entity、Evidence、Report 数据表。

## 图1：系统总体架构图

用途说明：展示当前系统分层、核心运行组件及外部依赖；括号内为真实工程模块。

```mermaid
flowchart TB
    subgraph U["用户层"]
        User["业务用户"]
        Input["用户输入<br/>文字 / WebM 音频"]
        WebUI["Web 前端<br/>frontend/src/app/page.tsx"]
        ReportView["报告查看<br/>详细报告 / 行动简报"]
        User --> Input --> WebUI
        WebUI --> ReportView
    end

    subgraph A["应用层"]
        FastAPI["FastAPI Backend<br/>app.main:app"]
        IntakeRouter["API Router<br/>app.api.intake.router"]
        TaskRouter["API Router<br/>app.api.tasks.router"]
        Auth["Authentication<br/>当前未实现"]
        TaskManagement["Task Management<br/>TaskRepository / IntakeSessionRepository"]
        FastAPI --> IntakeRouter
        FastAPI --> TaskRouter
        IntakeRouter --> TaskManagement
        TaskRouter --> TaskManagement
        Auth -.->|"未来规划"| FastAPI
    end

    subgraph Q["任务层"]
        ResearchWorkflow["Research Workflow<br/>ResearchPipeline.run"]
        CeleryWorker["Celery Worker<br/>run_research_pipeline<br/>run_intake_audio_transcription"]
        Queue["Celery Queue<br/>broker / result backend"]
        Queue --> CeleryWorker --> ResearchWorkflow
    end

    subgraph G["Agent 层：逻辑角色，非独立进程"]
        IntakeAgent["信息采集 Agent<br/>IntakeAgent"]
        IdentityAgent["身份确认 Agent<br/>IntakeEntityCandidateService<br/>EntityResolver"]
        TurnAgent["行动决策 Agent<br/>AgentNodes.agent_turn"]
        EvidenceAgent["歧义证据核验 Agent<br/>AgentNodes.evidence_verify"]
        FinalAgent["最终综合 Agent<br/>AgentNodes.final_synthesis"]
    end

    subgraph T["工具层"]
        TavilyClient["Tavily Search<br/>TavilyClient"]
        MCPClient["MCP Client<br/>ProjectMcpClient"]
        MCPServer["MCP Server<br/>mcp_server.server"]
        DBTools["数据库查询工具<br/>TaskRepository / ProjectRepository"]
        MCPClient --> MCPServer --> DBTools
    end

    subgraph D["数据层"]
        PostgreSQL["PostgreSQL + pgvector<br/>业务状态与内部项目数据"]
        Redis["Redis 7<br/>Celery broker / result backend"]
        FileStore["Docker Volumes<br/>audio_data / model_cache"]
    end

    subgraph X["外部系统"]
        LLMGateway["LLM Gateway<br/>MiniMax-M3 / OpenAI-compatible API"]
        WebSearch["Web Search<br/>api.tavily.com"]
        EnterpriseDB["企业内部数据库<br/>Demo 中由 PostgreSQL 种子表模拟"]
    end

    WebUI -->|"REST JSON / multipart"| FastAPI
    TaskManagement --> PostgreSQL
    IntakeRouter --> IntakeAgent
    IntakeRouter --> IdentityAgent
    IntakeRouter --> Queue
    TaskRouter --> Queue
    Queue --> Redis

    ResearchWorkflow --> TurnAgent
    ResearchWorkflow --> EvidenceAgent
    ResearchWorkflow --> FinalAgent
    IntakeAgent --> LLMGateway
    TurnAgent --> LLMGateway
    EvidenceAgent --> LLMGateway
    FinalAgent --> LLMGateway

    IdentityAgent --> MCPClient
    IdentityAgent --> TavilyClient
    ResearchWorkflow --> TavilyClient
    ResearchWorkflow --> MCPClient
    TavilyClient --> WebSearch
    DBTools --> PostgreSQL
    PostgreSQL --- EnterpriseDB
    CeleryWorker --> FileStore
    ResearchWorkflow --> PostgreSQL
    TaskRouter -->|"TaskResponse 报告字段"| ReportView
```

模块说明：用户输入经 `page.tsx` 进入两个 FastAPI Router；Router 管理会话和任务，Celery 异步运行 `ResearchPipeline`；Agent 只产生结构化决策，工具负责实际检索；结果写入 PostgreSQL，并通过任务查询接口返回前端。

## 图2：业务流程图（用户视角）

用途说明：只描述用户能感知的业务过程，不体现代码模块。

```mermaid
flowchart TD
    Start(["开始"])
    Input["输入会面信息<br/>人物、企业、活动、关注方向"]
    Collect["信息采集<br/>补充缺失的会面背景"]
    Resolve["身份确认<br/>核对人物与企业标准身份"]
    NeedConfirm{"身份是否唯一且信息完整？"}
    UserConfirm["用户选择候选<br/>或手工补充身份"]
    Ready["用户确认开始分析"]
    CreateTask["创建研究任务"]
    PublicResearch["公开信息研究"]
    InternalSearch["查询内部项目与可用资源"]
    Association["分析公开信息与内部资源关联"]
    Report["生成详细报告与行动简报"]
    Display["展示分析进度和最终结果"]
    End(["结束"])

    Start --> Input --> Collect --> Resolve --> NeedConfirm
    NeedConfirm -->|"否"| UserConfirm --> Resolve
    NeedConfirm -->|"是"| Ready --> CreateTask
    CreateTask --> PublicResearch --> InternalSearch
    InternalSearch --> Association --> Report --> Display --> End
```

模块说明：输入为用户提供的会面信息；输出为可追溯的公开研究、内部项目匹配和行动建议。身份不完整时流程回到用户确认，用户明确点击开始分析后才创建研究任务。

## 图3：后端服务架构图

用途说明：展示 FastAPI 后端当前真实代码组织和调用方向。工程没有独立的 `service/task_service.py`、`workflow/` 或 `agents/` 目录，下图映射到实际类。

```mermaid
flowchart TB
    Main["FastAPI<br/>backend/app/main.py<br/>app.main:app"]

    subgraph API["api/"]
        IntakeAPI["app.api.intake.router<br/>chat / activity / audio / confirm / start-analysis"]
        TaskAPI["app.api.tasks.router<br/>text / audio / get / confirm / cancel"]
        ReportAPI["报告读取<br/>GET /api/v1/tasks/{task_id}<br/>当前无独立 Report Router"]
    end

    subgraph Services["services/"]
        IntakeService["IntakeAgent<br/>信息采集与就绪复核"]
        IdentityService["IntakeEntityCandidateService<br/>EntityResolver"]
        TaskService["TaskRepository<br/>IntakeSessionRepository"]
        ResearchService["RuleExtractor<br/>研究前置处理"]
        ReportService["ReportRenderer<br/>Jinja2 渲染"]
    end

    subgraph Workflow["tasks/：工作流编排"]
        CeleryApp["celery_app"]
        IntakeAudioTask["run_intake_audio_transcription"]
        PipelineTask["run_research_pipeline"]
        Pipeline["ResearchPipeline"]
        AgentLoop["AgentLoopRunner<br/>动作校验 / 循环保护"]
        AgentTools["AgentToolExecutor<br/>工具 → Observation"]
        Deterministic["ProjectRanker<br/>ResourceAssociationBuilder"]
        CeleryApp --> IntakeAudioTask
        CeleryApp --> PipelineTask --> Pipeline
        Pipeline --> AgentLoop --> AgentTools
        Pipeline --> Deterministic
    end

    subgraph Agents["Agent 模块"]
        StructuredLLM["StructuredLLM"]
        IntakeAgentNode["IntakeAgent"]
        AgentNodes["AgentNodes<br/>agent_turn / evidence_verify / final_synthesis"]
        IntakeAgentNode --> StructuredLLM
        AgentNodes --> StructuredLLM
    end

    subgraph Tools["工具适配器"]
        Whisper["LocalWhisperTranscriber"]
        Tavily["TavilyClient"]
        MCP["ProjectMcpClient"]
    end

    subgraph Models["Pydantic schemas"]
        IntakeSchemas["schemas/intake.py"]
        TaskSchemas["schemas/task.py"]
    end

    subgraph Database["database/"]
        Repositories["database.py<br/>SessionLocal / repositories"]
        ORM["models/database.py<br/>SQLAlchemy ORM models"]
        PG["PostgreSQL"]
        Repositories --> ORM --> PG
    end

    Main --> IntakeAPI
    Main --> TaskAPI
    TaskAPI --- ReportAPI

    IntakeAPI --> IntakeService
    IntakeAPI --> IdentityService
    IntakeAPI --> TaskService
    IntakeAPI --> IntakeAudioTask
    IntakeAPI --> PipelineTask
    TaskAPI --> TaskService
    TaskAPI --> PipelineTask

    IntakeService --> IntakeAgentNode
    IdentityService --> Tavily
    IdentityService --> MCP
    Pipeline --> ResearchService
    Pipeline --> AgentNodes
    Pipeline --> Whisper
    Pipeline --> Tavily
    Pipeline --> MCP
    Pipeline --> ReportService
    Pipeline --> TaskService

    IntakeAPI --> IntakeSchemas
    TaskAPI --> TaskSchemas
    Pipeline --> TaskSchemas
    TaskService --> Repositories
```

模块说明：API 输入是 Pydantic 请求模型，输出是 Intake/Task 响应模型；`ResearchPipeline` 统一进入 `AgentLoopRunner`，由模型决定动作意图，规则代码生成工具参数并整理结果；`ProjectRanker` 和 `ResourceAssociationBuilder` 确定性运行，`ReportRenderer` 只渲染内容；所有持久化经 Repository 和 ORM 完成。

## 图4：Agent Workflow 详细图

用途说明：展示 `ResearchPipeline.run` 的控制流，并明确代码、LLM 和用户确认边界。

```mermaid
flowchart TD
    Input["[代码] Input<br/>task_id"]
    Load["[代码] TaskRepository.get<br/>加载 ResearchTask"]
    HasContext{"[代码] 是否已有 confirmed_context？"}

    Restore["[代码] 从 Intake snapshot<br/>恢复 ConfirmedContext"]
    Extract["[代码] RuleExtractor.extract"]
    Understand["[规则] fallback_understanding<br/>生成兼容 IntentUnderstanding"]
    Resolve["[代码] EntityResolver.resolve"]
    ConfirmNeeded{"[代码] 是否需要身份确认？"}
    UserConfirm["[用户] 选择候选或手工填写"]
    PersistConfirm["[代码] 保存 NEEDS_CONFIRMATION<br/>等待确认后重新调度"]

    Context["[代码] AgentContextBuilder<br/>裁剪 Context / Observation"]
    Turn["[LLM] agent_turn<br/>只选择允许的 AgentAction"]
    Validate{"[代码] AgentActionValidator<br/>动作是否合法？"}
    Fallback["[规则] 确定性 fallback action"]
    Action{"AgentAction"}

    WebPlan["[规则] fallback_web_plan<br/>生成 Tavily 查询"]
    WebTool["[工具] TavilyClient<br/>search → extract"]
    EvidenceRoute["[规则] 证据候选分流<br/>accepted / rejected / ambiguous"]
    WebVerify["[LLM] evidence_verify<br/>仅核验 ambiguous"]
    WebObservation["[代码] 标准化 / 去重 / 裁剪<br/>公开 Observation"]

    ProjectPlan["[规则] fallback_project_query<br/>生成 MCP 参数"]
    ProjectTool["[工具] ProjectMcpClient.search_projects"]
    ProjectObservation["[代码] 标准化 / 去重 / 裁剪<br/>项目 Observation"]

    Rerank["[规则] ProjectRanker<br/>确定性评分与 reason_codes"]
    Match["[规则] ResourceAssociationBuilder<br/>整理关系、缺口与风险"]
    Content["[LLM] final_synthesis<br/>只综合白名单材料"]
    Render["[代码] ReportRenderer<br/>Jinja2 渲染"]
    Complete["[代码] 保存 COMPLETED<br/>详细报告 / 行动简报"]

    Input --> Load --> HasContext
    HasContext -->|"是"| Restore --> Context
    HasContext -->|"否"| Extract --> Understand --> Resolve --> ConfirmNeeded
    ConfirmNeeded -->|"是"| PersistConfirm --> UserConfirm --> Resolve
    ConfirmNeeded -->|"否"| Context

    Context --> Turn --> Validate
    Validate -->|"否"| Fallback --> Action
    Validate -->|"是"| Action
    Action -->|"SEARCH_PUBLIC"| WebPlan --> WebTool --> EvidenceRoute
    EvidenceRoute -->|"ambiguous"| WebVerify --> WebObservation
    EvidenceRoute -->|"规则已处理"| WebObservation
    Action -->|"SEARCH_INTERNAL"| ProjectPlan --> ProjectTool --> ProjectObservation
    WebObservation --> Context
    ProjectObservation --> Context
    Action -->|"SYNTHESIZE / FINISH"| Rerank --> Match --> Content --> Render --> Complete
    Action -->|"ASK_USER"| PersistConfirm
```

模块说明：业务 LLM 只保留 `agent_turn`、`evidence_verify` 和 `final_synthesis`。`agent_turn` 决定下一步意图但不生成工具参数；Tavily/MCP 参数、项目排序和资源关联均由确定性代码完成。工具原始结果经过标准化、去重和裁剪后形成 `Observation`，再进入下一轮 Context。循环受最大轮数、最大工具调用数和重复动作保护；身份无法确定时任务进入 `NEEDS_CONFIRMATION`。

## 图5：数据库 ER 关系图

用途说明：同时展示当前物理模型和建议的规范化模型。名称带 `_PROPOSED` 的实体当前不存在。

```mermaid
erDiagram
    IntakeSession["IntakeSession（信息采集会话）"] {
        string id PK "（信息采集会话ID）"
        string status "（状态）"
        jsonb messages "（消息记录）"
        jsonb structured_context "（结构化上下文）"
        jsonb confirmation_request "（确认请求）"
        string research_task_id UK "（研究任务ID）"
    }

    ResearchTask["ResearchTask（研究任务）"] {
        string id PK "（研究任务ID）"
        string intake_session_id UK "（信息采集会话ID）"
        string status "（状态）"
        json input_snapshot "（输入快照）"
        json confirmed_context "（已确认上下文）"
        json public_claims "（公开信息事实）"
        json internal_results "（内部项目查询结果）"
        json association_analysis "（关联分析）"
        text detailed_report_markdown "（详细报告Markdown）"
        text action_brief_markdown "（行动说明Markdown）"
    }

    IntakeAudioJob["IntakeAudioJob（信息采集音频任务）"] {
        string id PK "（音频任务ID）"
        string session_id "（信息采集会话ID）"
        string status "（状态）"
        text audio_path "（音频文件路径）"
        text transcript "（音频转写文本）"
    }

    LlmCallLog["LlmCallLog（大模型调用日志）"] {
        string id PK "（调用日志ID）"
        string task_id "（研究任务ID）"
        string node_name "（节点名称）"
        string status "（调用状态）"
        int latency_ms "（调用耗时，毫秒）"
    }

    Customer["Customer（客户）"] {
        string customer_id PK "（客户ID）"
        string customer_name "（客户名称）"
    }

    CustomerContact["CustomerContact（客户联系人）"] {
        string contact_id PK "（联系人ID）"
        string customer_id FK "（客户ID）"
        string contact_name "（联系人姓名）"
        string job_title "（职位）"
    }

    InternalProject["InternalProject（内部项目）"] {
        string project_id PK "（项目ID）"
        string customer_id FK "（客户ID）"
        string customer_contact_id FK "（客户联系人ID）"
        string sales_rep_id FK "（销售代表ID）"
        string project_name "（项目名称）"
        vector project_embedding "（项目向量）"
    }

    SalesManager["SalesManager（销售经理）"] {
        string manager_id PK "（销售经理ID）"
    }

    SalesRepresentative["SalesRepresentative（销售代表）"] {
        string sales_rep_id PK "（销售代表ID）"
        string manager_id FK "（销售经理ID）"
    }

    ProjectStatusHistory["ProjectStatusHistory（项目状态历史）"] {
        int history_id PK "（历史记录ID）"
        string project_id FK "（项目ID）"
    }

    UserAccount_PROPOSED["UserAccount_PROPOSED（用户账户，建议新增）"] {
        string id PK "（用户账户ID）"
        string login_name UK "（登录名称）"
        string display_name "（显示名称）"
    }

    ConversationMessage_PROPOSED["ConversationMessage_PROPOSED（会话消息，建议新增）"] {
        string id PK "（会话消息ID）"
        string intake_session_id FK "（信息采集会话ID）"
        string role "（消息角色）"
        text content "（消息内容）"
    }

    CanonicalEntity_PROPOSED["CanonicalEntity_PROPOSED（标准实体，建议新增）"] {
        string id PK "（标准实体ID）"
        string entity_type "（实体类型）"
        string canonical_name "（标准名称）"
    }

    ResearchEntity_PROPOSED["ResearchEntity_PROPOSED（研究实体关联，建议新增）"] {
        string task_id FK "（研究任务ID）"
        string entity_id FK "（标准实体ID）"
        string confirmed_by "（确认方式）"
    }

    Evidence_PROPOSED["Evidence_PROPOSED（证据，建议新增）"] {
        string id PK "（证据ID）"
        string task_id FK "（研究任务ID）"
        string source_type "（来源类型）"
        text claim "（事实陈述）"
        text source_url "（来源链接）"
    }

    ProjectMatch_PROPOSED["ProjectMatch_PROPOSED（项目匹配，建议新增）"] {
        string task_id FK "（研究任务ID）"
        string project_id FK "（项目ID）"
        float relevance_score "（相关性评分）"
    }

    Report_PROPOSED["Report_PROPOSED（报告，建议新增）"] {
        string id PK "（报告ID）"
        string task_id FK "（研究任务ID）"
        string report_type "（报告类型）"
        text markdown "（Markdown报告内容）"
    }

    IntakeSession ||--o| ResearchTask : "逻辑一对一"
    IntakeSession ||--o{ IntakeAudioJob : "session_id 逻辑关联"
    ResearchTask ||--o{ LlmCallLog : "task_id 逻辑关联"

    Customer ||--o{ CustomerContact : "has（拥有）"
    Customer ||--o{ InternalProject : "owns（拥有）"
    CustomerContact ||--o{ InternalProject : "contacts（负责联系）"
    SalesManager ||--o{ SalesRepresentative : "manages（管理）"
    SalesRepresentative ||--o{ InternalProject : "owns（负责）"
    InternalProject ||--o{ ProjectStatusHistory : "records（记录）"

    UserAccount_PROPOSED ||--o{ IntakeSession : "owns（拥有）"
    IntakeSession ||--o{ ConversationMessage_PROPOSED : "contains（包含）"
    ResearchTask ||--o{ ResearchEntity_PROPOSED : "identifies（识别）"
    CanonicalEntity_PROPOSED ||--o{ ResearchEntity_PROPOSED : "referenced_by（被引用）"
    ResearchTask ||--o{ Evidence_PROPOSED : "collects（收集）"
    ResearchTask ||--o{ ProjectMatch_PROPOSED : "matches（匹配）"
    InternalProject ||--o{ ProjectMatch_PROPOSED : "referenced_by（被引用）"
    ResearchTask ||--o{ Report_PROPOSED : "generates（生成）"
```

模块说明：当前会话消息、实体、证据、项目匹配和报告主要内嵌在 JSON/Text 字段中；应用表之间多数只有逻辑 ID，没有数据库外键。建议新增用户、消息、标准实体、证据、项目匹配和报告表，以支持权限、审计、检索和版本管理。

## 图6：外部工具调用关系图

用途说明：展示 Agent、工具适配器和外部资源的访问边界，强调 LLM 不直接访问数据库或外部工具。

```mermaid
flowchart TB
    subgraph Decision["决策层"]
        IntakeAgent["IntakeAgent"]
        AgentNodes["AgentNodes<br/>agent_turn / evidence_verify / final_synthesis"]
        LLM["StructuredLLM<br/>输入：受控上下文<br/>输出：Pydantic 结构化结果"]
        IntakeAgent --> LLM
        AgentNodes --> LLM
    end

    subgraph Control["代码控制层"]
        IntakeCandidates["IntakeEntityCandidateService"]
        Pipeline["ResearchPipeline"]
        AgentLoop["AgentLoopRunner<br/>Context → Action → Observation"]
        ToolExecutor["AgentToolExecutor<br/>规则生成工具参数"]
        Validation["证据分流 / ProjectRanker<br/>ResourceAssociationBuilder"]
        IntakeCandidates --> Validation
        Pipeline --> AgentLoop --> ToolExecutor
        Pipeline --> Validation
    end

    subgraph Tools["Tool Layer"]
        TavilyClient["TavilyClient<br/>search / extract"]
        MCPClient["ProjectMcpClient<br/>MCP tool call"]
        Repositories["TaskRepository<br/>IntakeSessionRepository"]
        ProjectRepository["ProjectRepository<br/>只读内部查询"]
    end

    subgraph Resources["External Resource / Data"]
        TavilyAPI["api.tavily.com<br/>公开网页搜索"]
        MCPServer["FastMCP Server<br/>find_entity_candidates<br/>search_projects"]
        AppDB["PostgreSQL<br/>研究任务与会话"]
        EnterpriseDB["PostgreSQL 内部业务表<br/>resource_reader 只读账号"]
    end

    IntakeAgent --> IntakeCandidates
    AgentNodes --> AgentLoop
    IntakeCandidates --> TavilyClient
    IntakeCandidates --> MCPClient
    ToolExecutor --> TavilyClient
    ToolExecutor --> MCPClient
    Pipeline --> Repositories

    TavilyClient --> TavilyAPI
    MCPClient --> MCPServer --> ProjectRepository --> EnterpriseDB
    Repositories --> AppDB

    LLM -.->|"禁止：不能直接访问"| AppDB
    LLM -.->|"禁止：不能直接访问"| EnterpriseDB
    LLM -.->|"禁止：不能直接调用"| TavilyAPI
```

模块说明：`agent_turn` 输入经过裁剪的 `AgentContext`，只输出动作意图；`evidence_verify` 只接收规则无法判断的证据候选；`final_synthesis` 只接收已确认材料。LLM 不生成 Tavily 查询、MCP 参数或项目排序，真正的网络和数据库访问由 Python 工具类完成；内部业务查询必须经过 MCP Server 的 `ProjectRepository` 和只读账号。

## 图7：部署架构图

用途说明：区分当前 Docker Compose 开发部署与尚未实现的生产部署。

```mermaid
flowchart TB
    subgraph DEV["当前开发环境"]
        Developer["Developer"]
        Git["Git Repository<br/>resource-agent-demo"]
        Compose["Docker Compose"]

        BrowserDev["Browser"]
        FrontendDev["frontend<br/>Next.js :3000"]
        BackendDev["backend<br/>Uvicorn/FastAPI :8000"]
        WorkerDev["worker<br/>Celery --pool=solo"]
        MCPDev["mcp-server<br/>FastMCP :8001"]
        RedisDev["redis:7-alpine"]
        PGDev["pgvector/pgvector:pg16"]
        VolumesDev["audio_data / model_cache / postgres_data"]

        Developer --> Git --> Compose
        Compose --> FrontendDev
        Compose --> BackendDev
        Compose --> WorkerDev
        Compose --> MCPDev
        Compose --> RedisDev
        Compose --> PGDev

        BrowserDev -->|"HTTP :3000"| FrontendDev
        FrontendDev -->|"REST :8000"| BackendDev
        BackendDev -->|"enqueue"| RedisDev
        WorkerDev -->|"consume / result"| RedisDev
        BackendDev --> PGDev
        WorkerDev --> PGDev
        WorkerDev --> MCPDev --> PGDev
        WorkerDev --> VolumesDev
        PGDev --> VolumesDev
    end

    subgraph PROD["生产环境：未来规划，当前仓库未实现"]
        BrowserProd["Browser"]
        Gateway["Nginx / API Gateway<br/>TLS、路由、限流"]
        AuthProd["Authentication / Authorization"]
        FrontendProd["Next.js Runtime 或静态托管"]
        FastAPIProd["FastAPI Replicas"]
        RedisProd["Managed Redis<br/>broker / result backend"]
        WorkerProd["Celery Worker Pool"]
        MCPProd["Private MCP Server"]
        PGProd["Managed PostgreSQL / pgvector"]
        ObjectStore["Object Storage<br/>音频与导出报告"]
        Observability["Logs / Metrics / Tracing"]

        BrowserProd --> Gateway
        Gateway --> FrontendProd
        Gateway --> AuthProd --> FastAPIProd
        FastAPIProd -->|"enqueue"| RedisProd
        RedisProd -->|"consume"| WorkerProd
        FastAPIProd --> PGProd
        WorkerProd --> PGProd
        WorkerProd --> MCPProd --> PGProd
        WorkerProd --> ObjectStore
        FastAPIProd --> Observability
        WorkerProd --> Observability
        MCPProd --> Observability
    end
```

模块说明：当前环境通过 Docker Compose 暴露 `3000/8000/8001`，没有反向代理、TLS、认证或集中可观测性；生产建议由网关统一接入，FastAPI 只负责入队，Celery 从 Redis 消费任务，PostgreSQL 和对象存储负责持久化。


## 图8 resource_agent ER 关系图
```mermaid
erDiagram
    customers {
        varchar customer_id PK "（客户ID）"
        varchar customer_name UK "（客户名称）"
        varchar industry "（所属行业）"
        varchar region_name "（区域名称）"
        varchar account_tier "（客户等级）"
        timestamptz created_at "（创建时间）"
    }

    customer_contacts {
        varchar contact_id PK "（联系人ID）"
        varchar customer_id FK "（客户ID）"
        varchar contact_name "（联系人姓名）"
        varchar job_title "（职位）"
        varchar phone "（电话）"
        varchar email "（邮箱）"
        boolean is_primary "（是否主要联系人）"
        timestamptz created_at "（创建时间）"
    }

    sales_managers {
        varchar manager_id PK "（销售经理ID）"
        varchar manager_name "（销售经理姓名）"
        varchar region_name "（负责区域）"
        varchar phone "（电话）"
        varchar email "（邮箱）"
        boolean active "（是否在职）"
    }

    sales_representatives {
        varchar sales_rep_id PK "（销售人员ID）"
        varchar manager_id FK "（销售经理ID）"
        varchar sales_rep_name "（销售人员姓名）"
        varchar territory "（负责区域）"
        boolean active "（是否在职）"
        date hired_on "（入职日期）"
    }

    internal_projects {
        varchar project_id PK "（项目ID）"
        varchar project_name "（项目名称）"
        varchar customer_id FK "（客户ID）"
        varchar customer_contact_id FK "（客户联系人ID）"
        varchar sales_rep_id FK "（销售负责人ID）"
        varchar customer_name "（客户名称快照）"
        varchar contact_name "（联系人姓名快照）"
        varchar status "（项目状态）"
        varchar project_stage "（项目阶段）"
        varchar health_status "（项目健康状态）"
        varchar priority "（优先级）"
        numeric contract_value "（合同金额）"
        smallint win_probability "（赢单概率）"
        date start_date "（开始日期）"
        date end_date "（结束日期）"
        date last_activity_date "（最近活动日期）"
        date next_followup_date "（下次跟进日期）"
        text description "（项目描述）"
        vector project_embedding "（项目文本向量）"
    }

    project_status_history {
        bigint history_id PK "（状态历史ID）"
        varchar project_id FK "（项目ID）"
        varchar status "（项目状态）"
        varchar project_stage "（项目阶段）"
        varchar health_status "（健康状态）"
        timestamptz changed_at "（变更时间）"
        varchar changed_by "（变更人）"
        text change_note "（变更说明）"
    }

    entity_aliases {
        varchar candidate_id PK "（候选实体ID）"
        varchar entity_type "（实体类型）"
        varchar canonical_name "（标准名称）"
        varchar alias "（别名）"
        varchar organization_name "（所属组织）"
        varchar title "（职位）"
        varchar region "（区域）"
    }

    research_tasks {
        varchar id PK "（研究任务ID）"
        varchar intake_session_id UK "（信息采集会话ID）"
        varchar status "（任务状态）"
        varchar input_type "（输入类型）"
        text input_text "（输入文本）"
        jsonb input_snapshot "（输入快照）"
        jsonb confirmed_context "（已确认上下文）"
        jsonb public_claims "（公开信息声明）"
        jsonb internal_results "（内部项目结果）"
        jsonb ranked_internal_results "（内部项目排序结果）"
        jsonb association_analysis "（资源关联分析）"
        jsonb generated_report_content "（结构化报告内容）"
        text detailed_report_markdown "（详细报告）"
        text action_brief_markdown "（行动简报）"
        jsonb degraded_nodes "（降级节点）"
        timestamptz created_at "（创建时间）"
        timestamptz updated_at "（更新时间）"
    }

    llm_call_logs {
        varchar id PK "（调用日志ID）"
        varchar task_id "（研究任务ID）"
        varchar node_name "（Agent节点名称）"
        varchar model "（模型名称）"
        varchar status "（调用状态）"
        varchar response_id "（模型响应ID）"
        varchar prompt_version "（提示词版本）"
        integer latency_ms "（调用耗时毫秒）"
        integer input_tokens "（输入Token数）"
        integer output_tokens "（输出Token数）"
        varchar error_type "（错误类型）"
        text error_message "（错误信息）"
        timestamptz created_at "（创建时间）"
    }

    customers ||--o{ customer_contacts : "拥有联系人"
    customers ||--o{ internal_projects : "拥有项目"
    customer_contacts ||--o{ internal_projects : "关联项目"
    sales_managers ||--o{ sales_representatives : "管理"
    sales_representatives ||--o{ internal_projects : "负责项目"
    internal_projects ||--o{ project_status_history : "记录状态变化"
    research_tasks ||--o{ llm_call_logs : "逻辑关联，无数据库外键"
```



注意：entity_aliases 当前没有外键；research_tasks 与 llm_call_logs 通过 task_id 逻辑关联，但数据库未建立外键。当前运行库中也没有代码模型里定义的 intake_sessions 和 intake_audio_jobs 表。
