# 资源推动 Agent 使用手册

## 1. 产品用途

资源推动 Agent 用于在会面或项目推进前收集人物、企业和议题信息，检索公开资料及内部项目，最后生成详细报告和行动说明。

当前产品只有一个固定结构的 Default Agent。信息采集、身份确认、公开检索、内部项目检索和报告生成的流程由代码定义；管理员可以调整各节点的 Prompt 和模型，但不能在后台改变流程拓扑、工具权限或运行逻辑。

## 2. 访问与账号

启动系统后访问：

- 业务页面：`http://localhost:3000`
- 管理后台：`http://localhost:3000/admin`
- 后端 API 文档：`http://localhost:8000/docs`

登录页支持使用账号或邮箱登录。当前本地管理员账号由部署人员维护；首次登录后应避免在共享环境中继续使用弱密码。

角色权限：

- `ADMIN`：使用业务功能、查看自己的历史会话、管理并测试 Draft 配置；
- `MEMBER`：使用业务功能和自己的历史会话，运行稳定的 Published 配置；
- `SYSTEM`：关闭认证时使用的系统身份。

页面右上角的齿轮按钮只对 `ADMIN` 或 `SYSTEM` 显示，并且需要启用 `AGENT_ADMIN_ENABLED`。

## 3. 完成一次资源调查

### 3.1 新建调查

1. 登录后进入业务页面。
2. 点击右上角“重新开始”，清除当前页面状态并创建一轮新调查。
3. 在输入框中描述会面对象、企业和准备讨论或推动的事项，例如：

```text
今晚和中建二局的刘希川一起吃饭，想了解可以推动的项目。
```

4. 信息不足时，根据 Agent 的追问补充姓名、企业、职务或讨论目标。

尽量提供准确全名和企业全称。相同姓名或简称可能触发候选身份确认。

### 3.2 确认身份与信息

Agent 会先查询内部候选，必要时再通过公开信息核验身份。出现候选列表时：

1. 选择正确人物；或
2. 手工补充正确姓名、企业和职务；
3. 检查最终确认摘要；
4. 确认无误后进入“信息已确认”。

最终确认前仍可继续补充信息。不要在人物或企业不确定时直接开始分析，否则检索结果可能偏离目标。

### 3.3 开始分析

点击“开始分析”后，系统按固定流程执行：

```text
公开信息检索
  → 必要时核验证据歧义
  → 内部项目检索
  → 项目排序与资源关联
  → 综合分析
  → 生成报告
```

页面会持续展示阶段进度。任务执行期间可以取消；单个外部服务不可用时，允许降级的节点会记录事件并继续运行。

### 3.4 查看报告与继续问答

任务完成后切换到“分析报告”，可查看：

- 详细报告：公开证据、内部项目、资源关联、风险和建议；
- 行动说明：适合会前快速阅读的重点动作。

报告生成后可以在当前任务中继续提问。回答仅基于当前任务保存的上下文和分析结果，不会改变原报告或重新启动调查。

## 4. 历史会话

点击页面右上角的历史图标打开“我的调查”。选择一条记录可恢复对应 Intake 会话或研究任务。

每个用户只能查看自己的会话。恢复历史任务时继续使用该任务启动时冻结的 Snapshot，不会改用管理员后来调整的新配置。

“重新开始”只清空当前浏览器页面中的任务指针并开始新调查，不会删除数据库中的历史记录。

## 5. 语音输入

业务页面支持录制或上传音频。流程为：

```text
上传音频 → 后台转写 → 用户校对 → 确认文本 → 进入信息采集
```

转写文本必须由用户确认后才能作为调查输入。转写失败时可在页面重试；仍失败时检查 Worker 状态和 Whisper 模型卷。

## 6. 管理后台概念

管理后台只管理固定 Default Agent，核心对象如下：

```text
ModelConnection：Provider、Base URL 和 API Key Secret
ModelProfile：Model ID、API Mode 和模型参数
Prompt Working Copy：Draft 中可反复保存和测试的提示词工作稿
Published Version：稳定、不可变的配置检查点
AgentRun Snapshot：每个任务启动时冻结的真实运行配置
```

配置生效规则：

```text
ADMIN/SYSTEM 新任务：有 Draft 时使用 Draft，否则使用 Published
MEMBER 新任务：使用 Published
已有任务：始终使用创建时的 AgentRun Snapshot
```

因此管理员修改 Draft 后无需 Publish 就能用一个全新任务测试；Publish 用于保存稳定版本，而不是普通保存按钮。

## 7. 管理模型

### 7.1 导入模型配置

“模型”页面支持粘贴 TOML、JSON 或 ENV，并自动转换为现有 `ModelConnection + ModelProfile`。

TOML 示例：

```toml
model_provider = "OpenAI"
model = "gpt-5.5"
model_reasoning_effort = "xhigh"
disable_response_storage = true

[model_providers.OpenAI]
name = "OpenAI"
base_url = "https://vftsub.vf-tech.cn"
wire_api = "responses"
requires_openai_auth = true
```

操作步骤：

1. 点击“导入配置”；
2. 粘贴配置并点击“识别配置”；
3. 检查 Provider、Base URL、API Mode、Model ID 和参数；
4. 单独填写 API Key；
5. 确认名称和 slug 后保存。

API Key 只写入加密 Secret Store，不会在页面回显。粘贴内容中的密钥值不会被导入。`network_access`、`features` 等客户端字段会被列为已忽略，不会进入模型运行参数。

支持的通用参数包括：

- `reasoning_effort`
- `max_output_tokens`
- `timeout_seconds`
- `max_retries`
- `temperature`
- `top_p`
- `store=false`

当前协议 Adapter 支持 `openai` 和 `openai_compatible`。同协议的新模型不需要开发模型专用 Adapter；只有接入完全不同的原生协议才需要增加协议 Adapter。

### 7.2 测试连接和轮换密钥

连接行的试管图标用于测试连接。连接失败时依次检查：

1. Base URL 是否正确，是否需要 `/v1`；
2. API Key 是否有效；
3. 模型域名是否在可信列表中；
4. 代理是否实现模型列表接口。

部分代理能正常调用模型但没有实现 Models API，此时测试连接可能返回 404。轮换密钥只更新 Secret 引用指向的当前密钥，不会把密钥明文写入历史 Snapshot。

### 7.3 给节点切换模型

1. 确认当前存在 Draft；
2. 打开“节点配置”；
3. 在目标节点选择一个可选模型；
4. 新建 ADMIN 调查并验证效果；
5. 已有任务保持原 Snapshot；
6. 调试稳定后再发布。

节点来回切换只更新 Draft 绑定，不会创建新的 ModelProfileRevision。当前不提供物理删除模型功能；下线前应先解除节点绑定，历史版本引用的数据必须保留。

## 8. 管理 Prompt

1. 打开“Prompt”页面并选择固定节点；
2. 查看当前工作稿、基线 Revision 和稳定历史 Revision；
3. 点击编辑并选择“保存工作稿”；
4. 新建 ADMIN 调查测试最新内容；
5. 需要时选择历史 Revision 作为工作稿基线；
6. “丢弃工作稿”可恢复到当前基线内容。

连续保存工作稿不会增加 PromptRevision 数量，也不会覆盖稳定历史。只有发布时，真正变化的工作稿才冻结为新的不可变 Revision。

保存和运行都会校验节点归属、Placeholder、Skill 及代码拥有的输出契约。Prompt 不能借由文本修改固定 Node Registry 或结构化输出边界。

## 9. 版本、Diff 与发布

发布前在“版本”页面查看 Draft 与当前 Published 的差异，重点核对：

- 各节点 Prompt 内容和 hash；
- 各节点 Model Profile；
- 固定 Logical Tool Binding；
- 静态校验结果。

Publish 会把通过校验的 Draft 冻结为 Published Version，并保存配置 hash 和发布备注。发布后不会自动创建下一份 Draft。

需要恢复历史稳定状态时，将历史 Published 配置复制到当前 Draft，先创建新任务验证，再重新发布；原历史版本保持不变。

## 10. 启动与部署配置

真实运行配置写在项目根目录 `.env`，`.env.example` 只提供字段示例，不能替代 `.env`，也不应包含真实密钥。

常用配置：

```dotenv
AUTH_ENABLED=true
AUTH_ALLOW_REGISTRATION=true
AGENT_ADMIN_ENABLED=true
AGENT_SECRET_KEY=<固定的 Fernet 密钥>
AGENT_TRUSTED_MODEL_HOSTS=model.example.com
```

生成 `AGENT_SECRET_KEY`：

```powershell
docker compose exec -T backend python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

写入 `.env` 后重新创建相关服务：

```powershell
docker compose up -d --force-recreate backend worker
```

`AGENT_SECRET_KEY` 用于加密数据库中的 API Key，写入 Secret 后不要随意更换，否则已有密钥无法解密。只应将确认可信的内网或 Fake IP 域名加入 `AGENT_TRUSTED_MODEL_HOSTS`，不要无条件开放所有内网地址。

完整启动：

```powershell
docker compose up -d --build
docker compose ps
```

## 11. 常见问题

### 页面提示 `Failed to fetch`

检查 Backend 是否健康、浏览器访问的 API 地址是否正确，以及前后端端口是否可达。

### 提示 `AGENT_SECRET_KEY is required`

按照第 10 节生成固定 Fernet 密钥，写入 `.env` 后重新创建 Backend 和 Worker。

### 提示模型地址不是 global address

域名解析到了内网地址或代理 Fake IP。确认域名可信后加入 `AGENT_TRUSTED_MODEL_HOSTS`，再重新创建服务。

### 新任务没有使用刚修改的 Draft

确认当前用户角色为 `ADMIN/SYSTEM`，并且点击“重新开始”创建了全新 Intake。已有 Intake 或 AgentRun 必须继续使用原 Snapshot。

### 修改 Prompt 后无法发布

先查看发布预检错误。常见原因包括非法 Placeholder、缺少 Skill、输出契约不匹配、模型绑定不完整、Tool Binding 不完整或配置 hash 不一致。

### 历史 Revision 数量怎么看

Prompt 页面选择节点后，稳定历史列表中的条目数就是该 PromptDefinition 的 Revision 数量。反复点击“保存工作稿”后该数量应保持不变；发布真正变化的工作稿后才会增加。

## 12. 测试与排障命令

完整后端测试：

```powershell
docker compose run --rm --no-deps backend pytest -q
```

前端生产构建：

```powershell
cd frontend
npm run build
```

检查服务与日志：

```powershell
docker compose ps
docker compose logs --tail=100 backend worker frontend
```

提交前检查：

```powershell
git diff --check
git status --short --branch
```
