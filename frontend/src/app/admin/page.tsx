"use client";

import {
  ArrowLeft,
  Bot,
  Check,
  ChevronRight,
  Database,
  FileText,
  Gauge,
  LoaderCircle,
  Network,
  Plus,
  Rocket,
  Save,
  Server,
  Settings,
  SlidersHorizontal,
  TestTube2,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import styles from "./admin.module.css";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const apiFetch = (path: string, init?: RequestInit) =>
  fetch(`${API_BASE}${path}`, { ...init, credentials: "include" });

type User = { user_id?: string; display_name?: string; role?: string; agent_admin_enabled: boolean };
type Version = { id: string; agent_definition_id: string; version: number; status: string; config_hash: string };
type NodeBinding = { node_key: string; output_schema: string; conditional: boolean; allows_tools: boolean; model_profile_revision_id?: string; model_id: string; provider: string; prompt_revision_id?: string; prompt_version?: number; prompt_source?: string; allowed_tools: string[] };
type ToolBinding = { logical_tool_key: string; tool_mapping_revision_id: string; remote_tool_name: string; adapter_key: string; allowed_nodes: string[] };
type VersionDetail = Version & { config_schema_version: number; loop: Record<string, number | boolean>; output: { formats: string[]; evidence_validation_required: boolean; templates?: { name: string; path: string }[] }; nodes: NodeBinding[]; tools: ToolBinding[] };
type AgentSummary = { id: string; name: string; slug: string; status: string; published_version: Version; draft_version?: Version };
type AgentDetail = { id: string; name: string; slug: string; status: string; published_version: VersionDetail; draft_version?: VersionDetail };
type ModelConnection = { id: string; name: string; slug: string; active_revision_id: string; revision_version: number; provider: string; base_url: string; secret_ref: string };
type ModelProfile = { id: string; name: string; slug: string; active_revision_id: string; revision_version: number; connection_revision_id: string; model_id: string; api_mode: string; parameters: Record<string, unknown> };
type PromptRevision = { id: string; version: number; content: string; validation_report: Record<string, unknown>; status: string };
type PromptDefinition = { id: string; name: string; slug: string; node_key: string; active_revision: PromptRevision };
type McpServer = { id: string; name: string; slug: string; active_revision_id: string; revision_version: number; url: string; authentication_type: string; secret_ref?: string; timeout_seconds: number };
type ToolMapping = { id: string; name: string; logical_tool_key: string; active_revision_id: string; revision_version: number; mcp_server_revision_id: string; remote_tool_name: string; adapter_key: string; input_mapping: Record<string, unknown>; output_mapping: Record<string, unknown>; timeout_seconds: number };
type DiscoveredTool = { name: string; description?: string; input_schema: Record<string, unknown> };
type Tab = "overview" | "nodes" | "models" | "prompts" | "mcp" | "runtime";

const NODE_LABELS: Record<string, string> = {
  intake_chat: "Intake 对话",
  intake_agent: "Intake Agent",
  intake_identity_initialize: "身份初始化",
  intake_identity_update: "身份更新",
  intake_followup: "追问生成",
  intake_identity_normalize: "身份标准化",
  intake_readiness: "就绪判断",
  intake_final_confirmation: "最终确认",
  evidence_verify: "公开证据核验",
  final_synthesis: "最终报告生成",
  analysis_chat: "报告对话",
};

const TABS: { key: Tab; label: string; icon: typeof Settings }[] = [
  { key: "overview", label: "概览", icon: Gauge },
  { key: "nodes", label: "节点配置", icon: Network },
  { key: "models", label: "模型", icon: SlidersHorizontal },
  { key: "prompts", label: "Prompt", icon: FileText },
  { key: "mcp", label: "MCP / Tools", icon: Server },
  { key: "runtime", label: "Loop / 输出", icon: Settings },
];

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await apiFetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? `请求失败 (${response.status})`);
  return payload as T;
}

export default function AdminPage() {
  const [user, setUser] = useState<User | null>(null);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [agent, setAgent] = useState<AgentDetail | null>(null);
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [prompts, setPrompts] = useState<PromptDefinition[]>([]);
  const [servers, setServers] = useState<McpServer[]>([]);
  const [mappings, setMappings] = useState<ToolMapping[]>([]);
  const [tab, setTab] = useState<Tab>("overview");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadAgent = useCallback(async (agentId: string) => {
    setAgent(await jsonRequest<AgentDetail>(`/api/v1/admin/agents/${agentId}`));
  }, []);

  const loadResources = useCallback(async () => {
    const [nextConnections, nextProfiles, nextPrompts, nextServers, nextMappings] = await Promise.all([
      jsonRequest<ModelConnection[]>("/api/v1/admin/model-connections"),
      jsonRequest<ModelProfile[]>("/api/v1/admin/model-profiles"),
      jsonRequest<PromptDefinition[]>("/api/v1/admin/prompts"),
      jsonRequest<McpServer[]>("/api/v1/admin/mcp-servers"),
      jsonRequest<ToolMapping[]>("/api/v1/admin/tool-mappings"),
    ]);
    setConnections(nextConnections);
    setProfiles(nextProfiles);
    setPrompts(nextPrompts);
    setServers(nextServers);
    setMappings(nextMappings);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const me = await jsonRequest<User>("/api/v1/auth/me");
      setUser(me);
      if (!me.agent_admin_enabled || !["ADMIN", "SYSTEM"].includes(me.role ?? "")) {
        throw new Error("当前账户没有 Agent 管理权限");
      }
      const nextAgents = await jsonRequest<AgentSummary[]>("/api/v1/admin/agents");
      setAgents(nextAgents);
      if (nextAgents[0]) await Promise.all([loadAgent(nextAgents[0].id), loadResources()]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "管理配置加载失败");
    } finally {
      setLoading(false);
    }
  }, [loadAgent, loadResources]);

  useEffect(() => { void load(); }, [load]);

  const draft = agent?.draft_version;
  const active = draft ?? agent?.published_version;

  const mutate = async (label: string, action: () => Promise<unknown>, refreshResources = false) => {
    if (!agent) return;
    setBusy(label); setError(""); setNotice("");
    try {
      await action();
      if (refreshResources) await loadResources();
      const nextAgents = await jsonRequest<AgentSummary[]>("/api/v1/admin/agents");
      setAgents(nextAgents);
      await loadAgent(agent.id);
      setNotice(`${label}完成`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : `${label}失败`);
    } finally { setBusy(""); }
  };

  if (loading) return <main className={styles.center}><LoaderCircle className={styles.spin} /></main>;
  if (!agent || !active) return <main className={styles.center}><div><h1>无法打开管理后台</h1><p>{error || "没有可管理的 Agent"}</p><a href="/">返回 Agent</a></div></main>;

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <div className={styles.brand}><Bot size={22} /><div><strong>Agent 管理</strong><span>{agent.name}</span></div></div>
        <div className={styles.headerActions}>
          <span className={styles.user}>{user?.display_name}</span>
          <a className={styles.iconButton} href="/" aria-label="返回业务页面" title="返回业务页面"><ArrowLeft size={18} /></a>
        </div>
      </header>

      <div className={styles.layout}>
        <aside className={styles.sidebar}>
          <label className={styles.agentSelectLabel}>Agent</label>
          <select value={agent.id} onChange={(event) => void loadAgent(event.target.value)}>
            {agents.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
          <nav className={styles.nav}>
            {TABS.map((item) => { const Icon = item.icon; return <button key={item.key} className={tab === item.key ? styles.activeNav : ""} onClick={() => setTab(item.key)}><Icon size={17} /><span>{item.label}</span><ChevronRight size={14} /></button>; })}
          </nav>
          <div className={styles.versionPanel}>
            <span>当前编辑</span><strong>{draft ? `草稿 v${draft.version}` : `已发布 v${active.version}`}</strong>
            <small>{active.config_hash.slice(0, 12)}</small>
          </div>
        </aside>

        <section className={styles.content}>
          <div className={styles.pageHeading}>
            <div><span className={styles.eyebrow}>{TABS.find((item) => item.key === tab)?.label}</span><h1>{agent.name}</h1><p>配置只在发布后影响新任务，运行中的任务继续使用原 Snapshot。</p></div>
            <div className={styles.actions}>
              {!draft && <button className={styles.secondaryButton} disabled={Boolean(busy)} onClick={() => void mutate("创建草稿", () => jsonRequest(`/api/v1/admin/agents/${agent.id}/drafts`, { method: "POST" }))}><Plus size={16} />创建草稿</button>}
              {draft && <button className={styles.primaryButton} disabled={Boolean(busy)} onClick={() => void mutate("发布", () => jsonRequest(`/api/v1/admin/agent-versions/${draft.id}/publish`, { method: "POST" }))}><Rocket size={16} />发布 v{draft.version}</button>}
            </div>
          </div>
          {error && <div className={styles.error}>{error}</div>}
          {notice && <div className={styles.notice}><Check size={15} />{notice}</div>}
          {tab === "overview" && <Overview agent={agent} active={active} connections={connections} profiles={profiles} prompts={prompts} servers={servers} mappings={mappings} />}
          {tab === "nodes" && <NodesPanel version={active} draft={draft} profiles={profiles} prompts={prompts} busy={busy} mutate={mutate} />}
          {tab === "models" && <ModelsPanel connections={connections} profiles={profiles} busy={busy} mutate={mutate} />}
          {tab === "prompts" && <PromptsPanel prompts={prompts} busy={busy} mutate={mutate} />}
          {tab === "mcp" && <McpPanel servers={servers} mappings={mappings} version={active} draft={draft} busy={busy} mutate={mutate} />}
          {tab === "runtime" && <RuntimePanel version={active} draft={draft} busy={busy} mutate={mutate} />}
        </section>
      </div>
    </main>
  );
}

function Overview({ agent, active, connections, profiles, prompts, servers, mappings }: { agent: AgentDetail; active: VersionDetail; connections: ModelConnection[]; profiles: ModelProfile[]; prompts: PromptDefinition[]; servers: McpServer[]; mappings: ToolMapping[] }) {
  const metrics = [["节点", active.nodes.length], ["模型配置", profiles.length], ["Prompt", prompts.length], ["MCP Server", servers.length], ["Logical Tool", mappings.length]];
  return <div className={styles.stack}>
    <section className={styles.metricBand}>{metrics.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</section>
    <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Definition</span><h2>版本状态</h2></div></div><dl className={styles.definitionGrid}><div><dt>Slug</dt><dd>{agent.slug}</dd></div><div><dt>Schema</dt><dd>v{active.config_schema_version}</dd></div><div><dt>已发布版本</dt><dd>v{agent.published_version.version}</dd></div><div><dt>草稿版本</dt><dd>{agent.draft_version ? `v${agent.draft_version.version}` : "无"}</dd></div><div><dt>模型连接</dt><dd>{connections.length}</dd></div><div><dt>配置哈希</dt><dd className={styles.mono}>{active.config_hash}</dd></div></dl></section>
    <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Topology</span><h2>节点拓扑</h2></div></div><div className={styles.nodeStrip}>{active.nodes.map((node) => <div key={node.node_key}><strong>{NODE_LABELS[node.node_key] ?? node.node_key}</strong><span>{node.model_id}</span></div>)}</div></section>
  </div>;
}

function NodesPanel({ version, draft, profiles, prompts, busy, mutate }: { version: VersionDetail; draft?: VersionDetail; profiles: ModelProfile[]; prompts: PromptDefinition[]; busy: string; mutate: (label: string, action: () => Promise<unknown>) => Promise<void> }) {
  return <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Node Registry</span><h2>节点模型与 Prompt</h2></div><p>{draft ? "修改会写入当前草稿。" : "创建草稿后才能修改绑定。"}</p></div><div className={styles.tableWrap}><table><thead><tr><th>节点</th><th>输出</th><th>模型</th><th>Prompt</th></tr></thead><tbody>{version.nodes.map((node) => {
    const nodePrompts = prompts.filter((item) => item.node_key === node.node_key);
    return <tr key={node.node_key}><td><strong>{NODE_LABELS[node.node_key] ?? node.node_key}</strong><small>{node.conditional ? "按需调用" : "固定能力"}</small></td><td className={styles.mono}>{node.output_schema}</td><td><select disabled={!draft || Boolean(busy)} value={node.model_profile_revision_id ?? ""} onChange={(event) => void mutate("绑定模型", () => jsonRequest(`/api/v1/admin/agent-versions/${draft!.id}/nodes/${node.node_key}/model`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_profile_revision_id: event.target.value }) }))}><option value="">环境默认 · {node.model_id}</option>{profiles.map((profile) => <option key={profile.id} value={profile.active_revision_id}>{profile.name} · {profile.model_id}</option>)}</select></td><td><select disabled={!draft || Boolean(busy)} value={node.prompt_revision_id ?? ""} onChange={(event) => void mutate("绑定 Prompt", () => jsonRequest(`/api/v1/admin/agent-versions/${draft!.id}/nodes/${node.node_key}/prompt`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt_revision_id: event.target.value }) }))}>{nodePrompts.map((prompt) => <option key={prompt.id} value={prompt.active_revision.id}>{prompt.name} · v{prompt.active_revision.version}</option>)}</select></td></tr>;
  })}</tbody></table></div></section>;
}

function ModelsPanel({ connections, profiles, busy, mutate }: { connections: ModelConnection[]; profiles: ModelProfile[]; busy: string; mutate: (label: string, action: () => Promise<unknown>, refresh?: boolean) => Promise<void> }) {
  const [connection, setConnection] = useState({ name: "", slug: "", provider: "openai_compatible", base_url: "", api_key: "" });
  const [profile, setProfile] = useState({ name: "", slug: "", connection_revision_id: connections[0]?.active_revision_id ?? "", model_id: "", api_mode: "chat_completions", parameters: '{"temperature":0.2,"max_output_tokens":8000}' });
  useEffect(() => { if (!profile.connection_revision_id && connections[0]) setProfile((item) => ({ ...item, connection_revision_id: connections[0].active_revision_id })); }, [connections, profile.connection_revision_id]);
  const createConnection = (event: FormEvent) => { event.preventDefault(); void mutate("创建模型连接", () => jsonRequest("/api/v1/admin/model-connections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(connection) }), true); };
  const createProfile = (event: FormEvent) => { event.preventDefault(); let parameters: Record<string, unknown>; try { parameters = JSON.parse(profile.parameters); } catch { return; } void mutate("创建模型配置", () => jsonRequest("/api/v1/admin/model-profiles", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...profile, parameters }) }), true); };
  return <div className={styles.stack}><section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Connections</span><h2>模型供应商连接</h2></div></div><div className={styles.resourceList}>{connections.map((item) => <div key={item.id}><div><strong>{item.name}</strong><span>{item.provider} · revision {item.revision_version}</span></div><div><code>{item.base_url}</code><small>{item.secret_ref}</small></div><button className={styles.iconButton} disabled={Boolean(busy)} title="测试连接" onClick={() => void mutate("测试连接", () => jsonRequest(`/api/v1/admin/model-connections/${item.id}/test`, { method: "POST" }))}><TestTube2 size={16} /></button></div>)}</div><form className={styles.inlineForm} onSubmit={createConnection}><input required placeholder="连接名称" value={connection.name} onChange={(e) => setConnection({ ...connection, name: e.target.value })} /><input required placeholder="slug" value={connection.slug} onChange={(e) => setConnection({ ...connection, slug: e.target.value })} /><input required placeholder="Base URL" value={connection.base_url} onChange={(e) => setConnection({ ...connection, base_url: e.target.value })} /><input required type="password" placeholder="API Key" value={connection.api_key} onChange={(e) => setConnection({ ...connection, api_key: e.target.value })} /><button className={styles.secondaryButton} disabled={Boolean(busy)}><Plus size={15} />添加连接</button></form></section>
  <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Profiles</span><h2>模型调用配置</h2></div></div><div className={styles.resourceList}>{profiles.map((item) => <div key={item.id}><div><strong>{item.name}</strong><span>revision {item.revision_version}</span></div><div><code>{item.model_id}</code><small>{item.api_mode} · {JSON.stringify(item.parameters)}</small></div></div>)}</div><form className={styles.formGrid} onSubmit={createProfile}><input required placeholder="配置名称" value={profile.name} onChange={(e) => setProfile({ ...profile, name: e.target.value })} /><input required placeholder="slug" value={profile.slug} onChange={(e) => setProfile({ ...profile, slug: e.target.value })} /><select value={profile.connection_revision_id} onChange={(e) => setProfile({ ...profile, connection_revision_id: e.target.value })}>{connections.map((item) => <option key={item.id} value={item.active_revision_id}>{item.name}</option>)}</select><input required placeholder="模型 ID" value={profile.model_id} onChange={(e) => setProfile({ ...profile, model_id: e.target.value })} /><textarea value={profile.parameters} onChange={(e) => setProfile({ ...profile, parameters: e.target.value })} /><button className={styles.secondaryButton} disabled={Boolean(busy)}><Plus size={15} />添加模型配置</button></form></section></div>;
}

function PromptsPanel({ prompts, busy, mutate }: { prompts: PromptDefinition[]; busy: string; mutate: (label: string, action: () => Promise<unknown>, refresh?: boolean) => Promise<void> }) {
  const [selectedId, setSelectedId] = useState(prompts[0]?.id ?? "");
  const selected = prompts.find((item) => item.id === selectedId) ?? prompts[0];
  const [content, setContent] = useState(selected?.active_revision.content ?? "");
  useEffect(() => { if (selected) setContent(selected.active_revision.content); }, [selected]);
  if (!selected) return <section className={styles.section}>暂无 Prompt。</section>;
  return <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Versioned Prompts</span><h2>Prompt 编辑与校验</h2></div><select value={selected.id} onChange={(e) => setSelectedId(e.target.value)}>{prompts.map((item) => <option key={item.id} value={item.id}>{NODE_LABELS[item.node_key] ?? item.node_key} · {item.name}</option>)}</select></div><div className={styles.promptMeta}><span>节点 <code>{selected.node_key}</code></span><span>当前版本 v{selected.active_revision.version}</span><span>状态 {selected.active_revision.status}</span></div><textarea className={styles.promptEditor} value={content} onChange={(e) => setContent(e.target.value)} spellCheck={false} /><div className={styles.actions}><button className={styles.secondaryButton} disabled={Boolean(busy)} onClick={() => void mutate("静态校验", () => jsonRequest(`/api/v1/admin/prompt-revisions/${selected.active_revision.id}/validate`, { method: "POST" }))}><TestTube2 size={15} />校验当前版本</button><button className={styles.primaryButton} disabled={Boolean(busy) || content === selected.active_revision.content} onClick={() => void mutate("保存 Prompt 版本", () => jsonRequest(`/api/v1/admin/prompts/${selected.id}/revisions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content }) }), true)}><Save size={15} />保存新版本</button></div></section>;
}

function McpPanel({ servers, mappings, version, draft, busy, mutate }: { servers: McpServer[]; mappings: ToolMapping[]; version: VersionDetail; draft?: VersionDetail; busy: string; mutate: (label: string, action: () => Promise<unknown>, refresh?: boolean) => Promise<void> }) {
  const [server, setServer] = useState({ name: "", slug: "", url: "", authentication_type: "none", api_token: "", timeout_seconds: 10 });
  const [mapping, setMapping] = useState({ name: "", logical_tool_key: "", mcp_server_revision_id: servers[0]?.active_revision_id ?? "", remote_tool_name: "", adapter_key: "declarative", input_mapping: "{}", output_mapping: "{}", timeout_seconds: 10 });
  const [discovered, setDiscovered] = useState<DiscoveredTool[]>([]);
  const [discoverError, setDiscoverError] = useState("");
  useEffect(() => { if (!mapping.mcp_server_revision_id && servers[0]) setMapping((item) => ({ ...item, mcp_server_revision_id: servers[0].active_revision_id })); }, [servers, mapping.mcp_server_revision_id]);
  const createServer = (event: FormEvent) => { event.preventDefault(); void mutate("添加 MCP Server", () => jsonRequest("/api/v1/admin/mcp-servers", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...server, api_token: server.api_token || null }) }), true); };
  const createMapping = (event: FormEvent) => { event.preventDefault(); let input_mapping; let output_mapping; try { input_mapping = JSON.parse(mapping.input_mapping); output_mapping = JSON.parse(mapping.output_mapping); } catch { return; } void mutate("添加 Tool Mapping", () => jsonRequest("/api/v1/admin/tool-mappings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...mapping, input_mapping, output_mapping }) }), true); };
  const discover = async (serverId: string) => { setDiscoverError(""); try { setDiscovered(await jsonRequest<DiscoveredTool[]>(`/api/v1/admin/mcp-servers/${serverId}/discover-tools`, { method: "POST" })); } catch (reason) { setDiscoverError(reason instanceof Error ? reason.message : "工具发现失败"); } };
  return <div className={styles.stack}><section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Registry</span><h2>MCP Server</h2></div></div><div className={styles.resourceList}>{servers.map((item) => <div key={item.id}><div><strong>{item.name}</strong><span>{item.authentication_type} · {item.timeout_seconds}s</span></div><div><code>{item.url}</code><small>{item.secret_ref ?? "无认证"}</small></div><button className={styles.secondaryButton} disabled={Boolean(busy)} onClick={() => void discover(item.id)}><Database size={15} />发现工具</button></div>)}</div>{discoverError && <div className={styles.error}>{discoverError}</div>}{discovered.length > 0 && <div className={styles.discovered}>{discovered.map((tool) => <button key={tool.name} onClick={() => setMapping({ ...mapping, remote_tool_name: tool.name })}><strong>{tool.name}</strong><span>{tool.description}</span></button>)}</div>}<form className={styles.formGrid} onSubmit={createServer}><input required placeholder="Server 名称" value={server.name} onChange={(e) => setServer({ ...server, name: e.target.value })} /><input required placeholder="slug" value={server.slug} onChange={(e) => setServer({ ...server, slug: e.target.value })} /><input required placeholder="MCP URL" value={server.url} onChange={(e) => setServer({ ...server, url: e.target.value })} /><select value={server.authentication_type} onChange={(e) => setServer({ ...server, authentication_type: e.target.value })}><option value="none">无认证</option><option value="bearer">Bearer Token</option></select><input type="password" placeholder="Token（可选）" value={server.api_token} onChange={(e) => setServer({ ...server, api_token: e.target.value })} /><button className={styles.secondaryButton} disabled={Boolean(busy)}><Plus size={15} />添加 Server</button></form></section>
  <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Logical Tools</span><h2>Tool Mapping 与节点权限</h2></div></div><div className={styles.resourceList}>{mappings.map((item) => <ToolBindingEditor key={item.id} mapping={item} binding={version.tools.find((tool) => tool.logical_tool_key === item.logical_tool_key)} callers={[...version.nodes.map((node) => node.node_key), "research_pipeline"]} draft={draft} busy={busy} mutate={mutate} />)}</div><form className={styles.formGrid} onSubmit={createMapping}><input required placeholder="Mapping 名称" value={mapping.name} onChange={(e) => setMapping({ ...mapping, name: e.target.value })} /><input required placeholder="Logical Tool Key" value={mapping.logical_tool_key} onChange={(e) => setMapping({ ...mapping, logical_tool_key: e.target.value })} /><select value={mapping.mcp_server_revision_id} onChange={(e) => setMapping({ ...mapping, mcp_server_revision_id: e.target.value })}>{servers.map((item) => <option key={item.id} value={item.active_revision_id}>{item.name}</option>)}</select><input required placeholder="远端 Tool 名称" value={mapping.remote_tool_name} onChange={(e) => setMapping({ ...mapping, remote_tool_name: e.target.value })} /><textarea value={mapping.input_mapping} onChange={(e) => setMapping({ ...mapping, input_mapping: e.target.value })} /><textarea value={mapping.output_mapping} onChange={(e) => setMapping({ ...mapping, output_mapping: e.target.value })} /><button className={styles.secondaryButton} disabled={Boolean(busy)}><Plus size={15} />添加 Mapping</button></form></section></div>;
}

function ToolBindingEditor({ mapping, binding, callers, draft, busy, mutate }: { mapping: ToolMapping; binding?: ToolBinding; callers: string[]; draft?: VersionDetail; busy: string; mutate: (label: string, action: () => Promise<unknown>) => Promise<void> }) {
  const [allowed, setAllowed] = useState<string[]>(binding?.allowed_nodes ?? ["research_pipeline"]);
  useEffect(() => setAllowed(binding?.allowed_nodes ?? ["research_pipeline"]), [binding]);
  return <div><div><strong>{mapping.logical_tool_key}</strong><span>{mapping.name} · revision {mapping.revision_version}</span></div><div><code>{mapping.remote_tool_name}</code><select multiple className={styles.multiSelect} disabled={!draft} value={allowed} onChange={(event) => setAllowed(Array.from(event.target.selectedOptions, (option) => option.value))}>{callers.map((caller) => <option key={caller} value={caller}>{NODE_LABELS[caller] ?? caller}</option>)}</select></div><button className={styles.secondaryButton} disabled={!draft || Boolean(busy) || allowed.length === 0} onClick={() => void mutate("绑定 Tool", () => jsonRequest(`/api/v1/admin/agent-versions/${draft!.id}/tools/${mapping.logical_tool_key}/binding`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tool_mapping_revision_id: mapping.active_revision_id, allowed_nodes: allowed }) }))}><Save size={15} />绑定</button></div>;
}

function RuntimePanel({ version, draft, busy, mutate }: { version: VersionDetail; draft?: VersionDetail; busy: string; mutate: (label: string, action: () => Promise<unknown>) => Promise<void> }) {
  const [loop, setLoop] = useState(version.loop);
  const [formats, setFormats] = useState(version.output.formats);
  const [evidence, setEvidence] = useState(version.output.evidence_validation_required);
  useEffect(() => { setLoop(version.loop); setFormats(version.output.formats); setEvidence(version.output.evidence_validation_required); }, [version]);
  const numberField = (key: string, label: string, min: number, max: number, step = 1) => <label><span>{label}</span><input type="number" min={min} max={max} step={step} value={Number(loop[key])} disabled={!draft} onChange={(e) => setLoop({ ...loop, [key]: Number(e.target.value) })} /></label>;
  const toggle = (key: string, label: string) => <label className={styles.toggle}><input type="checkbox" checked={Boolean(loop[key])} disabled={!draft} onChange={(e) => setLoop({ ...loop, [key]: e.target.checked })} /><span>{label}</span></label>;
  return <div className={styles.stack}><section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Agent Loop</span><h2>循环与决策边界</h2></div></div><div className={styles.runtimeGrid}>{numberField("max_loops", "最大循环轮次", 1, 50)}{numberField("max_tool_calls", "最大 Tool 调用", 1, 100)}{numberField("max_repeated_actions", "重复动作上限", 1, 20)}{numberField("identity_auto_accept_threshold", "身份自动接受阈值", 0, 1, 0.01)}</div><div className={styles.toggleGrid}>{toggle("intake_agent_v2_enabled", "Intake Agent V2")}{toggle("intake_entity_resolution_enabled", "身份解析")}{toggle("intake_react_enabled", "ReAct Loop")}</div></section><section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Output</span><h2>输出与证据策略</h2></div></div><div className={styles.toggleGrid}><label className={styles.toggle}><input type="checkbox" disabled={!draft} checked={formats.includes("detailed_markdown")} onChange={(e) => setFormats(changeList(formats, "detailed_markdown", e.target.checked))} /><span>详细 Markdown 报告</span></label><label className={styles.toggle}><input type="checkbox" disabled={!draft} checked={formats.includes("action_brief_markdown")} onChange={(e) => setFormats(changeList(formats, "action_brief_markdown", e.target.checked))} /><span>行动简报</span></label><label className={styles.toggle}><input type="checkbox" disabled={!draft} checked={evidence} onChange={(e) => setEvidence(e.target.checked)} /><span>发布前要求证据校验</span></label></div><div className={styles.templateList}>{version.output.templates?.map((template) => <div key={template.name}><strong>{template.name}</strong><code>{template.path}</code></div>)}</div></section><div className={styles.actions}><button className={styles.primaryButton} disabled={!draft || Boolean(busy) || formats.length === 0} onClick={() => void mutate("保存运行配置", () => jsonRequest(`/api/v1/admin/agent-versions/${draft!.id}/runtime-config`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ loop, output: { formats, evidence_validation_required: evidence } }) }))}><Save size={15} />保存到草稿</button></div></div>;
}

function changeList(items: string[], value: string, enabled: boolean) {
  return enabled ? Array.from(new Set([...items, value])) : items.filter((item) => item !== value);
}
