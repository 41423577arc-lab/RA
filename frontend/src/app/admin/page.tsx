"use client";

import {
  ArrowLeft,
  Bot,
  Check,
  ChevronRight,
  FileText,
  Gauge,
  History,
  KeyRound,
  Link2,
  LoaderCircle,
  Network,
  Pencil,
  Plus,
  Rocket,
  RotateCcw,
  Save,
  SlidersHorizontal,
  TestTube2,
  Trash2,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useState } from "react";
import styles from "./admin.module.css";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const apiFetch = (path: string, init?: RequestInit) =>
  fetch(`${API_BASE}${path}`, { ...init, credentials: "include" });

type User = { user_id?: string; display_name?: string; role?: string; agent_admin_enabled: boolean };
type Version = { id: string; agent_definition_id: string; version: number; status: string; config_hash: string; release_note?: string; published_at?: string };
type NodeBinding = { node_key: string; output_schema: string; conditional: boolean; allows_tools: boolean; model_profile_revision_id?: string; model_id: string; provider: string; prompt_definition_id?: string; prompt_revision_id?: string; prompt_version?: number; prompt_source?: string; prompt_config: PromptConfig; allowed_tools: string[] };
type ToolBinding = { logical_tool_key: string; tool_mapping_revision_id: string; remote_tool_name: string; adapter_key: string; allowed_nodes: string[] };
type VersionDetail = Version & { config_schema_version: number; nodes: NodeBinding[]; tools: ToolBinding[] };
type AgentDetail = { id: string; name: string; slug: string; status: string; published_version: VersionDetail; draft_version?: VersionDetail };
type ModelConnection = { id: string; name: string; slug: string; active_revision_id: string; revision_version: number; provider: string; base_url: string; secret_ref: string };
type ModelProfile = { id: string; name: string; slug: string; active_revision_id: string; revision_version: number; connection_revision_id: string; model_id: string; api_mode: string; parameters: Record<string, unknown> };
type PromptSkill = { revision_id: string; name: string; content_hash: string; content: string };
type PromptValidation = { valid?: boolean; node_key?: string; output_schema?: string; output_schema_boundary?: string; required_variables?: string[]; skill_names?: string[] };
type PromptConfig = { prompt_definition_id?: string; revision_id?: string; base_revision_id?: string; version?: number; node_key?: string; content: string; content_hash: string; config_hash?: string; required_variables: string[]; skills: PromptSkill[]; validation_report: PromptValidation; smoke_test_status?: string; source?: string; working?: boolean };
type PromptRevision = { id: string; prompt_definition_id: string; version: number; content: string; content_hash: string; required_variables: string[]; skills: PromptSkill[]; validation_report: PromptValidation; smoke_test_status: string; source: string; status: string };
type PromptDefinition = { id: string; name: string; slug: string; node_key: string; active_revision: PromptRevision };
type NodeDiff = { node_key: string; prompt_changed: boolean; draft_prompt_revision_id?: string; published_prompt_revision_id?: string; prompt_content_hash_changed: boolean; model_changed: boolean; draft_model_name: string; published_model_name: string; draft_model_revision_id?: string; published_model_revision_id?: string };
type ToolDiff = { logical_tool_key: string; changed: boolean };
type ConfigDiff = { has_draft: boolean; has_changes: boolean; draft_version?: number; published_version: number; nodes: NodeDiff[]; tools: ToolDiff[] };
type Tab = "overview" | "nodes" | "models" | "prompts" | "versions";

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

const TABS: { key: Tab; label: string; icon: typeof Gauge }[] = [
  { key: "overview", label: "概览", icon: Gauge },
  { key: "nodes", label: "节点配置", icon: Network },
  { key: "models", label: "模型", icon: SlidersHorizontal },
  { key: "prompts", label: "Prompt", icon: FileText },
  { key: "versions", label: "版本", icon: History },
];

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await apiFetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? `请求失败 (${response.status})`);
  return payload as T;
}

export default function AdminPage() {
  const [user, setUser] = useState<User | null>(null);
  const [agent, setAgent] = useState<AgentDetail | null>(null);
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [prompts, setPrompts] = useState<PromptDefinition[]>([]);
  const [tab, setTab] = useState<Tab>("overview");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadAgent = useCallback(async () => {
    setAgent(await jsonRequest<AgentDetail>("/api/v1/admin/agent"));
  }, []);

  const loadResources = useCallback(async () => {
    const [nextConnections, nextProfiles, nextPrompts] = await Promise.all([
      jsonRequest<ModelConnection[]>("/api/v1/admin/model-connections"),
      jsonRequest<ModelProfile[]>("/api/v1/admin/model-profiles"),
      jsonRequest<PromptDefinition[]>("/api/v1/admin/prompts"),
    ]);
    setConnections(nextConnections);
    setProfiles(nextProfiles);
    setPrompts(nextPrompts);
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
      await Promise.all([loadAgent(), loadResources()]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "管理配置加载失败");
    } finally {
      setLoading(false);
    }
  }, [loadAgent, loadResources]);

  useEffect(() => { void load(); }, [load]);

  const draft = agent?.draft_version;
  const active = draft ?? agent?.published_version;
  const publishVersion = draft && agent
    ? Math.max(draft.version, agent.published_version.version + 1)
    : undefined;
  const mutate = async (label: string, action: () => Promise<unknown>, refreshResources = false) => {
    if (!agent) return;
    setBusy(label); setError(""); setNotice("");
    try {
      await action();
      if (refreshResources) await loadResources();
      await loadAgent();
      setNotice(`${label}完成`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : `${label}失败`);
    } finally { setBusy(""); }
  };
  const publishDraft = async () => {
    if (!draft) return;
    const releaseNote = window.prompt("发布备注（可选）", "");
    if (releaseNote === null) return;
    await mutate("发布", () => jsonRequest(`/api/v1/admin/agent-versions/${draft.id}/publish`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ release_note: releaseNote.trim() || null }),
    }));
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
            <div><span className={styles.eyebrow}>{TABS.find((item) => item.key === tab)?.label}</span><h1>{agent.name}</h1><p>当前草稿用于下一次新任务，已经开始的任务继续使用原 Snapshot。</p></div>
            <div className={styles.actions}>
              {!draft && <button className={styles.secondaryButton} disabled={Boolean(busy)} onClick={() => void mutate("创建草稿", () => jsonRequest(`/api/v1/admin/agents/${agent.id}/drafts`, { method: "POST" }))}><Plus size={16} />创建草稿</button>}
              {draft && <button className={styles.primaryButton} title="冻结当前工作配置并发布为稳定版本" disabled={Boolean(busy)} onClick={() => void publishDraft()}><Rocket size={16} />发布为 v{publishVersion}</button>}
            </div>
          </div>
          {error && <div className={styles.error}>{error}</div>}
          {notice && <div className={styles.notice}><Check size={15} />{notice}</div>}
          {tab === "overview" && <Overview agent={agent} active={active} connections={connections} profiles={profiles} prompts={prompts} />}
          {tab === "nodes" && <NodesPanel version={active} draft={draft} profiles={profiles} prompts={prompts} busy={busy} mutate={mutate} />}
          {tab === "models" && <ModelsPanel connections={connections} profiles={profiles} busy={busy} mutate={mutate} />}
          {tab === "prompts" && <PromptsPanel version={active} draft={draft} prompts={prompts} busy={busy} mutate={mutate} />}
          {tab === "versions" && <VersionsPanel agent={agent} busy={busy} mutate={mutate} />}
        </section>
      </div>
    </main>
  );
}

function VersionsPanel({ agent, busy, mutate }: { agent: AgentDetail; busy: string; mutate: (label: string, action: () => Promise<unknown>) => Promise<void> }) {
  const [versions, setVersions] = useState<Version[]>([]);
  const [diff, setDiff] = useState<ConfigDiff | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError("");
    Promise.all([
      jsonRequest<Version[]>("/api/v1/admin/agent/versions"),
      jsonRequest<ConfigDiff>("/api/v1/admin/agent/diff"),
    ]).then(([history, currentDiff]) => {
      if (!cancelled) { setVersions(history); setDiff(currentDiff); }
    }).catch((reason) => {
      if (!cancelled) setLoadError(reason instanceof Error ? reason.message : "版本信息加载失败");
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [agent.draft_version?.config_hash, agent.published_version.config_hash]);
  const restore = async (version: Version) => {
    const overwriting = Boolean(agent.draft_version);
    if (!window.confirm(overwriting ? "将使用此历史版本覆盖当前工作配置，现有草稿修改会丢失。" : "将以此历史版本创建新的工作草稿。")) return;
    await mutate(`恢复 v${version.version}`, () => jsonRequest(`/api/v1/admin/agent/versions/${version.id}/restore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm_overwrite: overwriting }),
    }));
  };
  if (loading) return <section className={styles.section}><LoaderCircle className={styles.spin} size={18} /></section>;
  if (loadError) return <section className={styles.section}>{loadError}</section>;
  return <div className={styles.stack}>
    <section className={styles.section}>
      <div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Draft vs Published</span><h2>当前配置差异</h2></div><p>{diff?.has_draft ? `草稿 v${diff.draft_version} 对比已发布 v${diff.published_version}` : "当前没有草稿。"}</p></div>
      {diff?.has_draft && <div className={styles.tableWrap}><table><thead><tr><th>节点</th><th>Prompt</th><th>模型</th></tr></thead><tbody>{diff.nodes.map((item) => <tr key={item.node_key}><td><strong>{NODE_LABELS[item.node_key] ?? item.node_key}</strong></td><td>{item.prompt_changed ? "已修改" : "未变化"}<small>{item.prompt_content_hash_changed ? "内容 Hash 已变化" : "内容 Hash 相同"}</small></td><td>{item.model_changed ? <><strong>{item.published_model_name}</strong><small>→ {item.draft_model_name}</small></> : <><span>未变化</span><small>{item.draft_model_name}</small></>}</td></tr>)}</tbody></table></div>}
      {diff?.has_draft && diff.tools.some((item) => item.changed) && <div className={styles.resourceList}>{diff.tools.map((item) => <div key={item.logical_tool_key}><div><strong>{item.logical_tool_key}</strong><span>{item.changed ? "绑定已变化" : "未变化"}</span></div></div>)}</div>}
    </section>
    <section className={styles.section}>
      <div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Published Checkpoints</span><h2>稳定版本历史</h2></div><p>恢复只会写入当前工作草稿，不会立即改变正式版本。</p></div>
      <div className={styles.resourceList}>{versions.map((version) => <div key={version.id}><div><strong>v{version.version}{version.id === agent.published_version.id ? " · 当前正式" : ""}</strong><span>{version.published_at ? new Date(version.published_at).toLocaleString("zh-CN") : ""}</span></div><div><code>{version.config_hash.slice(0, 16)}</code><small>{version.release_note || "无发布备注"}</small></div><button className={styles.iconButton} disabled={Boolean(busy)} title="恢复到工作草稿" onClick={() => void restore(version)}><RotateCcw size={16} /></button></div>)}</div>
    </section>
  </div>;
}

function Overview({ agent, active, connections, profiles, prompts }: { agent: AgentDetail; active: VersionDetail; connections: ModelConnection[]; profiles: ModelProfile[]; prompts: PromptDefinition[] }) {
  const metrics = [["节点", active.nodes.length], ["模型配置", profiles.length], ["Prompt", prompts.length], ["固定 Logical Tool", active.tools.length]];
  return <div className={styles.stack}>
    <section className={styles.metricBand}>{metrics.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</section>
    <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Definition</span><h2>版本状态</h2></div></div><dl className={styles.definitionGrid}><div><dt>Slug</dt><dd>{agent.slug}</dd></div><div><dt>Schema</dt><dd>v{active.config_schema_version}</dd></div><div><dt>已发布版本</dt><dd>v{agent.published_version.version}</dd></div><div><dt>草稿版本</dt><dd>{agent.draft_version ? `v${agent.draft_version.version}` : "无"}</dd></div><div><dt>模型连接</dt><dd>{connections.length}</dd></div><div><dt>配置哈希</dt><dd className={styles.mono}>{active.config_hash}</dd></div></dl></section>
    <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Topology</span><h2>节点拓扑</h2></div></div><div className={styles.nodeStrip}>{active.nodes.map((node) => <div key={node.node_key}><strong>{NODE_LABELS[node.node_key] ?? node.node_key}</strong><span>{node.model_id}</span></div>)}</div></section>
    <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Logical Tools</span><h2>固定工具绑定</h2></div><p>工具注册与权限由代码和现有配置底座管理，此处仅展示当前版本。</p></div><div className={styles.resourceList}>{active.tools.map((tool) => <div key={tool.logical_tool_key}><div><strong>{tool.logical_tool_key}</strong><span>{tool.adapter_key}</span></div><div><code>{tool.remote_tool_name}</code><small>{tool.allowed_nodes.map((node) => NODE_LABELS[node] ?? node).join("、")}</small></div></div>)}</div></section>
  </div>;
}

function NodesPanel({ version, draft, profiles, prompts, busy, mutate }: { version: VersionDetail; draft?: VersionDetail; profiles: ModelProfile[]; prompts: PromptDefinition[]; busy: string; mutate: (label: string, action: () => Promise<unknown>) => Promise<void> }) {
  return <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Node Registry</span><h2>节点模型与 Prompt</h2></div><p>{draft ? "模型修改会写入当前草稿。" : "创建草稿后才能修改模型绑定。"}</p></div><div className={styles.tableWrap}><table><thead><tr><th>节点</th><th>输出</th><th>模型</th><th>Prompt</th></tr></thead><tbody>{version.nodes.map((node) => {
    const prompt = prompts.find((item) => item.id === node.prompt_definition_id);
    return <tr key={node.node_key}><td><strong>{NODE_LABELS[node.node_key] ?? node.node_key}</strong><small>{node.conditional ? "按需调用" : "固定能力"}</small></td><td className={styles.mono}>{node.output_schema}</td><td><select disabled={!draft || Boolean(busy)} value={node.model_profile_revision_id ?? ""} onChange={(event) => void mutate("绑定模型", () => jsonRequest(`/api/v1/admin/agent-versions/${draft!.id}/nodes/${node.node_key}/model`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_profile_revision_id: event.target.value }) }))}><option value="">环境默认 · {node.model_id}</option>{profiles.map((profile) => <option key={profile.id} value={profile.active_revision_id}>{profile.name} · {profile.model_id}</option>)}</select></td><td><div className={styles.promptBinding}><strong>{prompt?.name ?? "固定节点 Prompt"}</strong><small>{node.prompt_config.working ? "工作稿" : "稳定副本"} · 基于 v{node.prompt_version ?? "-"}</small></div></td></tr>;
  })}</tbody></table></div></section>;
}

function ModelsPanel({ connections, profiles, busy, mutate }: { connections: ModelConnection[]; profiles: ModelProfile[]; busy: string; mutate: (label: string, action: () => Promise<unknown>, refresh?: boolean) => Promise<void> }) {
  const [connection, setConnection] = useState({ name: "", slug: "", provider: "openai_compatible", base_url: "", api_key: "" });
  const [profile, setProfile] = useState({ name: "", slug: "", connection_revision_id: connections[0]?.active_revision_id ?? "", model_id: "", api_mode: "chat_completions" });
  const [connectionEdits, setConnectionEdits] = useState<Record<string, { name: string; provider: string; base_url: string }>>({});
  const [profileEdits, setProfileEdits] = useState<Record<string, { name: string; connection_revision_id: string; model_id: string; api_mode: string }>>({});
  const [secretEdits, setSecretEdits] = useState<Record<string, string>>({});
  useEffect(() => {
    setConnectionEdits(Object.fromEntries(connections.map((item) => [item.id, { name: item.name, provider: item.provider, base_url: item.base_url }])));
  }, [connections]);
  useEffect(() => {
    setProfileEdits(Object.fromEntries(profiles.map((item) => [item.id, { name: item.name, connection_revision_id: item.connection_revision_id, model_id: item.model_id, api_mode: item.api_mode }])));
    if (!profile.connection_revision_id && connections[0]) setProfile((item) => ({ ...item, connection_revision_id: connections[0].active_revision_id }));
  }, [connections, profile.connection_revision_id, profiles]);
  const createConnection = (event: FormEvent) => { event.preventDefault(); void mutate("创建模型连接", () => jsonRequest("/api/v1/admin/model-connections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(connection) }), true); };
  const createProfile = (event: FormEvent) => { event.preventDefault(); void mutate("创建模型配置", () => jsonRequest("/api/v1/admin/model-profiles", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...profile, parameters: {} }) }), true); };
  const saveConnection = (item: ModelConnection) => { const edit = connectionEdits[item.id]; if (!edit) return; void mutate("保存模型连接", () => jsonRequest(`/api/v1/admin/model-connections/${item.id}/revisions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...edit, secret_ref: item.secret_ref }) }), true); };
  const rotateSecret = (item: ModelConnection) => { const apiKey = secretEdits[item.id]?.trim(); if (!apiKey) return; void mutate("轮换 API Key", () => jsonRequest(`/api/v1/admin/model-connections/${item.id}/rotate-secret`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: apiKey }) }), true).then(() => setSecretEdits((values) => ({ ...values, [item.id]: "" }))); };
  const saveProfile = (item: ModelProfile) => { const edit = profileEdits[item.id]; if (!edit) return; void mutate("保存模型配置", () => jsonRequest(`/api/v1/admin/model-profiles/${item.id}/revisions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...edit, parameters: item.parameters }) }), true); };
  return <div className={styles.stack}>
    <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Connections</span><h2>模型供应商连接</h2></div><p>API Key 只写入 Secret Store，页面不会回显明文。</p></div>
      <div className={styles.stack}>{connections.map((item) => { const edit = connectionEdits[item.id]; if (!edit) return null; const secretIsManaged = item.secret_ref.startsWith("db:"); return <form key={item.id} className={styles.formGrid} onSubmit={(event) => { event.preventDefault(); saveConnection(item); }}><input required aria-label="连接名称" value={edit.name} onChange={(event) => setConnectionEdits({ ...connectionEdits, [item.id]: { ...edit, name: event.target.value } })} /><select aria-label="Provider" value={edit.provider} onChange={(event) => setConnectionEdits({ ...connectionEdits, [item.id]: { ...edit, provider: event.target.value } })}><option value="openai_compatible">OpenAI Compatible</option><option value="openai">OpenAI</option></select><input required aria-label="Base URL" value={edit.base_url} onChange={(event) => setConnectionEdits({ ...connectionEdits, [item.id]: { ...edit, base_url: event.target.value } })} /><input type="password" disabled={!secretIsManaged} placeholder={secretIsManaged ? "输入新 API Key 进行轮换" : "环境变量密钥在部署配置中轮换"} value={secretEdits[item.id] ?? ""} onChange={(event) => setSecretEdits({ ...secretEdits, [item.id]: event.target.value })} /><div className={styles.actions}><button className={styles.iconButton} type="button" disabled={Boolean(busy)} title="测试连接" onClick={() => void mutate("测试连接", () => jsonRequest(`/api/v1/admin/model-connections/${item.id}/test`, { method: "POST" }))}><TestTube2 size={16} /></button><button className={styles.secondaryButton} type="button" disabled={Boolean(busy) || !secretIsManaged || !secretEdits[item.id]?.trim()} onClick={() => rotateSecret(item)}><KeyRound size={15} />轮换密钥</button><button className={styles.secondaryButton} disabled={Boolean(busy)}><Save size={15} />保存连接</button></div><small>连接 Revision v{item.revision_version} · {item.secret_ref}</small></form>; })}</div>
      <form className={styles.inlineForm} onSubmit={createConnection}><input required placeholder="连接名称" value={connection.name} onChange={(e) => setConnection({ ...connection, name: e.target.value })} /><input required placeholder="slug" value={connection.slug} onChange={(e) => setConnection({ ...connection, slug: e.target.value })} /><select value={connection.provider} onChange={(e) => setConnection({ ...connection, provider: e.target.value })}><option value="openai_compatible">OpenAI Compatible</option><option value="openai">OpenAI</option></select><input required placeholder="Base URL" value={connection.base_url} onChange={(e) => setConnection({ ...connection, base_url: e.target.value })} /><input required type="password" placeholder="API Key" value={connection.api_key} onChange={(e) => setConnection({ ...connection, api_key: e.target.value })} /><button className={styles.secondaryButton} disabled={Boolean(busy)}><Plus size={15} />添加连接</button></form>
    </section>
    <section className={styles.section}><div className={styles.sectionHeading}><div><span className={styles.eyebrow}>Profiles</span><h2>可选模型</h2></div><p>节点只切换已配置模型，不创建新的模型 Revision。</p></div>
      <div className={styles.stack}>{profiles.map((item) => { const edit = profileEdits[item.id]; if (!edit) return null; const activeIds = new Set(connections.map((connectionItem) => connectionItem.active_revision_id)); return <form key={item.id} className={styles.formGrid} onSubmit={(event) => { event.preventDefault(); saveProfile(item); }}><input required aria-label="模型名称" value={edit.name} onChange={(event) => setProfileEdits({ ...profileEdits, [item.id]: { ...edit, name: event.target.value } })} /><select value={edit.connection_revision_id} onChange={(event) => setProfileEdits({ ...profileEdits, [item.id]: { ...edit, connection_revision_id: event.target.value } })}>{!activeIds.has(edit.connection_revision_id) && <option value={edit.connection_revision_id}>当前历史连接 Revision</option>}{connections.map((connectionItem) => <option key={connectionItem.id} value={connectionItem.active_revision_id}>{connectionItem.name}</option>)}</select><input required aria-label="模型 ID" value={edit.model_id} onChange={(event) => setProfileEdits({ ...profileEdits, [item.id]: { ...edit, model_id: event.target.value } })} /><button className={styles.secondaryButton} disabled={Boolean(busy)}><Save size={15} />保存模型</button><small>Profile Revision v{item.revision_version}</small></form>; })}</div>
      <form className={styles.inlineForm} onSubmit={createProfile}><input required placeholder="模型名称" value={profile.name} onChange={(e) => setProfile({ ...profile, name: e.target.value })} /><input required placeholder="slug" value={profile.slug} onChange={(e) => setProfile({ ...profile, slug: e.target.value })} /><select required value={profile.connection_revision_id} onChange={(e) => setProfile({ ...profile, connection_revision_id: e.target.value })}>{connections.map((item) => <option key={item.id} value={item.active_revision_id}>{item.name}</option>)}</select><input required placeholder="模型 ID" value={profile.model_id} onChange={(e) => setProfile({ ...profile, model_id: e.target.value })} /><button className={styles.secondaryButton} disabled={Boolean(busy)}><Plus size={15} />添加模型</button></form>
    </section>
  </div>;
}

function PromptsPanel({ version, draft, prompts, busy, mutate }: { version: VersionDetail; draft?: VersionDetail; prompts: PromptDefinition[]; busy: string; mutate: (label: string, action: () => Promise<unknown>, refresh?: boolean) => Promise<void> }) {
  const [selectedNodeKey, setSelectedNodeKey] = useState(version.nodes[0]?.node_key ?? "");
  const [selectedDefinitionId, setSelectedDefinitionId] = useState("");
  const [revisions, setRevisions] = useState<PromptRevision[]>([]);
  const [selectedRevisionId, setSelectedRevisionId] = useState("working");
  const [content, setContent] = useState("");
  const [editing, setEditing] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");

  useEffect(() => {
    if (!version.nodes.some((item) => item.node_key === selectedNodeKey)) {
      setSelectedNodeKey(version.nodes[0]?.node_key ?? "");
    }
  }, [selectedNodeKey, version.nodes]);

  const node = version.nodes.find((item) => item.node_key === selectedNodeKey) ?? version.nodes[0];
  const nodeDefinitions = prompts.filter((item) => item.node_key === node?.node_key);
  const boundDefinition = nodeDefinitions.find((item) => item.id === node?.prompt_definition_id);
  const currentPrompt = node?.prompt_config;

  useEffect(() => {
    const preferredDefinition = boundDefinition ?? nodeDefinitions[0];
    setSelectedDefinitionId(preferredDefinition?.id ?? "");
    setSelectedRevisionId("working");
    setEditing(false);
  }, [boundDefinition, node?.node_key, prompts]);

  const definition = nodeDefinitions.find((item) => item.id === selectedDefinitionId) ?? boundDefinition ?? nodeDefinitions[0];

  useEffect(() => {
    if (!definition) {
      setRevisions([]);
      return;
    }
    let cancelled = false;
    setHistoryLoading(true);
    setHistoryError("");
    void jsonRequest<PromptRevision[]>(`/api/v1/admin/prompts/${definition.id}/revisions`)
      .then((items) => { if (!cancelled) setRevisions(items); })
      .catch((reason) => { if (!cancelled) setHistoryError(reason instanceof Error ? reason.message : "Prompt 历史加载失败"); })
      .finally(() => { if (!cancelled) setHistoryLoading(false); });
    return () => { cancelled = true; };
  }, [definition]);

  useEffect(() => {
    if (selectedRevisionId === "working") setContent(currentPrompt?.content ?? "");
  }, [currentPrompt?.content, currentPrompt?.content_hash, selectedRevisionId]);

  const selectedRevision = revisions.find((item) => item.id === selectedRevisionId);
  const showingWorking = selectedRevisionId === "working";
  const selectRevision = (revision: PromptRevision) => {
    setSelectedRevisionId(revision.id);
    setContent(revision.content);
    setEditing(false);
  };
  const showWorkingCopy = () => {
    setSelectedRevisionId("working");
    setContent(currentPrompt?.content ?? "");
    setEditing(false);
  };
  const saveWorkingCopy = async () => {
    if (!draft || !node) return;
    await mutate("保存工作稿", () => jsonRequest(`/api/v1/admin/agent-versions/${draft.id}/nodes/${node.node_key}/prompt-working-copy`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    }));
    setSelectedRevisionId("working");
    setEditing(false);
  };
  const useRevisionAsWorkingCopy = async (revision: PromptRevision) => {
    if (!draft || !node) return;
    await mutate("建立工作稿", () => jsonRequest(`/api/v1/admin/agent-versions/${draft.id}/nodes/${node.node_key}/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_revision_id: revision.id }),
    }));
    setSelectedDefinitionId(revision.prompt_definition_id);
    setSelectedRevisionId("working");
    setEditing(false);
  };
  const discardWorkingCopy = async () => {
    if (!draft || !node || !currentPrompt?.working) return;
    if (!window.confirm("将丢弃当前未发布的 Prompt 修改并恢复到稳定版本。")) return;
    await mutate("丢弃工作稿", () => jsonRequest(`/api/v1/admin/agent-versions/${draft.id}/nodes/${node.node_key}/prompt-working-copy/discard`, {
      method: "POST",
    }));
    setSelectedRevisionId("working");
    setEditing(false);
  };

  if (!node || !definition) return <section className={styles.section}>暂无可管理的 Prompt。</section>;

  const promptDetail = showingWorking ? currentPrompt : selectedRevision;
  const validation = promptDetail?.validation_report;
  const skills = promptDetail?.skills ?? [];
  const requiredVariables = promptDetail?.required_variables ?? [];
  return <section className={styles.section}>
    <div className={styles.sectionHeading}>
      <div><span className={styles.eyebrow}>Prompt Experiments</span><h2>Prompt 工作稿与稳定历史</h2></div>
      {nodeDefinitions.length > 1 && <label className={styles.compatDefinition}><span>兼容 Definition</span><select value={definition.id} onChange={(event) => { setSelectedDefinitionId(event.target.value); setSelectedRevisionId(event.target.value === boundDefinition?.id ? "working" : ""); setEditing(false); }}>{nodeDefinitions.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
    </div>
    <div className={styles.promptWorkspace}>
      <div className={styles.promptNodeRail}>
        <span className={styles.railLabel}>固定节点</span>
        {version.nodes.map((item) => <button key={item.node_key} className={item.node_key === node.node_key ? styles.selectedRailItem : ""} onClick={() => { setSelectedNodeKey(item.node_key); setSelectedRevisionId("working"); setEditing(false); }}><strong>{NODE_LABELS[item.node_key] ?? item.node_key}</strong><small>{item.prompt_config.working ? "工作稿" : "稳定副本"} · 基于 v{item.prompt_version ?? "-"}</small></button>)}
      </div>
      <div className={styles.promptRevisionRail}>
        <div className={styles.railHeading}><History size={15} /><div><strong>{definition.name}</strong><small>{definition.slug}</small></div></div>
        {definition.id === boundDefinition?.id && <button className={`${styles.revisionItem} ${showingWorking ? styles.selectedRevision : ""}`} onClick={showWorkingCopy}><span><strong>{draft ? "当前工作稿" : "当前已发布"}</strong>{currentPrompt?.working && <em>Working</em>}</span><small>基于稳定 Revision v{node.prompt_version ?? "-"}</small></button>}
        {historyLoading && <div className={styles.railState}><LoaderCircle className={styles.spin} size={17} /></div>}
        {historyError && <div className={styles.railError}>{historyError}</div>}
        {!historyLoading && revisions.map((revision) => <button key={revision.id} className={`${styles.revisionItem} ${revision.id === selectedRevision?.id ? styles.selectedRevision : ""}`} onClick={() => selectRevision(revision)}><span><strong>稳定 v{revision.version}</strong>{revision.id === node.prompt_revision_id && <em>工作稿基线</em>}</span><small>{revision.id === definition.active_revision.id ? "当前稳定 Revision" : revision.source}</small></button>)}
      </div>
      <div className={styles.promptRevisionDetail}>
        {promptDetail && <>
          <div className={styles.revisionHeader}>
            <div><span className={styles.eyebrow}>{showingWorking ? "Working Prompt" : `Stable Revision v${selectedRevision?.version}`}</span><h3>{NODE_LABELS[node.node_key] ?? node.node_key}</h3></div>
            <div className={styles.revisionStatus}><span>{showingWorking ? (draft ? "DRAFT" : "PUBLISHED") : "READ ONLY"}</span>{showingWorking && currentPrompt?.working && <strong>工作稿</strong>}</div>
          </div>
          <dl className={styles.promptRevisionMeta}>
            <div><dt>Content Hash</dt><dd className={styles.mono}>{promptDetail.content_hash}</dd></div>
            <div><dt>基线</dt><dd>Revision v{showingWorking ? node.prompt_version : selectedRevision?.version}</dd></div>
            <div><dt>静态校验</dt><dd>{validation?.valid ? "通过" : "未通过"}</dd></div>
            <div><dt>输出契约</dt><dd>{validation?.output_schema ?? node.output_schema}</dd></div>
            <div><dt>契约边界</dt><dd>{validation?.output_schema_boundary ?? "code_owned"}</dd></div>
            <div><dt>Placeholder</dt><dd>{requiredVariables.length ? requiredVariables.join("、") : "无"}</dd></div>
            <div><dt>Skills</dt><dd>{skills.length ? skills.map((skill) => skill.name).join("、") : "无"}</dd></div>
            <div><dt>状态</dt><dd>{showingWorking ? "下一次新任务直接使用" : "稳定历史只读"}</dd></div>
          </dl>
          <textarea className={styles.promptEditor} value={content} readOnly={!showingWorking || !editing} onChange={(event) => setContent(event.target.value)} spellCheck={false} />
          {skills.length > 0 && <div className={styles.promptSkills}><h3>代码约束 Skills</h3>{skills.map((skill) => <details key={skill.revision_id}><summary>{skill.name}</summary><pre>{skill.content}</pre></details>)}</div>}
          <div className={styles.actions}>
            {!showingWorking && selectedRevision && <button className={styles.secondaryButton} disabled={Boolean(busy)} onClick={() => void mutate("校验稳定 Revision", () => jsonRequest(`/api/v1/admin/prompt-revisions/${selectedRevision.id}/validate`, { method: "POST" }))}><TestTube2 size={15} />重新校验</button>}
            {showingWorking && !editing && <button className={styles.secondaryButton} disabled={!draft || Boolean(busy)} onClick={() => setEditing(true)}><Pencil size={15} />编辑工作稿</button>}
            {showingWorking && editing && <button className={styles.primaryButton} disabled={!draft || Boolean(busy) || content === currentPrompt?.content} onClick={() => void saveWorkingCopy()}><Save size={15} />保存工作稿</button>}
            {showingWorking && currentPrompt?.working && <button className={styles.secondaryButton} disabled={!draft || Boolean(busy)} onClick={() => void discardWorkingCopy()}><Trash2 size={15} />丢弃工作稿</button>}
            {!showingWorking && selectedRevision && <button className={styles.primaryButton} title={draft ? "复制此稳定 Revision 到当前工作稿" : "请先创建 Agent Draft"} disabled={!draft || Boolean(busy)} onClick={() => void useRevisionAsWorkingCopy(selectedRevision)}><Link2 size={15} />以此版本建立工作稿</button>}
          </div>
        </>}
      </div>
    </div>
  </section>;
}
