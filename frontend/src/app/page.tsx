"use client";

import {
  AlertCircle,
  Check,
  LoaderCircle,
  MessageSquare,
  RotateCcw,
  Search,
  Send
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type InputType = "text" | "audio";
type ReportTab = "detailed" | "action";
type ChatMessage = { role: "user" | "assistant"; content: string };

type Person = { name?: string; organization?: string; title?: string };
type EntityMention = {
  mention: string;
  canonical_name?: string;
  resolution: "CONFIRMED" | "NEEDS_CONFIRMATION" | "MISSING";
};
type Task = {
  task_id: string;
  status: string;
  input_type: InputType;
  input_text?: string;
  extracted_info?: {
    event_type: string;
    event_time?: string;
    event_location?: string;
    people: Person[];
    keywords: string[];
  };
  llm_understanding?: {
    people: EntityMention[];
    organizations: EntityMention[];
  };
  web_search_status?: string;
  web_fetch_status?: string;
  internal_search_status?: string;
  confirmation_request?: {
    version: number;
    items: Array<{
      mention: string;
      entity_type: string;
      candidates: Array<{
        candidate_id: string;
        canonical_name: string;
        organization?: string;
        title?: string;
        region?: string;
        reason: string;
        confidence: number;
        source_url?: string;
        evidence_quote?: string;
      }>;
    }>;
  };
  detailed_report_markdown?: string;
  action_brief_markdown?: string;
  report_markdown?: string;
  degraded_nodes?: string[];
  error_message?: string;
};

type IntakeResponse = {
  session_id: string;
  assistant_reply: string;
  analysis_input: string;
  ready_to_analyze: boolean;
  missing_information: string[];
};

const INITIAL_MESSAGE: ChatMessage = {
  role: "assistant",
  content: "请告诉我这次要了解的人物、企业，以及准备讨论或推动的事情。"
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED", "NEEDS_CONFIRMATION"]);
const STATUS_LABELS: Record<string, string> = {
  PENDING: "任务已创建",
  TRANSCRIBING: "正在识别语音",
  CONTEXT_EXTRACTING: "正在识别人物与企业",
  EXTRACTING: "正在提取关键信息",
  RULE_EXTRACTING: "正在提取关键信息",
  LLM_UNDERSTANDING: "正在理解任务意图",
  RESOLVING_ENTITIES: "正在核对人物身份",
  NEEDS_CONFIRMATION: "需要补充或确认信息",
  PLANNING_WEB_SEARCH: "正在规划公开检索",
  WEB_SEARCHING: "正在搜索公开信息",
  WEB_FETCHING: "正在抓取网页正文",
  VERIFYING_WEB_RESULTS: "正在核验网页身份",
  PLANNING_PROJECT_SEARCH: "正在扩展项目条件",
  PROJECT_SEARCHING: "正在检索内部项目",
  RERANKING_PROJECTS: "正在评估项目价值",
  ANALYZING_ASSOCIATIONS: "正在关联公开与内部信息",
  GENERATING_REPORT_CONTENT: "正在生成报告内容",
  GENERATING: "正在生成总结",
  RENDERING_REPORT: "正在排版报告",
  COMPLETED: "分析完成",
  FAILED: "分析失败",
  CANCELLED: "任务已取消"
};
const STATUS_ORDER = [
  "PENDING",
  "TRANSCRIBING",
  "CONTEXT_EXTRACTING",
  "PLANNING_WEB_SEARCH",
  "WEB_SEARCHING",
  "WEB_FETCHING",
  "VERIFYING_WEB_RESULTS",
  "PLANNING_PROJECT_SEARCH",
  "PROJECT_SEARCHING",
  "RERANKING_PROJECTS",
  "ANALYZING_ASSOCIATIONS",
  "GENERATING_REPORT_CONTENT",
  "RENDERING_REPORT",
  "COMPLETED"
];

function safeReportUrl(url: string) {
  return /^https?:\/\//i.test(url) ? url : "";
}

export default function Home() {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE]);
  const [chatInput, setChatInput] = useState("");
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [analysisInput, setAnalysisInput] = useState("");
  const [readyToAnalyze, setReadyToAnalyze] = useState(false);
  const [missingInformation, setMissingInformation] = useState<string[]>([]);
  const [isChatting, setIsChatting] = useState(false);
  const [task, setTask] = useState<Task | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [reportTab, setReportTab] = useState<ReportTab>("detailed");
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [manualValues, setManualValues] = useState<Record<string, string>>({});
  const chatThreadRef = useRef<HTMLDivElement | null>(null);

  const fetchTask = useCallback(async (taskId: string) => {
    const response = await fetch(`${API_BASE}/api/v1/tasks/${taskId}`);
    if (!response.ok) throw new Error("任务状态获取失败");
    const nextTask = (await response.json()) as Task;
    setTask(nextTask);
    return nextTask;
  }, []);

  useEffect(() => {
    if (!task || TERMINAL.has(task.status)) return;
    const interval = setInterval(() => {
      fetchTask(task.task_id).catch((reason) => setError(reason.message));
    }, 2000);
    return () => clearInterval(interval);
  }, [fetchTask, task]);

  useEffect(() => {
    chatThreadRef.current?.scrollTo({ top: chatThreadRef.current.scrollHeight });
  }, [chatMessages, isChatting]);

  const sendChatMessage = async () => {
    const content = chatInput.trim();
    if (!content || isChatting) return;
    const nextMessages: ChatMessage[] = [...chatMessages, { role: "user", content }];
    setChatMessages(nextMessages);
    setChatInput("");
    setError("");
    setIsChatting(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/intake/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...(chatSessionId ? { session_id: chatSessionId } : {}),
          messages: nextMessages
        })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "对话助手暂时不可用");
      }
      const payload = (await response.json()) as IntakeResponse;
      setChatSessionId(payload.session_id);
      setAnalysisInput(payload.analysis_input);
      setReadyToAnalyze(payload.ready_to_analyze);
      setMissingInformation(payload.missing_information);
      setChatMessages((current) => [
        ...current,
        { role: "assistant", content: payload.assistant_reply }
      ]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "对话助手暂时不可用");
    } finally {
      setIsChatting(false);
    }
  };

  const startAnalysis = async () => {
    if (!analysisInput.trim()) {
      setError("请先通过对话提供本次分析的信息");
      return;
    }
    setError("");
    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/tasks/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: analysisInput.trim() })
      });
      if (!response.ok) throw new Error("任务创建失败");
      setTask(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务创建失败");
    } finally {
      setIsSubmitting(false);
    }
  };

  const reset = () => {
    setTask(null);
    setError("");
    setChatMessages([INITIAL_MESSAGE]);
    setChatInput("");
    setChatSessionId(null);
    setAnalysisInput("");
    setReadyToAnalyze(false);
    setMissingInformation([]);
    setSelections({});
    setManualValues({});
    setReportTab("detailed");
  };

  const confirmEntities = async () => {
    if (!task?.confirmation_request) return;
    const missing = task.confirmation_request.items.find(
      (item) => !selections[item.mention] && !manualValues[item.mention]?.trim()
    );
    if (missing) {
      setError(`请选择或填写“${missing.mention}”`);
      return;
    }
    setIsSubmitting(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/v1/tasks/${task.task_id}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmation_version: task.confirmation_request.version,
          selections: task.confirmation_request.items.map((item) => ({
            mention: item.mention,
            candidate_id: selections[item.mention] || null,
            manual_value: selections[item.mention] ? null : manualValues[item.mention]?.trim()
          }))
        })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "人物确认提交失败");
      }
      setTask(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "人物确认提交失败");
    } finally {
      setIsSubmitting(false);
    }
  };

  const visibleStatuses = task?.input_type === "text"
    ? STATUS_ORDER.filter((status) => status !== "TRANSCRIBING")
    : STATUS_ORDER;
  const visibleCurrentIndex = task ? visibleStatuses.indexOf(task.status) : -1;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><Search size={18} strokeWidth={2.2} /></div>
        <div>
          <h1>资源推动 Agent</h1>
          <p>企业人物与项目资源调查</p>
        </div>
        {task && (
          <button className="icon-button reset-button" onClick={reset} title="新建调查">
            <RotateCcw size={17} />
          </button>
        )}
      </header>

      <div className="workspace">
        <aside className="progress-panel" aria-label="处理进度">
          <div className="section-label">处理进度</div>
          <ol className="steps">
            {!task && (
              <li className="active">
                <span className="step-dot">1</span>
                <span>对话收集信息</span>
              </li>
            )}
            {task && visibleStatuses.slice(0, -1).map((status, index) => {
              const done = task?.status === "COMPLETED" || visibleCurrentIndex > index;
              const active = task?.status === status;
              return (
                <li key={status} className={done ? "done" : active ? "active" : ""}>
                  <span className="step-dot">{done ? <Check size={13} /> : index + 1}</span>
                  <span>{STATUS_LABELS[status]}</span>
                </li>
              );
            })}
          </ol>
        </aside>

        <section className="main-column">
          {!task && (
            <div className="input-panel chat-panel">
              <div className="panel-heading">
                <div>
                  <span className="section-label">信息采集</span>
                  <h2>会前调查对话</h2>
                </div>
                <div className={`intake-status ${readyToAnalyze ? "ready" : ""}`}>
                  {readyToAnalyze ? <Check size={14} /> : <MessageSquare size={14} />}
                  {readyToAnalyze ? "信息已基本齐全" : "正在收集信息"}
                </div>
              </div>

              <div className="chat-thread" ref={chatThreadRef} aria-live="polite">
                {chatMessages.map((message, index) => (
                  <div key={`${message.role}-${index}`} className={`chat-row ${message.role}`}>
                    <div className="chat-bubble">{message.content}</div>
                  </div>
                ))}
                {isChatting && (
                  <div className="chat-row assistant">
                    <div className="chat-bubble chat-typing"><LoaderCircle className="spin" size={15} />正在整理</div>
                  </div>
                )}
              </div>

              <div className="chat-composer">
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void sendChatMessage();
                    }
                  }}
                  maxLength={2000}
                  rows={3}
                  placeholder="输入本次会面或项目的信息"
                  disabled={isChatting || isSubmitting}
                />
                <button
                  className="send-button"
                  onClick={sendChatMessage}
                  disabled={!chatInput.trim() || isChatting || isSubmitting}
                  title="发送"
                >
                  <Send size={18} />
                </button>
              </div>

              <div className="chat-actions">
                <div className="intake-hint">
                  {missingInformation.length > 0
                    ? `待补充：${missingInformation.join("、")}`
                    : analysisInput
                      ? "可以继续补充，也可以立即开始分析"
                      : "发送第一条消息后可开始分析"}
                </div>
                <button className="primary-button" disabled={!analysisInput || isSubmitting || isChatting} onClick={startAnalysis}>
                  {isSubmitting ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}
                  立即分析
                </button>
              </div>
              {error && <div className="error-banner"><AlertCircle size={17} />{error}</div>}
            </div>
          )}

          {task && (
            <>
              <div className={`status-strip ${task.status.toLowerCase()}`}>
                {task.status === "COMPLETED" ? <Check size={18} /> : task.status === "FAILED" ? <AlertCircle size={18} /> : <LoaderCircle className="spin" size={18} />}
                <div>
                  <strong>{STATUS_LABELS[task.status] ?? task.status}</strong>
                  <span>任务 {task.task_id.slice(0, 8)}</span>
                </div>
              </div>

              {task.extracted_info && (
                <div className="facts-band">
                  <div><span>活动</span><strong>{task.extracted_info.event_type}</strong></div>
                  <div><span>时间</span><strong>{task.extracted_info.event_time ?? "未识别"}</strong></div>
                  <div>
                    <span>人物</span>
                    <strong>
                      {task.llm_understanding?.people.map((person) =>
                        `${person.canonical_name ?? person.mention}${person.resolution === "NEEDS_CONFIRMATION" ? "（待确认）" : ""}`
                      ).join("、") || "未识别"}
                    </strong>
                  </div>
                  <div>
                    <span>企业</span>
                    <strong>
                      {task.llm_understanding?.organizations.map((organization) =>
                        organization.canonical_name ?? organization.mention
                      ).join("、") || "未识别"}
                    </strong>
                  </div>
                </div>
              )}

              {task.status === "FAILED" && (
                <div className="error-banner"><AlertCircle size={17} />{task.error_message ?? "任务处理失败"}</div>
              )}

              {task.status === "NEEDS_CONFIRMATION" && error && (
                <div className="error-banner"><AlertCircle size={17} />{error}</div>
              )}

              {task.status === "NEEDS_CONFIRMATION" && task.confirmation_request && (
                <section className="confirmation-panel">
                  <div className="section-label">关键信息确认</div>
                  <h2>请确认候选项，或直接填写缺失的人物与企业信息</h2>
                  {task.confirmation_request.items.map((item) => (
                    <fieldset key={item.mention}>
                      <legend>{item.mention}</legend>
                      {item.candidates.map((candidate) => (
                        <label key={candidate.candidate_id} className="candidate-option">
                          <input
                            type="radio"
                            name={item.mention}
                            value={candidate.candidate_id}
                            checked={selections[item.mention] === candidate.candidate_id}
                            onChange={() => {
                              setSelections((current) => ({ ...current, [item.mention]: candidate.candidate_id }));
                              setManualValues((current) => ({ ...current, [item.mention]: "" }));
                            }}
                          />
                          <span>
                            <strong>{candidate.canonical_name}</strong>
                            <small>{[candidate.organization, candidate.title, candidate.region].filter(Boolean).join("｜")}</small>
                            <small>{candidate.reason}｜置信度 {(candidate.confidence * 100).toFixed(0)}%</small>
                            {candidate.evidence_quote && <small>依据：{candidate.evidence_quote}</small>}
                            {candidate.source_url && <small className="candidate-source">来源：{candidate.source_url}</small>}
                          </span>
                        </label>
                      ))}
                      <label className="manual-entry">
                        <span>手工填写{item.entity_type === "PERSON" ? "人物姓名" : "企业名称"}</span>
                        <input
                          type="text"
                          value={manualValues[item.mention] ?? ""}
                          placeholder={item.entity_type === "PERSON" ? "例如：王传福" : "例如：比亚迪股份有限公司"}
                          onChange={(event) => {
                            const value = event.target.value;
                            setManualValues((current) => ({ ...current, [item.mention]: value }));
                            if (value) {
                              setSelections((current) => ({ ...current, [item.mention]: "" }));
                            }
                          }}
                        />
                      </label>
                    </fieldset>
                  ))}
                  <button className="primary-button" disabled={isSubmitting} onClick={confirmEntities}>
                    {isSubmitting ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />}
                    确认并继续
                  </button>
                </section>
              )}

              {(task.detailed_report_markdown || task.report_markdown) && (
                <>
                  <div className="report-tabs" role="tablist">
                    <button className={reportTab === "detailed" ? "selected" : ""} onClick={() => setReportTab("detailed")}>详细报告</button>
                    <button className={reportTab === "action" ? "selected" : ""} onClick={() => setReportTab("action")}>行动说明</button>
                  </div>
                  {task.degraded_nodes && task.degraded_nodes.length > 0 && (
                    <div className="degraded-banner">部分智能分析已降级，报告保留可验证的规则、网页和内部项目结果。</div>
                  )}
                  <article className="report">
                    <ReactMarkdown urlTransform={safeReportUrl}>
                      {reportTab === "action" ? task.action_brief_markdown ?? "暂无行动说明" : task.detailed_report_markdown ?? task.report_markdown ?? ""}
                    </ReactMarkdown>
                  </article>
                </>
              )}
            </>
          )}
        </section>
      </div>
    </main>
  );
}
