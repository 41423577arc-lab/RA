"use client";

import {
  AlertCircle,
  Bot,
  Check,
  Database,
  Globe2,
  LoaderCircle,
  Mic,
  MessageSquare,
  RefreshCw,
  RotateCcw,
  Search,
  Send
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type InputType = "text" | "audio";
type ReportTab = "detailed" | "action";
type ChatMessage = { role: "user" | "assistant"; content: string };
type IntakeActivity = {
  session_id?: string;
  phase: "IDLE" | "THINKING" | "CHECKING_CONTEXT" | "CALLING_TOOL" | "PROCESSING_TOOL_RESULT" | "COMPLETED" | "FAILED";
  detail: string;
  active: boolean;
  tool_name?: string;
  sequence: number;
};
type IntakeAudioJob = {
  job_id: string;
  session_id: string;
  status: "QUEUED" | "TRANSCRIBING" | "NEEDS_REVIEW" | "TRANSCRIBED" | "FAILED";
  transcript?: string;
  corrected_transcript?: string;
  error_message?: string;
  retry_count: number;
};

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
  status: "COLLECTING" | "PROCESSING_AUDIO" | "NEEDS_CONFIRMATION" | "READY" | "STARTING_ANALYSIS" | "ANALYZING";
  version: number;
  messages?: ChatMessage[];
  research_task_id?: string;
  confirmation_request?: Task["confirmation_request"];
  active_audio_job?: IntakeAudioJob;
};

const INITIAL_MESSAGE: ChatMessage = {
  role: "assistant",
  content: "请告诉我这次要了解的人物、企业，以及准备讨论或推动的事情。"
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const INTAKE_SESSION_STORAGE_KEY = "resource-agent-intake-session-id";
const IDLE_ACTIVITY: IntakeActivity = {
  phase: "IDLE",
  detail: "大模型待命",
  active: false,
  sequence: 0
};
const TOOL_LABELS: Record<string, string> = {
  lookup_internal_identity: "内部身份查询",
  search_key_person_identity_web: "联网身份补全"
};
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
  PLANNING_PROJECT_SEARCH: "正在准备内部项目检索",
  PROJECT_SEARCHING: "正在检索内部项目",
  RERANKING_PROJECTS: "正在评估项目价值",
  ANALYZING_ASSOCIATIONS: "正在关联关键人与内部项目",
  GENERATING_REPORT_CONTENT: "正在生成报告内容",
  GENERATING: "正在生成总结",
  RENDERING_REPORT: "正在排版报告",
  COMPLETED: "分析完成",
  FAILED: "分析失败",
  CANCELLED: "任务已取消"
};
const CURRENT_STATUS_ORDER = [
  "PENDING",
  "TRANSCRIBING",
  "CONTEXT_EXTRACTING",
  "PLANNING_PROJECT_SEARCH",
  "PROJECT_SEARCHING",
  "RERANKING_PROJECTS",
  "ANALYZING_ASSOCIATIONS",
  "GENERATING_REPORT_CONTENT",
  "RENDERING_REPORT",
  "COMPLETED"
];
const LEGACY_STATUS_ORDER = [
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
const LEGACY_WEB_STATUSES = new Set([
  "PLANNING_WEB_SEARCH",
  "WEB_SEARCHING",
  "WEB_FETCHING",
  "VERIFYING_WEB_RESULTS"
]);

function safeReportUrl(url: string) {
  return /^https?:\/\//i.test(url) ? url : "";
}

export default function Home() {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE]);
  const [chatInput, setChatInput] = useState("");
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [chatSessionVersion, setChatSessionVersion] = useState<number | null>(null);
  const [analysisInput, setAnalysisInput] = useState("");
  const [readyToAnalyze, setReadyToAnalyze] = useState(false);
  const [missingInformation, setMissingInformation] = useState<string[]>([]);
  const [intakeConfirmationRequest, setIntakeConfirmationRequest] = useState<Task["confirmation_request"]>();
  const [audioJob, setAudioJob] = useState<IntakeAudioJob>();
  const [audioTranscript, setAudioTranscript] = useState("");
  const [isChatting, setIsChatting] = useState(false);
  const [intakeActivity, setIntakeActivity] = useState<IntakeActivity>(IDLE_ACTIVITY);
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
    const sessionId = window.localStorage.getItem(INTAKE_SESSION_STORAGE_KEY);
    if (!sessionId) return;
    fetch(`${API_BASE}/api/v1/intake/${sessionId}`)
      .then(async (response) => {
        if (!response.ok) {
          window.localStorage.removeItem(INTAKE_SESSION_STORAGE_KEY);
          return null;
        }
        return (await response.json()) as IntakeResponse;
      })
      .then((payload) => {
        if (!payload) return;
        setChatSessionId(payload.session_id);
        setChatSessionVersion(payload.version);
        setAnalysisInput(payload.analysis_input);
        setReadyToAnalyze(payload.ready_to_analyze);
        setMissingInformation(payload.missing_information);
        setIntakeConfirmationRequest(payload.confirmation_request);
        if (payload.active_audio_job) {
          setAudioJob(payload.active_audio_job);
          setAudioTranscript(payload.active_audio_job.transcript ?? "");
        }
        if (payload.messages?.length) setChatMessages(payload.messages);
        if (payload.research_task_id) {
          void fetchTask(payload.research_task_id).catch((reason) => setError(reason.message));
        }
      })
      .catch(() => window.localStorage.removeItem(INTAKE_SESSION_STORAGE_KEY));
  }, [fetchTask]);

  useEffect(() => {
    if (!chatSessionId || !audioJob || !new Set(["QUEUED", "TRANSCRIBING"]).has(audioJob.status)) return;
    const interval = window.setInterval(() => {
      fetch(`${API_BASE}/api/v1/intake/${chatSessionId}/audio/${audioJob.job_id}`)
        .then(async (response) => {
          if (!response.ok) throw new Error("音频状态获取失败");
          return (await response.json()) as IntakeAudioJob;
        })
        .then((job) => {
          setAudioJob(job);
          if (job.transcript) setAudioTranscript(job.transcript);
        })
        .catch((reason) => setError(reason.message));
    }, 1500);
    return () => window.clearInterval(interval);
  }, [audioJob, chatSessionId]);

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

  useEffect(() => {
    if (!chatSessionId || !isChatting) return;
    let cancelled = false;
    const fetchActivity = async () => {
      const response = await fetch(`${API_BASE}/api/v1/intake/${chatSessionId}/activity`);
      if (!response.ok) return;
      const activity = (await response.json()) as IntakeActivity;
      if (!cancelled) setIntakeActivity(activity);
    };
    void fetchActivity().catch(() => undefined);
    const interval = window.setInterval(() => {
      void fetchActivity().catch(() => undefined);
    }, 400);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [chatSessionId, isChatting]);

  const sendChatMessage = async (contentOverride?: string, audioJobId?: string) => {
    const content = (contentOverride ?? chatInput).trim();
    if (!content || isChatting) return;
    const nextMessages: ChatMessage[] = [...chatMessages, { role: "user", content }];
    setChatMessages(nextMessages);
    setChatInput("");
    setError("");
    setIsChatting(true);
    const requestSessionId = chatSessionId ?? window.crypto.randomUUID();
    if (!chatSessionId) setChatSessionId(requestSessionId);
    setIntakeActivity({
      session_id: requestSessionId,
      phase: "THINKING",
      detail: "正在提交对话",
      active: true,
      sequence: 0
    });
    try {
      const response = await fetch(`${API_BASE}/api/v1/intake/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: requestSessionId,
          messages: nextMessages,
          ...(audioJobId ? { audio_job_id: audioJobId } : {})
        })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "对话助手暂时不可用");
      }
      const payload = (await response.json()) as IntakeResponse;
      setChatSessionId(payload.session_id);
      setChatSessionVersion(payload.version);
      window.localStorage.setItem(INTAKE_SESSION_STORAGE_KEY, payload.session_id);
      setAnalysisInput(payload.analysis_input);
      setReadyToAnalyze(payload.ready_to_analyze);
      setMissingInformation(payload.missing_information);
      setIntakeConfirmationRequest(payload.confirmation_request);
      setChatMessages((current) => [
        ...current,
        { role: "assistant", content: payload.assistant_reply }
      ]);
      setIntakeActivity({
        session_id: payload.session_id,
        phase: "COMPLETED",
        detail: "本轮对话处理完成",
        active: false,
        sequence: intakeActivity.sequence + 1
      });
      if (audioJobId) {
        setAudioJob(undefined);
        setAudioTranscript("");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "对话助手暂时不可用");
      setIntakeActivity({
        session_id: requestSessionId,
        phase: "FAILED",
        detail: "本轮对话处理失败",
        active: false,
        sequence: intakeActivity.sequence + 1
      });
    } finally {
      setIsChatting(false);
    }
  };

  const startAnalysis = async () => {
    if (!chatSessionId || !readyToAnalyze || !analysisInput.trim()) {
      setError("请先通过对话提供本次分析的信息");
      return;
    }
    setError("");
    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/intake/${chatSessionId}/start-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_version: chatSessionVersion })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? "任务创建失败");
      }
      setTask(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务创建失败");
    } finally {
      setIsSubmitting(false);
    }
  };

  const reset = () => {
    if (
      (chatSessionId || task) &&
      !window.confirm("将清空当前页面并新建调查；已经创建的分析任务不会被取消。是否继续？")
    ) {
      return;
    }
    setTask(null);
    setError("");
    setChatMessages([INITIAL_MESSAGE]);
    setChatInput("");
    setChatSessionId(null);
    setChatSessionVersion(null);
    window.localStorage.removeItem(INTAKE_SESSION_STORAGE_KEY);
    setAnalysisInput("");
    setReadyToAnalyze(false);
    setMissingInformation([]);
    setIntakeConfirmationRequest(undefined);
    setAudioJob(undefined);
    setAudioTranscript("");
    setSelections({});
    setManualValues({});
    setReportTab("detailed");
    setIntakeActivity(IDLE_ACTIVITY);
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

  const confirmIntakeEntities = async () => {
    if (!chatSessionId || !intakeConfirmationRequest) return;
    const missing = intakeConfirmationRequest.items.find(
      (item) => !selections[item.mention] && !manualValues[item.mention]?.trim()
    );
    if (missing) {
      setError(`请选择或填写“${missing.mention}”`);
      return;
    }
    setIsSubmitting(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/v1/intake/${chatSessionId}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmation_version: intakeConfirmationRequest.version,
          selections: intakeConfirmationRequest.items.map((item) => ({
            mention: item.mention,
            candidate_id: selections[item.mention] || null,
            manual_value: selections[item.mention] ? null : manualValues[item.mention]?.trim()
          }))
        })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail ?? "身份确认提交失败");
      const intake = payload as IntakeResponse;
      setChatSessionVersion(intake.version);
      setAnalysisInput(intake.analysis_input);
      setReadyToAnalyze(intake.ready_to_analyze);
      setMissingInformation(intake.missing_information);
      setIntakeConfirmationRequest(intake.confirmation_request);
      if (intake.messages?.length) setChatMessages(intake.messages);
      setSelections({});
      setManualValues({});
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "身份确认提交失败");
    } finally {
      setIsSubmitting(false);
    }
  };

  const uploadAudio = async (file: File) => {
    const sessionId = chatSessionId ?? crypto.randomUUID();
    setChatSessionId(sessionId);
    window.localStorage.setItem(INTAKE_SESSION_STORAGE_KEY, sessionId);
    setError("");
    const form = new FormData();
    form.append("audio", file, file.name || "recording.webm");
    try {
      const response = await fetch(`${API_BASE}/api/v1/intake/${sessionId}/audio`, {
        method: "POST",
        body: form
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail ?? "音频上传失败");
      setAudioJob(payload as IntakeAudioJob);
      setReadyToAnalyze(false);
      setMissingInformation(["等待音频转写和确认"]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "音频上传失败");
    }
  };

  const retryAudio = async () => {
    if (!chatSessionId || !audioJob) return;
    const response = await fetch(
      `${API_BASE}/api/v1/intake/${chatSessionId}/audio/${audioJob.job_id}/retry`,
      { method: "POST" }
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setError(payload.detail ?? "音频重试失败");
      return;
    }
    setAudioJob(payload as IntakeAudioJob);
  };

  const usesLegacyWebProgress = Boolean(
    task && (
      LEGACY_WEB_STATUSES.has(task.status)
      || task.web_search_status === "SUCCESS"
      || task.web_search_status === "FAILED"
    )
  );
  const statusOrder = usesLegacyWebProgress ? LEGACY_STATUS_ORDER : CURRENT_STATUS_ORDER;
  const visibleStatuses = task?.input_type === "text"
    ? statusOrder.filter((status) => status !== "TRANSCRIBING")
    : statusOrder;
  const visibleCurrentIndex = task ? visibleStatuses.indexOf(task.status) : -1;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><Search size={18} strokeWidth={2.2} /></div>
        <div>
          <h1>资源推动 Agent</h1>
          <p>企业人物与项目资源调查</p>
        </div>
        <button className="reset-button" onClick={reset} title="清空当前页面并新建调查">
          <RotateCcw size={16} />
          <span>重新开始</span>
        </button>
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

              <div
                className={`agent-activity ${intakeActivity.active ? "active" : ""} ${intakeActivity.phase.toLowerCase()}`}
                aria-live="polite"
              >
                <span className="activity-icon" aria-hidden="true">
                  {intakeActivity.active ? (
                    intakeActivity.tool_name === "lookup_internal_identity" ? <Database size={15} />
                      : intakeActivity.tool_name === "search_key_person_identity_web" ? <Globe2 size={15} />
                        : <LoaderCircle className="spin" size={15} />
                  ) : intakeActivity.phase === "COMPLETED" ? <Check size={15} /> : <Bot size={15} />}
                </span>
                <span className="activity-copy">
                  <strong>{intakeActivity.detail}</strong>
                  {intakeActivity.tool_name && (
                    <small>{TOOL_LABELS[intakeActivity.tool_name] ?? intakeActivity.tool_name}</small>
                  )}
                </span>
                <span className="activity-state">{intakeActivity.active ? "运行中" : intakeActivity.phase === "COMPLETED" ? "已完成" : "待命"}</span>
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

              <div className="audio-controls">
                <label className="icon-button" title="上传 WebM 录音">
                  <Mic size={17} />
                  <input
                    type="file"
                    accept="audio/webm"
                    hidden
                    disabled={isSubmitting || Boolean(audioJob)}
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void uploadAudio(file);
                      event.target.value = "";
                    }}
                  />
                </label>
                {audioJob && <span>音频：{audioJob.status}</span>}
                {audioJob?.status === "FAILED" && (
                  <button className="icon-button" onClick={() => void retryAudio()} title="重试转写">
                    <RefreshCw size={17} />
                  </button>
                )}
              </div>

              {audioJob?.status === "NEEDS_REVIEW" && (
                <div className="audio-review">
                  <textarea
                    value={audioTranscript}
                    onChange={(event) => setAudioTranscript(event.target.value)}
                    rows={4}
                    maxLength={10000}
                  />
                  <button
                    className="primary-button"
                    disabled={!audioTranscript.trim() || isChatting}
                    onClick={() => void sendChatMessage(audioTranscript, audioJob.job_id)}
                  >
                    <Check size={17} />
                    确认转写
                  </button>
                </div>
              )}

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
                  onClick={() => void sendChatMessage()}
                  disabled={!chatInput.trim() || isChatting || isSubmitting}
                  title="发送"
                >
                  <Send size={18} />
                </button>
              </div>

              {intakeConfirmationRequest && (
                <section className="confirmation-panel">
                  <div className="section-label">身份确认</div>
                  {intakeConfirmationRequest.items.map((item) => (
                    <fieldset key={item.mention}>
                      <legend>{item.mention}</legend>
                      {item.candidates.map((candidate) => (
                        <label key={candidate.candidate_id} className="candidate-option">
                          <input
                            type="radio"
                            name={`intake-${item.mention}`}
                            checked={selections[item.mention] === candidate.candidate_id}
                            onChange={() => {
                              setSelections((current) => ({ ...current, [item.mention]: candidate.candidate_id }));
                              setManualValues((current) => ({ ...current, [item.mention]: "" }));
                            }}
                          />
                          <span>
                            <strong>{candidate.canonical_name}</strong>
                            <small>{[candidate.organization, candidate.title, candidate.region].filter(Boolean).join(" · ")}</small>
                            <small>{candidate.reason}</small>
                            {candidate.evidence_quote && <small>依据：{candidate.evidence_quote}</small>}
                            {candidate.source_url && <small className="candidate-source">来源：{candidate.source_url}</small>}
                          </span>
                        </label>
                      ))}
                      <label className="manual-entry">
                        <span>手工填写确认名称</span>
                        <input
                          type="text"
                          value={manualValues[item.mention] ?? ""}
                          onChange={(event) => {
                            const value = event.target.value;
                            setManualValues((current) => ({ ...current, [item.mention]: value }));
                            if (value) setSelections((current) => ({ ...current, [item.mention]: "" }));
                          }}
                        />
                      </label>
                    </fieldset>
                  ))}
                  <button className="primary-button" disabled={isSubmitting} onClick={confirmIntakeEntities}>
                    {isSubmitting ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />}
                    确认身份
                  </button>
                </section>
              )}

              <div className="chat-actions">
                <div className="intake-hint">
                  {missingInformation.length > 0
                    ? `待补充：${missingInformation.join("、")}`
                    : readyToAnalyze
                      ? "可以继续补充，也可以立即开始分析"
                      : "正在确认信息是否完整"}
                </div>
                <button className="primary-button" disabled={!readyToAnalyze || !analysisInput || isSubmitting || isChatting} onClick={startAnalysis}>
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
