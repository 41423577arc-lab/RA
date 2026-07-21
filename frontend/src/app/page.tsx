"use client";

import {
  AlertCircle,
  Check,
  FileText,
  LoaderCircle,
  Mic,
  RotateCcw,
  Search,
  Square,
  X
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type Mode = "text" | "audio";
type ReportTab = "detailed" | "action";

type Person = { name?: string; organization?: string; title?: string };
type EntityMention = {
  mention: string;
  canonical_name?: string;
  needs_confirmation?: boolean;
};
type Task = {
  task_id: string;
  status: string;
  input_type: Mode;
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

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function safeReportUrl(url: string) {
  return /^https?:\/\//i.test(url) ? url : "";
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState("");
  const [task, setTask] = useState<Task | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [error, setError] = useState("");
  const [reportTab, setReportTab] = useState<ReportTab>("detailed");
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [manualValues, setManualValues] = useState<Record<string, string>>({});
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const cancelledRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const createTextTask = async () => {
    if (!text.trim()) {
      setError("请输入要分析的内容");
      return;
    }
    setError("");
    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/tasks/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() })
      });
      if (!response.ok) throw new Error("任务创建失败");
      setTask(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务创建失败");
    } finally {
      setIsSubmitting(false);
    }
  };

  const uploadRecording = async (blob: Blob) => {
    setIsSubmitting(true);
    setError("");
    const body = new FormData();
    body.append("audio", blob, "recording.webm");
    try {
      const response = await fetch(`${API_BASE}/api/v1/tasks/audio`, {
        method: "POST",
        body
      });
      if (!response.ok) throw new Error("语音上传失败，请重新录制");
      setTask(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "语音上传失败，请重新录制");
    } finally {
      setIsSubmitting(false);
    }
  };

  const stopTracks = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    setIsRecording(false);
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      setError("当前浏览器不支持网页录音，请使用最新版桌面端 Chrome");
      return;
    }
    if (!MediaRecorder.isTypeSupported("audio/webm")) {
      setError("当前浏览器不支持 WebM 录音，请使用最新版桌面端 Chrome");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];
      cancelledRef.current = false;
      setRecordingSeconds(0);
      setError("");
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        stopTracks();
        if (!cancelledRef.current && blob.size) void uploadRecording(blob);
      };
      recorder.start(1000);
      setIsRecording(true);
      timerRef.current = setInterval(() => {
        setRecordingSeconds((current) => {
          if (current + 1 >= 900) recorder.stop();
          return Math.min(current + 1, 900);
        });
      }, 1000);
    } catch {
      setError("无法使用麦克风，请在浏览器设置中允许麦克风权限");
      stopTracks();
    }
  };

  const stopRecording = () => {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  };

  const cancelRecording = () => {
    cancelledRef.current = true;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  };

  const reset = () => {
    setTask(null);
    setError("");
    setRecordingSeconds(0);
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
            {visibleStatuses.slice(0, -1).map((status, index) => {
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
            <div className="input-panel">
              <div className="panel-heading">
                <div>
                  <span className="section-label">新建调查</span>
                  <h2>录入活动信息</h2>
                </div>
                <div className="mode-control" role="tablist">
                  <button disabled={isRecording || isSubmitting} className={mode === "text" ? "selected" : ""} onClick={() => setMode("text")}>
                    <FileText size={16} />文字
                  </button>
                  <button disabled={isRecording || isSubmitting} className={mode === "audio" ? "selected" : ""} onClick={() => setMode("audio")}>
                    <Mic size={16} />语音
                  </button>
                </div>
              </div>

              {mode === "text" ? (
                <div className="text-entry">
                  <textarea
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    maxLength={10000}
                    placeholder="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。"
                  />
                  <div className="entry-footer">
                    <span>{text.length.toLocaleString()} / 10,000</span>
                    <button className="primary-button" disabled={isSubmitting} onClick={createTextTask}>
                      {isSubmitting ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}
                      开始分析
                    </button>
                  </div>
                </div>
              ) : (
                <div className={`voice-entry ${isRecording ? "recording" : ""}`}>
                  <button
                    className="record-button"
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isSubmitting}
                    title={isRecording ? "停止并上传" : "开始录音"}
                  >
                    {isRecording ? <Square size={25} fill="currentColor" /> : <Mic size={29} />}
                  </button>
                  <strong>{isRecording ? "正在录音" : isSubmitting ? "正在上传" : "准备录音"}</strong>
                  <span className="timer">{formatDuration(recordingSeconds)}</span>
                  {isRecording && (
                    <button className="cancel-button" onClick={cancelRecording} title="取消录音">
                      <X size={16} />取消
                    </button>
                  )}
                </div>
              )}
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
                        `${person.canonical_name ?? person.mention}${person.needs_confirmation ? "（待确认）" : ""}`
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
