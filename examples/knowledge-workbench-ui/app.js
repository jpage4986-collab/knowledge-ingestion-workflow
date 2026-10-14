const state = {
  datasets: [],
  sandboxes: [],
  summary: null,
  selectedDataset: "",
  selectedSandbox: "",
  documents: [],
  selectedDocument: null,
  page: 1,
  pages: 1,
  total: 0,
  query: "",
  run: "",
  loading: false,
  candidates: [],
  selectedFiles: [],
  monitorToken: 0,
  jobPollToken: 0,
  ingestionJob: null,
  importing: false,
  discovering: false,
  modelGate: null,
  restoredJobs: [],
};

const runLabels = {
  DONE: "解析完成",
  RUNNING: "处理中",
  UNSTART: "待解析",
  FAIL: "解析失败",
  CANCEL: "已取消",
  NO_CHUNKS: "无有效分块",
  UNKNOWN: "未知状态",
};

async function requestJson(url) {
  const response = await fetch(url, { method: "GET", headers: { Accept: "application/json" }, cache: "no-store" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function postJson(url, data) {
  const response = await fetch(url, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(data),
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function number(value) {
  return new Intl.NumberFormat("zh-CN").format(Number(value) || 0);
}

function dateTime(value) {
  const timestamp = Number(value);
  if (!timestamp) return "—";
  const date = new Date(timestamp < 10_000_000_000 ? timestamp * 1000 : timestamp);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
}

function fileSize(value) {
  const size = Number(value) || 0;
  if (size < 1024) return `${number(size)} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 ** 2).toFixed(1)} MB`;
}

function statusName(run) {
  const key = String(run || "UNKNOWN").toUpperCase();
  return runLabels[key] || key;
}

function statusClass(run) {
  const key = String(run || "UNKNOWN").toUpperCase();
  if (key === "DONE") return "done";
  if (key === "RUNNING") return "running";
  if (key === "FAIL" || key === "NO_CHUNKS") return "failed";
  return "waiting";
}

function setConnection(element, stateName, text) {
  element.dataset.state = stateName;
  element.querySelector("small, span:last-child")?.replaceChildren(document.createTextNode(text));
}

function renderConnection(status) {
  const ragflow = status.ragflow || {};
  const mineru = status.mineru || {};
  const gate = status.model_gate || {};
  const connected = ragflow.status === "online";
  state.modelGate = gate;
  const topbar = document.querySelector("#topbarStatus");
  topbar.dataset.state = connected ? "online" : "offline";
  document.querySelector("#topbarStatusText").textContent = connected
    ? `RAGFlow ${ragflow.version || "在线"} · ${ragflow.dataset_count} 个知识库`
    : `RAGFlow 未连接 · ${ragflow.message || "检查服务连接"}`;
  document.querySelector("#sidebarDot").className = `mode-dot ${connected ? "is-online" : "is-offline"}`;
  document.querySelector("#sidebarConnection").textContent = connected ? "RAGFlow 已连接" : "连接异常";
  document.querySelector("#sidebarScope").textContent = connected ? `${number(ragflow.dataset_count)} 个知识库已接入` : "知识库服务暂不可用";
  document.querySelector("#footerVersion").textContent = `RAGFlow ${ragflow.version || "—"}`;

  const gateLine = gate.ready
    ? `解析与向量配置 · 已读取 ${gate.configurations?.length || 1} 个知识库`
    : "模型 · 未读取到实际配置（解析/向量化暂缓，只读可用）";
  document.querySelector("#pipelineConfig").textContent = connected
    ? `RAGFlow ${ragflow.version || "在线"} · ${mineru.status === "online" ? "MinerU 在线" : "MinerU 离线，新沙盒使用 DeepDOC"} · ${gateLine}`
    : `RAGFlow 未连接 · ${gateLine}`;
  document.querySelector("#createSandboxButton").disabled = !connected || !state.datasets.length;
}

function renderKpis(datasets, summary) {
  const totals = datasets.reduce((acc, item) => {
    acc.documents += Number(item.document_count) || 0;
    acc.chunks += Number(item.chunk_count) || 0;
    return acc;
  }, { documents: 0, chunks: 0 });
  document.querySelector("#kpiDatasets").textContent = number(datasets.length);
  document.querySelector("#kpiDocuments").textContent = number(totals.documents);
  document.querySelector("#kpiChunks").textContent = number(totals.chunks);
  document.querySelector("#kpiParsed").textContent = summary
    ? `${number(summary.completed_count)} / ${number(summary.document_count)}`
    : "—";
}

function makeDatasetRow(dataset, maxDocs, maxChunks) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `distribution-row${dataset.id === state.selectedDataset ? " is-selected" : ""}`;
  button.dataset.datasetId = dataset.id;
  button.setAttribute("aria-label", `查看${dataset.name}，${dataset.document_count} 篇文档，${dataset.chunk_count} 个分块`);

  const name = document.createElement("span");
  name.className = "distribution-name";
  name.textContent = dataset.is_sandbox ? `${dataset.name} · 隔离库` : dataset.name;
  const bars = document.createElement("span");
  bars.className = "distribution-bars";
  const docs = document.createElement("span");
  docs.className = "distribution-track";
  const docsBar = document.createElement("i");
  docsBar.className = "docs-bar";
  docsBar.style.setProperty("--bar-scale", Math.max(0.01, dataset.document_count / maxDocs));
  docs.append(docsBar);
  const chunks = document.createElement("span");
  chunks.className = "distribution-track chunks-track";
  const chunksBar = document.createElement("i");
  chunksBar.className = "chunks-bar";
  chunksBar.style.setProperty("--bar-scale", Math.max(0.01, dataset.chunk_count / maxChunks));
  chunks.append(chunksBar);
  bars.append(docs, chunks);
  const values = document.createElement("span");
  values.className = "distribution-values";
  for (const value of [dataset.document_count, dataset.chunk_count]) {
    const span = document.createElement("span");
    span.textContent = number(value);
    values.append(span);
  }
  button.append(name, bars, values);
  return button;
}

function renderDatasetChart(datasets) {
  const root = document.querySelector("#datasetChart");
  root.replaceChildren();
  if (!datasets.length) {
    root.append(emptyMessage("没有可显示的知识库"));
    return;
  }
  const sorted = [...datasets].sort((a, b) => b.document_count - a.document_count);
  const maxDocs = Math.max(1, ...sorted.map((item) => item.document_count));
  const maxChunks = Math.max(1, ...sorted.map((item) => item.chunk_count));
  for (const dataset of sorted) root.append(makeDatasetRow(dataset, maxDocs, maxChunks));
}

function renderStatusChart(summary) {
  const root = document.querySelector("#statusChart");
  root.replaceChildren();
  if (!summary) {
    root.append(emptyMessage("暂无解析状态"));
    return;
  }
  const runs = Object.entries(summary.run_counts || {}).sort((a, b) => b[1] - a[1]);
  const total = Number(summary.document_count) || 0;
  document.querySelector("#statusTotal").textContent = `${number(total)} 篇`;
  if (!runs.length) {
    root.append(emptyMessage("暂无文档状态"));
    return;
  }
  const stack = document.createElement("div");
  stack.className = "status-stack";
  stack.setAttribute("role", "img");
  stack.setAttribute("aria-label", runs.map(([run, count]) => `${statusName(run)} ${count} 篇`).join("，"));
  for (const [run, count] of runs) {
    const segment = document.createElement("i");
    segment.className = `status-segment ${statusClass(run)}`;
    segment.style.width = `${total ? (count / total) * 100 : 0}%`;
    segment.title = `${statusName(run)}：${number(count)} 篇`;
    stack.append(segment);
  }
  root.append(stack);
  const legend = document.createElement("div");
  legend.className = "status-legend";
  for (const [run, count] of runs) {
    const row = document.createElement("div");
    row.className = "status-legend-row";
    const dot = document.createElement("i");
    dot.className = `status-dot ${statusClass(run)}`;
    const label = document.createElement("span");
    label.textContent = statusName(run);
    const value = document.createElement("strong");
    value.textContent = `${number(count)} · ${total ? Math.round((count / total) * 100) : 0}%`;
    row.append(dot, label, value);
    legend.append(row);
  }
  root.append(legend);
}

function renderDatasetSelect(datasets) {
  const select = document.querySelector("#datasetSelect");
  select.replaceChildren();
  for (const dataset of datasets) {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.name}${dataset.is_sandbox ? " · 隔离库" : ""} · ${number(dataset.document_count)} 篇`;
    select.append(option);
  }
  select.disabled = !datasets.length;
  if (datasets.length) select.value = state.selectedDataset;
}

function setFeedback(message, stateName = "info") {
  const target = document.querySelector("#intakeFeedback");
  target.textContent = message;
  target.dataset.state = stateName;
}

function addTrace(message, stateName = "info") {
  document.querySelector("#runTraceText").textContent = `${dateTime(Date.now())} · ${message}`;
  document.querySelector("#runTrace").dataset.state = stateName;
}

function setPipelineStep(step, stateName, message) {
  const node = document.querySelector(`[data-step="${step}"]`);
  if (!node) return;
  node.dataset.state = stateName;
  const small = node.querySelector("small");
  if (small && message) small.textContent = message;
}

const jobStages = [
  { key: "discover", title: "并行检索", detail: "OpenAlex · arXiv", icon: "search" },
  { key: "source_review", title: "来源审查", detail: "公开来源 · 去重", icon: "review" },
  { key: "acquire", title: "并行采集", detail: "下载 · 文件校验", icon: "download" },
  { key: "parse", title: "解析入库", detail: "RAGFlow Parser", icon: "parse" },
  { key: "chunk_review", title: "分块核验", detail: "实际内容抽查", icon: "chunks" },
];

const checkLabels = { source: "来源", file: "采集", parse: "解析", chunks: "分块" };
const checkSymbols = { passed: "✓", running: "◌", needs_review: "!", failed: "×", waiting: "·", skipped: "–" };
const checkStates = {
  passed: "通过", running: "进行中", needs_review: "复核", failed: "失败", waiting: "等待", skipped: "跳过",
};

function stageIcon(kind) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.5");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  const paths = {
    search: [["circle", { cx: 10.5, cy: 10.5, r: 6.5 }], ["path", { d: "m15.5 15.5 5 5" }]],
    review: [["path", { d: "M12 3 19 6v5c0 4.4-2.8 7.6-7 10-4.2-2.4-7-5.6-7-10V6l7-3Z" }], ["path", { d: "m9 12 2 2 4-4" }]],
    download: [["path", { d: "M12 3v12m-4-4 4 4 4-4" }], ["path", { d: "M5 17v3h14v-3" }]],
    parse: [["path", { d: "M6 3h8l4 4v14H6z" }], ["path", { d: "M14 3v5h5M9 12h6m-6 3h6m-6 3h4" }]],
    chunks: [["rect", { x: 4, y: 4, width: 16, height: 5, rx: 1 }], ["rect", { x: 4, y: 11, width: 11, height: 4, rx: 1 }], ["rect", { x: 4, y: 17, width: 14, height: 3, rx: 1 }]],
  };
  for (const [tag, attrs] of paths[kind] || []) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    svg.append(node);
  }
  return svg;
}

function renderJobCheck(check, key) {
  const value = check || { state: "waiting", detail: "等待此阶段" };
  const stateName = value.state || "waiting";
  const cell = document.createElement("span");
  cell.className = "job-check";
  cell.dataset.state = stateName;
  cell.title = value.detail || checkStates[stateName] || "";
  const icon = document.createElement("i");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = checkSymbols[stateName] || "·";
  const label = document.createElement("span");
  label.textContent = checkLabels[key];
  const state = document.createElement("small");
  state.textContent = checkStates[stateName] || "等待";
  cell.append(icon, label, state);
  return cell;
}

function renderJob(job) {
  state.ingestionJob = job || null;
  const root = document.querySelector("#jobStages");
  const itemRoot = document.querySelector("#jobItems");
  const badge = document.querySelector("#jobStatusBadge");
  const progress = document.querySelector(".job-progress-track");
  root.replaceChildren();
  itemRoot.replaceChildren();
  if (!job) {
    badge.dataset.state = "idle";
    badge.textContent = "待命";
    document.querySelector("#jobTarget").textContent = state.selectedSandbox ? `目标 · ${document.querySelector("#sandboxSelect").selectedOptions[0]?.textContent || "隔离知识库"}` : "选择隔离知识库开始入库";
    document.querySelector("#jobProgressText").textContent = "等待入库任务";
    document.querySelector("#jobProgressMeta").textContent = "执行记录与审查结果将在此汇总";
    document.querySelector("#jobProgressBar").style.transform = "scaleX(0)";
    progress.setAttribute("aria-valuenow", "0");
    const idleMetrics = document.querySelector("#jobMetrics");
    idleMetrics.replaceChildren();
    for (const label of ["资料", "通过", "待复核", "失败 / 跳过"]) {
      const metric = document.createElement("div");
      metric.className = "job-metric";
      const name = document.createElement("span");
      name.textContent = label;
      const amount = document.createElement("strong");
      amount.textContent = "—";
      metric.append(name, amount);
      idleMetrics.append(metric);
    }
    for (const stage of jobStages) root.append(makeStageNode(stage, "waiting", "等待"));
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "table-message";
    cell.textContent = "开始一次全自动入库或本地并行导入，任务明细会显示在这里。";
    row.append(cell);
    itemRoot.append(row);
    updateAutoControls();
    return;
  }

  const items = job.items || [];
  const counts = Object.fromEntries(["completed", "running", "needs_review", "failed", "skipped", "queued"].map((name) => [name, items.filter((item) => item.status === name).length]));
  const active = ["queued", "running"].includes(job.status);
  const labels = { queued: "排队中", running: "执行中", completed: "已完成", completed_with_review: "已完成 · 有待复核", failed: "执行失败" };
  badge.dataset.state = job.status === "completed" ? "done" : job.status === "completed_with_review" || job.status === "failed" ? "review" : active ? "running" : "idle";
  badge.textContent = labels[job.status] || job.status || "执行中";
  document.querySelector("#jobTarget").textContent = `目标 · ${job.dataset_name || "隔离知识库"}`;
  document.querySelector("#jobProgressText").textContent = job.message || labels[job.status] || "入库处理中";
  document.querySelector("#jobProgressMeta").textContent = `${job.mode === "local" ? "本地资料" : "开放文献"} · ${items.length || "—"} 项 · 并行度 ${job.parallelism || 1}`;

  const stageIndex = Math.max(0, jobStages.findIndex((stage) => stage.key === job.stage));
  const stageCheck = { source_review: "source", acquire: "file", parse: "parse", chunk_review: "chunks" };
  const stageCounts = jobStages.map((stage) => stage.key === "discover" ? items.length : items.filter((item) => item.checks?.[stageCheck[stage.key]]?.state === "passed").length);
  jobStages.forEach((stage, index) => {
    let stateName = index < stageIndex ? "done" : index === stageIndex && active ? "running" : "waiting";
    if (!active && ["completed", "completed_with_review"].includes(job.status)) stateName = "done";
    if (!active && job.status === "completed_with_review" && stageCheck[stage.key]) {
      const checks = items.map((item) => item.checks?.[stageCheck[stage.key]]?.state);
      if (checks.includes("failed")) stateName = "failed";
      else if (checks.includes("needs_review") || checks.every((value) => !value || value === "waiting")) stateName = "waiting";
    }
    if (job.status === "failed" && index === stageIndex) stateName = "failed";
    const detail = index === stageIndex && active ? `${items.filter((item) => item.stage === stage.key).length} 项 · 进行中` : `${stageCounts[index]} 项`;
    root.append(makeStageNode(job.mode === "local" && stage.key === "discover" ? { ...stage, title: "本地发现", detail: "已接入资料" } : stage, stateName, detail));
  });

  const metrics = document.querySelector("#jobMetrics");
  metrics.replaceChildren();
  const metricValues = [
    ["资料", items.length], ["通过", counts.completed], ["待复核", counts.needs_review], ["失败 / 跳过", counts.failed + counts.skipped],
  ];
  for (const [label, value] of metricValues) {
    const metric = document.createElement("div");
    metric.className = "job-metric";
    const name = document.createElement("span");
    name.textContent = label;
    const amount = document.createElement("strong");
    amount.textContent = number(value);
    metric.append(name, amount);
    metrics.append(metric);
  }
  let percent = 0;
  if (items.length) percent = Math.round(items.reduce((sum, item) => sum + Math.max(0, Math.min(100, Number(item.progress) || 0)), 0) / items.length);
  else if (job.stage === "discover") percent = 5;
  if (["completed", "completed_with_review", "failed"].includes(job.status)) percent = 100;
  document.querySelector("#jobProgressBar").style.transform = `scaleX(${percent / 100})`;
  document.querySelector("#jobProgressBar").style.backgroundColor = job.status === "failed" ? "var(--danger, #a3442e)" : "var(--green, #426c55)";
  progress.setAttribute("aria-valuenow", String(percent));

  if (!items.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "table-message";
    cell.textContent = job.stage === "discover" && active ? "正在并行检索并整理来源记录…" : job.message || "任务启动中…";
    row.append(cell);
    itemRoot.append(row);
  }
  for (const item of items) {
    const row = document.createElement("tr");
    row.className = "job-item-row";
    const sourceCell = document.createElement("td");
    const source = document.createElement("div");
    source.className = "job-item-title";
    source.textContent = item.title || "未命名资料";
    const sourceMeta = document.createElement("small");
    sourceMeta.textContent = [item.source, item.year, item.doi].filter(Boolean).join(" · ");
    sourceCell.append(source, sourceMeta);
    row.append(sourceCell);
    for (const key of ["source", "file", "parse", "chunks"]) {
      const cell = document.createElement("td");
      cell.append(renderJobCheck(item.checks?.[key], key));
      row.append(cell);
    }
    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = "job-item-status";
    status.dataset.state = item.status || "queued";
    status.textContent = item.message || item.status || "排队中";
    statusCell.append(status);
    if (item.document_ids?.length) {
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "job-inspect";
      inspect.dataset.title = item.title || "";
      inspect.textContent = "查看文档与分块";
      statusCell.append(inspect);
    }
    row.append(statusCell);
    itemRoot.append(row);
  }
  updateAutoControls();
}

function makeStageNode(stage, stateName, detail) {
  const node = document.createElement("article");
  node.className = "stage-node";
  node.dataset.state = stateName;
  const icon = document.createElement("span");
  icon.className = "stage-icon";
  icon.append(stageIcon(stage.icon));
  const copy = document.createElement("span");
  copy.className = "stage-copy";
  const title = document.createElement("strong");
  title.textContent = stage.title;
  const sub = document.createElement("small");
  sub.textContent = detail === "等待" ? stage.detail : detail;
  copy.append(title, sub);
  const mark = document.createElement("i");
  mark.className = "stage-mark";
  mark.textContent = stateName === "done" ? "✓" : stateName === "running" ? "◌" : stateName === "failed" ? "×" : "·";
  node.append(icon, copy, mark);
  return node;
}

function updateAutoControls() {
  const connected = document.querySelector("#topbarStatus").dataset.state === "online";
  const active = ["queued", "running"].includes(state.ingestionJob?.status);
  const hasTarget = Boolean(state.selectedSandbox);
  document.querySelector("#autoIngestButton").disabled = !connected || !hasTarget || active || state.importing;
  document.querySelector("#uploadFilesButton").disabled = !hasTarget || !state.selectedFiles.length || active || state.importing;
  document.querySelector("#createSandboxButton").disabled = !connected || !state.datasets.length || active;
}

async function pollIngestionJob(jobId, token = ++state.jobPollToken) {
  state.jobPollToken = token;
  while (token === state.jobPollToken) {
    try {
      const job = await requestJson(`/api/ingestion-jobs/${encodeURIComponent(jobId)}`);
      renderJob(job);
      if (!["queued", "running"].includes(job.status)) {
        if (job.dataset_id) {
          state.selectedDataset = job.dataset_id;
          await loadDashboard({ force: true });
        }
        return;
      }
    } catch (error) {
      document.querySelector("#jobProgressText").textContent = `任务状态读取失败 · ${error.message}`;
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1400));
  }
}

async function restoreRecentIngestionJob() {
  try {
    const payload = await requestJson("/api/ingestion-jobs");
    const requestedJob = new URLSearchParams(window.location.search).get("job");
    const job = (payload.items || []).find((item) => item.id === requestedJob) || (payload.items || [])[0];
    if (job) {
      if (state.datasets.some((dataset) => dataset.id === job.dataset_id)) {
        state.selectedDataset = job.dataset_id;
        if (state.sandboxes.some((dataset) => dataset.id === job.dataset_id)) state.selectedSandbox = job.dataset_id;
        await loadDashboard();
      }
      renderJob(job);
      if (["queued", "running"].includes(job.status)) pollIngestionJob(job.id);
    }
  } catch {
    // Status inventory remains available when no job history is present.
  }
  loadRestoredJobs();
}

async function loadRestoredJobs() {
  try {
    const payload = await requestJson("/api/ingestion-jobs/restored");
    state.restoredJobs = payload.items || [];
  } catch {
    state.restoredJobs = [];
  }
  renderRestoredJobs();
}

function renderRestoredJobs() {
  const root = document.querySelector("#restoredJobList");
  if (!root) return;
  root.replaceChildren();
  const jobs = state.restoredJobs || [];
  const note = document.querySelector("#restoredJobsNote");
  if (note) {
    note.textContent = jobs.length
      ? `${number(jobs.length)} 份任务已保存，可查看状态并继续结果核验。`
      : "任务历史保存在本机索引中，重启后可查看已完成与失败项。";
  }
  if (!jobs.length) {
    root.append(emptyMessage("重启后没有需要恢复的入库任务。"));
    return;
  }
  for (const job of jobs.slice(0, 8)) {
    const row = document.createElement("article");
    row.className = "restored-job-row";
    row.dataset.jobId = job.id;
    const heading = document.createElement("div");
    heading.className = "restored-job-heading";
    const title = document.createElement("strong");
    title.textContent = job.dataset_name || job.dataset_id || "隔离知识库";
    const status = document.createElement("span");
    status.className = `restored-job-status ${job.status === "completed" ? "done" : job.status === "failed" ? "failed" : job.status === "interrupted" ? "interrupted" : "review"}`;
    status.textContent = job.status === "completed" ? "已完成" : job.status === "failed" ? "执行失败" : job.status === "interrupted" ? "已中断" : "有待复核";
    heading.append(title, status);
    const meta = document.createElement("small");
    const gateNote = job.model_gate && !job.model_gate.ready ? " · 当时未读取到实际解析/向量化配置" : "";
    meta.textContent = `${job.mode === "local" ? "本地资料" : "开放文献"} · ${job.items?.length || 0} 项 · ${job.status === "interrupted" ? "已中断，可恢复继续检查" : "更新于 " + dateTime(job.updated_at || job.created_at)}${gateNote}`;
    const actions = document.createElement("div");
    actions.className = "restored-job-actions";
    const isInterrupted = job.status === "interrupted";
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "quiet-btn";
    retry.textContent = "继续结果核验";
    retry.dataset.jobId = job.id;
    retry.dataset.kind = "resume";
    actions.append(retry);
    row.append(heading, meta, actions);
    root.append(row);
  }
}

async function retryRestoredJob(jobId, kind = "retry") {
  if (!jobId || state.importing) return;
  state.importing = true;
  setFeedback(kind === "resume" ? "正在恢复这份中断任务…" : "正在恢复这份任务…", "loading");
  try {
    const job = await postJson(kind === "resume" ? "/api/ingestion-jobs/resume" : "/api/ingestion-jobs/retry", { job_id: jobId });
    if (job.resumable === false) {
      setFeedback(job.message || "无法安全恢复：缺少原始本地字节且尚无已上传文档", "error");
      return;
    }
    const active = job.job || job;
    state.ingestionJob = active;
    state.selectedSandbox = active.dataset_id;
    state.selectedDataset = active.dataset_id;
    renderJob(active);
    addTrace(kind === "resume" ? `已恢复中断任务 ${jobId}，继续检查已上传文档` : `已恢复任务 ${jobId}，重新执行中`, "success");
    setFeedback(kind === "resume" ? "任务已恢复，继续检查解析与分块。" : "任务已重新排队，页面会跟随新的执行状态。", "success");
    document.querySelector("#pipeline").scrollIntoView({ behavior: "smooth", block: "start" });
    pollIngestionJob(active.id);
  } catch (error) {
    setFeedback(`恢复失败：${error.message}`, "error");
  } finally {
    state.importing = false;
    updateAutoControls();
  }
}

async function startOnlineIngestion() {
  if (!state.selectedSandbox || state.importing || ["queued", "running"].includes(state.ingestionJob?.status)) return;
  const query = document.querySelector("#topicQuery").value.trim();
  if (query.length < 2) {
    setFeedback("主题至少输入 2 个字符。", "error");
    return;
  }
  state.importing = true;
  updateAutoControls();
  setFeedback("已提交全自动入库任务。", "loading");
  try {
    const job = await postJson(`/api/sandboxes/${encodeURIComponent(state.selectedSandbox)}/auto`, {
      query,
      limit: Number(document.querySelector("#autoLimit").value) || 6,
      parallelism: Number(document.querySelector("#parallelismSelect").value) || 3,
    });
    addTrace(`全自动入库已启动 · ${query}`, "success");
    renderJob(job);
    document.querySelector("#pipeline").scrollIntoView({ behavior: "smooth", block: "start" });
    pollIngestionJob(job.id);
  } catch (error) {
    setFeedback(`无法开始入库：${error.message}`, "error");
  } finally {
    state.importing = false;
    updateAutoControls();
  }
}

function renderSandboxControls() {
  const template = document.querySelector("#templateSelect");
  const previousTemplate = template.value;
  template.replaceChildren();
  for (const dataset of state.datasets) {
    template.append(new Option(`${dataset.name} · ${dataset.chunk_method || "解析模板"}`, dataset.id));
  }
  if (state.datasets.length) {
    const suggested = state.datasets.find((item) => /论文|手册|paper|manual/i.test(item.name));
    template.value = state.datasets.some((item) => item.id === previousTemplate)
      ? previousTemplate
      : (suggested || state.datasets[0]).id;
  }
  template.disabled = !state.datasets.length;

  const select = document.querySelector("#sandboxSelect");
  const selected = state.selectedSandbox;
  select.replaceChildren();
  if (!state.sandboxes.length) {
    select.append(new Option("新建隔离知识库以开始", ""));
    state.selectedSandbox = "";
  } else {
    for (const sandbox of state.sandboxes) select.append(new Option(`${sandbox.name} · ${number(sandbox.document_count)} 篇`, sandbox.id));
    state.selectedSandbox = state.sandboxes.some((item) => item.id === selected) ? selected : state.sandboxes[0].id;
    select.value = state.selectedSandbox;
  }
  select.disabled = !state.sandboxes.length;
  const connected = document.querySelector("#topbarStatus").dataset.state === "online";
  setPipelineStep("publish", connected ? "waiting" : "offline", connected ? (state.selectedSandbox ? "入库目标已选择" : "等待选择入库目标") : "服务未连接");
  document.querySelector("#createSandboxButton").disabled = !state.datasets.length || document.querySelector("#topbarStatus").dataset.state !== "online";
  renderCandidates();
  renderSelectedFiles();
  updateAutoControls();
}

function renderCandidates() {
  const root = document.querySelector("#discoveryResults");
  root.replaceChildren();
  if (!state.candidates.length) {
    root.append(emptyMessage("输入专业主题，检索可获取的开放文献。"));
    return;
  }
  for (const candidate of state.candidates) {
    const article = document.createElement("article");
    article.className = "candidate-row";
    const heading = document.createElement("div");
    heading.className = "candidate-heading";
    const title = document.createElement("strong");
    title.textContent = candidate.title || "未命名文献";
    const source = document.createElement("span");
    source.textContent = `${candidate.source || "开放文献"}${candidate.year ? ` · ${candidate.year}` : ""}`;
    heading.append(title, source);
    const authors = document.createElement("p");
    authors.className = "candidate-authors";
    authors.textContent = candidate.authors || candidate.doi || "作者信息未提供";
    article.append(heading, authors);
    if (candidate.abstract) {
      const abstract = document.createElement("p");
      abstract.className = "candidate-abstract";
      abstract.textContent = candidate.abstract;
      article.append(abstract);
    }
    const actions = document.createElement("div");
    actions.className = "candidate-actions";
    if (candidate.landing_url && /^https?:\/\//i.test(candidate.landing_url)) {
      const link = document.createElement("a");
      link.href = candidate.landing_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "查看来源";
      actions.append(link);
    }
    const importButton = document.createElement("button");
    importButton.className = "quiet-btn import-candidate";
    importButton.type = "button";
    importButton.disabled = !state.selectedSandbox;
    importButton.dataset.pdfUrl = candidate.pdf_url;
    importButton.dataset.title = candidate.title || "开放文献";
    importButton.textContent = state.selectedSandbox ? "导入并解析" : "先新建知识库";
    actions.append(importButton);
    article.append(actions);
    root.append(article);
  }
}

function renderSelectedFiles() {
  const root = document.querySelector("#selectedFiles");
  root.replaceChildren();
  if (!state.selectedFiles.length) {
    root.append(emptyMessage("所选资料会显示在这里。"));
  } else {
    const summary = document.createElement("p");
    summary.className = "file-summary";
    summary.textContent = `已选 ${state.selectedFiles.length} 个文件 · ${fileSize(state.selectedFiles.reduce((sum, file) => sum + file.size, 0))}`;
    root.append(summary);
    const list = document.createElement("ul");
    for (const file of state.selectedFiles) {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = file.webkitRelativePath || file.name;
      const size = document.createElement("small");
      size.textContent = fileSize(file.size);
      item.append(name, size);
      list.append(item);
    }
    root.append(list);
  }
  updateAutoControls();
}

function emptyMessage(message) {
  const p = document.createElement("p");
  p.className = "empty-message";
  p.textContent = message;
  return p;
}

function renderDocuments(payload) {
  const body = document.querySelector("#documentRows");
  body.replaceChildren();
  state.documents = payload.items || [];
  state.total = Number(payload.total) || 0;
  state.page = Number(payload.page) || 1;
  state.pages = Number(payload.pages) || 1;
  document.querySelector("#recordCount").textContent = `${number(state.total)} 条记录`;
  document.querySelector("#pageMeta").textContent = `第 ${number(state.page)} / ${number(state.pages)} 页 · 每页 ${number(payload.page_size)} 条`;
  document.querySelector("#prevPage").disabled = state.loading || state.page <= 1;
  document.querySelector("#nextPage").disabled = state.loading || state.page >= state.pages;
  if (!state.documents.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.className = "table-message";
    cell.textContent = state.query || state.run ? "没有符合条件的文档。" : "这个知识库还没有文档。";
    row.append(cell);
    body.append(row);
    return;
  }
  for (const doc of state.documents) {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.className = `document-row${state.selectedDocument?.id === doc.id ? " is-selected" : ""}`;
    row.dataset.documentId = doc.id;
    const name = document.createElement("td");
    const nameWrap = document.createElement("span");
    nameWrap.className = "table-document-name";
    nameWrap.textContent = doc.name;
    const source = document.createElement("small");
    source.textContent = doc.source_type || "RAGFlow 文档";
    nameWrap.append(source);
    name.append(nameWrap);
    const format = document.createElement("td");
    format.textContent = (doc.suffix || doc.type || "—").replace(/^\./, "").toUpperCase();
    const statusCell = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = `state-pill ${statusClass(doc.run)}`;
    const dot = document.createElement("i");
    const statusText = document.createElement("span");
    statusText.textContent = `${statusName(doc.run)}${doc.progress < 100 && doc.run === "RUNNING" ? ` ${doc.progress}%` : ""}`;
    pill.append(dot, statusText);
    statusCell.append(pill);
    const chunks = document.createElement("td");
    chunks.textContent = number(doc.chunk_count);
    const updated = document.createElement("td");
    updated.textContent = dateTime(doc.update_time || doc.create_time);
    row.append(name, format, statusCell, chunks, updated);
    body.append(row);
  }
}

function setTableLoading(message = "正在读取文档…") {
  state.loading = true;
  const body = document.querySelector("#documentRows");
  body.replaceChildren();
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 5;
  cell.className = "table-message";
  cell.textContent = message;
  row.append(cell);
  body.append(row);
  document.querySelector("#prevPage").disabled = true;
  document.querySelector("#nextPage").disabled = true;
}

async function loadDocuments(page = 1) {
  if (!state.selectedDataset) return;
  state.loading = false;
  setTableLoading("正在读取文档记录…");
  const params = new URLSearchParams({ page: String(page), page_size: "25" });
  if (state.query) params.set("q", state.query);
  if (state.run) params.set("run", state.run);
  try {
    const payload = await requestJson(`/api/datasets/${encodeURIComponent(state.selectedDataset)}/documents?${params}`);
    state.loading = false;
    renderDocuments(payload);
    if (state.selectedDocument && !state.documents.some((doc) => doc.id === state.selectedDocument.id)) {
      clearDocumentDetail();
    }
    if (!state.selectedDocument && state.documents.length) {
      const firstDocument = state.documents.find((doc) => String(doc.run).toUpperCase() === "DONE" && Number(doc.chunk_count) > 0) || state.documents[0];
      document.querySelectorAll("#documentRows .document-row").forEach((row) => {
        row.classList.toggle("is-selected", row.dataset.documentId === firstDocument.id);
      });
      await loadChunks(firstDocument);
    }
  } catch (error) {
    state.loading = false;
    setTableLoading(`读取失败：${error.message}`);
    document.querySelector("#recordCount").textContent = "读取失败";
  }
}

function clearDocumentDetail() {
  state.selectedDocument = null;
  document.querySelector("#detailTitle").textContent = "选择一篇文档";
  document.querySelector("#detailSubtitle").textContent = "文档属性与解析分块。";
  const metadata = document.querySelector("#detailMetadata");
  metadata.replaceChildren();
  for (const label of ["解析状态", "分块", "文件大小", "来源类型"]) {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = "—";
    row.append(dt, dd);
    metadata.append(row);
  }
  document.querySelector("#chunkCount").textContent = "选择文档查看";
  const chunks = document.querySelector("#chunkList");
  chunks.replaceChildren(emptyMessage("选择文档查看分块内容。"));
}

function renderDocumentDetail(doc) {
  state.selectedDocument = doc;
  document.querySelectorAll("#documentRows .document-row").forEach((row) => {
    row.classList.toggle("is-selected", row.dataset.documentId === doc.id);
  });
  document.querySelector("#detailTitle").textContent = doc.name;
  document.querySelector("#detailSubtitle").textContent = `最后更新 ${dateTime(doc.update_time || doc.create_time)}`;
  const metadata = document.querySelector("#detailMetadata");
  metadata.replaceChildren();
  const values = [
    ["解析状态", statusName(doc.run)],
    ["分块", number(doc.chunk_count)],
    ["文件大小", fileSize(doc.size)],
    ["来源类型", doc.source_type || "—"],
  ];
  for (const [label, value] of values) {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    row.append(dt, dd);
    metadata.append(row);
  }
  document.querySelector("#chunkCount").textContent = "正在加载…";
  const list = document.querySelector("#chunkList");
  list.replaceChildren(emptyMessage("正在读取分块内容…"));
}

async function loadChunks(doc) {
  renderDocumentDetail(doc);
  const list = document.querySelector("#chunkList");
  try {
    const payload = await requestJson(`/api/datasets/${encodeURIComponent(state.selectedDataset)}/documents/${encodeURIComponent(doc.id)}/chunks?page=1&page_size=5`);
    if (state.selectedDocument?.id !== doc.id) return;
    list.replaceChildren();
    document.querySelector("#chunkCount").textContent = `${number(payload.total)} 个分块 · 显示前 ${payload.items.length} 个`;
    setPipelineStep("review", payload.items.length ? "online" : "waiting", payload.items.length ? `${number(payload.total)} 个分块可查看` : "暂无可显示分块");
    if (!payload.items.length) {
      list.append(emptyMessage("该文档没有可读取的分块。"));
      return;
    }
    payload.items.forEach((chunk, index) => {
      const article = document.createElement("article");
      article.className = "chunk-card";
      const heading = document.createElement("div");
      heading.className = "chunk-card-heading";
      const label = document.createElement("strong");
      label.textContent = `分块 ${index + 1}`;
      const page = document.createElement("span");
      page.textContent = chunk.page != null ? `页 ${chunk.page}` : "位置未标注";
      heading.append(label, page);
      const text = document.createElement("div");
      text.className = "chunk-content";
      const content = chunk.content || "（空分块）";
      if (/<table[\s>]/i.test(content)) {
        const allowed = new Set(["TABLE", "CAPTION", "THEAD", "TBODY", "TFOOT", "TR", "TD", "TH"]);
        const inertMarkup = content.replace(/<\/?([a-z][a-z0-9-]*)\b[^>]*>/gi, (tag, name) => {
          if (!allowed.has(name.toUpperCase())) return "";
          const spans = [...tag.matchAll(/\b(rowspan|colspan)\s*=\s*["']?(\d+)/gi)]
            .filter((match) => Number(match[2]) > 0 && Number(match[2]) <= 20)
            .map((match) => ` ${match[1].toLowerCase()}="${match[2]}"`).join("");
          return `<${tag.startsWith("</") ? "/" : ""}${name.toLowerCase()}${spans}>`;
        });
        const parsed = new DOMParser().parseFromString(inertMarkup, "text/html");
        const appendSafe = (source, destination) => {
          for (const child of source.childNodes) {
            if (child.nodeType === Node.TEXT_NODE) destination.append(document.createTextNode(child.textContent));
            else if (allowed.has(child.nodeName)) {
              const clean = document.createElement(child.nodeName.toLowerCase());
              for (const attribute of ["rowspan", "colspan"]) {
                const value = Number(child.getAttribute(attribute));
                if (value > 0 && value <= 20) clean.setAttribute(attribute, String(value));
              }
              appendSafe(child, clean);
              destination.append(clean);
            } else if (!["SCRIPT", "STYLE", "IFRAME", "IMG", "OBJECT"].includes(child.nodeName)) {
              destination.append(document.createTextNode(child.textContent));
            }
          }
        };
        appendSafe(parsed.body, text);
      } else text.textContent = content;
      article.append(heading, text);
      if (chunk.truncated) {
        const note = document.createElement("small");
        note.textContent = "内容较长，当前显示截取文本。";
        article.append(note);
      }
      if (chunk.keywords?.length) {
        const keywords = document.createElement("div");
        keywords.className = "chunk-keywords";
        keywords.textContent = `关键词：${chunk.keywords.join(" · ")}`;
        article.append(keywords);
      }
      list.append(article);
    });
  } catch (error) {
    document.querySelector("#chunkCount").textContent = "读取失败";
    list.replaceChildren(emptyMessage(`读取分块失败：${error.message}`));
    setPipelineStep("review", "offline", "分块读取失败");
  }
}

async function loadDashboard({ force = false } = {}) {
  const refresh = document.querySelector("#refreshButton");
  refresh.disabled = true;
  refresh.classList.add("is-loading");
  try {
    const suffix = force ? "?refresh=1" : "";
    const [status, inventory, sandboxInventory] = await Promise.all([
      requestJson(`/api/status${suffix}`),
      requestJson(`/api/datasets${suffix}`),
      requestJson("/api/sandboxes"),
    ]);
    state.datasets = inventory.items || [];
    state.sandboxes = sandboxInventory.items || [];
    renderConnection(status);
    const queryDataset = new URLSearchParams(window.location.search).get("dataset");
    if (!state.datasets.some((item) => item.id === state.selectedDataset)) {
      state.selectedDataset = state.datasets.some((item) => item.id === queryDataset)
        ? queryDataset
        : [...state.datasets].sort((a, b) => b.document_count - a.document_count)[0]?.id || "";
    }
    renderDatasetSelect(state.datasets);
    renderSandboxControls();
    if (!state.ingestionJob) renderJob(null);
    state.summary = await requestJson(`/api/summary${suffix}`);
    renderKpis(state.datasets, state.summary);
    renderDatasetChart(state.datasets);
    renderStatusChart(state.summary);
    document.querySelector("#searchInput").disabled = !state.selectedDataset;
    document.querySelector("#runSelect").disabled = !state.selectedDataset;
    document.querySelector("#clearFilters").disabled = !state.selectedDataset;
    const runSelect = document.querySelector("#runSelect");
    const selectedRun = state.run;
    runSelect.replaceChildren(new Option("全部状态", ""));
    for (const run of Object.keys(state.summary.run_counts || {}).sort()) {
      runSelect.append(new Option(statusName(run), run));
    }
    runSelect.value = selectedRun;
    if (state.selectedDataset) await loadDocuments(1);
    else {
      setTableLoading("RAGFlow 中没有知识库。");
      document.querySelector("#recordCount").textContent = "0 条记录";
    }
  } catch (error) {
    document.querySelector("#topbarStatus").dataset.state = "offline";
    document.querySelector("#topbarStatusText").textContent = "RAGFlow 未连接";
    document.querySelector("#sidebarDot").className = "mode-dot is-offline";
    document.querySelector("#sidebarConnection").textContent = "连接异常";
    document.querySelector("#sidebarScope").textContent = "知识库服务暂不可用";
    document.querySelector("#pipelineConfig").textContent = "RAGFlow 未连接";
    document.querySelector("#datasetChart").replaceChildren(emptyMessage("知识库数据暂时无法读取"));
    document.querySelector("#statusChart").replaceChildren(emptyMessage("解析状态暂时无法读取"));
    setTableLoading("连接失败，请检查服务状态后刷新。");
  } finally {
    refresh.disabled = false;
    refresh.classList.remove("is-loading");
  }
}

document.querySelector("#datasetSelect").addEventListener("change", (event) => {
  state.selectedDataset = event.target.value;
  state.query = "";
  state.run = "";
  state.page = 1;
  document.querySelector("#searchInput").value = "";
  document.querySelector("#runSelect").value = "";
  history.replaceState(null, "", `?dataset=${encodeURIComponent(state.selectedDataset)}#documents`);
  renderDatasetChart(state.datasets);
  clearDocumentDetail();
  loadDocuments(1);
});

document.querySelector("#datasetChart").addEventListener("click", (event) => {
  const button = event.target.closest("[data-dataset-id]");
  if (!button) return;
  const select = document.querySelector("#datasetSelect");
  select.value = button.dataset.datasetId;
  select.dispatchEvent(new Event("change", { bubbles: true }));
});

let searchTimer;
document.querySelector("#searchInput").addEventListener("input", (event) => {
  window.clearTimeout(searchTimer);
  state.query = event.target.value.trim();
  searchTimer = window.setTimeout(() => loadDocuments(1), 350);
});

document.querySelector("#runSelect").addEventListener("change", (event) => {
  state.run = event.target.value;
  loadDocuments(1);
});

document.querySelector("#clearFilters").addEventListener("click", () => {
  state.query = "";
  state.run = "";
  document.querySelector("#searchInput").value = "";
  document.querySelector("#runSelect").value = "";
  loadDocuments(1);
});

document.querySelector("#documentRows").addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-document-id]");
  if (!row) return;
  const doc = state.documents.find((item) => item.id === row.dataset.documentId);
  if (doc) loadChunks(doc);
});

document.querySelector("#documentRows").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest("tr[data-document-id]");
  if (!row) return;
  event.preventDefault();
  const doc = state.documents.find((item) => item.id === row.dataset.documentId);
  if (doc) loadChunks(doc);
});

document.querySelector("#prevPage").addEventListener("click", () => loadDocuments(state.page - 1));
document.querySelector("#nextPage").addEventListener("click", () => loadDocuments(state.page + 1));
document.querySelector("#refreshButton").addEventListener("click", () => loadDashboard({ force: true }).then(loadRestoredJobs));

document.querySelector("#restoredJobList").addEventListener("click", (event) => {
  const button = event.target.closest(".restored-job-actions button");
  if (!button) return;
  retryRestoredJob(button.dataset.jobId, button.dataset.kind === "resume" ? "resume" : "retry");
});

document.querySelector("#jobItems").addEventListener("click", async (event) => {
  const button = event.target.closest(".job-inspect");
  if (!button || !state.ingestionJob?.dataset_id) return;
  state.selectedDataset = state.ingestionJob.dataset_id;
  state.query = button.dataset.title || "";
  state.run = "";
  document.querySelector("#datasetSelect").value = state.selectedDataset;
  document.querySelector("#searchInput").value = state.query;
  document.querySelector("#runSelect").value = "";
  history.replaceState(null, "", `?dataset=${encodeURIComponent(state.selectedDataset)}#documents`);
  renderDatasetChart(state.datasets);
  clearDocumentDetail();
  await loadDocuments(1);
  document.querySelector("#documents").scrollIntoView({ behavior: "smooth", block: "start" });
});

for (const tab of document.querySelectorAll("[data-source-tab]")) {
  tab.addEventListener("click", () => {
    const mode = tab.dataset.sourceTab;
    for (const choice of document.querySelectorAll("[data-source-tab]")) {
      const active = choice.dataset.sourceTab === mode;
      choice.classList.toggle("is-active", active);
      choice.setAttribute("aria-selected", String(active));
    }
    document.querySelector("#onlinePane").hidden = mode !== "online";
    document.querySelector("#localPane").hidden = mode !== "local";
  });
}

document.querySelector("#createSandboxButton").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  if (!document.querySelector("#templateSelect").value) return;
  button.disabled = true;
  button.textContent = "正在创建…";
  setFeedback("正在创建知识库…", "loading");
  try {
    const sandbox = await postJson("/api/sandboxes", { template_id: document.querySelector("#templateSelect").value });
    state.jobPollToken += 1;
    state.ingestionJob = null;
    state.selectedSandbox = sandbox.id;
    state.selectedDataset = sandbox.id;
    addTrace(`已创建知识库 ${sandbox.name}`, "success");
    setFeedback(`“${sandbox.name}”已创建，可以开始导入资料。${sandbox.parser_note ? ` ${sandbox.parser_note}` : ""}`, "success");
    await loadDashboard({ force: true });
    document.querySelector("#documents").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setFeedback(`创建失败：${error.message}`, "error");
  } finally {
    button.textContent = "新建隔离知识库";
    button.disabled = !state.datasets.length || document.querySelector("#topbarStatus").dataset.state !== "online";
  }
});

document.querySelector("#sandboxSelect").addEventListener("change", async (event) => {
  state.selectedSandbox = event.target.value;
  state.selectedDataset = event.target.value;
  renderCandidates();
  renderSelectedFiles();
  if (state.selectedSandbox) {
    setFeedback(`入库目标：${event.target.selectedOptions[0]?.textContent || "隔离知识库"}`, "success");
    await loadDashboard({ force: true });
    document.querySelector("#documents").scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

document.querySelector("#discoverForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.discovering) return;
  const query = document.querySelector("#topicQuery").value.trim();
  const button = document.querySelector("#discoverButton");
  if (query.length < 2) {
    setFeedback("主题至少输入 2 个字符。", "error");
    return;
  }
  state.discovering = true;
  button.disabled = true;
  button.textContent = "检索中…";
  setPipelineStep("source", "running", "正在搜索开放文献");
  document.querySelector("#discoveryResults").replaceChildren(emptyMessage("正在检索 OpenAlex 与 arXiv…"));
  try {
    const result = await postJson("/api/discover", { query, limit: 8 });
    state.candidates = result.items || [];
    renderCandidates();
    const suffix = result.errors?.length ? `；${result.errors.join("、")} 暂不可用` : "";
    setFeedback(`检索完成，找到 ${number(state.candidates.length)} 条带开放 PDF 的记录${suffix}。`, state.candidates.length ? "success" : "info");
    setPipelineStep("source", "online", `找到 ${number(state.candidates.length)} 条开放文献`);
    addTrace(`主题检索“${query}”，找到 ${state.candidates.length} 条可下载记录`, "success");
  } catch (error) {
    state.candidates = [];
    renderCandidates();
    setFeedback(`检索失败：${error.message}。可稍后重试，或改用本地资料。`, "error");
    setPipelineStep("source", "offline", "检索失败，可重试");
  } finally {
    state.discovering = false;
    button.disabled = false;
    button.textContent = "只检索";
  }
});

document.querySelector("#autoIngestButton").addEventListener("click", startOnlineIngestion);

function selectLocalFiles(input) {
  const allowed = new Set([".pdf", ".txt", ".md", ".docx", ".pptx", ".xlsx", ".csv", ".html"]);
  const incoming = Array.from(input.files || []);
  const issues = [];
  const valid = [];
  let totalBytes = 0;
  for (const file of incoming) {
    const extension = `.${file.name.split(".").pop().toLowerCase()}`;
    if (!allowed.has(extension)) issues.push(`${file.name}：格式不支持`);
    else if (!file.size || file.size > 50 * 1024 * 1024) issues.push(`${file.name}：文件为空或超过 50 MB`);
    else if (totalBytes + file.size > 95 * 1024 * 1024) issues.push("每批文件总量不能超过 95 MB");
    else {
      totalBytes += file.size;
      valid.push(file);
    }
  }
  state.selectedFiles = valid.slice(0, 10);
  if (valid.length > 10) issues.push("每次最多导入 10 个文件，已保留前 10 个");
  renderSelectedFiles();
  if (issues.length) setFeedback(issues.slice(0, 3).join("；"), "error");
  else if (valid.length) setFeedback(`已选择 ${valid.length} 个文件，选择入库目标后即可导入。`, "info");
  if (valid.length) {
    setPipelineStep("source", "online", `${valid.length} 个本地文件已选择`);
    setPipelineStep("acquire", "waiting", "等待导入");
  }
  input.value = "";
}

document.querySelector("#filePicker").addEventListener("change", (event) => selectLocalFiles(event.currentTarget));
document.querySelector("#folderPicker").addEventListener("change", (event) => selectLocalFiles(event.currentTarget));

document.querySelector("#uploadFilesButton").addEventListener("click", async (event) => {
  if (!state.selectedSandbox || !state.selectedFiles.length || state.importing || ["queued", "running"].includes(state.ingestionJob?.status)) return;
  state.importing = true;
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = `正在提交 ${state.selectedFiles.length} 个文件…`;
  setFeedback("正在创建并行入库任务…", "loading");
  try {
    const form = new FormData();
    state.selectedFiles.forEach((file) => form.append("files", file, file.name));
    const parallelism = Number(document.querySelector("#localParallelism").value) || 3;
    const response = await fetch(`/api/sandboxes/${encodeURIComponent(state.selectedSandbox)}/auto-files?parallelism=${parallelism}`, { method: "POST", body: form, cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.selectedDataset = state.selectedSandbox;
    state.selectedFiles = [];
    renderSelectedFiles();
    addTrace(`本地并行入库已启动 · ${payload.items?.length || 0} 项`, "success");
    setFeedback("本地资料已进入并行采集与审查流程。", "success");
    renderJob(payload);
    document.querySelector("#pipeline").scrollIntoView({ behavior: "smooth", block: "start" });
    pollIngestionJob(payload.id);
  } catch (error) {
    setFeedback(`导入失败：${error.message}`, "error");
  } finally {
    state.importing = false;
    button.textContent = "并行导入并审查";
    renderSelectedFiles();
  }
});

document.querySelector("#discoveryResults").addEventListener("click", async (event) => {
  const button = event.target.closest(".import-candidate");
  if (!button || !state.selectedSandbox || state.importing) return;
  state.importing = true;
  button.disabled = true;
  button.textContent = "下载与解析中…";
  setFeedback("正在下载开放 PDF 并导入知识库…", "loading");
  setPipelineStep("acquire", "running", "下载开放 PDF");
  try {
    const result = await postJson(`/api/sandboxes/${encodeURIComponent(state.selectedSandbox)}/remote`, {
      url: button.dataset.pdfUrl,
      name: button.dataset.title,
    });
    const ids = (result.items || []).map((item) => item.id).filter(Boolean);
    setPipelineStep("acquire", "online", "开放 PDF 已下载");
    setPipelineStep("publish", "online", "文档已加入知识库");
    setPipelineStep("parse", result.parse_started ? "running" : "waiting", result.parse_started ? "正在解析文档" : "等待解析任务");
    setFeedback(result.message || (result.parse_started ? "资料已导入，正在解析。" : "资料已导入，等待解析任务。"), result.parse_started ? "success" : "info");
    addTrace(`在线文献“${button.dataset.title}”已下载并提交解析`, "success");
    state.selectedDataset = state.selectedSandbox;
    await loadDashboard({ force: true });
    if (ids.length) monitorParse(state.selectedSandbox, ids);
  } catch (error) {
    setFeedback(`在线导入失败：${error.message}`, "error");
    setPipelineStep("acquire", "offline", "下载失败，可查看来源后重试");
  } finally {
    state.importing = false;
    renderCandidates();
  }
});

async function monitorParse(datasetId, documentIds) {
  const token = ++state.monitorToken;
  const terminal = new Set(["DONE", "FAIL", "CANCEL"]);
  setPipelineStep("parse", "running", "正在解析文档");
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 2500));
    if (token !== state.monitorToken) return;
    try {
      const params = new URLSearchParams({ page: "1", page_size: "100" });
      const payload = await requestJson(`/api/datasets/${encodeURIComponent(datasetId)}/documents?${params}`);
      const found = new Map((payload.items || []).map((item) => [item.id, item]));
      const docs = documentIds.map((id) => found.get(id)).filter(Boolean);
      if (docs.length) renderDocuments(payload);
      if (docs.length === documentIds.length && docs.every((doc) => terminal.has(String(doc.run).toUpperCase()))) {
        const completed = docs.filter((doc) => String(doc.run).toUpperCase() === "DONE").length;
        const failed = docs.length - completed;
        const message = failed ? `解析完成：${completed} 个成功，${failed} 个失败。` : `${completed} 个文件解析完成，可查看文档分块。`;
        setFeedback(message, failed ? "warning" : "success");
        setPipelineStep("parse", failed ? "offline" : "online", failed ? `${failed} 个文档解析失败` : `${completed} 个文档解析完成`);
        setPipelineStep("review", completed ? "online" : "offline", completed ? `${completed} 个文档可查看分块` : "无解析完成的文档");
        addTrace(message, failed ? "warning" : "success");
        await loadDashboard({ force: true });
        return;
      }
      setFeedback("文档正在解析，完成后将自动更新状态。", "loading");
    } catch (error) {
      setFeedback(`状态同步失败：${error.message}。刷新后可查看最新进度。`, "warning");
      return;
    }
  }
  setFeedback("解析时间较长，可刷新查看当前文档状态。", "warning");
}

loadDashboard().then(restoreRecentIngestionJob);
