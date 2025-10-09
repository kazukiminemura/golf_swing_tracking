const uploadForm = document.getElementById("upload-form");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const jobsTableBody = document.querySelector("#jobs-table tbody");
const resultSection = document.getElementById("results-section");
const resultVideo = document.getElementById("result-video");
const trajectoryCanvas = document.getElementById("trajectory-canvas");
const trajectoryPlot = document.getElementById("trajectory-plot");
const maxSpeedEl = document.getElementById("max-speed");
const avgSpeedEl = document.getElementById("avg-speed");
const impactFrameEl = document.getElementById("impact-frame");
const durationEl = document.getElementById("duration");
const downloadVideoLink = document.getElementById("download-video");
const downloadCsvLink = document.getElementById("download-csv");
const downloadJsonLink = document.getElementById("download-json");
const downloadStatsLink = document.getElementById("download-stats");

let jobSockets = {};
let currentJobId = null;
let currentTrajectory = [];
let currentStats = null;
let overlayAnimationId = null;
let trajectorySegments = [];

const trajectoryCtx = trajectoryCanvas.getContext("2d");
const trajectoryPlotCtx = trajectoryPlot ? trajectoryPlot.getContext("2d") : null;

const BACKSWING_COLOUR = "#42b2ff";
const DOWNSWING_COLOUR = "#ff7854";
const PATH_LINE_WIDTH = 3;
const DIRECTION_THRESHOLD = 1;

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("video-input");
  const deviceSelect = document.getElementById("device-select");
  if (!fileInput.files.length) {
    uploadStatus.textContent = "Please choose a video file.";
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);
  if (deviceSelect.value) {
    formData.append("device", deviceSelect.value);
  }

  uploadButton.disabled = true;
  uploadStatus.textContent = "Uploading...";

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }

    const data = await response.json();
    uploadStatus.textContent = `Job ${data.job_id} queued`;
    fileInput.value = "";
    await refreshJobs();
    openJobSocket(data.job_id);
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = err.message || "Unexpected error";
  } finally {
    uploadButton.disabled = false;
  }
});

async function refreshJobs() {
  const response = await fetch("/api/jobs");
  if (!response.ok) {
    console.error("Failed to fetch jobs");
    return;
  }
  const data = await response.json();
  renderJobs(data.jobs);
}

function renderJobs(jobs) {
  jobsTableBody.innerHTML = "";
  jobs.forEach((job) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><button class="link-btn" data-job="${job.id}">${job.id.slice(0, 8)}</button></td>
      <td><span class="badge ${job.status}">${job.status}</span></td>
      <td>${job.device}</td>
      <td>${Math.round(job.progress * 100)}%</td>
      <td>${new Date(job.updated_at).toLocaleTimeString()}</td>
      <td>
        <button class="small-btn" data-action="view" data-job="${job.id}">View</button>
        <button class="small-btn danger" data-action="delete" data-job="${job.id}">Delete</button>
      </td>
    `;
    jobsTableBody.appendChild(tr);

    if (!jobSockets[job.id] && (job.status === "QUEUED" || job.status === "RUNNING")) {
      openJobSocket(job.id);
    }
  });
}

jobsTableBody.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const jobId = target.dataset.job;
  if (!jobId) return;

  if (target.dataset.action === "view") {
    viewJob(jobId);
  } else if (target.dataset.action === "delete") {
    await deleteJob(jobId);
    await refreshJobs();
  } else {
    viewJob(jobId);
  }
});

async function viewJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) {
      throw new Error("Job not found");
    }
    const detail = await response.json();
    if (detail.job.status !== "SUCCEEDED") {
      uploadStatus.textContent = `Job ${jobId} is ${detail.job.status}`;
      return;
    }
    await loadResults(jobId);
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = err.message || "Unable to load job";
  }
}

async function deleteJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (!response.ok) {
      throw new Error("Delete failed");
    }
    if (jobSockets[jobId]) {
      jobSockets[jobId].close();
      delete jobSockets[jobId];
    }
    if (currentJobId === jobId) {
      resultSection.hidden = true;
      currentJobId = null;
    }
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = err.message || "Delete failed";
  }
}

async function loadResults(jobId) {
  const [detailResp, trajectoryResp, statsResp] = await Promise.all([
    fetch(`/api/jobs/${jobId}/results`),
    fetch(`/api/jobs/${jobId}/trajectory.json`),
    fetch(`/api/jobs/${jobId}/stats.json`),
  ]);

  if (!detailResp.ok || !trajectoryResp.ok || !statsResp.ok) {
    throw new Error("Failed to load analysis artifacts");
  }

  const detail = await detailResp.json();
  currentTrajectory = await trajectoryResp.json();
  currentStats = await statsResp.json();
  computeTrajectorySegments();
  currentJobId = jobId;

  const baseUrl = `/api/jobs/${jobId}`;
  const videoUrl = `${baseUrl}/video?ts=${Date.now()}`;
  stopOverlayLoop();
  resultVideo.pause();
  resultVideo.src = videoUrl;
  resultVideo.load();
  downloadVideoLink.href = `${baseUrl}/video`;
  downloadCsvLink.href = `${baseUrl}/trajectory.csv`;
  downloadJsonLink.href = `${baseUrl}/trajectory.json`;
  downloadStatsLink.href = `${baseUrl}/stats.json`;

  maxSpeedEl.textContent = (detail.stats.max_speed_mps || 0).toFixed(2);
  avgSpeedEl.textContent = (detail.stats.avg_speed_mps || 0).toFixed(2);
  impactFrameEl.textContent = detail.stats.impact_frame ?? "-";
  durationEl.textContent = (detail.stats.duration_s || 0).toFixed(2);

  resultSection.hidden = false;
  resizeTrajectoryCanvas();
  renderTrajectoryPlot();
  drawTrajectory();

  await refreshJobs();
}

function openJobSocket(jobId) {
  const socketUrl = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/jobs/${jobId}`;
  const ws = new WebSocket(socketUrl);
  jobSockets[jobId] = ws;

  ws.onmessage = async (event) => {
    try {
      const data = JSON.parse(event.data);
      await refreshJobs();
      if (data.status === "SUCCEEDED") {
        await loadResults(jobId);
        ws.close();
      }
    } catch (err) {
      console.error("Failed to handle websocket message", err);
    }
  };

  ws.onclose = () => {
    delete jobSockets[jobId];
  };
}

function resizeTrajectoryCanvas() {
  const width = resultVideo.videoWidth || 640;
  const height = resultVideo.videoHeight || 360;
  trajectoryCanvas.width = width;
  trajectoryCanvas.height = height;
}

function computeTrajectorySegments() {
  trajectorySegments = [];
  if (!currentTrajectory.length) return;
  const sorted = [...currentTrajectory].sort((a, b) => a.frame - b.frame);
  let prevDirection = null;
  let topIndex = null;
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const dy = sorted[i + 1].y - sorted[i].y;
    let current = prevDirection;
    if (dy <= -DIRECTION_THRESHOLD) current = "up";
    else if (dy >= DIRECTION_THRESHOLD) current = "down";
    if (prevDirection === "up" && current === "down" && topIndex === null) {
      topIndex = i + 1;
    }
    if (current) prevDirection = current;
  }
  if (topIndex === null) {
    let minIdx = 0;
    for (let i = 1; i < sorted.length; i += 1) {
      if (sorted[i].y < sorted[minIdx].y) minIdx = i;
    }
    topIndex = minIdx;
  }
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const start = sorted[i];
    const end = sorted[i + 1];
    if (end.frame === start.frame) continue;
    const phase = i + 1 <= topIndex ? "backswing" : "downswing";
    trajectorySegments.push({ start, end, phase });
  }
}

function drawTrajectory() {
  if (!trajectoryCtx) return;
  trajectoryCtx.clearRect(0, 0, trajectoryCanvas.width, trajectoryCanvas.height);
  if (!currentTrajectory.length || !currentStats) return;

  const fps = currentStats.fps || 30;
  const currentFrame = Math.floor(resultVideo.currentTime * fps);
  const points = currentTrajectory.filter((p) => p.frame <= currentFrame);
  if (points.length === 0) {
    return;
  }

  trajectoryCtx.lineWidth = PATH_LINE_WIDTH;
  trajectoryCtx.lineCap = "round";
  trajectoryCtx.lineJoin = "round";

  trajectorySegments.forEach((segment) => {
    const { start, end, phase } = segment;
    if (currentFrame < start.frame) return;
    const colour = phase === "backswing" ? BACKSWING_COLOUR : DOWNSWING_COLOUR;
    let endX = end.x;
    let endY = end.y;
    if (currentFrame < end.frame) {
      const span = Math.max(1, end.frame - start.frame);
      const ratio = Math.max(0, Math.min(1, (currentFrame - start.frame) / span));
      endX = start.x + (end.x - start.x) * ratio;
      endY = start.y + (end.y - start.y) * ratio;
    }
    trajectoryCtx.beginPath();
    trajectoryCtx.moveTo(start.x, start.y);
    trajectoryCtx.lineTo(endX, endY);
    trajectoryCtx.strokeStyle = colour;
    trajectoryCtx.stroke();
  });

  const lastSegment = trajectorySegments
    .filter((segment) => currentFrame >= segment.start.frame)
    .slice(-1)[0];
  let lastPoint = points[points.length - 1];
  if (lastSegment && currentFrame > lastSegment.start.frame && currentFrame < lastSegment.end.frame) {
    const span = Math.max(1, lastSegment.end.frame - lastSegment.start.frame);
    const ratio = Math.max(0, Math.min(1, (currentFrame - lastSegment.start.frame) / span));
    lastPoint = {
      x: lastSegment.start.x + (lastSegment.end.x - lastSegment.start.x) * ratio,
      y: lastSegment.start.y + (lastSegment.end.y - lastSegment.start.y) * ratio,
    };
  }

  trajectoryCtx.fillStyle = "#e74c3c";
  trajectoryCtx.beginPath();
  trajectoryCtx.arc(lastPoint.x, lastPoint.y, 8, 0, Math.PI * 2);
  trajectoryCtx.fill();
}

function renderTrajectoryPlot() {
  if (!trajectoryPlotCtx || !currentTrajectory.length) {
    if (trajectoryPlotCtx) trajectoryPlotCtx.clearRect(0, 0, trajectoryPlot.width, trajectoryPlot.height);
    return;
  }

  const padding = 20;
  const width = trajectoryPlot.width;
  const height = trajectoryPlot.height;
  const videoWidth = currentStats?.width || (Math.max(...currentTrajectory.map((p) => p.x)) + 1);
  const videoHeight = currentStats?.height || (Math.max(...currentTrajectory.map((p) => p.y)) + 1);

  trajectoryPlotCtx.clearRect(0, 0, width, height);

  const usableWidth = Math.max(1, videoWidth);
  const usableHeight = Math.max(1, videoHeight);
  const scaleX = (width - padding * 2) / usableWidth;
  const scaleY = (height - padding * 2) / usableHeight;

  const toCanvas = (point) => ({
    x: padding + Math.max(0, Math.min(point.x, usableWidth)) * scaleX,
    y: padding + Math.max(0, Math.min(point.y, usableHeight)) * scaleY,
  });

  trajectoryPlotCtx.lineWidth = 2;
  trajectoryPlotCtx.lineCap = "round";
  trajectoryPlotCtx.lineJoin = "round";
  trajectorySegments.forEach(({ start, end, phase }) => {
    const startPoint = toCanvas(start);
    const endPoint = toCanvas(end);
    trajectoryPlotCtx.beginPath();
    trajectoryPlotCtx.moveTo(startPoint.x, startPoint.y);
    trajectoryPlotCtx.lineTo(endPoint.x, endPoint.y);
    trajectoryPlotCtx.strokeStyle = phase === "backswing" ? BACKSWING_COLOUR : DOWNSWING_COLOUR;
    trajectoryPlotCtx.stroke();
  });

  // Draw start and end points
  const startPoint = toCanvas(currentTrajectory[0]);
  const endPoint = toCanvas(currentTrajectory[currentTrajectory.length - 1]);
  trajectoryPlotCtx.fillStyle = "#2ecc71";
  trajectoryPlotCtx.beginPath();
  trajectoryPlotCtx.arc(startPoint.x, startPoint.y, 6, 0, Math.PI * 2);
  trajectoryPlotCtx.fill();

  trajectoryPlotCtx.fillStyle = "#e74c3c";
  trajectoryPlotCtx.beginPath();
  trajectoryPlotCtx.arc(endPoint.x, endPoint.y, 6, 0, Math.PI * 2);
  trajectoryPlotCtx.fill();
}

function startOverlayLoop() {
  stopOverlayLoop();
  const loop = () => {
    drawTrajectory();
    overlayAnimationId = requestAnimationFrame(loop);
  };
  overlayAnimationId = requestAnimationFrame(loop);
}

function stopOverlayLoop() {
  if (overlayAnimationId) {
    cancelAnimationFrame(overlayAnimationId);
    overlayAnimationId = null;
  }
  if (trajectoryCtx) {
    trajectoryCtx.clearRect(0, 0, trajectoryCanvas.width, trajectoryCanvas.height);
  }
}

resultVideo.addEventListener("loadedmetadata", () => {
  resizeTrajectoryCanvas();
  drawTrajectory();
});

resultVideo.addEventListener("loadeddata", () => {
  drawTrajectory();
});

resultVideo.addEventListener("seeked", () => {
  drawTrajectory();
});

resultVideo.addEventListener("play", () => {
  startOverlayLoop();
});

resultVideo.addEventListener("pause", () => {
  stopOverlayLoop();
  drawTrajectory();
});

resultVideo.addEventListener("ended", () => {
  stopOverlayLoop();
  drawTrajectory();
});

setInterval(refreshJobs, 5000);
refreshJobs();
