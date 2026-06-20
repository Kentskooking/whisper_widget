from __future__ import annotations

from pathlib import Path
import os
from queue import Queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONVERT_TIMEOUT_SECONDS = 180


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whisper Widget Web</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #181b20;
      --panel-2: #20242b;
      --text: #f2f4f8;
      --muted: #aab2c0;
      --border: #343a45;
      --green: #2f8f5b;
      --green-hover: #27784d;
      --red: #b94444;
      --red-hover: #9f3838;
      --blue: #4176d8;
      --blue-hover: #345fb0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(900px, 100%);
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 14px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 4px 0;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 650;
    }
    .status {
      min-height: 22px;
      color: var(--muted);
      text-align: right;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
    }
    button {
      min-width: 112px;
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    #recordButton { background: var(--green); }
    #recordButton:hover:not(:disabled) { background: var(--green-hover); }
    #stopButton { background: var(--red); }
    #stopButton:hover:not(:disabled) { background: var(--red-hover); }
    #copyButton { background: var(--blue); }
    #copyButton:hover:not(:disabled) { background: var(--blue-hover); }
    label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      min-height: 42px;
      padding: 0 4px;
      user-select: none;
    }
    input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--blue);
    }
    #waveform {
      width: 100%;
      height: 34px;
      display: block;
      margin: -2px 0 -4px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #14171c;
    }
    textarea {
      width: 100%;
      min-height: min(54vh, 520px);
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      padding: 14px;
      font: 16px/1.5 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    }
    .meta {
      min-height: 22px;
      color: var(--muted);
    }
    .error { color: #ffb4b4; }
    .recording { color: #ffd596; }
    @media (max-width: 620px) {
      main { padding: 14px; }
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .status { text-align: left; }
      button { flex: 1 1 130px; }
      label { flex-basis: 100%; }
      #waveform { height: 30px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Whisper Widget Web</h1>
      <div id="status" class="status">Checking widget</div>
    </header>
    <section class="toolbar" aria-label="Recorder controls">
      <button id="recordButton" type="button">Record</button>
      <button id="stopButton" type="button" disabled>Stop</button>
      <button id="copyButton" type="button" disabled>Copy</button>
      <label><input id="autoCopy" type="checkbox"> Auto-copy</label>
    </section>
    <canvas id="waveform" aria-hidden="true"></canvas>
    <textarea id="transcript" spellcheck="true" placeholder="Transcript"></textarea>
    <div id="meta" class="meta"></div>
  </main>
  <script>
    const recordButton = document.getElementById("recordButton");
    const stopButton = document.getElementById("stopButton");
    const copyButton = document.getElementById("copyButton");
    const autoCopy = document.getElementById("autoCopy");
    const statusEl = document.getElementById("status");
    const transcriptEl = document.getElementById("transcript");
    const metaEl = document.getElementById("meta");
    const waveformCanvas = document.getElementById("waveform");
    const waveformCtx = waveformCanvas.getContext("2d");

    let mediaStream = null;
    let audioContext = null;
    let sourceNode = null;
    let workletNode = null;
    let muteNode = null;
    let chunkConfig = null;
    let sessionId = "";
    let recordingStartedAt = 0;
    let pollingTimer = null;
    let chunkSessionActive = false;
    let stopRequested = false;
    let finalCopied = false;
    let sampleBuffer = [];
    let bufferStartSample = 0;
    let totalSamplesReceived = 0;
    let lastChunkEndSample = 0;
    let nextChunkIndex = 0;
    let chunksQueued = 0;
    let chunksUploaded = 0;
    let uploadChain = Promise.resolve();
    let uploadFailure = null;
    let resampler = null;
    let waveformLevels = [];
    let waveformAnimationId = 0;
    let waveformActive = false;
    const waveformMaxLevels = 180;

    function setStatus(text, className = "") {
      statusEl.textContent = text;
      statusEl.className = className ? `status ${className}` : "status";
    }

    function sleep(ms) {
      return new Promise((resolve) => window.setTimeout(resolve, ms));
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, { cache: "no-store", ...options });
      let data = {};
      try {
        data = await response.json();
      } catch (error) {
        data = {};
      }
      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      return data;
    }

    function requireOk(data) {
      if (data.ok === false) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    async function refreshHealth() {
      if (chunkSessionActive || sessionId) {
        return;
      }
      try {
        const response = await fetch("/health", { cache: "no-store" });
        const data = await response.json();
        if (data.busy) {
          setStatus("Widget busy");
        } else if (data.model_ready) {
          setStatus(`Ready: ${data.model} on ${data.device}`);
        } else {
          setStatus("Widget loading model");
        }
      } catch (error) {
        setStatus("Widget unavailable", "error");
      }
    }

    function resetRecordingState() {
      clearStatusPolling();
      chunkSessionActive = false;
      stopRequested = false;
      finalCopied = false;
      sessionId = "";
      chunkConfig = null;
      sampleBuffer = [];
      bufferStartSample = 0;
      totalSamplesReceived = 0;
      lastChunkEndSample = 0;
      nextChunkIndex = 0;
      chunksQueued = 0;
      chunksUploaded = 0;
      uploadFailure = null;
      uploadChain = Promise.resolve();
      resampler = null;
      resetWaveform();
    }

    function floatToInt16Sample(value) {
      const sample = Math.max(-1, Math.min(1, value || 0));
      return sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767);
    }

    function floatsToInt16(floatSamples) {
      const output = new Int16Array(floatSamples.length);
      for (let i = 0; i < floatSamples.length; i += 1) {
        output[i] = floatToInt16Sample(floatSamples[i]);
      }
      return output;
    }

    function resizeWaveformCanvas() {
      const pixelRatio = window.devicePixelRatio || 1;
      const rect = waveformCanvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width * pixelRatio));
      const height = Math.max(1, Math.round(rect.height * pixelRatio));
      if (waveformCanvas.width !== width || waveformCanvas.height !== height) {
        waveformCanvas.width = width;
        waveformCanvas.height = height;
      }
      waveformCtx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }

    function resetWaveform() {
      waveformActive = false;
      waveformLevels = new Array(waveformMaxLevels).fill(0);
      stopWaveformAnimation();
      drawWaveform();
    }

    function startWaveform() {
      waveformActive = true;
      startWaveformAnimation();
    }

    function stopWaveform() {
      waveformActive = false;
      startWaveformAnimation();
    }

    function startWaveformAnimation() {
      if (!waveformAnimationId) {
        waveformAnimationId = window.requestAnimationFrame(renderWaveformFrame);
      }
    }

    function stopWaveformAnimation() {
      if (waveformAnimationId) {
        window.cancelAnimationFrame(waveformAnimationId);
      }
      waveformAnimationId = 0;
    }

    function renderWaveformFrame() {
      waveformAnimationId = 0;
      let maxLevel = 0;
      if (!waveformActive) {
        for (let index = 0; index < waveformLevels.length; index += 1) {
          waveformLevels[index] *= 0.82;
          maxLevel = Math.max(maxLevel, waveformLevels[index]);
        }
      }
      drawWaveform();
      if (waveformActive || maxLevel > 0.003) {
        startWaveformAnimation();
      }
    }

    function drawWaveform() {
      resizeWaveformCanvas();
      const width = waveformCanvas.clientWidth;
      const height = waveformCanvas.clientHeight;
      const mid = height / 2;
      waveformCtx.clearRect(0, 0, width, height);
      waveformCtx.fillStyle = "#14171c";
      waveformCtx.fillRect(0, 0, width, height);
      waveformCtx.strokeStyle = "rgba(170, 178, 192, 0.18)";
      waveformCtx.lineWidth = 1;
      waveformCtx.beginPath();
      waveformCtx.moveTo(10, mid);
      waveformCtx.lineTo(width - 10, mid);
      waveformCtx.stroke();

      const usableWidth = Math.max(1, width - 20);
      const step = usableWidth / Math.max(1, waveformLevels.length - 1);
      waveformCtx.strokeStyle = waveformActive
        ? "rgba(73, 190, 125, 0.88)"
        : "rgba(139, 153, 173, 0.5)";
      waveformCtx.lineWidth = Math.max(1, Math.min(3, step * 0.48));
      waveformCtx.beginPath();
      for (let index = 0; index < waveformLevels.length; index += 1) {
        const level = Math.min(1, Math.max(0, waveformLevels[index]));
        const x = 10 + index * step;
        const amp = Math.max(1, level * height * 0.42);
        waveformCtx.moveTo(x, mid - amp);
        waveformCtx.lineTo(x, mid + amp);
      }
      waveformCtx.stroke();
    }

    function feedWaveform(floatSamples) {
      if (!floatSamples || !floatSamples.length) {
        return;
      }

      let peak = 0;
      for (let index = 0; index < floatSamples.length; index += 1) {
        peak = Math.max(peak, Math.abs(floatSamples[index] || 0));
      }

      waveformLevels.push(Math.min(1, peak * 3.5));
      while (waveformLevels.length > waveformMaxLevels) {
        waveformLevels.shift();
      }
    }

    class PcmResampler {
      constructor(sourceRate, targetRate) {
        this.sourceRate = sourceRate;
        this.targetRate = targetRate;
        this.ratio = sourceRate / targetRate;
        this.remainder = new Float32Array(0);
        this.position = 0;
      }

      combine(input) {
        if (!this.remainder.length) {
          return input;
        }
        const combined = new Float32Array(this.remainder.length + input.length);
        combined.set(this.remainder, 0);
        combined.set(input, this.remainder.length);
        return combined;
      }

      process(input) {
        if (!input || !input.length) {
          return new Int16Array(0);
        }

        const data = this.combine(input);
        const output = [];
        while (this.position + 1 < data.length) {
          const index = Math.floor(this.position);
          const fraction = this.position - index;
          const value = data[index] + (data[index + 1] - data[index]) * fraction;
          output.push(floatToInt16Sample(value));
          this.position += this.ratio;
        }

        const keepIndex = Math.max(0, Math.floor(this.position));
        this.remainder = data.slice(keepIndex);
        this.position -= keepIndex;
        return Int16Array.from(output);
      }

      flush() {
        if (!this.remainder.length) {
          return new Int16Array(0);
        }

        const data = new Float32Array(this.remainder.length + 1);
        data.set(this.remainder, 0);
        data[data.length - 1] = this.remainder[this.remainder.length - 1];
        const output = [];
        while (this.position < this.remainder.length) {
          const index = Math.floor(this.position);
          const fraction = this.position - index;
          const value = data[index] + (data[index + 1] - data[index]) * fraction;
          output.push(floatToInt16Sample(value));
          this.position += this.ratio;
        }

        this.remainder = new Float32Array(0);
        this.position = 0;
        return Int16Array.from(output);
      }
    }

    function recorderWorkletSource() {
      return `
        class PcmCaptureProcessor extends AudioWorkletProcessor {
          process(inputs) {
            const input = inputs[0];
            if (!input || !input.length || !input[0] || !input[0].length) {
              return true;
            }

            const frames = input[0].length;
            const mixed = new Float32Array(frames);
            for (let channel = 0; channel < input.length; channel += 1) {
              const channelData = input[channel];
              for (let index = 0; index < frames; index += 1) {
                mixed[index] += channelData[index] / input.length;
              }
            }
            this.port.postMessage(mixed, [mixed.buffer]);
            return true;
          }
        }

        registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
      `;
    }

    async function setupAudioPipeline() {
      if (!window.AudioContext && !window.webkitAudioContext) {
        throw new Error("AudioContext unavailable");
      }

      const ContextClass = window.AudioContext || window.webkitAudioContext;
      const targetRate = Number(chunkConfig.sample_rate);
      audioContext = new ContextClass({ sampleRate: targetRate });
      if (!audioContext.audioWorklet) {
        throw new Error("AudioWorklet unavailable in this browser");
      }

      const workletBlob = new Blob([recorderWorkletSource()], { type: "application/javascript" });
      const workletUrl = URL.createObjectURL(workletBlob);
      try {
        await audioContext.audioWorklet.addModule(workletUrl);
      } finally {
        URL.revokeObjectURL(workletUrl);
      }

      if (audioContext.sampleRate !== targetRate) {
        resampler = new PcmResampler(audioContext.sampleRate, targetRate);
      }

      sourceNode = audioContext.createMediaStreamSource(mediaStream);
      workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
      muteNode = audioContext.createGain();
      muteNode.gain.value = 0;
      workletNode.port.onmessage = (event) => appendFloatSamples(event.data);
      sourceNode.connect(workletNode);
      workletNode.connect(muteNode);
      muteNode.connect(audioContext.destination);
    }

    async function startRecording() {
      if (!window.isSecureContext) {
        setStatus("Use HTTPS or localhost for microphone", "error");
        return;
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("Microphone API unavailable", "error");
        return;
      }

      try {
        resetRecordingState();
        transcriptEl.value = "";
        metaEl.textContent = "";
        copyButton.disabled = true;
        recordButton.disabled = true;
        stopButton.disabled = true;
        setStatus("Opening microphone");

        const configPayload = requireOk(await requestJson("/api/chunked/config"));
        chunkConfig = configPayload.config;

        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: { ideal: 1 },
            sampleRate: { ideal: Number(chunkConfig.sample_rate) },
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        await setupAudioPipeline();

        const startPayload = requireOk(await requestJson("/api/chunked/start", { method: "POST" }));
        sessionId = startPayload.session_id;
        chunkConfig = startPayload.config || chunkConfig;
        chunkSessionActive = true;
        recordingStartedAt = performance.now();
        startStatusPolling();
        await audioContext.resume();
        startWaveform();

        stopButton.disabled = false;
        setStatus("Recording", "recording");
      } catch (error) {
        setStatus(error.message || "Recording failed", "error");
        if (sessionId) {
          await cancelCurrentSession();
        }
        await closeAudioPipeline();
        recordButton.disabled = false;
        stopButton.disabled = true;
        chunkSessionActive = false;
        sessionId = "";
      }
    }

    async function stopRecording() {
      if (stopRequested) {
        return;
      }

      stopRequested = true;
      stopButton.disabled = true;
      setStatus("Finalizing chunks");
      try {
        await closeAudioPipeline();
        finalizeBufferedChunks();
        await uploadChain;
        if (uploadFailure) {
          throw uploadFailure;
        }

        if (sessionId) {
          const data = requireOk(
            await requestJson(`/api/chunked/${encodeURIComponent(sessionId)}/finish`, {
              method: "POST"
            })
          );
          await applySessionStatus(data);
          await waitForFinalStatus();
        }
      } catch (error) {
        setStatus(error.message || "Transcription failed", "error");
        await cancelCurrentSession();
        recordButton.disabled = false;
        stopButton.disabled = true;
        chunkSessionActive = false;
        sessionId = "";
        clearStatusPolling();
      }
    }

    async function closeAudioPipeline() {
      try {
        if (sourceNode) {
          sourceNode.disconnect();
        }
      } catch (error) {
      }
      try {
        if (workletNode) {
          workletNode.disconnect();
        }
      } catch (error) {
      }
      try {
        if (muteNode) {
          muteNode.disconnect();
        }
      } catch (error) {
      }
      try {
        if (audioContext && audioContext.state !== "closed") {
          await audioContext.close();
        }
      } catch (error) {
      }
      if (resampler) {
        appendPcmSamples(resampler.flush());
      }
      if (workletNode) {
        workletNode.port.onmessage = null;
      }
      if (mediaStream) {
        for (const track of mediaStream.getTracks()) {
          track.stop();
        }
      }
      stopWaveform();
      mediaStream = null;
      audioContext = null;
      sourceNode = null;
      workletNode = null;
      muteNode = null;
    }

    function appendFloatSamples(floatSamples) {
      if (!chunkConfig || !floatSamples || !floatSamples.length) {
        return;
      }
      feedWaveform(floatSamples);
      const pcmSamples = resampler ? resampler.process(floatSamples) : floatsToInt16(floatSamples);
      appendPcmSamples(pcmSamples);
    }

    function appendPcmSamples(pcmSamples) {
      if (!pcmSamples || !pcmSamples.length) {
        return;
      }
      for (let i = 0; i < pcmSamples.length; i += 1) {
        sampleBuffer.push(pcmSamples[i]);
      }
      totalSamplesReceived += pcmSamples.length;
      flushCompleteChunks();
      updateCaptureMeta();
    }

    function flushCompleteChunks() {
      const chunkSamples = Number(chunkConfig.chunk_samples);
      const overlapSamples = Number(chunkConfig.overlap_samples);

      while (totalSamplesReceived - bufferStartSample >= chunkSamples) {
        const startSample = bufferStartSample;
        const endSample = startSample + chunkSamples;
        const samples = Int16Array.from(sampleBuffer.slice(0, chunkSamples));
        queueChunk(startSample, endSample, false, samples);

        const keepStartSample = endSample - overlapSamples;
        const dropSamples = keepStartSample - bufferStartSample;
        sampleBuffer.splice(0, dropSamples);
        bufferStartSample = keepStartSample;
      }
    }

    function finalizeBufferedChunks() {
      if (totalSamplesReceived > lastChunkEndSample && sampleBuffer.length) {
        const sampleCount = totalSamplesReceived - bufferStartSample;
        const samples = Int16Array.from(sampleBuffer.slice(0, sampleCount));
        queueChunk(bufferStartSample, totalSamplesReceived, true, samples);
        sampleBuffer = [];
        bufferStartSample = totalSamplesReceived;
      }
    }

    function queueChunk(startSample, endSample, isFinal, samples) {
      const chunk = {
        index: nextChunkIndex,
        startSample,
        endSample,
        isFinal,
        samples
      };
      nextChunkIndex += 1;
      chunksQueued += 1;
      lastChunkEndSample = Math.max(lastChunkEndSample, endSample);
      uploadChain = uploadChain
        .then(() => uploadChunk(chunk))
        .catch((error) => {
          uploadFailure = error;
          setStatus(error.message || "Chunk upload failed", "error");
          throw error;
        });
      updateCaptureMeta();
    }

    async function uploadChunk(chunk) {
      if (!sessionId) {
        throw new Error("Chunk session has not started");
      }

      const formData = new FormData();
      formData.append("chunk_index", String(chunk.index));
      formData.append("start_sample", String(chunk.startSample));
      formData.append("end_sample", String(chunk.endSample));
      formData.append("is_final", chunk.isFinal ? "true" : "false");
      formData.append(
        "audio",
        buildWavBlob(chunk.samples, Number(chunkConfig.sample_rate)),
        `chunk_${String(chunk.index).padStart(4, "0")}_${String(chunk.startSample).padStart(10, "0")}_${String(chunk.endSample).padStart(10, "0")}.wav`
      );

      const data = requireOk(
        await requestJson(`/api/chunked/${encodeURIComponent(sessionId)}/chunks`, {
          method: "POST",
          body: formData
        })
      );
      chunksUploaded += 1;
      await applySessionStatus(data);
    }

    function buildWavBlob(samples, sampleRate) {
      const bytesPerSample = 2;
      const channelCount = 1;
      const dataBytes = samples.length * bytesPerSample;
      const buffer = new ArrayBuffer(44 + dataBytes);
      const view = new DataView(buffer);

      writeAscii(view, 0, "RIFF");
      view.setUint32(4, 36 + dataBytes, true);
      writeAscii(view, 8, "WAVE");
      writeAscii(view, 12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, channelCount, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * channelCount * bytesPerSample, true);
      view.setUint16(32, channelCount * bytesPerSample, true);
      view.setUint16(34, 16, true);
      writeAscii(view, 36, "data");
      view.setUint32(40, dataBytes, true);

      for (let i = 0; i < samples.length; i += 1) {
        view.setInt16(44 + i * 2, samples[i], true);
      }

      return new Blob([buffer], { type: "audio/wav" });
    }

    function writeAscii(view, offset, text) {
      for (let index = 0; index < text.length; index += 1) {
        view.setUint8(offset + index, text.charCodeAt(index));
      }
    }

    function updateCaptureMeta(statusPayload = null) {
      if (!chunkConfig) {
        return;
      }
      const capturedSeconds = totalSamplesReceived / Number(chunkConfig.sample_rate);
      const localPart = `Captured ${capturedSeconds.toFixed(1)}s / queued ${chunksQueued} / uploaded ${chunksUploaded}`;
      if (!statusPayload) {
        metaEl.textContent = localPart;
        return;
      }
      metaEl.textContent = `${localPart} / server ${statusPayload.chunks_transcribed}/${statusPayload.chunks_received}`;
    }

    function startStatusPolling() {
      clearStatusPolling();
      pollingTimer = window.setInterval(pollSessionStatus, 1000);
    }

    function clearStatusPolling() {
      if (pollingTimer) {
        window.clearInterval(pollingTimer);
      }
      pollingTimer = null;
    }

    async function pollSessionStatus() {
      if (!sessionId) {
        return;
      }
      try {
        const data = await requestJson(`/api/chunked/${encodeURIComponent(sessionId)}`);
        await applySessionStatus(data);
      } catch (error) {
        if (chunkSessionActive) {
          setStatus(error.message || "Status unavailable", "error");
        }
      }
    }

    async function applySessionStatus(data) {
      updateCaptureMeta(data);
      const text = data.final ? (data.final_text || "") : (data.partial_text || "");
      transcriptEl.value = text;
      copyButton.disabled = !text;

      if (data.status === "failed" || data.status === "cancelled" || data.ok === false) {
        setStatus(data.error || "Transcription failed", "error");
        chunkSessionActive = false;
        clearStatusPolling();
        recordButton.disabled = false;
        stopButton.disabled = true;
        sessionId = "";
        return;
      }

      if (data.final) {
        chunkSessionActive = false;
        clearStatusPolling();
        recordButton.disabled = false;
        stopButton.disabled = true;
        sessionId = "";
        if (text) {
          setStatus("Ready");
          await maybeAutoCopyFinal(text);
        } else {
          setStatus("No text returned");
        }
        return;
      }

      if (stopRequested) {
        setStatus("Transcribing final chunks");
      } else {
        setStatus("Recording", "recording");
      }
    }

    async function waitForFinalStatus() {
      const started = performance.now();
      const timeoutMs = 60 * 60 * 1000;
      while (sessionId && performance.now() - started < timeoutMs) {
        const data = await requestJson(`/api/chunked/${encodeURIComponent(sessionId)}`);
        await applySessionStatus(data);
        if (data.final || data.status === "failed" || data.status === "cancelled") {
          return data;
        }
        await sleep(1000);
      }
      if (sessionId) {
        throw new Error("Timed out waiting for final transcript");
      }
      return null;
    }

    async function maybeAutoCopyFinal(text) {
      if (finalCopied || !autoCopy.checked || !text) {
        return;
      }
      finalCopied = true;
      try {
        await navigator.clipboard.writeText(text);
        setStatus("Copied");
      } catch (error) {
        setStatus("Ready");
      }
    }

    async function cancelCurrentSession() {
      if (!sessionId) {
        return;
      }
      try {
        await fetch(`/api/chunked/${encodeURIComponent(sessionId)}`, {
          method: "DELETE",
          cache: "no-store"
        });
      } catch (error) {
      }
    }

    async function copyTranscript() {
      const text = transcriptEl.value;
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        setStatus("Copied");
      } catch (error) {
        transcriptEl.focus();
        transcriptEl.select();
        setStatus("Select text to copy", "error");
      }
    }

    window.addEventListener("pagehide", () => {
      if (sessionId && chunkSessionActive) {
        fetch(`/api/chunked/${encodeURIComponent(sessionId)}`, {
          method: "DELETE",
          keepalive: true
        }).catch(() => {});
      }
    });

    recordButton.addEventListener("click", startRecording);
    stopButton.addEventListener("click", stopRecording);
    copyButton.addEventListener("click", copyTranscript);
    window.addEventListener("resize", drawWaveform);
    resetWaveform();
    refreshHealth();
    window.setInterval(refreshHealth, 10000);
  </script>
</body>
</html>
"""


def safe_upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix or len(suffix) > 12:
        return ".webm"
    allowed = set(".abcdefghijklmnopqrstuvwxyz0123456789")
    if any(char not in allowed for char in suffix):
        return ".webm"
    return suffix


def save_upload(upload: UploadFile, output_dir: str) -> tuple[str, int]:
    suffix = safe_upload_suffix(upload.filename)
    output_path = os.path.join(output_dir, f"upload{suffix}")
    total_bytes = 0
    with open(output_path, "wb") as output:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise ValueError("Upload is too large")
            output.write(chunk)

    if total_bytes == 0:
        raise ValueError("Upload is empty")

    return output_path, total_bytes


def convert_to_wav(input_path: str, output_path: str) -> float:
    started = time.perf_counter()
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg was not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {CONVERT_TIMEOUT_SECONDS}s") from e

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()[-1200:]
        raise RuntimeError(f"ffmpeg conversion failed: {stderr or completed.returncode}")

    return time.perf_counter() - started


def validate_chunk_wav(path: str, sample_rate: int, channels: int, sample_width: int, frames: int):
    with wave.open(path, "rb") as reader:
        actual_channels = reader.getnchannels()
        actual_sample_width = reader.getsampwidth()
        actual_sample_rate = reader.getframerate()
        actual_frames = reader.getnframes()

    if actual_channels != channels:
        raise ValueError(f"Chunk channel mismatch: expected {channels}, got {actual_channels}")
    if actual_sample_width != sample_width:
        raise ValueError(
            f"Chunk sample width mismatch: expected {sample_width}, got {actual_sample_width}"
        )
    if actual_sample_rate != sample_rate:
        raise ValueError(f"Chunk sample rate mismatch: expected {sample_rate}, got {actual_sample_rate}")
    if actual_frames != frames:
        raise ValueError(f"Chunk frame mismatch: expected {frames}, got {actual_frames}")


def save_chunk_upload(upload: UploadFile, output_path: str) -> int:
    total_bytes = 0
    with open(output_path, "wb") as output:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise ValueError("Chunk upload is too large")
            output.write(chunk)

    if total_bytes == 0:
        raise ValueError("Chunk upload is empty")

    return total_bytes


class RemoteChunkSession:
    def __init__(self, manager, session_id: str, session_dir: str, config: dict):
        self.manager = manager
        self.widget = manager.widget
        self.session_id = session_id
        self.session_dir = session_dir
        self.config = dict(config)
        self.queue = Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"web_chunk_session_{session_id[:8]}",
        )
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.status = "recording"
        self.error = None
        self.next_expected_index = 0
        self.chunks_received = 0
        self.chunks_transcribed = 0
        self.final_chunk_seen = False
        self.final = False
        self.manager_finished = False
        self.partial_text = ""
        self.final_text = ""
        self.results = {}
        self.received_chunks = {}
        os.makedirs(self.session_dir, exist_ok=True)

    def start(self):
        self.thread.start()

    def status_payload(self):
        with self.lock:
            return {
                "ok": self.error is None,
                "session_id": self.session_id,
                "status": self.status,
                "error": self.error,
                "chunks_received": self.chunks_received,
                "chunks_transcribed": self.chunks_transcribed,
                "partial_text": self.partial_text,
                "final_text": self.final_text,
                "final": self.final,
                "config": dict(self.config),
            }

    def enqueue_chunk(
        self,
        chunk_index: int,
        start_sample: int,
        end_sample: int,
        is_final: bool,
        upload: UploadFile,
    ):
        sample_count = end_sample - start_sample
        if sample_count <= 0:
            raise ValueError("Chunk end_sample must be greater than start_sample")

        with self.lock:
            if self.status not in ("recording", "processing"):
                raise ValueError(f"Session is not accepting chunks: {self.status}")
            if self.final_chunk_seen:
                raise ValueError("Final chunk has already been received")
            if chunk_index != self.next_expected_index:
                raise ValueError(
                    f"Unexpected chunk index {chunk_index}; expected {self.next_expected_index}"
                )

            filename = f"chunk_{chunk_index:04d}_{start_sample:010d}_{end_sample:010d}.wav"
            chunk_path = os.path.join(self.session_dir, filename)

        upload_bytes = save_chunk_upload(upload, chunk_path)
        validate_chunk_wav(
            chunk_path,
            sample_rate=int(self.config["sample_rate"]),
            channels=int(self.config["channels"]),
            sample_width=int(self.config["sample_width"]),
            frames=sample_count,
        )

        chunk = {
            "index": chunk_index,
            "path": chunk_path,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "seconds": sample_count / float(self.config["sample_rate"]),
            "is_final": bool(is_final),
        }

        with self.lock:
            self.next_expected_index += 1
            self.chunks_received += 1
            self.received_chunks[chunk_index] = chunk
            self.updated_at = time.time()
            self.status = "processing"
            if is_final:
                self.final_chunk_seen = True

        self.manager.log_event(
            "web_chunk_received",
            session_id=self.session_id,
            chunk_index=chunk_index,
            start_sample=start_sample,
            end_sample=end_sample,
            final=is_final,
            bytes=upload_bytes,
            file=chunk_path,
        )
        self.queue.put(chunk)
        return self.status_payload()

    def finish(self):
        should_queue_finish = False
        with self.lock:
            if not self.final and not self.final_chunk_seen:
                if self.status not in ("recording", "processing"):
                    raise ValueError(f"Session cannot finish from state: {self.status}")
                self.final_chunk_seen = True
                self.status = "processing"
                self.updated_at = time.time()
                should_queue_finish = True

        if should_queue_finish:
            self.manager.log_event(
                "web_chunk_session_finish_requested",
                session_id=self.session_id,
                chunks_received=self.chunks_received,
            )
            self.queue.put({"finish": True})
        return self.status_payload()

    def cancel(self, reason: str = "cancelled"):
        should_cancel = False
        with self.lock:
            if not self.final:
                self.status = "cancelled"
                self.error = reason
                self.updated_at = time.time()
                should_cancel = True
        if should_cancel:
            self.queue.put(None)
            self.manager.finish_session(self, success=False, reason=reason)
        return self.status_payload()

    def _worker_loop(self):
        try:
            while True:
                chunk = self.queue.get()
                try:
                    if chunk is None:
                        return
                    if chunk.get("finish"):
                        self._complete_success()
                        return
                    self._process_chunk(chunk)
                    if chunk.get("is_final"):
                        self._complete_success()
                        return
                finally:
                    self.queue.task_done()
        except Exception as e:
            self._complete_failure(str(e))

    def _process_chunk(self, chunk):
        chunk_index = chunk.get("index")
        result = self.widget.process_and_transcribe_chunk(chunk)
        with self.lock:
            self.results[chunk_index] = result
            self.chunks_transcribed = len(self.results)
            ordered_results = [self.results[index] for index in sorted(self.results)]
            self.partial_text = self.widget.stitch_chunk_transcripts(ordered_results)
            self.updated_at = time.time()

        self.manager.log_event(
            "web_chunk_transcribed",
            session_id=self.session_id,
            chunk_index=chunk_index,
            ok=result.get("ok"),
            chars=len(result.get("text") or ""),
            final=chunk.get("is_final"),
        )

    def _complete_success(self):
        with self.lock:
            ordered_results = [self.results[index] for index in sorted(self.results)]
            failed_results = [result for result in ordered_results if not result.get("ok")]
            if failed_results:
                raise RuntimeError(f"chunk_transcription_failures={len(failed_results)}")

            final_text = self.widget.stitch_chunk_transcripts(ordered_results)
            self.partial_text = final_text
            self.final_text = final_text
            self.final = True
            self.status = "complete"
            self.updated_at = time.time()

        self.widget.write_chunk_transcript_debug(ordered_results, final_text, reason="web_success")
        if final_text:
            if hasattr(self.widget, "print_transcription_to_terminal"):
                self.widget.print_transcription_to_terminal(final_text)
            self.widget.log_transcription(final_text)
        self.manager.log_event(
            "web_chunk_session_complete",
            session_id=self.session_id,
            chunks=len(ordered_results),
            chars=len(final_text),
        )
        self.manager.finish_session(self, success=True, reason="complete")

    def _complete_failure(self, error: str):
        with self.lock:
            ordered_results = [self.results[index] for index in sorted(self.results)]
            self.status = "failed"
            self.error = error
            self.final = True
            self.updated_at = time.time()

        self.widget.write_chunk_transcript_debug(ordered_results, "", reason="web_failure")
        self.manager.log_event(
            "web_chunk_session_failed",
            session_id=self.session_id,
            error=error,
        )
        self.manager.finish_session(self, success=False, reason="failure")


class RemoteChunkSessionManager:
    def __init__(self, widget, work_dir: str, log_fn):
        self.widget = widget
        self.work_dir = work_dir
        self.log_fn = log_fn
        self.lock = threading.Lock()
        self.sessions = {}
        self.active_session_id = None
        os.makedirs(self.work_dir, exist_ok=True)

    def log_event(self, event, **fields):
        try:
            self.log_fn(event, **fields)
        except Exception:
            pass

    def chunk_config(self):
        return self.widget.web_chunk_config_payload()

    def start_session(self):
        with self.lock:
            if self.active_session_id:
                active = self.sessions.get(self.active_session_id)
                if active and active.status in ("recording", "processing"):
                    raise RuntimeError("Another chunked web session is already active")

            if self.widget.is_recording or self.widget.is_processing:
                raise RuntimeError("Widget is busy recording or processing")

            if not self.widget.web_pipeline_lock.acquire(blocking=False):
                raise RuntimeError("Widget web pipeline is busy")

            try:
                session_id = datetime_session_id()
                session_dir = os.path.join(self.work_dir, "chunked_sessions", session_id)
                session = RemoteChunkSession(
                    manager=self,
                    session_id=session_id,
                    session_dir=session_dir,
                    config=self.chunk_config(),
                )
                self.sessions[session_id] = session
                self.active_session_id = session_id
            except Exception:
                self.widget.web_pipeline_lock.release()
                raise

        self.widget.is_processing = True
        self.widget.ui_call(self.widget.update_ui_state, "processing")
        self.widget.write_runtime_heartbeat(reason="web_chunk_session_start")
        session.start()
        self.log_event("web_chunk_session_started", session_id=session_id, dir=session_dir)
        return session.status_payload()

    def get_session(self, session_id: str):
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def finish_session(self, session: RemoteChunkSession, success: bool, reason: str):
        cleanup_dir = session.session_dir
        with self.lock:
            if session.manager_finished:
                return
            session.manager_finished = True
            if self.active_session_id == session.session_id:
                self.active_session_id = None

        try:
            if success:
                shutil.rmtree(cleanup_dir)
                self.log_event(
                    "web_chunk_session_temp_removed",
                    session_id=session.session_id,
                    dir=cleanup_dir,
                )
            else:
                self.log_event(
                    "web_chunk_session_temp_preserved",
                    session_id=session.session_id,
                    dir=cleanup_dir,
                    reason=reason,
                )
        except Exception as e:
            self.log_event(
                "web_chunk_session_temp_cleanup_failed",
                session_id=session.session_id,
                dir=cleanup_dir,
                error=e,
            )

        self.widget.ui_call(self.widget.finish_processing)
        try:
            self.widget.web_pipeline_lock.release()
        except RuntimeError:
            pass


def datetime_session_id():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


class WhisperWidgetWebServer:
    def __init__(
        self,
        widget,
        host: str,
        port: int,
        work_dir: str,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        log_fn,
    ):
        self.widget = widget
        self.host = host
        self.port = int(port)
        self.work_dir = work_dir
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.log_fn = log_fn
        self.server = None
        self.thread = None
        self.chunk_sessions = RemoteChunkSessionManager(
            widget=self.widget,
            work_dir=self.work_dir,
            log_fn=self.log_fn,
        )
        os.makedirs(self.work_dir, exist_ok=True)

    def log_event(self, event, **fields):
        try:
            self.log_fn(event, **fields)
        except Exception:
            pass

    def create_app(self):
        app = FastAPI(title="Whisper Widget Web")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse(INDEX_HTML)

        @app.get("/health")
        def health():
            return self.widget.web_status_payload()

        @app.get("/api/chunked/config")
        def chunked_config():
            return {
                "ok": True,
                "config": self.chunk_sessions.chunk_config(),
            }

        @app.post("/api/chunked/start")
        def start_chunked():
            try:
                return self.chunk_sessions.start_session()
            except Exception as e:
                self.log_event("web_chunk_session_start_failed", error=e)
                return JSONResponse(
                    {
                        "ok": False,
                        "busy": True,
                        "error": str(e),
                    },
                    status_code=409,
                )

        @app.post("/api/chunked/{session_id}/chunks")
        def upload_chunk(
            session_id: str,
            chunk_index: int = Form(...),
            start_sample: int = Form(...),
            end_sample: int = Form(...),
            is_final: bool = Form(False),
            audio: UploadFile = File(...),
        ):
            try:
                session = self.chunk_sessions.get_session(session_id)
                return session.enqueue_chunk(
                    chunk_index=chunk_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    is_final=is_final,
                    upload=audio,
                )
            except KeyError:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "Unknown chunked transcription session",
                    },
                    status_code=404,
                )
            except ValueError as e:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": str(e),
                    },
                    status_code=400,
                )
            except Exception as e:
                self.log_event("web_chunk_upload_failed", session_id=session_id, error=e)
                return JSONResponse(
                    {
                        "ok": False,
                        "error": str(e),
                    },
                    status_code=500,
                )

        @app.post("/api/chunked/{session_id}/finish")
        def finish_chunked(session_id: str):
            try:
                session = self.chunk_sessions.get_session(session_id)
                return session.finish()
            except KeyError:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "Unknown chunked transcription session",
                    },
                    status_code=404,
                )
            except ValueError as e:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": str(e),
                    },
                    status_code=400,
                )
            except Exception as e:
                self.log_event("web_chunk_finish_failed", session_id=session_id, error=e)
                return JSONResponse(
                    {
                        "ok": False,
                        "error": str(e),
                    },
                    status_code=500,
                )

        @app.get("/api/chunked/{session_id}")
        def chunked_status(session_id: str):
            try:
                session = self.chunk_sessions.get_session(session_id)
                return session.status_payload()
            except KeyError:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "Unknown chunked transcription session",
                    },
                    status_code=404,
                )

        @app.delete("/api/chunked/{session_id}")
        def cancel_chunked(session_id: str):
            try:
                session = self.chunk_sessions.get_session(session_id)
                return session.cancel()
            except KeyError:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "Unknown chunked transcription session",
                    },
                    status_code=404,
                )

        @app.post("/api/transcribe")
        def transcribe(audio: UploadFile = File(...)):
            upload_root = os.path.join(self.work_dir, "uploads")
            os.makedirs(upload_root, exist_ok=True)
            request_dir = tempfile.mkdtemp(prefix="upload_", dir=upload_root)
            upload_path = None
            wav_path = os.path.join(request_dir, "converted.wav")
            success = False

            try:
                upload_path, upload_bytes = save_upload(audio, request_dir)
                self.log_event(
                    "web_widget_upload_saved",
                    file=upload_path,
                    bytes=upload_bytes,
                    content_type=audio.content_type,
                )
                convert_seconds = convert_to_wav(upload_path, wav_path)
                self.log_event(
                    "web_widget_upload_converted",
                    source=upload_path,
                    output=wav_path,
                    convert_seconds=f"{convert_seconds:.3f}",
                )
                result = self.widget.transcribe_web_wav(wav_path, source_label="web_upload")
                timings = dict(result.get("timings") or {})
                timings["convert_seconds"] = convert_seconds
                result["timings"] = timings
                success = bool(result.get("ok"))
                status_code = 200 if success else 409 if result.get("busy") else 500
                return JSONResponse(result, status_code=status_code)
            except Exception as e:
                self.log_event("web_widget_upload_transcription_failed", file=upload_path, error=e)
                return JSONResponse(
                    {
                        "ok": False,
                        "text": "",
                        "error": str(e),
                        "timings": {},
                    },
                    status_code=500,
                )
            finally:
                if success:
                    try:
                        shutil.rmtree(request_dir)
                        self.log_event("web_widget_upload_temp_removed", dir=request_dir)
                    except Exception as e:
                        self.log_event("web_widget_upload_temp_remove_failed", dir=request_dir, error=e)
                else:
                    self.log_event("web_widget_upload_temp_preserved", dir=request_dir)

        return app

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return

        import uvicorn

        app = self.create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            ssl_certfile=self.ssl_certfile,
            ssl_keyfile=self.ssl_keyfile,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run,
            daemon=True,
            name="widget_web_server",
        )
        self.thread.start()
        self.log_event("web_widget_server_start_requested", host=self.host, port=self.port)

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.log_event("web_widget_server_stop_requested", host=self.host, port=self.port)
