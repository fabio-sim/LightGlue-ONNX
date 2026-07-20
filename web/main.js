const ORT_VERSION = '1.27.0';
const RELEASE_TAG = 'v3.0';
const SAMPLE_LEFT = 'assets/marunouchi-left.jpg';
const SAMPLE_RIGHT = 'assets/marunouchi-right.jpg';
const ROTATION_STEP_DEGREES = 5;

const $ = (id) => document.getElementById(id);
const elements = {
  modelSource: $('model-source'),
  releaseModelField: $('release-model-field'),
  releaseModel: $('release-model'),
  localModelField: $('local-model-field'),
  localModel: $('local-model'),
  demoMode: $('demo-mode'),
  resolution: $('resolution'),
  power: $('power'),
  warmup: $('warmup'),
  runsField: $('runs-field'),
  runs: $('runs'),
  customImages: $('custom-images'),
  leftFile: $('left-file'),
  rightFile: $('right-file'),
  run: $('run'),
  stop: $('stop'),
  progress: $('progress'),
  progressLabel: $('progress-label'),
  readiness: $('readiness'),
  provider: $('metric-provider'),
  median: $('metric-median'),
  matches: $('metric-matches'),
  model: $('metric-model'),
  canvas: $('matches'),
  placeholder: $('canvas-placeholder'),
  status: $('status'),
};

const context = elements.canvas.getContext('2d');
const state = {
  sampleImages: null,
  customImages: null,
  customImageKey: null,
  session: null,
  sessionKey: null,
  provider: null,
  adapter: null,
  initializationMs: null,
  modelLoadMs: null,
  webgpuDevice: null,
  webgpuPreference: null,
  runToken: 0,
  busy: false,
};

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1)
  : 1;

function releaseFilename() {
  return `raco_aliked_lightglue_pipeline_k${elements.releaseModel.value}.onnx`;
}

function sessionPowerKey() {
  return state.webgpuDevice ? state.webgpuPreference : elements.power.value;
}

function modelLabel() {
  if (elements.modelSource.value === 'release') return `v3.0 · K${elements.releaseModel.value}`;
  return elements.localModel.files[0]?.name || 'local model';
}

function selectedDimensions() {
  const [width, height] = elements.resolution.value.split('x').map(Number);
  return {width, height};
}

function percentile(values, fraction) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1);
  return sorted[Math.max(0, index)];
}

function formatMilliseconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : '—';
}

function setProgress(value, label) {
  elements.progress.value = Math.max(0, Math.min(1, value));
  elements.progressLabel.textContent = label;
}

function setStatus(details) {
  elements.status.textContent = typeof details === 'string'
    ? details
    : JSON.stringify(details, null, 2);
}

function setBusy(busy, rotation = false) {
  state.busy = busy;
  elements.run.disabled = busy || !inputsReady();
  elements.stop.disabled = !busy || !rotation;
  for (const element of [
    elements.modelSource,
    elements.releaseModel,
    elements.localModel,
    elements.demoMode,
    elements.resolution,
    elements.power,
    elements.warmup,
    elements.runs,
    elements.leftFile,
    elements.rightFile,
  ]) {
    element.disabled = busy;
  }
}

function inputsReady() {
  const modelReady = elements.modelSource.value === 'release' || elements.localModel.files.length === 1;
  const imagesReady = elements.demoMode.value !== 'custom'
    ? state.sampleImages !== null
    : elements.leftFile.files.length === 1 && elements.rightFile.files.length === 1;
  return modelReady && imagesReady;
}

function refreshReadiness() {
  if (!state.sampleImages && elements.demoMode.value !== 'custom') {
    elements.readiness.textContent = 'Loading the sample pair…';
  } else if (!inputsReady()) {
    elements.readiness.textContent = 'Select the required local files.';
  } else {
    elements.readiness.textContent = 'Ready to run.';
  }
  elements.run.disabled = state.busy || !inputsReady();
}

function createSurface(width, height) {
  if ('OffscreenCanvas' in globalThis) return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

async function bitmapFromBlob(blob) {
  try {
    return await createImageBitmap(blob, {imageOrientation: 'from-image'});
  } catch {
    return createImageBitmap(blob);
  }
}

async function fetchBitmap(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Unable to load ${url}: HTTP ${response.status}`);
  return bitmapFromBlob(await response.blob());
}

function drawCover(draw, image, width, height) {
  const sourceAspect = image.width / image.height;
  const targetAspect = width / height;
  let sourceWidth = image.width;
  let sourceHeight = image.height;
  let sourceX = 0;
  let sourceY = 0;
  if (sourceAspect > targetAspect) {
    sourceWidth = image.height * targetAspect;
    sourceX = (image.width - sourceWidth) / 2;
  } else if (sourceAspect < targetAspect) {
    sourceHeight = image.width / targetAspect;
    sourceY = (image.height - sourceHeight) / 2;
  }
  draw.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, width, height);
}

function renderBitmap(bitmap, width, height) {
  const surface = createSurface(width, height);
  const draw = surface.getContext('2d', {willReadFrequently: true});
  draw.fillStyle = '#090d0c';
  draw.fillRect(0, 0, width, height);
  drawCover(draw, bitmap, width, height);
  return surface;
}

function renderRotated(base, angleDegrees) {
  const surface = createSurface(base.width, base.height);
  const draw = surface.getContext('2d', {willReadFrequently: true});
  draw.fillStyle = '#090d0c';
  draw.fillRect(0, 0, base.width, base.height);
  draw.translate(base.width / 2, base.height / 2);
  draw.rotate(angleDegrees * Math.PI / 180);
  draw.drawImage(base, -base.width / 2, -base.height / 2);
  return surface;
}

function surfaceRgba(surface) {
  const draw = surface.getContext('2d', {willReadFrequently: true});
  return draw.getImageData(0, 0, surface.width, surface.height).data;
}

function rgbNchw(left, right, width, height) {
  const area = width * height;
  const output = new Float32Array(2 * 3 * area);
  const images = [left, right];
  for (let batch = 0; batch < 2; batch += 1) {
    const rgba = images[batch];
    const base = batch * 3 * area;
    for (let pixel = 0; pixel < area; pixel += 1) {
      const source = pixel * 4;
      output[base + pixel] = rgba[source] / 255;
      output[base + area + pixel] = rgba[source + 1] / 255;
      output[base + 2 * area + pixel] = rgba[source + 2] / 255;
    }
  }
  return new ort.Tensor('float32', output, [2, 3, height, width]);
}

function hidePlaceholder() {
  elements.placeholder.classList.add('is-hidden');
}

function drawPair(left, right) {
  elements.canvas.width = left.width + right.width;
  elements.canvas.height = Math.max(left.height, right.height);
  context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  context.drawImage(left, 0, 0);
  context.drawImage(right, left.width, 0);
  hidePlaceholder();
}

function insideCircle(x, y, width, height) {
  const radius = Math.min(width, height) / 2;
  const dx = x - width / 2;
  const dy = y - height / 2;
  return dx * dx + dy * dy <= radius * radius;
}

function drawMatches(left, right, outputs, circleFilter = false) {
  const width = left.width;
  const height = left.height;
  drawPair(left, right);

  const keypoints = outputs.keypoints.data;
  const keypointCount = outputs.keypoints.dims[1];
  const matches = outputs.matches.data;
  const scores = outputs.mscores.data;
  const plotted = [];

  for (let row = 0; row < outputs.matches.dims[0]; row += 1) {
    const matchOffset = row * 3;
    if (Number(matches[matchOffset]) !== 0) continue;
    const first = Number(matches[matchOffset + 1]);
    const second = Number(matches[matchOffset + 2]);
    const leftOffset = first * 2;
    const rightOffset = (keypointCount + second) * 2;
    const x0 = keypoints[leftOffset];
    const y0 = keypoints[leftOffset + 1];
    const x1 = keypoints[rightOffset];
    const y1 = keypoints[rightOffset + 1];
    if (circleFilter && (!insideCircle(x0, y0, width, height) || !insideCircle(x1, y1, width, height))) {
      continue;
    }
    plotted.push({x0, y0, x1, y1, score: scores[row]});
  }

  context.lineWidth = Math.max(0.7, width / 1100);
  context.lineCap = 'round';
  for (const match of plotted) {
    const hue = 320 * Math.max(0, Math.min(1, match.x0 / width));
    const alpha = 0.22 + 0.5 * Math.max(0, Math.min(1, match.score));
    context.strokeStyle = `hsla(${hue}, 78%, 62%, ${alpha})`;
    context.beginPath();
    context.moveTo(match.x0, match.y0);
    context.lineTo(width + match.x1, match.y1);
    context.stroke();
  }

  const radius = Math.max(1.2, width / 430);
  for (const match of plotted) {
    const hue = 320 * Math.max(0, Math.min(1, match.x0 / width));
    context.fillStyle = `hsl(${hue}, 82%, 70%)`;
    context.beginPath();
    context.arc(match.x0, match.y0, radius, 0, 2 * Math.PI);
    context.arc(width + match.x1, match.y1, radius, 0, 2 * Math.PI);
    context.fill();
  }
  return plotted.length;
}

function disposeOutputs(outputs) {
  if (!outputs) return;
  for (const tensor of Object.values(outputs)) tensor?.dispose?.();
}

function assertCurrent(token) {
  if (token !== state.runToken) throw new DOMException('Run cancelled', 'AbortError');
}

async function readResponseWithProgress(response, token) {
  if (!response.ok) throw new Error(`Model request failed: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const bytes = total ? new Uint8Array(total) : null;
  const chunks = bytes ? null : [];
  let received = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    assertCurrent(token);
    if (bytes) bytes.set(value, received);
    else chunks.push(value);
    received += value.byteLength;
    const fraction = total ? received / total : 0;
    setProgress(fraction * 0.45, total
      ? `Downloading model · ${(100 * fraction).toFixed(0)}%`
      : `Downloading model · ${(received / 1048576).toFixed(1)} MB`);
  }
  if (bytes) return received === bytes.byteLength ? bytes : bytes.slice(0, received);
  const output = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

async function loadModel(token) {
  const start = performance.now();
  let bytes;
  let key;
  if (elements.modelSource.value === 'release') {
    const filename = releaseFilename();
    key = `release:${RELEASE_TAG}:${filename}:${sessionPowerKey()}`;
    if (state.sessionKey === key && state.session) return {key, bytes: null, loadMs: 0};
    const response = await fetch(`/release/${RELEASE_TAG}/${filename}`);
    bytes = await readResponseWithProgress(response, token);
  } else {
    const file = elements.localModel.files[0];
    if (!file) throw new Error('Select a local ONNX model.');
    key = `local:${file.name}:${file.size}:${file.lastModified}:${sessionPowerKey()}`;
    if (state.sessionKey === key && state.session) return {key, bytes: null, loadMs: 0};
    setProgress(0.15, 'Reading local model…');
    bytes = new Uint8Array(await file.arrayBuffer());
  }
  return {key, bytes, loadMs: performance.now() - start};
}

function adapterDetails(adapter) {
  if (!adapter) return null;
  const info = adapter.info;
  if (!info) return {available: true};
  const details = {};
  for (const key of ['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize']) {
    if (info[key] !== undefined && info[key] !== '') details[key] = info[key];
  }
  return Object.keys(details).length ? details : {available: true};
}

function adapterRequiredLimits(adapter) {
  const limits = {};
  for (const key of Object.getOwnPropertyNames(Object.getPrototypeOf(adapter.limits))) {
    const value = adapter.limits[key];
    if (key !== 'constructor' && typeof value === 'number') limits[key] = value;
  }
  return limits;
}

async function ensureWebGpuDevice() {
  if (state.webgpuDevice) {
    return {device: state.webgpuDevice, adapter: state.adapter};
  }
  const preference = elements.power.value;
  const adapterOptions = preference === 'default' ? {} : {powerPreference: preference};
  const adapter = await navigator.gpu.requestAdapter(adapterOptions);
  if (!adapter) throw new Error('No WebGPU adapter was returned.');
  const optionalFeatures = ['subgroups', 'shader-f16', 'timestamp-query'];
  const requiredFeatures = optionalFeatures.filter((feature) => adapter.features.has(feature));
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: adapterRequiredLimits(adapter),
  });
  // ORT 1.27 initializes custom devices correctly through the environment setter.
  // Passing the same device through the per-session option currently fails for this graph.
  ort.env.webgpu.device = device;
  state.webgpuDevice = device;
  state.webgpuPreference = preference;
  state.adapter = adapterDetails(adapter);
  return {device, adapter: state.adapter};
}

async function createRuntimeSession(bytes) {
  let webgpuError = null;
  if (navigator.gpu?.requestAdapter) {
    try {
      const {adapter} = await ensureWebGpuDevice();
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['webgpu', 'wasm'],
      });
      return {
        session,
        provider: 'WebGPU · WASM fallback',
        adapter,
      };
    } catch (error) {
      webgpuError = String(error?.message || error);
    }
  } else {
    webgpuError = 'WebGPU is unavailable in this browser or context.';
  }

  const session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm']});
  return {session, provider: 'WebAssembly', adapter: state.adapter, webgpuError};
}

async function ensureSession(token) {
  const model = await loadModel(token);
  assertCurrent(token);
  if (state.session && state.sessionKey === model.key) return state;

  setProgress(0.5, 'Initializing execution provider…');
  const start = performance.now();
  const runtime = await createRuntimeSession(model.bytes);
  assertCurrent(token);
  await state.session?.release?.();
  state.session = runtime.session;
  state.sessionKey = model.key;
  state.provider = runtime.provider;
  state.adapter = runtime.adapter;
  state.webgpuError = runtime.webgpuError || null;
  state.initializationMs = performance.now() - start;
  state.modelLoadMs = model.loadMs;
  return state;
}

async function selectedBitmaps() {
  if (elements.demoMode.value !== 'custom') return state.sampleImages;
  const [leftFile, rightFile] = [elements.leftFile.files[0], elements.rightFile.files[0]];
  if (!leftFile || !rightFile) throw new Error('Select both custom images.');
  const key = [leftFile, rightFile]
    .map((file) => `${file.name}:${file.size}:${file.lastModified}`)
    .join('|');
  if (state.customImageKey !== key) {
    for (const bitmap of state.customImages || []) bitmap.close?.();
    state.customImages = await Promise.all([bitmapFromBlob(leftFile), bitmapFromBlob(rightFile)]);
    state.customImageKey = key;
  }
  return state.customImages;
}

async function renderSelectedPair() {
  if (!inputsReady()) return;
  const {width, height} = selectedDimensions();
  const images = await selectedBitmaps();
  const rightBitmap = elements.demoMode.value === 'rotation' ? images[0] : images[1];
  drawPair(renderBitmap(images[0], width, height), renderBitmap(rightBitmap, width, height));
}

function runtimeSummary(extra = {}) {
  return {
    model: modelLabel(),
    inputShape: `[2, 3, ${selectedDimensions().height}, ${selectedDimensions().width}]`,
    executionProviderPreference: state.provider,
    webgpuAdapter: state.adapter,
    webgpuFallbackReason: state.webgpuError,
    requestedPowerPreference: elements.power.value,
    adapterPowerPreference: state.webgpuPreference,
    powerPreferenceNote: state.webgpuDevice && state.webgpuPreference !== elements.power.value
      ? 'Reload the page to select a different adapter after WebGPU has initialized.'
      : null,
    modelLoadMs: state.modelLoadMs,
    sessionInitializationMs: state.initializationMs,
    wasmThreads: ort.env.wasm.numThreads,
    transferScope: 'Each timed run includes CPU tensor upload and CPU output download; canvas preprocessing is separate.',
    ...extra,
  };
}

async function runPair(token) {
  const runtime = await ensureSession(token);
  assertCurrent(token);
  const {width, height} = selectedDimensions();
  const [leftBitmap, rightBitmap] = await selectedBitmaps();
  const preprocessingStart = performance.now();
  const left = renderBitmap(leftBitmap, width, height);
  const right = renderBitmap(rightBitmap, width, height);
  const images = rgbNchw(surfaceRgba(left), surfaceRgba(right), width, height);
  const preprocessingMs = performance.now() - preprocessingStart;
  const warmups = Math.max(0, Number(elements.warmup.value));
  const runs = Math.max(1, Number(elements.runs.value));

  let finalOutputs = null;
  try {
    for (let index = 0; index < warmups; index += 1) {
      setProgress(0.55 + 0.15 * ((index + 1) / Math.max(1, warmups)), `Warmup ${index + 1} / ${warmups}`);
      let outputs = null;
      try {
        outputs = await runtime.session.run({images});
        assertCurrent(token);
      } finally {
        disposeOutputs(outputs);
      }
    }

    const timings = [];
    for (let index = 0; index < runs; index += 1) {
      setProgress(0.7 + 0.3 * (index / runs), `Inference ${index + 1} / ${runs}`);
      const start = performance.now();
      const outputs = await runtime.session.run({images});
      timings.push(performance.now() - start);
      disposeOutputs(finalOutputs);
      finalOutputs = outputs;
      assertCurrent(token);
    }

    const plottedMatches = drawMatches(left, right, finalOutputs);
    const median = percentile(timings, 0.5);
    elements.provider.textContent = runtime.provider;
    elements.median.textContent = formatMilliseconds(median);
    elements.matches.textContent = plottedMatches.toLocaleString();
    elements.model.textContent = modelLabel();
    setStatus(runtimeSummary({
      demo: elements.demoMode.value,
      preprocessingMs,
      warmups,
      measuredRuns: runs,
      steadyStateMedianMs: median,
      steadyStateP95Ms: percentile(timings, 0.95),
      modelOutputMatches: finalOutputs.matches.dims[0],
      plottedMatches,
    }));
    setProgress(1, 'Complete');
  } finally {
    disposeOutputs(finalOutputs);
    images.dispose?.();
  }
}

async function runRotation(token) {
  const runtime = await ensureSession(token);
  assertCurrent(token);
  const {width, height} = selectedDimensions();
  if (width !== height) throw new Error('Rotation mode requires a square input dimension.');
  const base = renderBitmap(state.sampleImages[0], width, height);
  const leftRgba = surfaceRgba(base);
  const warmups = Math.max(0, Number(elements.warmup.value));

  for (let index = 0; index < warmups; index += 1) {
    setProgress(0.55 + 0.2 * ((index + 1) / Math.max(1, warmups)), `Rotation warmup ${index + 1} / ${warmups}`);
    const images = rgbNchw(leftRgba, leftRgba, width, height);
    let outputs = null;
    try {
      outputs = await runtime.session.run({images});
      assertCurrent(token);
    } finally {
      disposeOutputs(outputs);
      images.dispose?.();
    }
  }

  const timings = [];
  let frame = 0;
  while (token === state.runToken) {
    const angle = (frame * ROTATION_STEP_DEGREES) % 360;
    const right = renderRotated(base, angle);
    const preprocessingStart = performance.now();
    const images = rgbNchw(leftRgba, surfaceRgba(right), width, height);
    const preprocessingMs = performance.now() - preprocessingStart;
    const start = performance.now();
    let outputs = null;
    try {
      outputs = await runtime.session.run({images});
      const inferenceMs = performance.now() - start;
      assertCurrent(token);
      timings.push(inferenceMs);
      if (timings.length > 72) timings.shift();
      const plottedMatches = drawMatches(base, right, outputs, true);
      elements.provider.textContent = runtime.provider;
      elements.median.textContent = formatMilliseconds(percentile(timings, 0.5));
      elements.matches.textContent = plottedMatches.toLocaleString();
      elements.model.textContent = modelLabel();
      setProgress(angle / 360, `${angle}° · frame ${frame + 1}`);
      setStatus(runtimeSummary({
        demo: 'rotation sweep · same image',
        angleDegrees: angle,
        rotationStepDegrees: ROTATION_STEP_DEGREES,
        frame: frame + 1,
        preprocessingMs,
        latestInferenceMs: inferenceMs,
        rollingMedianInferenceMs: percentile(timings, 0.5),
        rollingP95InferenceMs: percentile(timings, 0.95),
        modelOutputMatches: outputs.matches.dims[0],
        plottedMatchesInsideCircle: plottedMatches,
      }));
    } finally {
      disposeOutputs(outputs);
      images.dispose?.();
    }
    frame += 1;
    // Leave a task boundary between inferences so Stop remains responsive even
    // when the WASM fallback occupies the main thread for most of each frame.
    await new Promise((resolve) => setTimeout(resolve, 75));
  }
}

async function startRun() {
  if (!inputsReady() || state.busy) return;
  const rotation = elements.demoMode.value === 'rotation';
  const token = ++state.runToken;
  setBusy(true, rotation);
  setProgress(0, 'Preparing…');
  try {
    if (rotation) await runRotation(token);
    else await runPair(token);
  } catch (error) {
    if (error?.name !== 'AbortError') {
      setProgress(0, 'Failed');
      setStatus(error?.stack || String(error));
    }
  } finally {
    if (token === state.runToken || rotation) {
      setBusy(false);
      if (rotation && token !== state.runToken) setProgress(0, 'Rotation stopped');
    }
  }
}

function stopRun() {
  state.runToken += 1;
  elements.stop.disabled = true;
  setProgress(0, 'Stopping after this inference…');
}

function updateModelSource() {
  const release = elements.modelSource.value === 'release';
  elements.releaseModelField.classList.toggle('is-hidden', !release);
  elements.localModelField.classList.toggle('is-hidden', release);
  refreshReadiness();
}

async function updateDemoMode() {
  const custom = elements.demoMode.value === 'custom';
  const rotation = elements.demoMode.value === 'rotation';
  elements.customImages.classList.toggle('is-hidden', !custom);
  elements.runsField.classList.toggle('is-hidden', rotation);
  elements.stop.classList.toggle('is-hidden', !rotation);
  if (rotation && selectedDimensions().width !== selectedDimensions().height) {
    elements.resolution.value = '512x512';
  }
  refreshReadiness();
  await renderSelectedPair();
}

elements.modelSource.addEventListener('change', updateModelSource);
elements.localModel.addEventListener('change', refreshReadiness);
elements.releaseModel.addEventListener('change', refreshReadiness);
elements.demoMode.addEventListener('change', () => updateDemoMode().catch((error) => setStatus(String(error))));
elements.resolution.addEventListener('change', () => renderSelectedPair().catch((error) => setStatus(String(error))));
elements.leftFile.addEventListener('change', () => {
  refreshReadiness();
  renderSelectedPair().catch((error) => setStatus(String(error)));
});
elements.rightFile.addEventListener('change', () => {
  refreshReadiness();
  renderSelectedPair().catch((error) => setStatus(String(error)));
});
elements.run.addEventListener('click', startRun);
elements.stop.addEventListener('click', stopRun);
addEventListener('pagehide', () => {
  state.session?.release?.();
  for (const bitmap of state.customImages || []) bitmap.close?.();
});

async function initialize() {
  try {
    state.sampleImages = await Promise.all([fetchBitmap(SAMPLE_LEFT), fetchBitmap(SAMPLE_RIGHT)]);
    await renderSelectedPair();
    setStatus({
      ready: true,
      sample: 'Marunouchi left/right pair',
      defaultModel: `raco_aliked_lightglue_pipeline_k${elements.releaseModel.value}.onnx`,
      modelDownload: 'Deferred until Run is selected.',
      webgpuAvailable: Boolean(navigator.gpu?.requestAdapter),
      wasmThreads: ort.env.wasm.numThreads,
    });
  } catch (error) {
    setStatus(error?.stack || String(error));
    elements.readiness.textContent = 'Unable to load the sample images.';
  } finally {
    refreshReadiness();
  }
}

initialize();
