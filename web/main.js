const $ = (id) => document.getElementById(id);
const status = $('status');
const canvas = $('matches');
const context = canvas.getContext('2d');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

async function decode(file, width, height) {
  const bitmap = await createImageBitmap(file);
  const surface = new OffscreenCanvas(width, height);
  const draw = surface.getContext('2d', {willReadFrequently: true});
  draw.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return {surface, rgba: draw.getImageData(0, 0, width, height).data};
}

function rgbNchw(left, right, width, height) {
  const area = width * height;
  const output = new Float32Array(2 * 3 * area);
  for (const [batch, rgba] of [left, right].entries()) {
    for (let pixel = 0; pixel < area; pixel += 1) {
      const source = pixel * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        output[batch * 3 * area + channel * area + pixel] = rgba[source + channel] / 255;
      }
    }
  }
  return new ort.Tensor('float32', output, [2, 3, height, width]);
}

function percentile(sorted, fraction) {
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

function drawMatches(left, right, width, height, outputs) {
  canvas.width = width * 2;
  canvas.height = height;
  context.drawImage(left, 0, 0);
  context.drawImage(right, width, 0);
  const keypoints = outputs.keypoints.data;
  const keypointCount = outputs.keypoints.dims[1];
  const matches = outputs.matches.data;
  context.strokeStyle = 'rgba(72, 255, 170, 0.65)';
  context.lineWidth = 0.75;
  for (let index = 0; index < matches.length; index += 3) {
    const batch = Number(matches[index]);
    if (batch !== 0) continue;
    const first = Number(matches[index + 1]);
    const second = Number(matches[index + 2]);
    const leftOffset = first * 2;
    const rightOffset = (keypointCount + second) * 2;
    context.beginPath();
    context.moveTo(keypoints[leftOffset], keypoints[leftOffset + 1]);
    context.lineTo(width + keypoints[rightOffset], keypoints[rightOffset + 1]);
    context.stroke();
  }
}

async function createSession(model) {
  const preference = $('power').value;
  if (preference !== 'default') ort.env.webgpu.powerPreference = preference;
  const bytes = await model.arrayBuffer();
  if ('gpu' in navigator) {
    try {
      const session = await ort.InferenceSession.create(bytes, {executionProviders: ['webgpu', 'wasm']});
      return {session, provider: 'webgpu (with wasm node fallback)'};
    } catch (error) {
      status.textContent = `WebGPU initialization failed; using WebAssembly.\n${error}\n`;
    }
  }
  return {
    session: await ort.InferenceSession.create(bytes, {executionProviders: ['wasm']}),
    provider: 'wasm',
  };
}

$('run').addEventListener('click', async () => {
  const [model, leftFile, rightFile] = [$('model').files[0], $('left').files[0], $('right').files[0]];
  if (!model || !leftFile || !rightFile) {
    status.textContent = 'Select a model and both images.';
    return;
  }
  $('run').disabled = true;
  try {
    const initializationStart = performance.now();
    const {session, provider} = await createSession(model);
    const initializationMs = performance.now() - initializationStart;
    const metadata = session.inputMetadata[0];
    const [, channels, height, width] = metadata.shape;
    if (channels !== 3 || !Number.isInteger(height) || !Number.isInteger(width)) {
      throw new Error(`The browser demo requires a static [2,3,H,W] model; received ${metadata.shape}.`);
    }
    const [left, right] = await Promise.all([decode(leftFile, width, height), decode(rightFile, width, height)]);
    const images = rgbNchw(left.rgba, right.rgba, width, height);
    const warmup = Number($('warmup').value);
    const runs = Number($('runs').value);
    let outputs;
    for (let index = 0; index < warmup; index += 1) outputs = await session.run({images});
    const timings = [];
    for (let index = 0; index < runs; index += 1) {
      const start = performance.now();
      outputs = await session.run({images});
      timings.push(performance.now() - start);
    }
    timings.sort((a, b) => a - b);
    drawMatches(left.surface, right.surface, width, height, outputs);
    const adapter = ort.env.webgpu?.adapter;
    const adapterInfo = adapter?.info ? {...adapter.info} : null;
    status.textContent = JSON.stringify({
      provider,
      powerPreference: $('power').value,
      adapter: adapterInfo,
      initializationMs,
      warmup,
      runs,
      steadyStateMedianMs: percentile(timings, 0.5),
      steadyStateP95Ms: percentile(timings, 0.95),
      matches: outputs.matches.dims[0],
      transferScope: 'CPU image upload and CPU output download are included in each run',
    }, null, 2);
  } catch (error) {
    status.textContent = String(error?.stack || error);
  } finally {
    $('run').disabled = false;
  }
});
