// Minimal WebGPU runtime for pywg.

export async function createContext(canvasOrNull = null) {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to acquire GPU adapter");
  }
  const device = await adapter.requestDevice();
  return { adapter, device, canvas: canvasOrNull };
}

export async function compileKernel(device, wgsl, entry = "main") {
  const module = device.createShaderModule({ code: wgsl });
  const pipeline = await device.createComputePipelineAsync({
    compute: { module, entryPoint: entry },
  });
  return pipeline;
}

export function bytesOf(typedArrayLike) {
  return typedArrayLike.byteLength;
}

export async function makeBuffer(device, typedArray, usage) {
  const buffer = device.createBuffer({
    size: typedArray.byteLength,
    usage,
    mappedAtCreation: true,
  });
  const arrayBuffer = buffer.getMappedRange();
  new typedArray.constructor(arrayBuffer).set(typedArray);
  buffer.unmap();
  return buffer;
}

export async function runDispatch(
  device,
  pipeline,
  bindGroup,
  x,
  y = 1,
  z = 1,
  workgroupSizeX = 256,
  workgroupSizeY = 1,
  workgroupSizeZ = 1
) {
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(x, y, z);
  passEncoder.end();
  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);
  await device.queue.onSubmittedWorkDone();
}

export async function readBuffer(device, buffer, byteLength) {
  const readBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([commandEncoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const copy = readBuffer.getMappedRange();
  const result = copy.slice(0);
  readBuffer.unmap();
  return result;
}
