export interface PyWGContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  canvas: HTMLCanvasElement | null;
}

export function createContext(canvasOrNull?: HTMLCanvasElement | null): Promise<PyWGContext>;
export function compileKernel(device: GPUDevice, wgsl: string, entry?: string): Promise<GPUComputePipeline>;
export function bytesOf(typedArrayLike: ArrayBufferView): number;
export function makeBuffer(device: GPUDevice, typedArray: ArrayBufferView, usage: GPUBufferUsageFlags): Promise<GPUBuffer>;
export function runDispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  x: number,
  y?: number,
  z?: number,
  workgroupSizeX?: number,
  workgroupSizeY?: number,
  workgroupSizeZ?: number
): Promise<void>;
export function readBuffer(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<ArrayBuffer>;
