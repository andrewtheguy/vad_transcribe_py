/**
 * AudioWorklet processor for real-time audio capture and processing.
 *
 * This processor:
 * 1. Mixes stereo channels to mono (if stereo input)
 * 2. Downsamples from native sample rate to 16kHz
 * 3. Accumulates samples into fixed-size chunks
 * 4. Converts to 16-bit PCM format
 * 5. Sends chunks to main thread via MessagePort
 */

// Type declarations for AudioWorklet global scope
declare const sampleRate: number
declare function registerProcessor(
  name: string,
  processorCtor: new (options?: AudioWorkletNodeOptions) => AudioWorkletProcessor,
): void

declare class AudioWorkletProcessor {
  readonly port: MessagePort
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean
}

const TARGET_SAMPLE_RATE = 16000
const TARGET_CHUNK_SAMPLES = 16384 // 4096 * 4, ~1 second at 16kHz

interface ProcessorOptions {
  targetSampleRate?: number
  targetChunkSamples?: number
}

/**
 * Clamp a value between min and max
 */
function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/**
 * Convert float32 audio samples to 16-bit PCM format
 */
function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(input.length * 2)
  const view = new DataView(buffer)
  for (let i = 0; i < input.length; i++) {
    const sample = clamp(input[i], -1, 1)
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }
  return buffer
}

/**
 * Downsample or upsample audio buffer to target sample rate
 */
function downsampleBuffer(
  input: Float32Array,
  inputSampleRate: number,
  targetSampleRate: number,
): Float32Array {
  if (targetSampleRate === inputSampleRate) {
    return input
  }

  if (targetSampleRate > inputSampleRate) {
    // Upsample using linear interpolation
    const ratio = targetSampleRate / inputSampleRate
    const newLength = Math.floor(input.length * ratio)
    const result = new Float32Array(newLength)
    for (let i = 0; i < newLength; i++) {
      const index = i / ratio
      const low = Math.floor(index)
      const high = Math.min(Math.ceil(index), input.length - 1)
      const frac = index - low
      result[i] = input[low] * (1 - frac) + input[high] * frac
    }
    return result
  }

  // Downsample using averaging-based decimation
  const ratio = inputSampleRate / targetSampleRate
  const newLength = Math.floor(input.length / ratio)
  const result = new Float32Array(newLength)
  let offsetResult = 0
  let offsetBuffer = 0

  while (offsetResult < newLength) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio)
    let accum = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
      accum += input[i]
      count++
    }
    result[offsetResult] = count > 0 ? accum / count : 0
    offsetResult++
    offsetBuffer = nextOffsetBuffer
  }

  return result
}

class AudioProcessor extends AudioWorkletProcessor {
  private targetSampleRate: number
  private targetChunkSamples: number
  private pendingSamples: Float32Array | null = null
  private sentSamples = 0

  constructor(options?: { processorOptions?: ProcessorOptions }) {
    super()
    this.targetSampleRate = options?.processorOptions?.targetSampleRate ?? TARGET_SAMPLE_RATE
    this.targetChunkSamples = options?.processorOptions?.targetChunkSamples ?? TARGET_CHUNK_SAMPLES
  }

  /**
   * Process audio buffers
   * Called by the browser's audio rendering thread at regular intervals
   */
  process(inputs: Float32Array[][], _outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
    const input = inputs[0]

    if (!input || input.length === 0) {
      return true // Keep processor alive
    }

    // Mix down to mono
    let monoData: Float32Array

    if (input.length === 1) {
      // Already mono, use as-is
      monoData = input[0]
    } else if (input.length === 2) {
      // Stereo: mix down by averaging left + right channels
      const left = input[0]
      const right = input[1]
      monoData = new Float32Array(left.length)
      for (let i = 0; i < left.length; i++) {
        monoData[i] = (left[i] + right[i]) / 2
      }
    } else {
      // Multi-channel: average all channels
      const channelCount = input.length
      const frameCount = input[0].length
      monoData = new Float32Array(frameCount)
      for (let frame = 0; frame < frameCount; frame++) {
        let sum = 0
        for (let channel = 0; channel < channelCount; channel++) {
          sum += input[channel][frame]
        }
        monoData[frame] = sum / channelCount
      }
    }

    // Downsample to target sample rate
    const downsampled = downsampleBuffer(monoData, sampleRate, this.targetSampleRate)

    // Accumulate samples
    this.accumulateSamples(downsampled)

    return true // Keep processor alive
  }

  /**
   * Accumulate samples and send chunks when ready
   */
  private accumulateSamples(samples: Float32Array): void {
    if (samples.length === 0) {
      return
    }

    // Merge with pending samples
    if (!this.pendingSamples || this.pendingSamples.length === 0) {
      this.pendingSamples = samples
    } else {
      const merged = new Float32Array(this.pendingSamples.length + samples.length)
      merged.set(this.pendingSamples)
      merged.set(samples, this.pendingSamples.length)
      this.pendingSamples = merged
    }

    // Flush complete chunks
    this.flushChunks()
  }

  /**
   * Flush complete chunks to main thread
   */
  private flushChunks(): void {
    let pending = this.pendingSamples

    if (!pending || pending.length === 0) {
      return
    }

    // Send complete chunks
    while (pending && pending.length >= this.targetChunkSamples) {
      const chunk = pending.slice(0, this.targetChunkSamples)

      // Calculate timestamp
      const chunkStart = this.sentSamples / this.targetSampleRate
      this.sentSamples += this.targetChunkSamples

      // Convert to 16-bit PCM
      const pcmData = floatTo16BitPCM(chunk)

      // Send to main thread (transfer ownership for zero-copy)
      this.port.postMessage(
        {
          type: 'chunk',
          chunk: pcmData,
          start: chunkStart,
        },
        [pcmData], // Transfer ownership
      )

      // Keep remaining samples
      pending = pending.length > this.targetChunkSamples
        ? pending.slice(this.targetChunkSamples)
        : null
    }

    this.pendingSamples = pending
  }
}

registerProcessor('audio-processor', AudioProcessor)
