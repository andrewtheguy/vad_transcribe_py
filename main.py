import argparse

import torch
from silero_vad import (load_silero_vad,
                        read_audio,
                        get_speech_timestamps,
                        save_audio,
                        VADIterator,
                        collect_chunks)

import soundfile as sf

from speech_detector.speech_detector import TARGET_SAMPLE_RATE, ffmpeg_get_16bit_pcm, pcm_s16le_to_float32

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--file', type=str, required=True)
    args = argparse.parse_args()

    with ffmpeg_get_16bit_pcm(args.file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
        data = stdout.read()
        audio = pcm_s16le_to_float32(data)

    model = load_silero_vad()
    window_size_samples = 512 if TARGET_SAMPLE_RATE == 16000 else 256

    #print(audio)

    # Convert to PyTorch tensor and reshape to (1, num_samples)
    # Silero typically expects a single-channel tensor with shape (1, samples)
    audio_tensor = torch.from_numpy(audio)

    speech_probs = []

    wav = audio_tensor

    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i + window_size_samples]
        if len(chunk) < window_size_samples:
            break
        speech_prob = model(chunk, TARGET_SAMPLE_RATE).item()
        speech_probs.append((i,speech_prob),)
    model.reset_states()  # reset model states after each audio

    # print indexes of speech_probs where probability is greater than 0.5
    seconds = [i/TARGET_SAMPLE_RATE for i, prob in speech_probs if prob > 0.5]
    print(seconds)
    #sf.write('output.wav', audio, TARGET_SAMPLE_RATE)