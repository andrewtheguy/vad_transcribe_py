import argparse
import os

import torch
from silero_vad import (load_silero_vad,
                        read_audio,
                        get_speech_timestamps,
                        save_audio,
                        VADIterator,
                        collect_chunks)


from speech_detector.speech_detector import TARGET_SAMPLE_RATE, ffmpeg_get_16bit_pcm, pcm_s16le_to_float32, \
    process_silero, record

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('action', type=str, choices=['file','mic'])
    argparse.add_argument('--file', type=str, required=False)
    args = argparse.parse_args()

    model = load_silero_vad()

    if args.action == 'file':
        if not args.file:
            print("Please provide a file path")
            exit(1)
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist")
            exit(1)

        with ffmpeg_get_16bit_pcm(args.file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
            data = stdout.read()
            audio = pcm_s16le_to_float32(data)
        process_silero(model, audio)
    elif args.action == 'mic':
        audio = record()
        process_silero(model, audio)
        #sf.write('output.wav', audio, TARGET_SAMPLE_RATE)