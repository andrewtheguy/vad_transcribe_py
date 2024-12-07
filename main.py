import argparse
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
        #print(audio)
        #sf.write('output.wav', audio, TARGET_SAMPLE_RATE)