voice activity detection on an audio stream using silero-vad and transcribing the detected voice segments using whispercpp

## transcribe from file (not tested)
```commandline
python main.py file --file /path/to/file
```
## transcribe from microphone
```commandline
python main.py mic
```

## transcribe from audio stream

```commandline
python main.py config --config configs/rthk2.toml
```