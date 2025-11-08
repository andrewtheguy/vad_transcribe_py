voice activity detection on an audio stream using silero-vad and transcribing the detected voice segments using whispercpp

## transcribe from file
```commandline
python main.py file --file /path/to/file --lang en --output /path/to/output.json
```
## transcribe from microphone (will save to database)
```commandline
python main.py mic --lang en
```

## transcribe from audio stream (will save to database)

```commandline
python main.py config --config configs/rthk2.toml
```