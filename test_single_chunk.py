"""Test single chunk inference to isolate realtime issues."""
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline

# Record 1 second of audio
print("Recording 1 second... (speak now!)")
duration = 1.0
sample_rate = 44100
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
sd.wait()
audio = audio.flatten()

print(f"Recorded: len={len(audio)}, min={audio.min():.4f}, max={audio.max():.4f}, rms={np.sqrt(np.mean(audio**2)):.4f}")

# Save input
wavfile.write("test_input.wav", sample_rate, audio)
print("Saved test_input.wav")

# Load pipeline
print("Loading pipeline...")
pipeline = RVCPipeline(
    model_path="model/kana/kana/voice.pth",
    device="xpu",
    use_f0=True,
)
pipeline.load()

# Infer
print("Running inference...")
output = pipeline.infer(audio, input_sr=sample_rate, pitch_shift=0)

print(f"Output: len={len(output)}, min={output.min():.4f}, max={output.max():.4f}")

# Save output
wavfile.write("test_output.wav", pipeline.sample_rate, output)
print(f"Saved test_output.wav (sample rate: {pipeline.sample_rate})")

print("\nPlay test_output.wav to check quality!")
