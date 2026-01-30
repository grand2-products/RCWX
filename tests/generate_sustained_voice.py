"""
Generate a sustained vowel from the sample voice file.

Extract a portion where the speaker is sustaining a sound.
"""
import numpy as np
from scipy.io import wavfile
from pathlib import Path


def find_sustained_region(audio: np.ndarray, sr: int, min_duration: float = 2.0) -> tuple[int, int]:
    """Find a region with sustained energy (no silence)."""
    window = int(sr * 0.02)  # 20ms
    threshold = 0.02

    # Calculate energy
    energies = []
    for i in range(0, len(audio) - window, window):
        energy = np.sqrt(np.mean(audio[i:i + window] ** 2))
        energies.append(energy)

    energies = np.array(energies)

    # Find longest continuous region above threshold
    above = energies > threshold
    best_start = 0
    best_len = 0
    current_start = 0
    current_len = 0

    for i, v in enumerate(above):
        if v:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > best_len:
                best_start = current_start
                best_len = current_len
            current_len = 0

    if current_len > best_len:
        best_start = current_start
        best_len = current_len

    start_sample = best_start * window
    end_sample = (best_start + best_len) * window

    return start_sample, end_sample


def main():
    input_path = Path("sample_data/kakita.wav")
    output_path = Path("sample_data/sustained_voice.wav")

    sr, audio = wavfile.read(input_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    print(f"Input: {input_path}")
    print(f"  Duration: {len(audio) / sr:.2f} sec")

    # Find sustained region
    start, end = find_sustained_region(audio, sr, min_duration=2.0)
    duration = (end - start) / sr

    print(f"  Sustained region: {start/sr:.2f}s - {end/sr:.2f}s ({duration:.2f}s)")

    if duration < 2.0:
        print("WARNING: No sustained region found, using first 3 seconds")
        start = 0
        end = min(int(sr * 3), len(audio))

    # Extract and save
    extracted = audio[start:end]

    # Extend to 5 seconds by repeating
    target_len = int(sr * 5)
    if len(extracted) < target_len:
        repeats = target_len // len(extracted) + 1
        extracted = np.tile(extracted, repeats)[:target_len]

    wavfile.write(output_path, sr, (extracted * 32767).astype(np.int16))
    print(f"Output: {output_path}")
    print(f"  Duration: {len(extracted) / sr:.2f} sec")


if __name__ == "__main__":
    main()
