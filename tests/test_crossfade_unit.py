"""
_apply_crossfade の単体テスト

realtime.py の _apply_crossfade が正しく動作するか確認
"""

import numpy as np


class MockConfig:
    use_sola = False
    sola_search_ratio = 0.25


class CrossfadeTester:
    """realtime.py の _apply_crossfade をシミュレート"""

    def __init__(self, cf_samples: int):
        self.output_crossfade_samples = cf_samples
        self.config = MockConfig()
        self._prev_overlap = None
        self.stats = type('obj', (object,), {'frames_processed': 0})()

    def _apply_crossfade(self, output: np.ndarray, trimmed_output: np.ndarray) -> np.ndarray:
        """realtime.py からコピー"""
        cf_samples = self.output_crossfade_samples

        if cf_samples == 0:
            return trimmed_output

        trimmed_len = len(trimmed_output)
        actual_cf = min(cf_samples, trimmed_len // 3)
        if actual_cf < 100:
            return trimmed_output

        # First chunk
        if self._prev_overlap is None:
            self._prev_overlap = trimmed_output[-actual_cf:].copy()
            return trimmed_output[:-actual_cf]

        # Ensure prev_overlap has correct size
        if len(self._prev_overlap) != actual_cf:
            if len(self._prev_overlap) > actual_cf:
                self._prev_overlap = self._prev_overlap[-actual_cf:]
            else:
                self._prev_overlap = np.pad(
                    self._prev_overlap,
                    (actual_cf - len(self._prev_overlap), 0)
                )

        # Create fade windows (linear crossfade - more stable for varying phase)
        fade_in = np.linspace(0, 1, actual_cf, dtype=np.float32)
        fade_out = np.linspace(1, 0, actual_cf, dtype=np.float32)

        prev_tail = self._prev_overlap
        curr_head = trimmed_output[:actual_cf]

        # Simple blend (no SOLA offset)
        blended = prev_tail * fade_out + curr_head * fade_in

        self._prev_overlap = trimmed_output[-actual_cf:].copy()

        # Build output
        result = np.concatenate([blended, trimmed_output[actual_cf:-actual_cf]])
        return result


def test_crossfade_continuity():
    """クロスフェードの連続性テスト"""
    print("=" * 60)
    print("_apply_crossfade 単体テスト")
    print("=" * 60)

    # パラメータ
    cf_samples = 2400  # 50ms @ 48kHz
    trimmed_len = 13920  # テストと同じ

    # 一定値のチャンクを作成（クロスフェードがうまくいけば連続するはず）
    chunks = []
    for i in range(5):
        # 各チャンクは異なる値を持つ（境界で急激な変化がないか確認）
        chunk = np.full(trimmed_len, 0.5, dtype=np.float32)
        chunks.append(chunk)

    tester = CrossfadeTester(cf_samples)
    outputs = []

    for i, chunk in enumerate(chunks):
        result = tester._apply_crossfade(None, chunk)
        outputs.append(result)
        print(f"Chunk {i}: input_len={len(chunk)}, output_len={len(result)}")

    # 連結
    full_output = np.concatenate(outputs)
    print(f"\n合計出力長: {len(full_output)}")

    # 連続性チェック: 一定値なら全て0.5のはず
    unique_values = np.unique(full_output)
    if len(unique_values) == 1 and unique_values[0] == 0.5:
        print("[PASS] constant value test: all 0.5")
    else:
        print(f"[FAIL] constant value test: unique values = {unique_values[:10]}...")

    # 位相変化のあるチャンクでテスト
    print("\n" + "-" * 60)
    print("位相変化テスト")
    print("-" * 60)

    tester2 = CrossfadeTester(cf_samples)
    outputs2 = []

    for i in range(5):
        # 各チャンクで位相がずれた正弦波
        t = np.arange(trimmed_len) / 48000
        phase = i * 0.1  # チャンクごとに位相がずれる
        chunk = np.sin(2 * np.pi * 440 * t + phase).astype(np.float32)
        result = tester2._apply_crossfade(None, chunk)
        outputs2.append(result)

    full_output2 = np.concatenate(outputs2)

    # エネルギー連続性チェック
    window = 480  # 10ms
    energies = []
    for i in range(0, len(full_output2) - window, window):
        energy = np.sqrt(np.mean(full_output2[i:i+window]**2))
        energies.append(energy)

    energies = np.array(energies)
    cv = np.std(energies) / np.mean(energies)
    min_ratio = np.min(energies) / np.mean(energies)

    print(f"CV: {cv:.3f} (target < 0.2)")
    print(f"min/mean: {min_ratio:.3f} (target > 0.5)")

    # 期待される出力長の計算
    # 最初: trimmed - cf
    # 2番目以降: cf (blended) + (trimmed - 2*cf) = trimmed - cf
    expected_total = (trimmed_len - cf_samples) * 5
    print(f"\n期待出力長: {expected_total}, 実際: {len(full_output)}")

    if len(full_output) == expected_total:
        print("[PASS] output length")
    else:
        print(f"[FAIL] output length: diff = {len(full_output) - expected_total}")


def test_realtime_simulation():
    """リアルタイム処理のシミュレーション"""
    print("\n" + "=" * 60)
    print("リアルタイム処理シミュレーション")
    print("=" * 60)

    # テストと同じパラメータ
    cf_samples = 2400
    context_samples = 4800  # 100ms @ 48kHz
    extra_samples = 960  # 20ms @ 48kHz

    # RVC出力のシミュレーション（一定振幅の正弦波）
    sr = 48000
    rvc_out_len = 25440  # テストと同じ

    tester = CrossfadeTester(cf_samples)
    outputs = []

    for chunk_num in range(8):
        # RVC出力（一定振幅）
        t = np.arange(rvc_out_len) / sr
        rvc_out = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

        # エッジトリム（修正後: 最初のチャンクも同じトリム）
        total_left_trim = context_samples + extra_samples
        total_right_trim = context_samples + extra_samples
        trimmed = rvc_out[total_left_trim:-total_right_trim]

        print(f"Chunk {chunk_num}: rvc_out={len(rvc_out)}, trimmed={len(trimmed)}", end="")

        # クロスフェード
        result = tester._apply_crossfade(None, trimmed)
        outputs.append(result)
        print(f", output={len(result)}")

    full_output = np.concatenate(outputs)
    print(f"\n合計出力長: {len(full_output)}")

    # エネルギー連続性
    window = 480
    energies = []
    for i in range(0, len(full_output) - window, window):
        energy = np.sqrt(np.mean(full_output[i:i+window]**2))
        energies.append(energy)

    energies = np.array(energies)
    mean_e = np.mean(energies)
    min_e = np.min(energies)
    cv = np.std(energies) / mean_e
    min_ratio = min_e / mean_e

    print(f"\nCV: {cv:.3f} (target < 0.2)")
    print(f"min/mean: {min_ratio:.3f} (target > 0.5)")

    if cv < 0.2 and min_ratio > 0.5:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        # 問題箇所を特定
        threshold = mean_e * 0.5
        for i, e in enumerate(energies):
            if e < threshold:
                time_ms = i * 10  # 10ms windows
                print(f"  エネルギー低下: {time_ms}ms, energy={e:.4f}")
        return False


if __name__ == "__main__":
    test_crossfade_continuity()
    test_realtime_simulation()
