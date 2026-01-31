"""
音声途切れ診断スクリプト

実際のRealtimeVoiceChangerを使用して、チャンク境界での
音声の途切れを可視化・測定します。
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger


def generate_test_signal(duration_sec: float = 5.0, sr: int = 48000) -> np.ndarray:
    """テスト信号生成: 安定した220Hz + 振幅変調"""
    t = np.arange(int(sr * duration_sec)) / sr
    # 基本波 220Hz
    fundamental = np.sin(2 * np.pi * 220 * t)
    # ゆっくりした振幅変調 (5Hz)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    return (fundamental * modulation * 0.5).astype(np.float32)


def detect_gaps(audio: np.ndarray, sr: int, threshold_db: float = -40) -> list[dict]:
    """音声の途切れ（gap）を検出"""
    # 短時間エネルギー (10ms window)
    window = int(sr * 0.01)
    energies = []
    positions = []

    for i in range(0, len(audio) - window, window // 2):
        energy = np.sqrt(np.mean(audio[i:i + window] ** 2))
        energy_db = 20 * np.log10(energy + 1e-10)
        energies.append(energy_db)
        positions.append(i / sr)  # 秒単位

    energies = np.array(energies)
    positions = np.array(positions)

    # 途切れの検出: エネルギーが閾値以下
    gaps = []
    in_gap = False
    gap_start = 0

    for i, (pos, energy) in enumerate(zip(positions, energies)):
        if energy < threshold_db:
            if not in_gap:
                gap_start = pos
                in_gap = True
        else:
            if in_gap:
                gaps.append({
                    'start': gap_start,
                    'end': pos,
                    'duration': pos - gap_start,
                    'min_energy': energies[i-1] if i > 0 else energy,
                })
                in_gap = False

    return gaps


def analyze_chunk_boundaries(
    audio: np.ndarray,
    sr: int,
    chunk_sec: float,
    boundary_window_ms: float = 50,
) -> list[dict]:
    """チャンク境界付近のエネルギーを分析"""
    chunk_samples = int(sr * chunk_sec)
    window_samples = int(sr * boundary_window_ms / 1000)

    boundaries = []
    chunk_num = 0

    for pos in range(chunk_samples, len(audio), chunk_samples):
        # 境界前後のエネルギー
        before_start = max(0, pos - window_samples)
        after_end = min(len(audio), pos + window_samples)

        energy_before = np.sqrt(np.mean(audio[before_start:pos] ** 2))
        energy_after = np.sqrt(np.mean(audio[pos:after_end] ** 2))

        # 位相連続性チェック（簡易版）
        if pos > 10 and pos + 10 < len(audio):
            # 境界前後10サンプルの相関
            correlation = np.corrcoef(
                audio[pos-10:pos],
                audio[pos:pos+10]
            )[0, 1]
        else:
            correlation = 1.0

        boundaries.append({
            'chunk': chunk_num,
            'position': pos / sr,
            'energy_before': energy_before,
            'energy_after': energy_after,
            'energy_ratio': energy_after / (energy_before + 1e-10),
            'correlation': correlation,
        })

        chunk_num += 1

    return boundaries


def main():
    print("=" * 80)
    print("音声途切れ診断スクリプト")
    print("=" * 80)

    # 設定読み込み
    config = RCWXConfig.load()
    model_path = config.last_model_path

    if not model_path or not Path(model_path).exists():
        print("❌ エラー: モデルが設定されていません")
        print("   rcwx を起動してモデルを選択してください")
        return

    print(f"📁 モデル: {Path(model_path).name}")

    # パイプライン初期化
    print("\n⏳ モデルをロード中...")
    pipeline = RVCPipeline(model_path, device="auto", use_compile=False)
    pipeline.load()

    # RealtimeVoiceChanger初期化
    rt_config = RealtimeConfig(
        input_device=None,  # ダミー入力
        output_device=None,  # ダミー出力
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=48000,
        chunk_sec=0.35,
        pitch_shift=0,
        use_f0=True,
        f0_method="rmvpe",
        use_feature_cache=True,
        use_sola=True,
        voice_gate_mode="off",
        denoise_enabled=False,
    )

    voice_changer = RealtimeVoiceChanger(pipeline, rt_config)

    # テスト信号生成
    duration = 5.0
    test_signal = generate_test_signal(duration, sr=48000)

    print(f"\n🎵 テスト信号生成:")
    print(f"   時間: {duration}秒")
    print(f"   サンプルレート: 48000Hz")
    print(f"   波形: 220Hz正弦波 + 5Hz振幅変調")

    # 手動でチャンク処理をシミュレート
    print(f"\n⚙️  処理中 (チャンクサイズ: {rt_config.chunk_sec}秒)...")

    mic_chunk_samples = int(48000 * rt_config.chunk_sec)
    output_chunks = []

    # ChunkBufferのシミュレーション
    from rcwx.audio.buffer import ChunkBuffer
    from rcwx.audio.resample import resample

    chunk_buffer = ChunkBuffer(
        mic_chunk_samples,
        crossfade_samples=0,
        context_samples=int(48000 * 0.05),
        lookahead_samples=0,
    )

    # 入力を分割して処理
    pos = 0
    chunk_count = 0

    voice_changer.pipeline.clear_cache()

    while pos < len(test_signal):
        # チャンクサイズ分取得
        end = min(pos + mic_chunk_samples, len(test_signal))
        chunk = test_signal[pos:end]

        if len(chunk) < mic_chunk_samples:
            # 最後のチャンクはパディング
            chunk = np.pad(chunk, (0, mic_chunk_samples - len(chunk)), mode='constant')

        # ChunkBufferに追加
        chunk_buffer.add_input(chunk)

        # チャンク取得可能なら処理
        if chunk_buffer.has_chunk():
            buffered_chunk = chunk_buffer.get_chunk()

            # リサンプル
            chunk_16k = resample(buffered_chunk, 48000, 16000)

            # RVC推論
            output = voice_changer.pipeline.infer(
                chunk_16k,
                input_sr=16000,
                pitch_shift=0,
                f0_method="rmvpe",
                use_feature_cache=True,
                voice_gate_mode="off",
            )

            # リサンプル
            output_48k = resample(output, voice_changer.pipeline.sample_rate, 48000)

            output_chunks.append(output_48k)
            chunk_count += 1

        pos = end

    # 単純結合（クロスフェードなし）
    output_simple = np.concatenate(output_chunks)

    print(f"   処理完了: {chunk_count}チャンク")
    print(f"   入力長: {len(test_signal)} samples ({len(test_signal)/48000:.2f}s)")
    print(f"   出力長: {len(output_simple)} samples ({len(output_simple)/48000:.2f}s)")

    # 分析
    print(f"\n📊 分析結果:")
    print("-" * 80)

    # 1. ギャップ検出
    gaps = detect_gaps(output_simple, 48000, threshold_db=-40)
    print(f"\n🔍 検出された途切れ (エネルギー < -40dB):")
    if gaps:
        print(f"   ⚠️  {len(gaps)}箇所で途切れを検出!")
        for i, gap in enumerate(gaps[:10]):  # 最初の10件
            print(f"   #{i+1}: {gap['start']:.3f}s - {gap['end']:.3f}s "
                  f"(継続時間: {gap['duration']*1000:.1f}ms, "
                  f"最小エネルギー: {gap['min_energy']:.1f}dB)")
        if len(gaps) > 10:
            print(f"   ... 他 {len(gaps)-10}箇所")
    else:
        print(f"   ✅ 途切れなし")

    # 2. チャンク境界分析
    boundaries = analyze_chunk_boundaries(output_simple, 48000, rt_config.chunk_sec)
    print(f"\n🔍 チャンク境界分析 ({len(boundaries)}箇所):")

    energy_drops = [b for b in boundaries if b['energy_ratio'] < 0.5]
    if energy_drops:
        print(f"   ⚠️  {len(energy_drops)}箇所でエネルギー低下 (ratio < 0.5)")
        for b in energy_drops[:5]:
            print(f"   チャンク{b['chunk']}: {b['position']:.3f}s, "
                  f"ratio={b['energy_ratio']:.3f}, corr={b['correlation']:.3f}")
    else:
        print(f"   ✅ 大きなエネルギー低下なし")

    # 3. 可視化
    print(f"\n📈 波形とエネルギーを可視化中...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 入力波形
    ax1 = axes[0]
    time_in = np.arange(len(test_signal)) / 48000
    ax1.plot(time_in, test_signal, linewidth=0.5, alpha=0.7)
    ax1.set_title("Input Signal (220Hz + 5Hz modulation)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # 出力波形
    ax2 = axes[1]
    time_out = np.arange(len(output_simple)) / 48000
    ax2.plot(time_out, output_simple, linewidth=0.5, alpha=0.7, color='orange')

    # チャンク境界を縦線で表示
    for b in boundaries:
        ax2.axvline(b['position'], color='red', alpha=0.3, linestyle='--', linewidth=1)

    # ギャップ領域をハイライト
    for gap in gaps:
        ax2.axvspan(gap['start'], gap['end'], alpha=0.2, color='red')

    ax2.set_title("Output Signal (with chunk boundaries and gaps)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Output', 'Chunk boundary', 'Gap'])

    # 短時間エネルギー
    ax3 = axes[2]
    window = int(48000 * 0.01)
    energies_out = []
    positions_out = []
    for i in range(0, len(output_simple) - window, window // 2):
        energy = np.sqrt(np.mean(output_simple[i:i + window] ** 2))
        energies_out.append(energy)
        positions_out.append(i / 48000)

    ax3.plot(positions_out, energies_out, linewidth=1, color='green')

    # チャンク境界を縦線で表示
    for b in boundaries:
        ax3.axvline(b['position'], color='red', alpha=0.3, linestyle='--', linewidth=1)

    ax3.set_title("Short-time Energy (10ms window)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("RMS Energy")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    plt.tight_layout()

    # 保存
    output_dir = Path(__file__).parent / "diagnostic_output"
    output_dir.mkdir(exist_ok=True)

    plot_path = output_dir / "chunk_gap_diagnosis.png"
    wav_path = output_dir / "output_with_gaps.wav"

    plt.savefig(plot_path, dpi=150)
    print(f"   📊 グラフ保存: {plot_path}")

    wavfile.write(wav_path, 48000, output_simple)
    print(f"   🎵 音声保存: {wav_path}")

    plt.show()

    # サマリー
    print(f"\n" + "=" * 80)
    print("診断サマリー:")
    print("=" * 80)
    print(f"途切れ検出: {len(gaps)}箇所")
    print(f"エネルギー低下: {len(energy_drops)}箇所 (チャンク境界)")

    if gaps or energy_drops:
        print(f"\n⚠️  問題が検出されました。以下を確認してください:")
        print(f"   1. SOLA (use_sola=True) が有効か")
        print(f"   2. Feature Cache (use_feature_cache=True) が有効か")
        print(f"   3. Voice Gate が 'off' または 'expand' か")
        print(f"   4. チャンクサイズが適切か (現在: {rt_config.chunk_sec}s)")
    else:
        print(f"\n✅ 大きな問題は検出されませんでした")

    print(f"\n詳細は以下を確認:")
    print(f"   グラフ: {plot_path}")
    print(f"   音声: {wav_path}")


if __name__ == "__main__":
    main()
