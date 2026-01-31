"""
チャンク連続性の統合テスト

RealtimeVoiceChangerの完全なフローをテストして、
チャンク間での音声途切れを検出します。
"""

import numpy as np
import pytest

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
from rcwx.audio.buffer import ChunkBuffer
from rcwx.audio.resample import resample


def generate_continuous_tone(duration_sec: float = 3.0, sr: int = 48000) -> np.ndarray:
    """連続した正弦波を生成（途切れ検出用）"""
    t = np.arange(int(sr * duration_sec)) / sr
    # 220Hz + ゆっくりした振幅変調
    fundamental = np.sin(2 * np.pi * 220 * t)
    modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 3 * t)
    return (fundamental * modulation * 0.5).astype(np.float32)


def detect_discontinuities(
    audio: np.ndarray,
    sr: int,
    energy_drop_threshold: float = 0.3,
    window_ms: float = 10,
) -> dict:
    """音声の不連続性を検出"""
    window = int(sr * window_ms / 1000)
    stride = window // 2

    energies = []
    for i in range(0, len(audio) - window, stride):
        energy = np.sqrt(np.mean(audio[i:i + window] ** 2))
        energies.append(energy)

    energies = np.array(energies)

    # 有声区間のエネルギー
    voiced = energies[energies > 0.01]
    if len(voiced) < 2:
        return {'has_gaps': False, 'gap_count': 0, 'min_ratio': 1.0}

    mean_energy = np.mean(voiced)
    min_energy = np.min(voiced)
    min_ratio = min_energy / mean_energy

    # エネルギーの急激な低下を検出
    energy_drops = 0
    for i in range(1, len(energies)):
        if energies[i-1] > 0.01:  # 有声区間から
            ratio = energies[i] / energies[i-1]
            if ratio < energy_drop_threshold:
                energy_drops += 1

    return {
        'has_gaps': energy_drops > 0,
        'gap_count': energy_drops,
        'min_ratio': float(min_ratio),
        'mean_energy': float(mean_energy),
    }


def process_with_realtime_simulation(
    audio_input: np.ndarray,
    pipeline: RVCPipeline,
    rt_config: RealtimeConfig,
) -> np.ndarray:
    """RealtimeVoiceChangerの処理を手動でシミュレート"""

    mic_chunk_samples = int(rt_config.mic_sample_rate * rt_config.chunk_sec)

    # ChunkBuffer初期化
    chunk_buffer = ChunkBuffer(
        mic_chunk_samples,
        crossfade_samples=0,
        context_samples=int(rt_config.mic_sample_rate * rt_config.context_sec),
        lookahead_samples=int(rt_config.mic_sample_rate * rt_config.lookahead_sec),
    )

    # SOLAState初期化
    from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade

    output_crossfade_samples = int(
        rt_config.output_sample_rate * rt_config.crossfade_sec
    )
    sola_state = SOLAState.create(
        output_crossfade_samples,
        rt_config.output_sample_rate,
    ) if rt_config.use_sola else None

    pipeline.clear_cache()
    outputs = []

    # 入力を分割して処理
    pos = 0
    while pos < len(audio_input):
        # チャンクサイズ分取得
        end = min(pos + mic_chunk_samples, len(audio_input))
        chunk = audio_input[pos:end]

        # 最後のチャンクはスキップ（不完全なため）
        if len(chunk) < mic_chunk_samples and pos > 0:
            break

        # パディングが必要なら
        if len(chunk) < mic_chunk_samples:
            chunk = np.pad(chunk, (0, mic_chunk_samples - len(chunk)), mode='constant')

        # ChunkBufferに追加
        chunk_buffer.add_input(chunk)

        # チャンク取得可能なら処理
        if chunk_buffer.has_chunk():
            buffered_chunk = chunk_buffer.get_chunk()

            # リサンプル: mic_sr -> input_sr
            chunk_16k = resample(
                buffered_chunk,
                rt_config.mic_sample_rate,
                rt_config.input_sample_rate,
            )

            # RVC推論
            output = pipeline.infer(
                chunk_16k,
                input_sr=rt_config.input_sample_rate,
                pitch_shift=rt_config.pitch_shift,
                f0_method=rt_config.f0_method if rt_config.use_f0 else "none",
                use_feature_cache=rt_config.use_feature_cache,
                voice_gate_mode=rt_config.voice_gate_mode,
            )

            # リサンプル: model_sr -> output_sr
            if pipeline.sample_rate != rt_config.output_sample_rate:
                output = resample(
                    output,
                    pipeline.sample_rate,
                    rt_config.output_sample_rate,
                )

            # SOLA適用
            if rt_config.use_sola and sola_state is not None:
                cf_result = apply_sola_crossfade(output, sola_state)
                output = cf_result.audio

            outputs.append(output)

        pos = end

    if not outputs:
        return np.array([], dtype=np.float32)

    return np.concatenate(outputs)


class TestChunkContinuity:
    """チャンク連続性の統合テスト"""

    @pytest.fixture(scope="class")
    def setup(self):
        """テスト用のセットアップ"""
        config = RCWXConfig.load()
        model_path = config.last_model_path

        if not model_path:
            pytest.skip("モデルが設定されていません")

        pipeline = RVCPipeline(model_path, device="auto", use_compile=False)
        pipeline.load()

        return {
            'pipeline': pipeline,
            'model_path': model_path,
        }

    def test_continuity_with_sola_and_cache(self, setup):
        """SOLA + Feature Cacheありでの連続性テスト"""
        pipeline = setup['pipeline']

        # テスト信号
        test_signal = generate_continuous_tone(3.0, sr=48000)

        # RealtimeConfig (SOLA + Feature Cache有効)
        rt_config = RealtimeConfig(
            mic_sample_rate=48000,
            input_sample_rate=16000,
            output_sample_rate=48000,
            chunk_sec=0.35,
            pitch_shift=0,
            use_f0=True,
            f0_method="rmvpe",
            use_feature_cache=True,
            use_sola=True,
            context_sec=0.05,
            crossfade_sec=0.05,
            voice_gate_mode="off",
        )

        # 処理
        output = process_with_realtime_simulation(test_signal, pipeline, rt_config)

        # 不連続性検出
        result = detect_discontinuities(output, rt_config.output_sample_rate)

        print(f"\n[SOLA=ON, Cache=ON]")
        print(f"  gap_count: {result['gap_count']}")
        print(f"  min_ratio: {result['min_ratio']:.3f}")
        print(f"  mean_energy: {result['mean_energy']:.4f}")

        # アサーション
        assert result['gap_count'] == 0, \
            f"途切れが検出されました: {result['gap_count']}箇所"
        assert result['min_ratio'] > 0.5, \
            f"エネルギーの急激な低下が検出されました: min/mean={result['min_ratio']:.3f}"

    def test_continuity_without_sola(self, setup):
        """SOLA無効時の連続性テスト（劣化を確認）"""
        pipeline = setup['pipeline']

        test_signal = generate_continuous_tone(3.0, sr=48000)

        rt_config = RealtimeConfig(
            mic_sample_rate=48000,
            input_sample_rate=16000,
            output_sample_rate=48000,
            chunk_sec=0.35,
            pitch_shift=0,
            use_f0=True,
            f0_method="rmvpe",
            use_feature_cache=True,
            use_sola=False,  # SOLA無効
            context_sec=0.05,
            crossfade_sec=0.05,
            voice_gate_mode="off",
        )

        output = process_with_realtime_simulation(test_signal, pipeline, rt_config)
        result = detect_discontinuities(output, rt_config.output_sample_rate)

        print(f"\n[SOLA=OFF, Cache=ON]")
        print(f"  gap_count: {result['gap_count']}")
        print(f"  min_ratio: {result['min_ratio']:.3f}")

        # SOLA無効では品質が劣る可能性があるが、完全に途切れるべきではない
        # 緩い基準でテスト
        assert result['gap_count'] < 10, \
            f"過度の途切れが検出されました: {result['gap_count']}箇所"

    def test_continuity_without_cache(self, setup):
        """Feature Cache無効時の連続性テスト"""
        pipeline = setup['pipeline']

        test_signal = generate_continuous_tone(3.0, sr=48000)

        rt_config = RealtimeConfig(
            mic_sample_rate=48000,
            input_sample_rate=16000,
            output_sample_rate=48000,
            chunk_sec=0.35,
            pitch_shift=0,
            use_f0=True,
            f0_method="rmvpe",
            use_feature_cache=False,  # Cache無効
            use_sola=True,
            context_sec=0.05,
            crossfade_sec=0.05,
            voice_gate_mode="off",
        )

        output = process_with_realtime_simulation(test_signal, pipeline, rt_config)
        result = detect_discontinuities(output, rt_config.output_sample_rate)

        print(f"\n[SOLA=ON, Cache=OFF]")
        print(f"  gap_count: {result['gap_count']}")
        print(f"  min_ratio: {result['min_ratio']:.3f}")

        # Cache無効でもSOLAがあれば途切れは少ないはず
        assert result['gap_count'] < 5, \
            f"途切れが検出されました: {result['gap_count']}箇所"

    def test_worst_case_no_continuity_features(self, setup):
        """最悪ケース: SOLA + Cache両方無効"""
        pipeline = setup['pipeline']

        test_signal = generate_continuous_tone(3.0, sr=48000)

        rt_config = RealtimeConfig(
            mic_sample_rate=48000,
            input_sample_rate=16000,
            output_sample_rate=48000,
            chunk_sec=0.35,
            pitch_shift=0,
            use_f0=True,
            f0_method="rmvpe",
            use_feature_cache=False,  # Cache無効
            use_sola=False,  # SOLA無効
            context_sec=0.05,
            crossfade_sec=0.05,
            voice_gate_mode="off",
        )

        output = process_with_realtime_simulation(test_signal, pipeline, rt_config)
        result = detect_discontinuities(output, rt_config.output_sample_rate)

        print(f"\n[SOLA=OFF, Cache=OFF] (最悪ケース)")
        print(f"  gap_count: {result['gap_count']}")
        print(f"  min_ratio: {result['min_ratio']:.3f}")

        # 最悪ケースでも、極端な途切れは許容しない
        # （w-okada styleのcontextがあるため）
        assert result['gap_count'] < 20, \
            f"過度の途切れが検出されました: {result['gap_count']}箇所"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
