"""
レイテンシ診断スクリプト

実際の設定とバッファ状態を確認して、高レイテンシの原因を特定します。
"""

import sys
from pathlib import Path

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
from rcwx.models.fcpe import is_fcpe_available


def main():
    print("=" * 80)
    print("レイテンシ診断スクリプト")
    print("=" * 80)
    print()

    # 設定読み込み
    config = RCWXConfig.load()

    print("[設定確認]")
    print("-" * 80)
    print(f"F0方式: {config.inference.f0_method}")
    print(f"チャンクサイズ: {config.audio.chunk_sec}秒 ({config.audio.chunk_sec * 1000:.0f}ms)")
    print(f"プリバッファ: {config.audio.prebuffer_chunks}チャンク")
    print(f"バッファマージン: {config.audio.buffer_margin}")
    print(f"コンテキスト: {config.inference.context_sec}秒")
    print(f"SOLA: {config.inference.use_sola}")
    print(f"Feature Cache: {config.inference.use_feature_cache}")
    print()

    # FCPE利用可能性確認
    fcpe_available = is_fcpe_available()
    print("[F0抽出エンジン]")
    print("-" * 80)
    print(f"FCPE利用可能: {fcpe_available}")

    if config.inference.f0_method == "fcpe" and not fcpe_available:
        print("[!] 警告: FCPEが設定されていますが、インストールされていません！")
        print("   -> RMVPEにフォールバックします（レイテンシ増加）")
        print("   -> インストール: uv sync --extra lowlatency")
    elif config.inference.f0_method == "rmvpe":
        print("[!] 注意: RMVPEモードです（高レイテンシ）")
        print("   -> 低レイテンシにするには: F0方式をFCPEに変更")
    print()

    # デバイス確認
    from rcwx.device import get_device, get_device_name

    device = get_device(config.device)
    device_name = get_device_name(device)

    print("[デバイス]")
    print("-" * 80)
    print(f"設定: {config.device}")
    print(f"実際: {device} ({device_name})")

    if device == "cpu":
        print("[!] 警告: CPU実行中！推論が遅くなります")
        print("   -> XPU/CUDAが利用可能か確認してください")
    print()

    # オーディオデバイス確認
    import sounddevice as sd

    print("[オーディオデバイス]")
    print("-" * 80)

    devices = sd.query_devices()
    default_in, default_out = sd.default.device

    if config.audio.input_device_name:
        print(f"入力: {config.audio.input_device_name}")
    else:
        if default_in is not None:
            print(f"入力: デフォルト [{default_in}] {devices[default_in]['name']}")
        else:
            print("入力: 未設定")

    if config.audio.output_device_name:
        print(f"出力: {config.audio.output_device_name}")
    else:
        if default_out is not None:
            print(f"出力: デフォルト [{default_out}] {devices[default_out]['name']}")
        else:
            print("出力: 未設定")
    print()

    # レイテンシ計算
    print("[理論レイテンシ計算]")
    print("-" * 80)

    # RealtimeConfig作成
    rt_config = RealtimeConfig(
        mic_sample_rate=config.audio.output_sample_rate,
        input_sample_rate=config.audio.sample_rate,
        output_sample_rate=config.audio.output_sample_rate,
        chunk_sec=config.audio.chunk_sec,
        pitch_shift=config.inference.pitch_shift,
        use_f0=config.inference.use_f0,
        f0_method=config.inference.f0_method,
        prebuffer_chunks=config.audio.prebuffer_chunks,
        buffer_margin=config.audio.buffer_margin,
        use_feature_cache=config.inference.use_feature_cache,
        use_sola=config.inference.use_sola,
        context_sec=config.inference.context_sec,
        crossfade_sec=config.inference.crossfade_sec,
        voice_gate_mode=config.inference.voice_gate_mode,
    )

    # チャンクサイズ検証（F0方式により自動調整される可能性）
    if rt_config.use_f0:
        if rt_config.f0_method == "fcpe" and fcpe_available:
            min_chunk = 0.10
        elif rt_config.f0_method == "rmvpe":
            min_chunk = 0.32
        else:
            min_chunk = 0.10

        actual_chunk = max(rt_config.chunk_sec, min_chunk + 0.03)
        if actual_chunk != rt_config.chunk_sec:
            print(f"[!] チャンクサイズ自動調整: {rt_config.chunk_sec:.3f}s -> {actual_chunk:.3f}s")
            rt_config.chunk_sec = actual_chunk

    # バッファ計算
    max_latency_sec = rt_config.chunk_sec * (rt_config.prebuffer_chunks + rt_config.buffer_margin)

    print(f"1. チャンクサイズ: {rt_config.chunk_sec * 1000:.0f}ms")
    print(f"2. 推論時間 (想定):")
    if rt_config.f0_method == "fcpe" and fcpe_available:
        print(f"   - HuBERT: 15-20ms")
        print(f"   - FCPE: 10-15ms")
        print(f"   - Synthesizer: 10-15ms")
        print(f"   - 合計: 35-50ms (XPU最適化)")
    elif rt_config.f0_method == "rmvpe":
        print(f"   - HuBERT: 15-20ms")
        print(f"   - RMVPE: 20-30ms")
        print(f"   - Synthesizer: 10-15ms")
        print(f"   - 合計: 45-65ms (XPU最適化)")
    else:
        print(f"   - HuBERT: 15-20ms")
        print(f"   - Synthesizer: 10-15ms")
        print(f"   - 合計: 25-35ms (F0なし)")

    print(f"3. 出力バッファ最大: {max_latency_sec * 1000:.0f}ms")
    print(f"   = chunk ({rt_config.chunk_sec:.3f}s) × (prebuffer ({rt_config.prebuffer_chunks}) + margin ({rt_config.buffer_margin}))")
    print(f"4. その他:")
    print(f"   - リサンプル: 5-10ms")
    print(f"   - SOLA: 3-5ms")

    # 総レイテンシ
    if rt_config.f0_method == "fcpe" and fcpe_available:
        infer_ms = 50
    elif rt_config.f0_method == "rmvpe":
        infer_ms = 65
    else:
        infer_ms = 35

    total_min = rt_config.chunk_sec * 1000 + infer_ms + 10
    total_max = rt_config.chunk_sec * 1000 + infer_ms + max_latency_sec * 1000 + 20

    print()
    print(f"[期待総レイテンシ] {total_min:.0f} - {total_max:.0f}ms")
    print()

    # 実測値との比較
    print("[!] 実測値: 推論200ms, レイテンシ2000ms と報告")
    print()

    # 問題診断
    print("[問題診断]")
    print("-" * 80)

    issues_found = False

    # 1. FCPE未インストール
    if config.inference.f0_method == "fcpe" and not fcpe_available:
        print("[X] 問題1: FCPEが未インストール")
        print("   -> RMVPEにフォールバック -> チャンクサイズ増加")
        print("   -> 解決策: uv sync --extra lowlatency")
        print()
        issues_found = True

    # 2. チャンクサイズが大きい
    if rt_config.chunk_sec > 0.20:
        print(f"[X] 問題2: チャンクサイズが大きい ({rt_config.chunk_sec:.3f}s)")
        print(f"   -> レイテンシ {rt_config.chunk_sec * 1000:.0f}ms 増加")
        if rt_config.f0_method == "fcpe" and fcpe_available:
            print(f"   -> 解決策: GUIでチャンクサイズを 0.15s に変更")
        elif rt_config.f0_method == "rmvpe":
            print(f"   -> 解決策: F0方式をFCPEに変更してチャンクサイズ削減")
        print()
        issues_found = True

    # 3. バッファマージンが大きい
    if rt_config.buffer_margin > 0.7:
        print(f"[X] 問題3: バッファマージンが大きい ({rt_config.buffer_margin})")
        print(f"   -> バッファ蓄積の可能性")
        print(f"   -> 解決策: buffer_margin を 0.5 に変更")
        print()
        issues_found = True

    # 4. プリバッファが大きい
    if rt_config.prebuffer_chunks > 2:
        print(f"[X] 問題4: プリバッファが大きい ({rt_config.prebuffer_chunks})")
        print(f"   -> レイテンシ {rt_config.prebuffer_chunks * rt_config.chunk_sec * 1000:.0f}ms 増加")
        print(f"   -> 解決策: prebuffer_chunks を 1 に変更")
        print()
        issues_found = True

    # 5. デバイスがCPU
    if device == "cpu":
        print("[X] 問題5: CPU実行中")
        print("   -> 推論が遅い (200ms = CPU実行の可能性)")
        print("   -> 解決策: XPU/CUDAを有効化")
        print()
        issues_found = True

    # 6. 推論時間が異常に長い
    if not issues_found:
        print("[!] 設定は正常ですが、実測レイテンシが異常に高い")
        print()
        print("可能性:")
        print("1. キューが詰まっている")
        print("   → GUIを再起動してください")
        print("2. sounddeviceのblocksizeが大きい")
        print("   → ログで確認: 'Audio config: ... blocksize=...'")
        print("3. バッファアンダーラン頻発")
        print("   → ログで確認: '[OUTPUT] Buffer underrun'")
        print("4. 推論が実際に遅い (CPU実行、メモリ不足)")
        print("   → タスクマネージャでGPU使用率を確認")
        print()

    # ログ確認推奨
    print("=" * 80)
    print("次のステップ:")
    print("=" * 80)
    print()
    print("1. ログを確認:")
    print("   uv run rcwx logs --tail 100")
    print()
    print("   確認項目:")
    print("   - '[INFER] Chunk #X: infer=XXXms' → 推論時間")
    print("   - '[INFER] ... latency=XXXms' → 総レイテンシ")
    print("   - '[INFER] ... buf=XXXXX' → バッファサンプル数")
    print("   - 'Chunk size X.XXs too small for ...' → 自動調整")
    print("   - '[OUTPUT] Buffer underrun' → バッファ不足")
    print()
    print("2. FCPE未インストールなら:")
    print("   uv sync --extra lowlatency")
    print()
    print("3. GUIを再起動:")
    print("   uv run rcwx")
    print()
    print("4. 設定を確認・調整:")
    print("   - F0方式: FCPE")
    print("   - チャンクサイズ: 150ms")
    print()


if __name__ == "__main__":
    main()
