# チャンク連続性テスト - 実行ガイド

音声途切れ問題を診断・修正するためのテストガイドです。

## 🚀 クイックスタート

### 1. 即座の診断（推奨: 最初にこれを実行）

```powershell
cd C:\lib\github\grand2-products\RCWX
uv run python tests/diagnose_chunk_gap.py
```

**何が起こるか:**
- 5秒の連続音声を生成
- RealtimeVoiceChangerの処理をシミュレート
- 途切れを可視化したグラフを表示・保存
- 問題箇所をリスト表示

**出力:**
- `tests/diagnostic_output/chunk_gap_diagnosis.png` - 波形グラフ
- `tests/diagnostic_output/output_with_gaps.wav` - 処理済み音声

**途切れが検出された場合:**
1. グラフの赤い領域 = 途切れ箇所
2. 赤い縦線 = チャンク境界
3. 途切れがチャンク境界付近なら、SOLA/Cacheの問題の可能性

---

### 2. 自動テスト（CI/CD用）

```powershell
# 全テスト実行
uv run pytest tests/test_chunk_continuity_integration.py -v -s

# 特定のテストのみ
uv run pytest tests/test_chunk_continuity_integration.py::TestChunkContinuity::test_continuity_with_sola_and_cache -v -s
```

**テストケース:**
1. `test_continuity_with_sola_and_cache` - 推奨設定（途切れなし期待）
2. `test_continuity_without_sola` - SOLA無効（品質劣化許容）
3. `test_continuity_without_cache` - Cache無効（軽微な劣化許容）
4. `test_worst_case_no_continuity_features` - 最悪ケース

---

## 🔍 問題パターンと対策

### パターン1: チャンク境界で途切れる

**症状:**
```
⚠️  5箇所で途切れを検出!
#1: 0.350s - 0.370s (継続時間: 20.0ms)
#2: 0.700s - 0.720s (継続時間: 20.0ms)
```

**原因:**
- SOLAが無効、または正しく機能していない
- クロスフェード長が不足

**対策:**
1. `use_sola=True` を確認
2. `crossfade_sec` を増やす (0.05 → 0.08)
3. SOLA実装を確認: `rcwx/audio/crossfade.py`

```python
# realtime.py で確認
if self.config.use_sola and self._sola_state is not None:
    cf_result = apply_sola_crossfade(output, self._sola_state)
    output = cf_result.audio
```

---

### パターン2: ランダムに途切れる（チャンク境界と無関係）

**症状:**
```
⚠️  10箇所で途切れを検出!
#1: 0.234s - 0.248s (継続時間: 14.0ms)
#3: 0.891s - 0.903s (継続時間: 12.0ms)
```

**原因:**
- Voice Gateが過度に音声をカット
- F0検出の失敗（無声と誤判定）

**対策:**
1. Voice Gateを変更
   - `strict` → `expand` または `off`
2. F0メソッドを変更
   - `rmvpe` → `fcpe` (より安定)
3. Energy Thresholdを下げる (0.05 → 0.03)

```python
# config.py
voice_gate_mode: str = "expand"  # または "off"
energy_threshold: float = 0.03
```

---

### パターン3: エネルギー低下（完全には途切れない）

**症状:**
```
🔍 チャンク境界分析 (10箇所):
   ⚠️  3箇所でエネルギー低下 (ratio < 0.5)
   チャンク1: 0.350s, ratio=0.35, corr=0.82
```

**原因:**
- Feature Cacheが無効、またはブレンディングが不十分
- HuBERT/F0のキャッシュサイズが小さすぎる

**対策:**
1. `use_feature_cache=True` を確認
2. キャッシュサイズを増やす

```python
# inference.py の _feature_cache_frames を調整
self._feature_cache_frames: int = 10  # 10 → 15
```

---

### パターン4: 位相不連続（correlation < 0.5）

**症状:**
```
チャンク1: 0.350s, ratio=0.95, corr=0.23
```

**原因:**
- SOLAの探索範囲が不足
- サンプルレート変換の精度問題

**対策:**
1. SOLA探索範囲を増やす

```python
# realtime.py
sola_search_ratio: float = 0.25  # 0.25 → 0.5
```

2. リサンプリングの精度を確認

```python
# resample.py で高品質リサンプリングに変更
from scipy.signal import resample as scipy_resample
return scipy_resample(audio, int(len(audio) * target_sr / orig_sr))
```

---

## 🛠️ デバッグ手順

### ステップ1: 診断スクリプト実行

```powershell
uv run python tests/diagnose_chunk_gap.py
```

結果を確認:
- 途切れ箇所がチャンク境界と一致するか？
- エネルギーグラフで急激な低下があるか？

---

### ステップ2: 設定を段階的に変更

1. **最小構成でテスト**

```python
rt_config = RealtimeConfig(
    use_sola=False,
    use_feature_cache=False,
    voice_gate_mode="off",
)
# 途切れが減る → SOLA/Cacheの実装問題
# 途切れが変わらない → 他の原因
```

2. **SOLAのみ有効**

```python
rt_config = RealtimeConfig(
    use_sola=True,
    use_feature_cache=False,
    voice_gate_mode="off",
)
# 途切れが減る → SOLA有効
```

3. **Cacheのみ有効**

```python
rt_config = RealtimeConfig(
    use_sola=False,
    use_feature_cache=True,
    voice_gate_mode="off",
)
# 途切れが減る → Cache有効
```

---

### ステップ3: 詳細ログ確認

```powershell
# 詳細ログ付きで実行
uv run rcwx --verbose

# ログ確認
uv run rcwx logs --tail 200
```

確認項目:
```
[SOLA] chunk=1, offset=23  # オフセットが0付近 → 位相整合失敗
[INFER] F0: voiced=150/200  # 有声比率が低い → Voice Gate問題
```

---

### ステップ4: コード修正箇所

**SOLA実装:**
- `rcwx/audio/crossfade.py:30-90` - `apply_sola_crossfade()`

**Feature Cache:**
- `rcwx/pipeline/inference.py:448-463` - HuBERTキャッシュブレンディング
- `rcwx/pipeline/inference.py:518-539` - F0キャッシュブレンディング

**Voice Gate:**
- `rcwx/pipeline/inference.py:633-693` - ゲート適用

---

## 📊 期待される結果

### 正常な場合

```
📊 分析結果:
🔍 検出された途切れ (エネルギー < -40dB):
   ✅ 途切れなし

🔍 チャンク境界分析 (10箇所):
   ✅ 大きなエネルギー低下なし

診断サマリー:
途切れ検出: 0箇所
エネルギー低下: 0箇所 (チャンク境界)
✅ 大きな問題は検出されませんでした
```

### 問題がある場合

```
📊 分析結果:
🔍 検出された途切れ (エネルギー < -40dB):
   ⚠️  8箇所で途切れを検出!

⚠️  問題が検出されました。以下を確認してください:
   1. SOLA (use_sola=True) が有効か
   2. Feature Cache (use_feature_cache=True) が有効か
   3. Voice Gate が 'off' または 'expand' か
   4. チャンクサイズが適切か (現在: 0.35s)
```

---

## 🔧 高度なデバッグ

### 実際のリアルタイム処理をテスト

```python
# tests/test_realtime_actual.py を作成
from rcwx.pipeline.realtime import RealtimeVoiceChanger

vc = RealtimeVoiceChanger(pipeline, config)

# 録音した音声を再生
recorded_output = []

def output_callback(frames):
    output = vc._on_audio_output(frames)
    recorded_output.extend(output)
    return output

# 実行...
```

### SOLA相関係数の監視

```python
# realtime.py の _inference_thread に追加
if self.stats.frames_processed % 10 == 0:
    logger.info(
        f"[SOLA] chunk={self.stats.frames_processed}, "
        f"offset={cf_result.sola_offset}, "
        f"corr={cf_result.sola_correlation:.3f}"  # 相関係数
    )
```

相関係数が低い (< 0.5) → 位相不整合

---

## 📝 テスト追加のガイドライン

新しいテストケースを追加する場合:

```python
def test_your_new_case(self, setup):
    """説明"""
    pipeline = setup['pipeline']

    # 1. テスト信号生成
    test_signal = generate_continuous_tone(duration, sr)

    # 2. 設定作成
    rt_config = RealtimeConfig(...)

    # 3. 処理実行
    output = process_with_realtime_simulation(
        test_signal, pipeline, rt_config
    )

    # 4. 検証
    result = detect_discontinuities(output, sr)
    assert result['gap_count'] < threshold, "メッセージ"
```

---

## 🎯 まとめ

1. **まず診断スクリプトを実行** → 問題を可視化
2. **自動テストで原因を特定** → SOLA/Cache/VoiceGate
3. **設定を調整** → 最適なパラメータを見つける
4. **必要ならコード修正** → 実装の改善

問題が解決しない場合は、診断結果のグラフとログを共有してください。
