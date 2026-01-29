# CLAUDE.md - RCWX Development Guide

## Project Overview

**RCWX** = RVC Real-time Voice Changer on Intel Arc (XPU)

RVCv2モデルを使用したリアルタイムボイスチェンジャー。Intel Arc GPU (XPU) に最適化されたフルスクラッチ実装。

## Quick Start

```powershell
# 依存関係インストール（PyTorch XPU版が自動的にインストールされる）
uv sync

# XPU確認
uv run python -c "import torch; print(f'XPU: {torch.xpu.is_available()}')"

# モデルダウンロード（HuBERT, RMVPE）
uv run rcwx download

# GUI起動
uv run rcwx
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RealtimeVoiceChanger                     │
├─────────────────────────────────────────────────────────────┤
│  AudioInput (48kHz)                                          │
│       │                                                      │
│       ▼                                                      │
│  ChunkBuffer (入力バッファリング)                            │
│       │                                                      │
│       ▼                                                      │
│  [Input Queue] ──► Inference Thread                          │
│                         │                                    │
│                    Resample 48k→16k                          │
│                         │                                    │
│                    Denoise (optional)                        │
│                         │                                    │
│                    RVCPipeline.infer()                       │
│                    ├─ HuBERT (特徴抽出)                      │
│                    ├─ RMVPE (F0抽出, optional)               │
│                    └─ Synthesizer (音声合成)                 │
│                         │                                    │
│                    Resample 40k→48k                          │
│                         │                                    │
│  [Output Queue] ◄───────┘                                    │
│       │                                                      │
│       ▼                                                      │
│  OutputBuffer (出力バッファリング)                           │
│       │                                                      │
│       ▼                                                      │
│  AudioOutput (48kHz)                                         │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
rcwx/
├── cli.py                 # CLIエントリポイント、ログ設定
├── config.py              # RCWXConfig (JSON永続化)
├── device.py              # XPU/CUDA/CPU選択
├── downloader.py          # HuggingFaceモデルダウンロード
├── audio/
│   ├── input.py           # AudioInput (sounddevice)
│   ├── output.py          # AudioOutput (sounddevice)
│   ├── buffer.py          # ChunkBuffer, OutputBuffer
│   ├── resample.py        # scipy.signal.resample_poly
│   └── denoise.py         # MLDenoiser, SpectralGateDenoiser
├── models/
│   ├── hubert.py          # HuBERTFeatureExtractor (transformers)
│   ├── rmvpe.py           # RMVPE F0抽出
│   ├── synthesizer.py     # SynthesizerLoader
│   └── infer_pack/        # RVC WebUIから移植したコアモジュール
├── pipeline/
│   ├── inference.py       # RVCPipeline (単発推論)
│   └── realtime.py        # RealtimeVoiceChanger (リアルタイム処理)
└── gui/
    ├── app.py             # RCWXApp (CustomTkinter)
    └── widgets/           # UIコンポーネント
```

## Key Configuration

### pyproject.toml - PyTorch XPU設定

```toml
[tool.uv]
environments = ["sys_platform == 'win32'"]
override-dependencies = [
    "triton-xpu; sys_platform == 'linux'",
    "pytorch-triton-xpu; sys_platform == 'linux'",
]

[tool.uv.sources]
torch = { index = "pytorch-xpu" }
torchaudio = { index = "pytorch-xpu" }

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```

### RealtimeConfig (realtime.py)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mic_sample_rate` | 48000 | マイク入力レート |
| `input_sample_rate` | 16000 | 処理レート (HuBERT/RMVPE用) |
| `output_sample_rate` | 48000 | 出力レート |
| `chunk_sec` | 0.35 | チャンクサイズ (RMVPE要件: >=0.32) |
| `crossfade_sec` | 0.05 | クロスフェード長 |
| `max_queue_size` | 2 | キュー最大サイズ |
| `prebuffer_chunks` | 1 | 出力開始前のプリバッファ |

## CLI Commands

```powershell
uv run rcwx              # GUI起動
uv run rcwx devices      # デバイス一覧
uv run rcwx download     # モデルダウンロード
uv run rcwx run in.wav model.pth -o out.wav --pitch 5
uv run rcwx info model.pth
uv run rcwx logs         # ログファイル一覧
uv run rcwx logs --tail 50   # 最新ログの末尾50行
uv run rcwx logs --open      # 最新ログを開く
```

## Log Investigation

### ログファイルの場所

```
~/.config/rcwx/logs/rcwx_YYYYMMDD_HHMMSS.log
```

### ログタグの意味

| タグ | 説明 |
|------|------|
| `[INPUT]` | 入力コールバック - バッファ状態、キューサイズ |
| `[OUTPUT]` | 出力コールバック - バッファ状態、ドロップ数 |
| `[INFER]` | 推論スレッド - 処理時間、レイテンシ |

### 正常動作時のログ例

```
[INPUT] received=4200, input_buffer=16800, input_queue=0
[INFER] Chunk #1: in=16800, out=14000, infer=45ms, latency=180ms, buf=14000, under=0, over=0
[OUTPUT] frames=4200, chunks_added=1, output_buffer=14000, output_queue=0, dropped=0
```

### 問題パターン

#### 1. バッファアンダーラン (音切れ)
```
[OUTPUT] Buffer underrun #1
[OUTPUT] Buffer underrun #2
```
**原因**: 推論が間に合っていない
**対策**: `chunk_sec` を増やす、F0なしモードにする

#### 2. バッファオーバーラン (遅延増加)
```
[INPUT] Queue full, dropping chunk
[INFER] ... over=5
```
**原因**: 推論が遅すぎる、または出力が消費されていない
**対策**: キューサイズ確認、出力デバイス確認

#### 3. 入力遅延 (チャンク蓄積)
```
[INPUT] Falling behind: queued 3 chunks at once
```
**原因**: 入力処理が追いついていない
**対策**: チャンクサイズ確認

## Known Issues

### 1. エコー/ループ問題 (調査中)

**症状**: 音声がエコーのように繰り返される

**調査ポイント**:
- VB-Cable等の仮想オーディオデバイスの設定
- 入出力デバイスのフィードバックループ
- クロスフェード処理 (現在無効化中)

**現在の対策**:
- クロスフェードを一時的に無効化 (`_apply_crossfade`で即return)
- `max_queue_size`: 4→2
- `prebuffer_chunks`: 2→1
- `max_latency_samples`: 1.5チャンク分

### 2. XPUが認識されない

```
XPU available: False
```

**確認**:
```powershell
uv run python -c "import torch; print(torch.__version__)"
# → 2.10.0+xpu であることを確認 (+cpu だと問題)
```

**対策**: `uv.lock` を削除して `uv sync` を再実行

### 3. uv run でパッケージが入れ替わる

`uv run` は `uv.lock` に同期するため、手動インストールしたパッケージが上書きされる。
`pyproject.toml` でXPUインデックスを設定済みなので `uv sync` で正しくインストールされる。

## Audio Device Setup

### 推奨構成

```
入力: 物理マイク (Fifine K420等)
出力: 物理スピーカー/ヘッドホン または VB-Cable Input
```

### VB-Cable使用時の注意

```
Discord等で使う場合:
  RCWX出力 → CABLE Input
  Discord入力 ← CABLE Output

注意: CABLE OutputをRCWX入力に設定するとフィードバックループが発生
```

## Development

### Lint & Format

```powershell
uv run ruff check rcwx
uv run ruff format rcwx
```

### デバッグ実行

```powershell
# 詳細ログ付きで実行
uv run rcwx --verbose

# ログ確認
uv run rcwx logs --tail 100
```

### 重要なクラス

| クラス | ファイル | 役割 |
|--------|----------|------|
| `RCWXApp` | gui/app.py | メインGUIアプリケーション |
| `RealtimeVoiceChanger` | pipeline/realtime.py | リアルタイム処理の統合 |
| `RVCPipeline` | pipeline/inference.py | RVC推論パイプライン |
| `ChunkBuffer` | audio/buffer.py | 入力チャンクバッファリング |
| `OutputBuffer` | audio/buffer.py | 出力バッファ (max_latency制御) |

## Model Files

| ファイル | 場所 | 用途 |
|----------|------|------|
| `hubert_base.pt` | ~/.cache/rcwx/models/ | HuBERT特徴抽出 |
| `rmvpe.pt` | ~/.cache/rcwx/models/ | F0抽出 |
| `*.pth` | 任意 | RVCモデル |
| `*.index` | モデルと同じディレクトリ | FAISSインデックス |

## Noise Cancellation

### 方式

| 方式 | 説明 |
|------|------|
| `auto` | ML利用可能ならML、なければSpectral |
| `ml` | Facebook Denoiser (PyTorch, CPU実行) |
| `spectral` | Spectral Gate (DSP) |

### 注意

- MLDenoiserはXPU非対応、常にCPUで実行
- RVC推論のデバイス設定には影響しない

## Troubleshooting

### フィードバック/エコー問題 (ピッチが蓄積する)

**症状**: ピッチシフト+5に設定しているのに、時間とともに+10, +15...と上がっていく

**原因**: 同じオーディオインターフェース（例: High Definition Audio Device）を入力と出力の両方に使用すると、ドライバー内部のモニタリング機能により出力が入力に戻る

**解決策**:
1. **異なるオーディオインターフェースを使用する**
   - 入力: USBマイク（例: Fifine K420）
   - 出力: オンボードスピーカー/ヘッドホン
2. `rcwx diagnose` で現在の設定を確認

### XPUが認識されない

```powershell
# 確認
uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"

# 期待: 2.x.x+xpu True
# NG: 2.x.x+cpu False
```

**解決策**: `uv sync --reinstall` でXPU版PyTorchを再インストール

### 推論が遅い

1. `--verbose` でログ確認
2. デノイズ無効化でテスト（MLデノイズはCPU実行で遅い場合がある）
3. チャンクサイズを増やす（レイテンシと引き換え）

## References

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [PyTorch XPU](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Facebook Denoiser](https://github.com/facebookresearch/denoiser)
- [ContentVec](https://github.com/auspicious3000/contentvec)
