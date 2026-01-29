# RCWX

RVC Real-time Voice Changer on Intel Arc (XPU)

## Features

- **Intel Arc GPU対応** - PyTorch XPU + torch.compile による高速推論
- **RVC v1/v2両対応** - 256次元・768次元特徴量モデルに対応
- **F0あり/なしモデル対応** - RMVPE F0抽出または低遅延モード
- **リアルタイム変換** - クロスフェード処理による低遅延・高品質変換
- **ノイズキャンセリング** - ML (Facebook Denoiser) / Spectral Gate 切替可能
- **CustomTkinter GUI** - モダンなダークテーマUI
- **フルスクラッチ実装** - rvc-python等の依存なし

## Requirements

- Windows 10/11
- Python 3.11 or 3.12
- Intel Arc GPU (A770, A750, B580, etc.) または CUDA GPU
- [uv](https://github.com/astral-sh/uv) パッケージマネージャ

## Installation

### Intel Arc GPU (XPU) の場合

```powershell
git clone https://github.com/yourname/rcwx.git
cd rcwx

# PyTorch XPU版を含む全依存関係をインストール
uv sync

# XPU確認
uv run python -c "import torch; print(f'XPU: {torch.xpu.is_available()}')"
```

> **Note**: `pyproject.toml` で PyTorch XPU インデックスが設定済みのため、`uv sync` だけで XPU 版がインストールされます。

### NVIDIA GPU (CUDA) の場合

pyproject.toml を編集して CUDA インデックスに変更:

```toml
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/cu124"  # XPU → CUDA に変更
explicit = true
```

```powershell
uv sync

# CUDA確認
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### CPU のみの場合

pyproject.toml の `[tool.uv.sources]` セクションを削除またはコメントアウト:

```powershell
uv sync

# CPU版が使用される
uv run python -c "import torch; print(torch.__version__)"
```

## Quick Start

```powershell
# 必要モデル (HuBERT, RMVPE) のダウンロード
uv run rcwx download

# GUI起動
uv run rcwx
```

## CLI Commands

```powershell
# デバイス一覧
uv run rcwx devices

# ファイル変換
uv run rcwx run input.wav model.pth -o output.wav --pitch 5

# モデル情報表示
uv run rcwx info model.pth
```

## GUI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  RCWX - RVC Voice Changer                              [─][□][×]│
├────────────────────────────────────────────────────────────────┤
│  [メイン] [オーディオ] [詳細設定]                               │
├────────────────────────────────────────────────────────────────┤
│  ■ モデル選択                                                   │
│  [▼ 元気系アニメボイス Kana        ] [開く...]                 │
│  状態: F0あり (RVCv2) | インデックス: あり                     │
│                                                                 │
│  ■ ピッチシフト                                                 │
│  -24 ────────────●──────────────────────── +24                 │
│  現在値: +5 半音                                                │
│                                                                 │
│  ■ F0モード                                                     │
│  (●) RMVPE (高品質)    ( ) なし (低遅延)                       │
│                                                                 │
│  ■ ノイズキャンセリング                                         │
│  [✓] 有効   方式: [auto ▼]                                     │
│                                                                 │
│              ┌────────────────────────┐                        │
│              │       ▶ 開始          │                        │
│              └────────────────────────┘                        │
├────────────────────────────────────────────────────────────────┤
│ デバイス: Intel Arc A770 | レイテンシ: 145ms | 推論: 45ms      │
└─────────────────────────────────────────────────────────────────┘
```

## Noise Cancellation

騒音環境でのマイク入力を改善するノイズキャンセリング機能:

| 方式 | 説明 | 用途 |
|------|------|------|
| `auto` | ML利用可能ならML、なければSpectral | 推奨 |
| `ml` | Facebook Denoiser (PyTorch) | 高品質、人声保持 |
| `spectral` | Spectral Gate (DSP) | 軽量、低遅延 |

- **ML方式**: 機械学習ベースで人間の声を認識・保持しながらノイズを除去
- **Spectral方式**: 周波数スペクトルの閾値処理による従来型ノイズ除去

## Project Structure

```
rcwx/
├── config.py              # 設定管理 (JSON永続化)
├── device.py              # デバイス選択 (xpu/cuda/cpu)
├── downloader.py          # HuggingFace モデルダウンロード
├── cli.py                 # CLI エントリポイント
├── audio/
│   ├── input.py           # マイク入力 (sounddevice)
│   ├── output.py          # 音声出力
│   ├── buffer.py          # クロスフェード付きバッファ
│   ├── resample.py        # リサンプリング
│   └── denoise.py         # ノイズキャンセリング (ML/Spectral)
├── models/
│   ├── hubert.py          # ContentVec 特徴抽出 (transformers)
│   ├── rmvpe.py           # RMVPE F0抽出
│   ├── synthesizer.py     # モデルローダー
│   └── infer_pack/        # RVC コアモジュール
├── pipeline/
│   ├── inference.py       # 推論パイプライン
│   └── realtime.py        # リアルタイム処理
└── gui/
    ├── app.py             # メインアプリケーション
    └── widgets/           # UIコンポーネント
```

## Latency

| 構成 | チャンクサイズ | 期待レイテンシ |
|------|---------------|---------------|
| F0あり (RMVPE) | 200ms | 150〜250ms |
| F0なし | 100ms | 80〜150ms |

## Supported Models

- RVC v1 (256-dim HuBERT features)
- RVC v2 (768-dim HuBERT features)
- F0モデル (NSF decoder)
- No-F0モデル (standard decoder)

モデルは [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) で作成したものを使用できます。

## Configuration (pyproject.toml)

PyTorch XPU 版は `[tool.uv]` セクションで設定:

```toml
[tool.uv]
# Windows限定 (triton-xpuはLinux専用)
environments = ["sys_platform == 'win32'"]

# triton-xpu依存を除外
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

## Development

```powershell
# 開発用依存関係をインストール
uv sync --extra dev

# Lint
ruff check rcwx

# Format
ruff format rcwx
```

## Troubleshooting

### XPU が認識されない

```powershell
# PyTorch バージョン確認
uv run python -c "import torch; print(torch.__version__)"
# → 2.10.0+xpu のように +xpu が付いていることを確認

# +cpu の場合は uv.lock を再生成
del uv.lock
uv sync
```

### uv run でパッケージが入れ替わる

`uv run` は `uv.lock` に同期するため、手動インストールしたパッケージが上書きされることがあります。
`pyproject.toml` で XPU インデックスを設定済みであれば `uv sync` で正しくインストールされます。

## License

This project is licensed under the MIT License.

### Third-Party Licenses

| Component | License | Source |
|-----------|---------|--------|
| RVC WebUI | MIT | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| ContentVec | MIT | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
| fairseq / HuBERT | MIT | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) |
| RMVPE | Apache 2.0 | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) |
| Facebook Denoiser | CC BY-NC 4.0 | [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) |
| sounddevice | MIT | [spatialaudio/python-sounddevice](https://github.com/spatialaudio/python-sounddevice) |
| PyTorch | BSD | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| CustomTkinter | MIT | [TomSchimansky/CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) |
| transformers | Apache 2.0 | [huggingface/transformers](https://github.com/huggingface/transformers) |

### Note on Model Files

RVC model files (`.pth`) are subject to their own licensing terms. Ensure you have appropriate rights for any voice models you use, including consent from the original voice owner.
