# RCWX - RVC Real-time Voice Changer on Intel Arc (XPU)

## 概要

**目的**: RVCv2モデルを使用したリアルタイムボイスチェンジャーをWindows Native + Intel Arc GPUで構築する

**アプローチ**: PyTorch XPU統一（A案）- 全コンポーネントをPyTorch XPU + torch.compileで実装

**対応モデル**: F0あり / F0なし（No-F0）両対応

---

## 環境要件

### ハードウェア
- Intel Arc GPU (A770, A750, B580, etc.)
- Resizable BAR 有効化推奨

### ソフトウェア（2026年1月時点）
- Windows 10/11
- Python 3.11 または 3.12
- **uv** (パッケージマネージャ)
- **oneAPI不要**: PyTorch XPU版は単独で動作

---

## PyTorch XPU サポート状況（2026年1月更新）

### 現状
| 項目 | 状況 |
|------|------|
| **PyTorch安定版** | 2.10（torch.xpuネイティブサポート） |
| **torch.compile (Windows XPU)** | PyTorch 2.7以降で対応済み |
| **IPEX** | **2026年3月末でEOL**、PyTorch本体に統合完了 |
| **対応データ型** | FP32, BF16, FP16, AMP |

### インストール方法（uv推奨）
```powershell
# uv環境作成
uv venv --python 3.12
.venv\Scripts\activate

# PyTorch XPU版（安定版）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 追加依存
uv pip install sounddevice numpy scipy

# 確認
python -c "import torch; print(f'XPU: {torch.xpu.is_available()}, Version: {torch.__version__}')"
```

### uvのインストール（未導入の場合）
```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# または winget
winget install astral-sh.uv
```

### XPU使用時の注意点
```python
# CUDAコードからの移行: cuda → xpu に置換するだけ
tensor = torch.tensor([1.0, 2.0]).to("xpu")
```

- **GradScalerは無効化必須**: Intel Arc A-Seriesでは`torch.amp.GradScaler`を使わない
- 一部OPがXPU未対応でCPUフォールバックする可能性あり
- **torch.compileの初回実行**: コンパイルに時間がかかるがキャッシュされる

---

## アーキテクチャ設計

### 全体構成

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RCWX Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐                                                  │
│   │  AudioInput  │  sounddevice (WASAPI/ASIO)                       │
│   │  16kHz mono  │                                                  │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐                                                  │
│   │ ChunkBuffer  │  0.1〜0.3秒チャンク + コンテキストバッファ         │
│   │              │  クロスフェード処理                               │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │                   RVC Inference Core                      │      │
│   │  ┌─────────────────────────────────────────────────────┐ │      │
│   │  │  HuBERT (ContentVec)                                │ │      │
│   │  │  hubert_base.pt → torch.compile()                   │ │      │
│   │  │  入力: audio [B, T] → 出力: features [B, T', 256]   │ │      │
│   │  └─────────────────────┬───────────────────────────────┘ │      │
│   │                        │                                  │      │
│   │           ┌────────────┴────────────┐                    │      │
│   │           │                         │                    │      │
│   │           ▼ (F0モデル)              ▼ (No-F0モデル)       │      │
│   │  ┌─────────────────┐       ┌─────────────────┐          │      │
│   │  │  RMVPE          │       │  Skip (f0=None) │          │      │
│   │  │  rmvpe.pt       │       │                 │          │      │
│   │  │  torch.compile()│       └────────┬────────┘          │      │
│   │  └────────┬────────┘                │                    │      │
│   │           │                         │                    │      │
│   │           └────────────┬────────────┘                    │      │
│   │                        │                                  │      │
│   │                        ▼                                  │      │
│   │  ┌─────────────────────────────────────────────────────┐ │      │
│   │  │  SynthesizerTrn (Generator)                         │ │      │
│   │  │  model.pth → torch.compile()                        │ │      │
│   │  │  入力: features, f0 (optional), speaker_id          │ │      │
│   │  │  出力: audio [B, T_out]                             │ │      │
│   │  └─────────────────────────────────────────────────────┘ │      │
│   └──────────────────────────────────────────────────────────┘      │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐                                                  │
│   │ AudioOutput  │  VB-Cable / 仮想オーディオデバイス               │
│   │ 48kHz/44.1kHz│  リサンプリング処理                              │
│   └──────────────┘                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### モジュール構成

```
rcwx/
├── __init__.py
├── config.py              # 設定管理
├── device.py              # デバイス選択 (xpu/cuda/cpu)
├── audio/
│   ├── __init__.py
│   ├── input.py           # マイク入力 (sounddevice)
│   ├── output.py          # 音声出力
│   ├── buffer.py          # チャンクバッファ・クロスフェード
│   └── resample.py        # リサンプリング
├── models/
│   ├── __init__.py
│   ├── hubert.py          # HuBERT/ContentVec ラッパー
│   ├── rmvpe.py           # RMVPE F0抽出
│   └── synthesizer.py     # SynthesizerTrn (Generator)
├── pipeline/
│   ├── __init__.py
│   ├── inference.py       # 推論パイプライン統合
│   └── realtime.py        # リアルタイムストリーム管理
└── cli.py                 # コマンドラインインターフェース
```

---

## コアコンポーネント詳細

### 1. デバイス選択 (`device.py`)

```python
import torch

def get_device() -> str:
    """利用可能な最適デバイスを返す"""
    if torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_dtype(device: str) -> torch.dtype:
    """デバイスに最適なデータ型を返す"""
    if device == "xpu":
        return torch.float16  # Arc GPUはFP16が効率的
    return torch.float32
```

### 2. HuBERT ラッパー (`models/hubert.py`)

```python
import torch
import torch.nn as nn

class HuBERTFeatureExtractor(nn.Module):
    """ContentVec特徴抽出器"""

    def __init__(self, model_path: str, device: str):
        super().__init__()
        self.device = device
        # fairseqまたはtransformers形式のロード
        self.model = self._load_model(model_path)
        self.model.to(device).eval()

    def _load_model(self, path: str):
        # HuBERT/ContentVecモデルのロード実装
        ...

    @torch.no_grad()
    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] 16kHz音声
        Returns:
            features: [B, T', 256] 特徴量
        """
        with torch.autocast(self.device, dtype=torch.float16):
            return self.model.extract_features(audio)
```

### 3. RMVPE F0抽出 (`models/rmvpe.py`)

```python
import torch
import torch.nn as nn

class RMVPE(nn.Module):
    """F0 (ピッチ) 抽出器"""

    def __init__(self, model_path: str, device: str):
        super().__init__()
        self.device = device
        self.model = self._load_model(model_path)
        self.model.to(device).eval()

    @torch.no_grad()
    def infer(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] 16kHz音声
        Returns:
            f0: [B, T'] ピッチ (Hz)
        """
        with torch.autocast(self.device, dtype=torch.float16):
            return self.model(audio)
```

### 4. 推論パイプライン (`pipeline/inference.py`)

```python
import torch
from typing import Optional

class RVCPipeline:
    """RVC推論パイプライン"""

    def __init__(
        self,
        hubert: HuBERTFeatureExtractor,
        synthesizer: SynthesizerTrn,
        rmvpe: Optional[RMVPE] = None,  # No-F0モデルの場合はNone
        device: str = "xpu",
        use_compile: bool = True,
    ):
        self.hubert = hubert
        self.synthesizer = synthesizer
        self.rmvpe = rmvpe
        self.device = device
        self.use_f0 = rmvpe is not None

        if use_compile:
            self._compile_models()

    def _compile_models(self):
        """torch.compileで最適化"""
        self.hubert = torch.compile(self.hubert, mode="reduce-overhead")
        self.synthesizer = torch.compile(self.synthesizer, mode="reduce-overhead")
        if self.rmvpe:
            self.rmvpe = torch.compile(self.rmvpe, mode="reduce-overhead")

    @torch.no_grad()
    def infer(
        self,
        audio: torch.Tensor,
        pitch_shift: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            audio: [T] 入力音声 (16kHz)
            pitch_shift: ピッチシフト (半音単位)
        Returns:
            output: [T_out] 変換後音声
        """
        audio = audio.unsqueeze(0).to(self.device)

        # 特徴抽出
        features = self.hubert.extract(audio)

        # F0抽出 (オプション)
        f0 = None
        if self.use_f0:
            f0 = self.rmvpe.infer(audio)
            if pitch_shift != 0:
                f0 = f0 * (2 ** (pitch_shift / 12))

        # 音声合成
        output = self.synthesizer.infer(features, f0=f0)

        return output.squeeze(0).cpu()
```

### 5. リアルタイムストリーム (`pipeline/realtime.py`)

```python
import sounddevice as sd
import numpy as np
import torch
from threading import Thread
from queue import Queue

class RealtimeVoiceChanger:
    """リアルタイム音声変換"""

    def __init__(
        self,
        pipeline: RVCPipeline,
        input_device: int = None,
        output_device: int = None,
        chunk_sec: float = 0.2,
        sample_rate: int = 16000,
    ):
        self.pipeline = pipeline
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * chunk_sec)

        self.input_device = input_device
        self.output_device = output_device

        self._running = False
        self._input_queue = Queue(maxsize=4)
        self._output_queue = Queue(maxsize=4)

    def _process_callback(self, indata, outdata, frames, time, status):
        """sounddeviceコールバック"""
        if status:
            print(f"Status: {status}")

        audio_in = indata[:, 0].copy()
        self._input_queue.put(audio_in)

        try:
            audio_out = self._output_queue.get_nowait()
            outdata[:, 0] = audio_out[:frames]
        except:
            outdata.fill(0)

    def _inference_thread(self):
        """推論スレッド"""
        while self._running:
            try:
                audio_in = self._input_queue.get(timeout=0.5)
                audio_tensor = torch.from_numpy(audio_in).float()
                audio_out = self.pipeline.infer(audio_tensor)
                self._output_queue.put(audio_out.numpy())
            except:
                pass

    def start(self):
        """変換開始"""
        self._running = True

        # 推論スレッド開始
        self._thread = Thread(target=self._inference_thread, daemon=True)
        self._thread.start()

        # オーディオストリーム開始
        self._stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=1,
            callback=self._process_callback,
            device=(self.input_device, self.output_device),
        )
        self._stream.start()
        print("変換中... Ctrl+C で停止")

    def stop(self):
        """変換停止"""
        self._running = False
        self._stream.stop()
        self._stream.close()
```

---

## レイテンシ設計

### 目標値
| 構成 | チャンクサイズ | 期待レイテンシ |
|------|---------------|---------------|
| F0あり（RMVPE）| 200ms | 170〜250ms |
| No-F0 | 100ms | 90〜150ms |
| ASIO使用 | 100ms | 70〜100ms |

### レイテンシ削減テクニック
1. **torch.compile**: `mode="reduce-overhead"` で推論オーバーヘッド削減
2. **FP16**: `torch.autocast("xpu")` で半精度推論
3. **チャンク重複**: クロスフェードで継ぎ目ノイズを軽減
4. **非同期処理**: 入出力とGPU推論を別スレッドで並列化

---

## 必要なモデルファイル

| ファイル | 用途 | 取得元 | 必須 |
|----------|------|--------|------|
| `hubert_base.pt` | 特徴抽出 | [HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI) | Yes |
| `rmvpe.pt` | F0抽出 | 同上 | F0モデルのみ |
| `*.pth` | RVCv2モデル | 任意 | Yes |
| `*.index` | 検索インデックス | 任意 | No |

---

## RVCコアモジュールの移植元

RVC WebUIから以下を参考に抽出：

```
Retrieval-based-Voice-Conversion-WebUI/
├── infer/
│   ├── lib/
│   │   └── infer_pack/        # モデル定義（SynthesizerTrn等）
│   │       ├── models.py      # ★ Generator本体
│   │       ├── attentions.py  # ★ Attention実装
│   │       ├── commons.py     # ★ ユーティリティ
│   │       ├── modules.py     # ★ サブモジュール
│   │       └── transforms.py  # ★ Piecewise Linear Flow
│   └── modules/
│       └── vc/
│           ├── modules.py     # VCクラス
│           └── pipeline.py    # 変換パイプライン
└── assets/
    └── hubert/                # HuBERTモデル
```

### 移植時の変更点
- `cuda` → `xpu` に置換
- `torch.cuda.amp` → `torch.autocast("xpu")` に変更
- 不要な学習コード削除
- fairseq依存を最小化

---

## パフォーマンス最適化

### torch.compile設定
```python
# mode選択
# - "default": バランス
# - "reduce-overhead": 推論向け（推奨）
# - "max-autotune": 最大性能だがコンパイル時間長い

model = torch.compile(model, mode="reduce-overhead")
```

### AMP (Automatic Mixed Precision)
```python
with torch.autocast("xpu", dtype=torch.float16):
    output = model(input)
```

### キャッシュ活用
```python
# torch.compileのキャッシュ有効化
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
```

---

## 依存関係

### requirements.txt
```
torch>=2.7
torchaudio>=2.7
sounddevice>=0.4.6
numpy>=1.24
scipy>=1.10
```

### オプション依存
```
faiss-cpu>=1.7.4    # indexファイル使用時
librosa>=0.10       # 追加の音声処理
```

---

## TODO（実装タスク）

### Phase 1: 環境構築
- [ ] PyTorch XPU版の動作確認
- [ ] `torch.xpu.is_available()` = True の確認
- [ ] torch.compileのXPU動作確認

### Phase 2: モデル移植
- [ ] HuBERTモデルのロード・XPU動作確認
- [ ] RMVPEのロード・XPU動作確認
- [ ] SynthesizerTrnの移植・XPU動作確認
- [ ] 単体推論テスト（ファイル入力→ファイル出力）

### Phase 3: リアルタイム実装
- [ ] sounddevice入出力実装
- [ ] チャンクバッファ実装
- [ ] パイプライン統合
- [ ] リアルタイム推論動作確認

### Phase 4: 最適化
- [ ] torch.compile適用
- [ ] FP16推論テスト
- [ ] レイテンシ測定
- [ ] チャンクサイズ調整

### Phase 5: 仕上げ
- [ ] CLIインターフェース
- [ ] VB-Cable連携テスト
- [ ] エラーハンドリング

---

## 参考リンク

### 公式ドキュメント
- [PyTorch XPU Getting Started](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [torch.compile on Windows XPU](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
- [RVC-Project GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

### モデルダウンロード
- [HuggingFace: VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main)

### PyPI軽量ラッパー（参考）
- [rvc-python](https://pypi.org/project/rvc-python/) - RVC推論ラッパー
- [infer-rvc-python](https://pypi.org/project/infer-rvc-python/) - 高速推論特化
- [tts-with-rvc-onnx](https://pypi.org/project/tts-with-rvc-onnx/) - ONNX推論

---

## 備考

- 更新日: 2026年1月29日
- Intel Extension for PyTorchは2026年3月末でEOL（PyTorch本体に統合済み）
- RVCv3が開発中だが、v2モデルは引き続き使用可能
- PyTorch 2.10でtorch.xpuが安定版に

## Sources (調査時参照)

- [PyTorch XPU Documentation](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Intel GPU Support in PyTorch 2.5](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/)
- [IPEX End of Life Announcement](https://github.com/intel/intel-extension-for-pytorch/issues/867)
- [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [RMVPE Paper](https://arxiv.org/abs/2306.15412)
