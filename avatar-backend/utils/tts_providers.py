from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
import torch


class BaseTTSProvider(ABC):
    """TTS provider 抽象基类。

    这里把“文本转语音”的能力统一收口成一个简单接口，目的是让上层推理逻辑
    不再感知底层到底是 Qwen 本地模型，还是阿里云在线 TTS。

    这样做的好处：
    1. 现有 `/api/text2audio` 与 `/api/text2avatar` 的 HTTP 协议不需要改；
    2. 默认 provider 仍然可以保持原有 Qwen 逻辑，避免影响现有功能；
    3. 未来如果还要继续接第三个 TTS 源，只需要新增一个 provider 类即可。
    """

    provider_name: str = "base"

    @abstractmethod
    def synthesize_batch(self, *, texts: list[str], language: str, ref_audio: str, ref_text: str) -> tuple[list[np.ndarray], int]:
        """把多段文本合成为多段音频。

        返回值约定为：
        - `list[np.ndarray]`：与 `texts` 顺序一致的音频数组；
        - `int`：采样率。

        之所以统一成这个接口，是因为上层逻辑需要：
        - `text2audio` 场景按句分批生成；
        - `text2avatar` 场景单段直接生成。
        """

    def synthesize_single(self, *, text: str, language: str, ref_audio: str, ref_text: str) -> tuple[np.ndarray, int]:
        """单段文本合成的默认实现。

        默认直接复用 `synthesize_batch`，这样 provider 只实现一套批量逻辑即可。
        """

        wavs, sample_rate = self.synthesize_batch(
            texts=[text],
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        if not wavs:
            raise ValueError(f"{self.provider_name} TTS 未返回音频数据")
        return wavs[0], sample_rate


class QwenTTSProvider(BaseTTSProvider):
    """现有 Qwen3-TTS provider。

    这是当前系统默认使用的实现，保持原有模型路径、推理参数和行为不变，
    以确保在没有显式切换 provider 时，系统表现与改造前一致。
    """

    provider_name = "qwen"

    def __init__(
        self,
        *,
        model_path: str,
        device_map: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        # 这里采用延迟导入，避免在仅使用阿里云 TTS 时也强制初始化 Qwen 相关依赖。
        # 这样既能保持原有 qwen 功能不变，又能让新增 provider 的依赖边界更清晰。
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

    def synthesize_batch(self, *, texts: list[str], language: str, ref_audio: str, ref_text: str) -> tuple[list[np.ndarray], int]:
        """复用原有 Qwen 声纹克隆逻辑。

        这里保留旧实现中的 `create_voice_clone_prompt + generate_voice_clone` 组合，
        以避免对当前线上质量、音色和用法造成影响。
        """

        if not texts:
            return [], 16000

        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        try:
            wavs, sample_rate = self.model.generate_voice_clone(
                text=texts,
                language=[language] * len(texts),
                voice_clone_prompt=prompt_items,
            )
            return [np.asarray(wav, dtype=np.float32) for wav in wavs], int(sample_rate)
        finally:
            del prompt_items
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def synthesize_single(self, *, text: str, language: str, ref_audio: str, ref_text: str) -> tuple[np.ndarray, int]:
        """单段场景仍走原有 `generate_voice_clone` 单次调用方式。

        `text2avatar` 当前原本就是单段直出，这里保持原行为，避免因为强行走批量接口
        带来兼容性风险。
        """

        wavs, sample_rate = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        if not wavs:
            raise ValueError("qwen TTS 未返回音频数据")
        return np.asarray(wavs[0], dtype=np.float32), int(sample_rate)


class AliyunTTSProvider(BaseTTSProvider):
    """阿里云 TTS provider。

    这里采用阿里云智能语音交互（NLS）REST 接口，原因是：
    1. 接入简单，只依赖标准库即可；
    2. 不会影响现有 Qwen provider；
    3. 可以通过环境变量独立配置，便于灰度切换。

    说明：
    - 该实现需要用户自行提供 `AppKey` 与 `Token`；
    - 当前主要面向“文本转音频”能力，`ref_audio/ref_text` 对阿里云 provider 不生效；
    - 为了和现有口型推理链路兼容，这里固定请求 `wav`，方便后续直接读取为 PCM 波形。
    """

    provider_name = "aliyun"

    def __init__(self) -> None:
        self.url = os.environ.get("ALIYUN_TTS_URL", "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/tts")
        self.app_key = os.environ.get("ALIYUN_TTS_APP_KEY", "").strip()
        self.token = os.environ.get("ALIYUN_TTS_TOKEN", "").strip()
        self.voice = os.environ.get("ALIYUN_TTS_VOICE", "") or None
        self.format = os.environ.get("ALIYUN_TTS_FORMAT", "wav").strip().lower() or "wav"
        self.sample_rate = int(os.environ.get("ALIYUN_TTS_SAMPLE_RATE", "16000"))
        self.speech_rate = int(os.environ.get("ALIYUN_TTS_SPEECH_RATE", "0"))
        self.pitch_rate = int(os.environ.get("ALIYUN_TTS_PITCH_RATE", "0"))
        self.volume = int(os.environ.get("ALIYUN_TTS_VOLUME", "50"))
        self.timeout = float(os.environ.get("ALIYUN_TTS_TIMEOUT", "60"))

        if not self.app_key:
            raise ValueError("ALIYUN_TTS_APP_KEY 未配置，无法启用 aliyun TTS provider")
        if not self.token:
            raise ValueError("ALIYUN_TTS_TOKEN 未配置，无法启用 aliyun TTS provider")
        if self.format != "wav":
            raise ValueError("当前 avatar-backend 仅支持 ALIYUN_TTS_FORMAT=wav，以兼容后续口型推理")

    def synthesize_batch(self, *, texts: list[str], language: str, ref_audio: str, ref_text: str) -> tuple[list[np.ndarray], int]:
        """逐段调用阿里云 TTS。

        这里故意采用“逐句请求”的方式，而不是尝试做 provider 内部的文本拼接，
        原因是上层已经按业务逻辑决定了句子切分策略；如果在 provider 内再次拼接，
        会改变现有的时长统计与分段文件输出行为。
        """

        wavs: list[np.ndarray] = []
        sample_rate: int | None = None
        for text in texts:
            audio_bytes = self._request_audio_bytes(text=text)
            wav, current_sr = self._decode_wav_bytes(audio_bytes)
            wavs.append(wav)
            sample_rate = current_sr
        return wavs, int(sample_rate or self.sample_rate)

    def _request_audio_bytes(self, *, text: str) -> bytes:
        """调用阿里云 RESTful TTS 接口并返回原始 WAV 字节流。"""

        payload: dict[str, object] = {
            "appkey": self.app_key,
            "token": self.token,
            "text": text,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "speech_rate": self.speech_rate,
            "pitch_rate": self.pitch_rate,
            "volume": self.volume,
        }
        if self.voice:
            payload["voice"] = self.voice

        request = Request(
            url=self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
                # 阿里云接口在异常场景下可能返回 JSON 错误体。
                # 这里先根据 Content-Type 做一次兜底判断，避免把错误 JSON 当成音频去解码。
                if "json" in content_type.lower():
                    detail = body.decode("utf-8", errors="ignore")
                    raise ValueError(f"阿里云 TTS 返回错误响应: {detail}")
                return body
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ValueError(f"阿里云 TTS 请求失败: HTTP {exc.code} {detail}") from exc
        except URLError as exc:
            raise ValueError(f"阿里云 TTS 服务不可达: {exc}") from exc

    @staticmethod
    def _decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """把阿里云返回的 WAV 字节流解码成 numpy 音频数组。"""

        wav, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            # 若阿里云侧返回多声道数据，这里统一折叠成单声道，
            # 以兼容当前数字人口型模型只接单通道输入的假设。
            wav = np.mean(wav, axis=1, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32), int(sample_rate)


def load_tts_provider() -> BaseTTSProvider:
    """按环境变量加载具体的 TTS provider。

    约定：
    - `TTS_PROVIDER=qwen`：保持原有本地 Qwen TTS；
    - `TTS_PROVIDER=aliyun`：切换到阿里云 REST TTS。

    默认值是 `qwen`，这是为了确保不配置任何新环境变量时，系统仍完全按照旧行为工作。
    """

    provider_name = os.environ.get("TTS_PROVIDER", "qwen").strip().lower() or "qwen"
    if provider_name == "qwen":
        return QwenTTSProvider(model_path="./Qwen3-TTS-12Hz-1.7B-Base")
    if provider_name == "aliyun":
        return AliyunTTSProvider()
    raise ValueError(f"不支持的 TTS_PROVIDER: {provider_name}")
