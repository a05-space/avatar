import runpy
import torch
import warnings

from qwen_tts import Qwen3TTSModel

from utils.infer import Audio2ExpressionInfer


_cfg = None
_infer_engine = None
_tts_model = None


def _load_streaming_config(config_path: str):
    data = runpy.run_path(config_path)
    cfg_data = {k: v for k, v in data.items() if not k.startswith("__")}

    class C:
        pass

    cfg = C()
    for k, v in cfg_data.items():
        setattr(cfg, k, v)
    return cfg


def load_config_and_model(config_path: str = "configs/config.py"):
    global _cfg, _infer_engine, _tts_model
    if _infer_engine is not None and _tts_model is not None:
        return _cfg, _infer_engine, _tts_model

    cfg = _load_streaming_config(config_path)
    infer_engine = Audio2ExpressionInfer(cfg, verbose=False)
    infer_engine.model.eval()

    tts_model = Qwen3TTSModel.from_pretrained(
        "./Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    _cfg = cfg
    _infer_engine = infer_engine
    _tts_model = tts_model
    return _cfg, _infer_engine, _tts_model


def get_config():
    global _cfg
    if _cfg is None:
        load_config_and_model()
    return _cfg


def get_infer_engine():
    global _infer_engine
    if _infer_engine is None:
        load_config_and_model()
    return _infer_engine


def get_tts_model():
    global _tts_model
    if _tts_model is None:
        load_config_and_model()
    return _tts_model


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
