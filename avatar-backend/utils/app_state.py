import runpy
import warnings

from utils.infer import Audio2ExpressionInfer
from utils.tts_providers import load_tts_provider


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

    # TTS provider 支持通过环境变量切换。
    # 默认值仍然是 qwen，因此不新增任何环境变量时，旧功能和旧行为都保持不变。
    tts_model = load_tts_provider()

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
