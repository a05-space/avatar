import io
import os
import math
import time
import uuid
import base64
import librosa
import numpy as np
import soundfile as sf
from openai import AsyncOpenAI
from collections import OrderedDict

import torch
import torch.nn.functional as F

from models import build_model
from utils.registry import Registry

from models.utils import smooth_mouth_movements, apply_frame_blending, apply_savitzky_golay_smoothing, \
    symmetrize_blendshapes, apply_random_eye_blinks_context, DEFAULT_CONTEXT, ARKitBlendShape

INFER = Registry("infer")

class InferBase:
    def __init__(self, cfg, model=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.cfg = cfg
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()

    def build_model(self):
        model = build_model(self.cfg.model)
        model = model.to(self.device)
        if self.device.type == 'cuda':
            checkpoint = torch.load(self.cfg.weight)
        else:
            checkpoint = torch.load(self.cfg.weight, map_location='cpu')
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            weight[key] = value
        model.load_state_dict(weight, strict=True)
        return model

@INFER.register_module("Audio2ExpressionInfer")
class Audio2ExpressionInfer(InferBase):
    def infer_streaming_audio(self,
                           audio: np.ndarray,
                           ssr: float,
                           context: dict):

        if context is None:
            context = DEFAULT_CONTEXT.copy()
        max_frame_length = 64

        frame_length = math.ceil(audio.shape[0] / ssr * 30)
        output_context = DEFAULT_CONTEXT.copy()

        volume = librosa.feature.rms(y=audio, frame_length=min(int(1 / 30 * ssr), len(audio)), hop_length=int(1 / 30 * ssr))[0]
        if volume.shape[0] > frame_length:
            volume = volume[:frame_length]

        in_audio = audio.copy()
        start_frame = int(max_frame_length - in_audio.shape[0] / self.cfg.audio_sr * 30)

        if context['is_initial_input'] or (context['previous_audio'] is None):
            blank_audio_length = self.cfg.audio_sr * max_frame_length // 30 - in_audio.shape[0]
            blank_audio = np.zeros(blank_audio_length, dtype=np.float32)

            input_audio = np.concatenate([blank_audio, in_audio])
            output_context['previous_audio'] = input_audio

        with torch.no_grad():
            input_dict = {}

            model_cfg = self.cfg.model
            backbone_cfg = model_cfg.get("backbone", {})
            num_identity_classes = backbone_cfg.get("num_identity_classes", 1)
            input_dict["id_idx"] = F.one_hot(
                torch.tensor(self.cfg.id_idx),
                num_identity_classes,
            ).to(self.device, non_blocking=True)[None, ...]

            input_dict["input_audio_array"] = torch.FloatTensor(input_audio).to(self.device, non_blocking=True)[
                None, ...
            ]
            output_dict = self.model(input_dict)
            out_exp = output_dict["pred_exp"].squeeze().cpu().numpy()[start_frame:, :]

        if context['previous_expression'] is None:
            out_exp = self.apply_expression_postprocessing(out_exp, audio_volume=volume)
        else:
            previous_length = context['previous_expression'].shape[0]
            out_exp = self.apply_expression_postprocessing(expression_params = np.concatenate([context['previous_expression'], out_exp], axis=0),
                                                           audio_volume=np.concatenate([context['previous_volume'], volume], axis=0),
                                                           processed_frames=previous_length)[previous_length:, :]

        if context['previous_expression'] is not None:
            output_context['previous_expression'] = np.concatenate([context['previous_expression'], out_exp], axis=0)[
                                                -max_frame_length:, :]
            output_context['previous_volume'] = np.concatenate([context['previous_volume'], volume], axis=0)[-max_frame_length:]
        else:
            output_context['previous_expression'] = out_exp.copy()
            output_context['previous_volume'] = volume.copy()

        output_context['first_input_flag'] = False

        return {"code": 0,
                "expression": out_exp,
                "headpose": None}, output_context

    @staticmethod
    def apply_expression_postprocessing(
            expression_params: np.ndarray,
            processed_frames: int = 0,
            audio_volume: np.ndarray = None
    ) -> np.ndarray:
        expression_params = smooth_mouth_movements(expression_params, processed_frames, audio_volume)
        expression_params = apply_frame_blending(expression_params, processed_frames)
        expression_params, _ = apply_savitzky_golay_smoothing(expression_params, window_length=5)
        expression_params = symmetrize_blendshapes(expression_params)
        expression_params = apply_random_eye_blinks_context(expression_params, processed_frames=processed_frames)

        return expression_params


def _build_animation_json(
    blendshape_weights: np.ndarray,
    blendshape_names: list[str],
    fps: float,
) -> dict:
    """将表情系数序列转换为前端可用的动画 JSON。"""
    frame_count = blendshape_weights.shape[0]

    frames = []
    for frame_idx in range(frame_count):
        frames.append(
            {
                "weights": blendshape_weights[frame_idx].tolist(),
                "time": frame_idx / fps,
                "rotation": [],
            }
        )

    return {
        "names": blendshape_names,
        "metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "blendshape_names": blendshape_names,
        },
        "frames": frames,
    }


def _process_audio_array(audio: np.ndarray, cfg, infer_engine) -> dict:
    gap = int(cfg.audio_sr)
    context = None
    all_exp = []
    
    t_start = time.time()

    for start in range(0, len(audio), gap):
        end = min(start + gap, len(audio))
        segment = audio[start:end]
        
        if len(segment) == 0:
            continue

        output, context = infer_engine.infer_streaming_audio(
            segment,
            cfg.audio_sr,
            context,
        )

        expression = output.get("expression")
        if expression is not None:
            all_exp.append(expression)

    all_exp_array = np.concatenate(all_exp, axis=0)
    fps = 30.0
    print(f"{time.time() - t_start:.6f}")

    return _build_animation_json(
        blendshape_weights=all_exp_array,
        blendshape_names=ARKitBlendShape,
        fps=fps,
    )


def _process_audio_inference(tmp_path: str, cfg, infer_engine) -> dict:
    audio, _ = librosa.load(tmp_path, sr=cfg.audio_sr)
    return _process_audio_array(audio, cfg, infer_engine)


def _process_text2audio_task(tts_model, text: str, language: str, ref_audio: str, ref_text: str):
    sentences = text.replace("\n", "").split("。")
    sentences = [s+'。' for s in sentences if s]
        
    languages = [language] * len(sentences)

    t_start = time.time()

    prompt_items = tts_model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    wavs, sr = tts_model.generate_voice_clone(
        text=sentences,
        language=languages,
        voice_clone_prompt=prompt_items,
    )

    del prompt_items
    torch.cuda.empty_cache()

    task_id = str(uuid.uuid4())
    base_dir = "./audio"
    task_dir = os.path.join(base_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)

    for i, wav in enumerate(wavs):
        file_path = os.path.join(task_dir, f"{i}.wav")
        sf.write(file_path, wav, sr)

    times = [len(wav) / sr for wav in wavs]
    print(f"{time.time() - t_start:.6f}")
    return sentences, times, task_id


def _process_text2avatar_task(tts_model, cfg, infer_engine, text: str, language: str, ref_audio: str, ref_text: str):
    t_start = time.time()
    wavs, sr = tts_model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    
    wav = wavs[0]

    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='WAV')
    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_base64 = f"data:audio/wav;base64,{audio_b64}"

    if sr != cfg.audio_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.audio_sr)
        
    animation_json = _process_audio_array(
        wav,
        cfg,
        infer_engine
    )
    print(f"{time.time() - t_start:.6f}")
    
    return {
        "expression": animation_json,
        "audioBase64": audio_base64,
    }


async def _process_llm_inference(request):
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = await client.chat.completions.create(
        model=request.model,
        messages=request.messages,
        stream=request.stream,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return response
