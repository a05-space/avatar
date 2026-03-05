import time
import numpy as np
from pydantic import BaseModel
from typing import Optional,Tuple
from scipy.signal import savgol_filter


ARKitLeftRightPair = [
        ("jawLeft", "jawRight"),
        ("mouthLeft", "mouthRight"),
        ("mouthSmileLeft", "mouthSmileRight"),
        ("mouthFrownLeft", "mouthFrownRight"),
        ("mouthDimpleLeft", "mouthDimpleRight"),
        ("mouthStretchLeft", "mouthStretchRight"),
        ("mouthPressLeft", "mouthPressRight"),
        ("mouthLowerDownLeft", "mouthLowerDownRight"),
        ("mouthUpperUpLeft", "mouthUpperUpRight"),
        ("cheekSquintLeft", "cheekSquintRight"),
        ("noseSneerLeft", "noseSneerRight"),
        ("browDownLeft", "browDownRight"),
        ("browOuterUpLeft", "browOuterUpRight"),
        ("eyeBlinkLeft","eyeBlinkRight"),
        ("eyeLookDownLeft","eyeLookDownRight"),
        ("eyeLookInLeft", "eyeLookInRight"),
        ("eyeLookOutLeft","eyeLookOutRight"),
        ("eyeLookUpLeft","eyeLookUpRight"),
        ("eyeSquintLeft","eyeSquintRight"),
        ("eyeWideLeft","eyeWideRight")
    ]

ARKitBlendShape =[
   "browDownLeft",
   "browDownRight",
   "browInnerUp",
   "browOuterUpLeft",
   "browOuterUpRight",
   "cheekPuff",
   "cheekSquintLeft",
   "cheekSquintRight",
   "eyeBlinkLeft",
   "eyeBlinkRight",
   "eyeLookDownLeft",
   "eyeLookDownRight",
   "eyeLookInLeft",
   "eyeLookInRight",
   "eyeLookOutLeft",
   "eyeLookOutRight",
   "eyeLookUpLeft",
   "eyeLookUpRight",
   "eyeSquintLeft",
   "eyeSquintRight",
   "eyeWideLeft",
   "eyeWideRight",
   "jawForward",
   "jawLeft",
   "jawOpen",
   "jawRight",
   "mouthClose",
   "mouthDimpleLeft",
   "mouthDimpleRight",
   "mouthFrownLeft",
   "mouthFrownRight",
   "mouthFunnel",
   "mouthLeft",
   "mouthLowerDownLeft",
   "mouthLowerDownRight",
   "mouthPressLeft",
   "mouthPressRight",
   "mouthPucker",
   "mouthRight",
   "mouthRollLower",
   "mouthRollUpper",
   "mouthShrugLower",
   "mouthShrugUpper",
   "mouthSmileLeft",
   "mouthSmileRight",
   "mouthStretchLeft",
   "mouthStretchRight",
   "mouthUpperUpLeft",
   "mouthUpperUpRight",
   "noseSneerLeft",
   "noseSneerRight",
   "tongueOut"
]

MOUTH_BLENDSHAPES = [ 
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "noseSneerLeft",
    "noseSneerRight",
    "cheekPuff",
]

DEFAULT_CONTEXT ={
    'is_initial_input': True,
    'previous_audio': None,
    'previous_expression': None,
    'previous_volume': None,
    'previous_headpose': None,
}

BLINK_PATTERNS = [
    np.array([0.365, 0.950, 0.956, 0.917, 0.367, 0.119, 0.025]),
    np.array([0.235, 0.910, 0.945, 0.778, 0.191, 0.235, 0.089]),
    np.array([0.870, 0.950, 0.949, 0.696, 0.191, 0.073, 0.007]),
    np.array([0.000, 0.557, 0.953, 0.942, 0.426, 0.148, 0.018])
]

def symmetrize_blendshapes(
        bs_params: np.ndarray,
        mode: str = "average",
        symmetric_pairs: list = ARKitLeftRightPair
) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(ARKitBlendShape)}

    symmetric_bs = bs_params.copy()

    valid_pairs = []
    for left, right in symmetric_pairs:
        left_idx = name_to_idx.get(left)
        right_idx = name_to_idx.get(right)
        if None not in (left_idx, right_idx):
            valid_pairs.append((left_idx, right_idx))

    for l_idx, r_idx in valid_pairs:
        left_col = symmetric_bs[:, l_idx]
        right_col = symmetric_bs[:, r_idx]

        new_vals = (left_col + right_col) / 2
        symmetric_bs[:, l_idx] = new_vals
        symmetric_bs[:, r_idx] = new_vals

    return symmetric_bs

def apply_random_eye_blinks_context(
        animation_params: np.ndarray,
        processed_frames: int = 0,
        intensity_range: tuple = (0.8, 1.0)
) -> np.ndarray:
    remaining_frames = animation_params.shape[0] - processed_frames

    min_blink_interval = 40
    max_blink_interval = 100

    previous_blink_indices = np.where(animation_params[:processed_frames, 8] > 0.5)[0]
    last_processed_blink = previous_blink_indices[-1] - 7 if previous_blink_indices.size > 0 else processed_frames

    blink_interval = np.random.randint(min_blink_interval, max_blink_interval)
    first_blink_start = max(0, blink_interval - last_processed_blink)

    if first_blink_start <= (remaining_frames - 7):
        blink_pattern = BLINK_PATTERNS[np.random.randint(0, 4)]
        intensity = np.random.uniform(*intensity_range)

        blink_start = processed_frames + first_blink_start
        blink_end = blink_start + 7

        animation_params[blink_start:blink_end, 8] = blink_pattern * intensity
        animation_params[blink_start:blink_end, 9] = blink_pattern * intensity

    return animation_params


def apply_savitzky_golay_smoothing(
        input_data: np.ndarray,
        window_length: int = 5,
        polyorder: int = 2,
        axis: int = 0,
        validate: bool = True
) -> Tuple[np.ndarray, Optional[float]]:
    original_dtype = input_data.dtype
    working_data = input_data.astype(np.float64)

    timer_start = time.perf_counter()

    smoothed_data = savgol_filter(working_data,
                                  window_length=window_length,
                                  polyorder=polyorder,
                                  axis=axis,
                                  mode='mirror')

    processing_time = time.perf_counter() - timer_start

    return (
        np.clip(smoothed_data,
                0.0,
                1.0
                ).astype(original_dtype),
        processing_time
    )


def _blend_region_start(
    array: np.ndarray,
    region: np.ndarray,
    processed_boundary: int,
    blend_frames: int
) -> None:
    blend_length = min(blend_frames, region[0] - processed_boundary)
    if blend_length <= 0:
        return

    pre_frame = array[region[0] - 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[0] + i] = pre_frame * (1 - weight) + array[region[0] + i] * weight

def _blend_region_end(
    array: np.ndarray,
    region: np.ndarray,
    blend_frames: int
) -> None:
    blend_length = min(blend_frames, array.shape[0] - region[-1] - 1)
    if blend_length <= 0:
        return

    post_frame = array[region[-1] + 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[-1] - i] = post_frame * (1 - weight) + array[region[-1] - i] * weight

def find_low_value_regions(
        signal: np.ndarray,
        threshold: float,
        min_region_length: int = 5
) -> list:
    low_value_indices = np.where(signal < threshold)[0]
    contiguous_regions = []
    current_region_length = 0
    region_start_idx = 0

    for i in range(1, len(low_value_indices)):
        if low_value_indices[i] != low_value_indices[i - 1] + 1:
            if current_region_length >= min_region_length:
                contiguous_regions.append(low_value_indices[region_start_idx:i])
            region_start_idx = i
            current_region_length = 0
        current_region_length += 1

    if current_region_length >= min_region_length:
        contiguous_regions.append(low_value_indices[region_start_idx:])

    return contiguous_regions


def smooth_mouth_movements(
        blend_shapes: np.ndarray,
        processed_frames: int,
        volume: np.ndarray = None,
        silence_threshold: float = 0.001,
        min_silence_duration: int = 7,
        blend_window: int = 3
) -> np.ndarray:
    silent_regions = find_low_value_regions(
        volume,
        threshold=silence_threshold,
        min_region_length=min_silence_duration
    )

    for region_indices in silent_regions:
        # Reduce mouth blend shapes in silent region
        mouth_blend_indices = [ARKitBlendShape.index(name) for name in MOUTH_BLENDSHAPES]
        for region_indice in region_indices.tolist():
            blend_shapes[region_indice, mouth_blend_indices] *= 0.1

        _blend_region_start(
            blend_shapes,
            region_indices,
            processed_frames,
            blend_window
        )

        _blend_region_end(
            blend_shapes,
            region_indices,
            blend_window
        )
    return blend_shapes


def apply_frame_blending(
        blend_shapes: np.ndarray,
        processed_frames: int,
        initial_blend_window: int = 3,
        subsequent_blend_window: int = 5
) -> np.ndarray:
    if processed_frames > 0:
        _blend_animation_segment(
            blend_shapes,
            transition_start=processed_frames,
            blend_window=subsequent_blend_window,
            reference_frame=blend_shapes[processed_frames - 1]
        )
    else:
        _blend_animation_segment(
            blend_shapes,
            transition_start=0,
            blend_window=initial_blend_window,
            reference_frame=np.zeros_like(blend_shapes[0])
        )
    return blend_shapes


def _blend_animation_segment(
        array: np.ndarray,
        transition_start: int,
        blend_window: int,
        reference_frame: np.ndarray
) -> None:
    actual_blend_length = min(blend_window, array.shape[0] - transition_start)

    for frame_offset in range(actual_blend_length):
        current_idx = transition_start + frame_offset
        blend_weight = (frame_offset + 1) / (actual_blend_length + 1)

        array[current_idx] = (reference_frame * (1 - blend_weight)
                              + array[current_idx] * blend_weight)


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "Chinese"
    ref_audio: Optional[str] = "./Qwen3-TTS-12Hz-1.7B-Base/LiBai_TTS_chinese.wav"
    ref_text: Optional[str] = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"


class LLMRequest(BaseModel):
    messages: list
    stream: bool = False
    model: str = "qwen3-vl-plus"
    temperature: float = 0.7
    max_tokens: int = 16384
