id_idx: int = 0
audio_sr: int = 16000
batch_size: int = 4

weight: str = "pretrained_models/avatar.tar"

model: dict[str, str | dict[str, str | int | bool]] = {
    "type": "Estimator",
    "backbone": {
        "type": "Audio2Expression",
        "pretrained_encoder_type": "wav2vec",
        "pretrained_encoder_path": "facebook/wav2vec2-base-960h",
        "wav2vec2_config_path": "configs/wav2vec2_config.json",
        "num_identity_classes": 12,
        "identity_feat_dim": 64,
        "hidden_dim": 512,
        "expression_dim": 52,
        "norm_type": "ln",
        "use_transformer": False,
        "num_attention_heads": 8,
        "num_transformer_layers": 6,
    },
}
