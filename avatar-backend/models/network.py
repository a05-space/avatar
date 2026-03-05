import math
import torch
import torch.nn as nn

from models.builder import MODELS
from models.encoder.wav2vec import Wav2Vec2Model

from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config


@MODELS.register_module("Audio2Expression")
class Audio2Expression(nn.Module):

    def __init__(self,
                 device: torch.device = None,
                 pretrained_encoder_type: str = 'wav2vec',
                 pretrained_encoder_path: str = '',
                 wav2vec2_config_path: str = '',
                 num_identity_classes: int = 0,
                 identity_feat_dim: int = 64,
                 hidden_dim: int = 512,
                 expression_dim: int = 52,
                 norm_type: str = 'ln',
                 decoder_depth: int = 3,
                 use_transformer: bool = False,
                 num_attention_heads: int = 8,
                 num_transformer_layers: int = 6,
                 ) -> None:
        super().__init__()

        self.device = device

        config = Wav2Vec2Config.from_pretrained(wav2vec2_config_path)
        self.audio_encoder = Wav2Vec2Model(config)
        encoder_output_dim = 768

        self.audio_encoder.feature_extractor._freeze_parameters()
        self.feature_projection = nn.Linear(encoder_output_dim, hidden_dim)

        self.identity_encoder = AudioIdentityEncoder(
            hidden_dim,
            num_identity_classes,
            identity_feat_dim,
            use_transformer,
            num_attention_heads,
            num_transformer_layers
        )

        self.decoder = nn.ModuleList([
            nn.Sequential(*[
                ConvNormRelu(hidden_dim, hidden_dim, norm=norm_type)
                for _ in range(decoder_depth)
            ])
        ])

        self.output_proj = nn.Linear(hidden_dim, expression_dim)

    def forward(self, input_dict):
        audio_length = input_dict['input_audio_array'].shape[1]
        time_steps = math.ceil(audio_length / 16000 * 30)

        audio_input = input_dict['input_audio_array'].flatten(start_dim=1)
        hidden_states = self.audio_encoder(audio_input, frame_num=time_steps).last_hidden_state

        audio_features = self.feature_projection(hidden_states).transpose(1, 2)
        audio_features = self.identity_encoder(audio_features, identity=input_dict['id_idx'])
        audio_features = self.decoder[0](audio_features)
        audio_features = audio_features.permute(0, 2, 1)

        expression_params = self.output_proj(audio_features)

        return torch.sigmoid(expression_params)


class AudioIdentityEncoder(nn.Module):

    def __init__(self,
                 hidden_dim,
                 num_identity_classes=0,
                 identity_feat_dim=64,
                 use_transformer=False,
                 num_attention_heads = 8,
                 num_transformer_layers = 6,
                 dropout_ratio=0.1,
                 ) -> None:
        super().__init__()
        in_dim = hidden_dim + identity_feat_dim

        self.id_mlp = nn.Conv1d(num_identity_classes, identity_feat_dim, 1, 1)
        self.first_net = SeqTranslator1D(in_dim, hidden_dim,
                                         min_layers_num=3,
                                         residual=True,
                                         norm='ln'
                                         )
        self.grus = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout_ratio)
        self.use_transformer = use_transformer

    def forward(self,
                audio_features: torch.Tensor,
                identity: torch.Tensor = None,
                time_steps: int = None) -> tuple:
        audio_features = self.dropout(audio_features)
        identity = identity.reshape(identity.shape[0], -1, 1).repeat(1, 1, audio_features.shape[2]).to(torch.float32)
        identity = self.id_mlp(identity)
        audio_features = torch.cat([audio_features, identity], dim=1)
        return self.first_net(audio_features)

class ConvNormRelu(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 type='1d',
                 leaky=False,
                 downsample=False,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 p=0,
                 groups=1,
                 residual=False,
                 norm='bn') -> None:
        super().__init__()

        self.residual = residual
        self.norm_type = norm

        if kernel_size is None and stride is None:
            kernel_size = 3
            stride = 1

        if padding is None:
            padding = int((kernel_size - stride) / 2)

        if self.residual:
            if in_channels == out_channels:
                self.residual_layer = nn.Identity()
            else:
                if type == '1d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )

        in_channels = in_channels * groups
        out_channels = out_channels * groups

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        out = self.dropout(self.conv(x))
        out = self.norm(out.transpose(1,2)).transpose(1,2)
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)

class SeqTranslator1D(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=None,
                 stride=None,
                 min_layers_num=None,
                 residual=True,
                 norm='bn'
                 ) -> None:
        super().__init__()
        conv_layers = nn.ModuleList([])
        conv_layers.append(ConvNormRelu(
            in_channels=C_in,
            out_channels=C_out,
            type='1d',
            kernel_size=kernel_size,
            stride=stride,
            residual=residual,
            norm=norm
        ))
        
        self.num_layers = 1
        if min_layers_num is not None and self.num_layers < min_layers_num:
            while self.num_layers < min_layers_num:
                conv_layers.append(ConvNormRelu(
                    in_channels=C_out,
                    out_channels=C_out,
                    type='1d',
                    kernel_size=kernel_size,
                    stride=stride,
                    residual=residual,
                    norm=norm
                ))
                self.num_layers += 1
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_layers(x)
