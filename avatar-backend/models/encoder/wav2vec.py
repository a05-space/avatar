import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput


class Wav2Vec2Model(Wav2Vec2Model):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.lm_head = nn.Linear(1024, 32)

    def linear_interpolation(self, features, output_len=None):
        output_features = F.interpolate(features.transpose(1, 2), size=output_len, align_corners=True, mode="linear")
        return output_features.transpose(1, 2)

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            frame_num=None
    ) -> BaseModelOutput:
        hidden_states = self.feature_extractor(input_values).transpose(1, 2)
        hidden_states = self.linear_interpolation(hidden_states, output_len=frame_num)
        hidden_states = self.feature_projection(hidden_states)[0]

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.use_return_dict,
        )

        return BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
