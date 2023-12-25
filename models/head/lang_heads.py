import torch
import torch.nn as nn

from transformers.activations import ACT2FN


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size=512, hidden_act='gelu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size=512, hidden_act='gelu', vocab_size=49408):  # CLIP:49408
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size=512, hidden_act='gelu', vocab_size=49408):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, hidden_act, vocab_size)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores