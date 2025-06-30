import torch
import torch.nn as nn
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.orig = orig
        self.lora_A = nn.Linear(orig.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, orig.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        self.orig.weight.requires_grad = False
        if self.orig.bias is not None:
            self.orig.bias.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.orig(x) + self.scaling * self.dropout(self.lora_B(self.lora_A(x)))

class PrefixEncoder(nn.Module):
    def __init__(self, num_prefix_tokens, hidden_size):
        super().__init__()
        self.prefix = nn.Embedding(num_prefix_tokens, hidden_size)
        nn.init.normal_(self.prefix.weight, std=0.02)

    def forward(self, bsz):
        tokens = torch.arange(self.prefix.num_embeddings, device=self.prefix.weight.device)
        tokens = tokens.unsqueeze(0).expand(bsz, -1)
        return self.prefix(tokens)

class PrefixLoRAModel(nn.Module):
    def __init__(
        self,
        model_name="prajjwal1/bert-tiny",
        num_prefix_tokens=10,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        num_labels=2,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_prefix_tokens = num_prefix_tokens

        hidden_size = self.base.bert.config.hidden_size
        self.prefix_encoder = PrefixEncoder(num_prefix_tokens, hidden_size)

        # Inject LoRA into attention
        for layer in self.base.bert.encoder.layer:
            for attr in ["query", "value"]:
                orig = getattr(layer.attention.self, attr)
                setattr(layer.attention.self, attr, LoRALinear(orig, lora_r, lora_alpha, lora_dropout))

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bsz = input_ids.shape[0]
        device = input_ids.device

        # Create prefix embeddings
        prefix_embed = self.prefix_encoder(bsz)  # [bsz, num_prefix_tokens, hidden_dim]

        # Word embeddings
        inputs_embeds = self.base.bert.embeddings.word_embeddings(input_ids)  # [bsz, seq_len, hidden_dim]

        # Combine: prefix + input
        inputs_embeds = torch.cat([prefix_embed, inputs_embeds], dim=1)  # [bsz, num_prefix + seq_len, hidden_dim]

        # Combine attention mask
        prefix_mask = torch.ones(bsz, self.num_prefix_tokens, device=device)  # [bsz, num_prefix_tokens]
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [bsz, num_prefix + seq_len]

        # Combine token_type_ids if present
        if token_type_ids is not None:
            token_type_prefix = torch.zeros(bsz, self.num_prefix_tokens, dtype=token_type_ids.dtype, device=device)
            token_type_ids = torch.cat([token_type_prefix, token_type_ids], dim=1)

        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

