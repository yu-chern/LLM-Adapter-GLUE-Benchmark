import torch
import torch.nn as nn
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.orig = orig_linear
        self.lora_A = nn.Linear(orig_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, orig_linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        # Freeze original weights
        self.orig.weight.requires_grad = False
        if self.orig.bias is not None:
            self.orig.bias.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.orig(x) + self.scaling * self.dropout(self.lora_B(self.lora_A(x)))

class SoftPromptLoRAModel(nn.Module):
    def __init__(
        self,
        model_name="prajjwal1/bert-tiny",
        num_virtual_tokens=10,
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

        # Soft Prompt: add learnable embeddings
        hidden_size = self.base.bert.embeddings.word_embeddings.embedding_dim
        self.soft_prompt = nn.Embedding(num_virtual_tokens, hidden_size)
        nn.init.normal_(self.soft_prompt.weight, std=0.02)
        self.num_virtual_tokens = num_virtual_tokens

        # Inject LoRA into attention (query/key/value)
        for layer in self.base.bert.encoder.layer:
            for attr in ["query", "value"]:
                orig = getattr(layer.attention.self, attr)
                setattr(layer.attention.self, attr, LoRALinear(orig, lora_r, lora_alpha, lora_dropout))

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bsz = input_ids.shape[0]
        device = input_ids.device

        # Create virtual token ids and embeddings
        soft_ids = torch.arange(self.num_virtual_tokens, device=device).unsqueeze(0).expand(bsz, -1)
        soft_embeds = self.soft_prompt(soft_ids)

        # Standard embeddings
        orig_embed = self.base.bert.embeddings.word_embeddings(input_ids)

        # Concatenate soft prompt and input embeddings
        inputs_embeds = torch.cat([soft_embeds, orig_embed], dim=1)

        # Update attention mask
        soft_mask = torch.ones(bsz, self.num_virtual_tokens, device=device)
        attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        # Update token_type_ids if present
        if token_type_ids is not None:
            token_type_ids = torch.cat([
                torch.zeros(bsz, self.num_virtual_tokens, dtype=token_type_ids.dtype, device=device),
                token_type_ids
            ], dim=1)

        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

