import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class SmartHeadConfig(PretrainedConfig):
    model_type = "smart_mean_cls_head"

    def __init__(
        self,
        encoder_name_or_path=None,
        encoder_hidden_size=128,
        head_hidden_size=256,
        head_dropout=0.15,
        max_length=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_name_or_path = encoder_name_or_path
        self.encoder_hidden_size = encoder_hidden_size
        self.head_hidden_size = head_hidden_size
        self.head_dropout = head_dropout
        self.max_length = max_length


class SmartHeadClassificationModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config.encoder_hidden_size
        h = config.head_hidden_size
        p = config.head_dropout
        self.in_norm = nn.LayerNorm(4 * H)
        self.in_proj = nn.Sequential(nn.Linear(4 * H, h), nn.GELU(), nn.Dropout(p))
        self.res_norm = nn.LayerNorm(h)
        self.res_dense1 = nn.Linear(h, h * 2)
        self.res_act = nn.GELU()
        self.res_drop = nn.Dropout(p)
        self.res_dense2 = nn.Linear(h * 2, h)
        self.out_norm = nn.LayerNorm(h)
        self.out_drop = nn.Dropout(p)
        self.classifier = nn.Linear(h, config.num_labels)

    def forward(self, pooled):
        H = pooled.shape[-1] // 2
        u = pooled[:, :H]
        v = pooled[:, H:]
        x = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        x = self.in_proj(self.in_norm(x))
        x = x + self.res_dense2(self.res_drop(self.res_act(self.res_dense1(self.res_norm(x)))))
        return self.classifier(self.out_drop(self.out_norm(x)))


class SmartHeadForSequenceClassification(PreTrainedModel):
    config_class = SmartHeadConfig
    _keys_to_ignore_on_load_missing = [r"encoder\."]

    @property
    def all_tied_weights_keys(self):
        return getattr(self, "_tied_weights_keys", None) or {}

    def __init__(self, config: SmartHeadConfig):
        super().__init__(config)
        self.config = config
        encoder_config = AutoConfig.from_pretrained(
            config.encoder_name_or_path, trust_remote_code=True
        )
        self.encoder = AutoModel.from_config(encoder_config, trust_remote_code=True)
        self.encoder.requires_grad_(False).eval()
        self.head = SmartHeadClassificationModule(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        if model.config.encoder_name_or_path:
            pretrained_encoder = AutoModel.from_pretrained(
                model.config.encoder_name_or_path,
                dtype=torch.float16,
                trust_remote_code=True,
            )
            model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)
            del pretrained_encoder
            torch.cuda.empty_cache()
            model.encoder.requires_grad_(False).eval()
        return model.eval()

    @torch.inference_mode()
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float()
        u = hs[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        v = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = torch.cat([u, v], dim=-1)
        return SequenceClassifierOutput(logits=self.head(pooled))
