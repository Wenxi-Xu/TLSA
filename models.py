import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class ProjectionHead(nn.Module):
    """Project BERT features to contrastive learning space."""
    def __init__(self, input_dim, output_dim, hidden_dim=None, use_mlp=False):
        super(ProjectionHead, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
            
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        return F.normalize(projected, dim=-1)





class TLSAModel(nn.Module):
    """TLSA model with shared BERT and separate projection heads."""
    def __init__(self, bert_model_path, proj_dim=256, temperature=0.07,
                 use_mlp_projection=False,
                 mlp_hidden_dim=None):
        super(TLSAModel, self).__init__()

        self.temperature = temperature

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.bert_config = self.bert.config

        self.text_projection_head = ProjectionHead(
            self.bert_config.hidden_size,
            proj_dim,
            hidden_dim=mlp_hidden_dim,
            use_mlp=use_mlp_projection
        )

        self.label_projection_head = ProjectionHead(
            self.bert_config.hidden_size,
            proj_dim,
            hidden_dim=mlp_hidden_dim,
            use_mlp=use_mlp_projection
        )

        self.configure_trainable_parameters()

    def configure_trainable_parameters(self):
        """Train only last 3 BERT layers and both projection heads."""
        for param in self.bert.parameters():
            param.requires_grad = False

        for i in range(3):
            layer_idx = self.bert_config.num_hidden_layers - 1 - i
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = True

        for param in self.text_projection_head.parameters():
            param.requires_grad = True
        for param in self.label_projection_head.parameters():
            param.requires_grad = True

    def get_pooled_output(self, input_ids, attention_mask, token_type_ids=None):
        """Get mean-pooled output over all tokens."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        last_hidden_state = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

        pooled_output = sum_embeddings / sum_mask

        return pooled_output

    def encode_text(self, input_ids, attention_mask, token_type_ids=None):
        """Encode text."""
        pooled_output = self.get_pooled_output(input_ids, attention_mask, token_type_ids)
        projected_output = self.text_projection_head(pooled_output)

        return projected_output

    def encode_labels(self, input_ids, attention_mask, token_type_ids=None):
        """Encode labels."""
        pooled_output = self.get_pooled_output(input_ids, attention_mask, token_type_ids)
        projected_output = self.label_projection_head(pooled_output)

        return projected_output

    def forward(self, text_input_ids, text_attention_mask, text_token_type_ids,
                label_input_ids, label_attention_mask, label_token_type_ids):
        """Forward pass returns text and label embeddings."""
        text_embeddings = self.encode_text(
            text_input_ids, text_attention_mask, text_token_type_ids
        )
        label_embeddings = self.encode_labels(
            label_input_ids, label_attention_mask, label_token_type_ids
        )

        return text_embeddings, label_embeddings


    def set_training_mode(self, phase='supervised'):
        """Set training mode (unified strategy for both phases)."""
        self.configure_trainable_parameters()


def create_tlsa_model(args, num_known_classes=None):
    """Create TLSA model."""
    model = TLSAModel(
        bert_model_path=args.bert_model,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        use_mlp_projection=getattr(args, 'use_mlp_projection', False),
        mlp_hidden_dim=getattr(args, 'mlp_hidden_dim', None)
    )

    return model