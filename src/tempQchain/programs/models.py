import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    ModernBertModel,
)


class BERTTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, _, question, story):
        encoded_input = self.tokenizer(question, story, padding="max_length", truncation=True)
        input_ids = encoded_input["input_ids"]
        return torch.LongTensor(input_ids)


class ModernBERTTokenizer:
    def __init__(self, special_tokens: list[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        if special_tokens:
            self.tokenizer.add_tokens(special_tokens)

    def __call__(self, _, question, story):
        encoded_input = self.tokenizer(
            question, story, padding="max_length", truncation=True, max_length=5000, return_tensors=None
        )

        input_ids = encoded_input["input_ids"]
        return torch.LongTensor(input_ids)


class MultipleClassYN(BertPreTrainedModel):
    def __init__(self, config, device="cpu", drp=False):
        super().__init__(config)

        if drp:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.cur_device = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)

        return output


class ModernBert(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base", num_classes=6, drp=False, device="cpu"):
        super().__init__()

        self.bert = ModernBertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        dropout_prob = 0.0 if drp else getattr(self.bert.config, "classifier_dropout", 0.0)
        self.dropout = nn.Dropout(dropout_prob)

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.device = device
        self.to(device)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # probs = self.softmax(logits)
        return logits


class MultipleClassYN_Hidden(BertPreTrainedModel):
    def __init__(self, config, device="cpu", drp=False):
        super().__init__(config)

        if drp:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.cur_device = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        return pooled_output


class ClassifyLayer(nn.Module):
    def __init__(self, hidden_size, device="cpu", drp=False):
        super().__init__()

        self.num_classes = 2
        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, pooled_output):
        output = self.classifier(pooled_output)

        return output


class ClassifyLayer2(nn.Module):
    def __init__(self, hidden_size, hidden_layer=1, device="cpu", drp=False):
        super().__init__()

        self.num_classes = 2
        layer_parameters = [hidden_size] + [256 for i in range(hidden_layer - 1)] + [self.num_classes]

        all_layer = []
        for i in range(len(layer_parameters) - 1):
            all_layer.append(nn.Linear(layer_parameters[i], layer_parameters[i + 1]))
        self.classifier = nn.Sequential(*all_layer)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, pooled_output):
        logits = self.classifier(pooled_output)
        # logits = self.sigmoid(logits)

        return logits
