from typing import List, Any

from torch import nn
import torch


"""
These models are used for tests only
They need to be in packages to be serializable/deserializable
by Pyspark (via pickle)
"""


class ToyModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(2, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, 10)
        self.softmax = nn.Softmax()

    def forward(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        features = torch.stack([
            torch.squeeze(feature1), torch.squeeze(feature2)
        ], dim=1)
        output1 = self.hidden1(features)
        output2 = self.hidden2(output1)
        return self.softmax(output2)


class Reducer(nn.Module):

    def __init__(self) -> None:
        super(Reducer, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x - y, 1)


class Translator(nn.Module):

    def __init__(self) -> None:
        super(Translator, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Tokenizer(nn.Module):

    vocab_en = {
        "hello": 1,
        "world": 2,
        "how": 3,
        "are": 4,
        "you": 5
    }

    vocab_fr = {
        1: "bonjour",
        2: "tout le monde",
        3: "comment",
        4: "vas",
        5: "tu"
    }

    default_en = 0

    default_fr = ""

    def __init__(self) -> None:
        super(Tokenizer, self).__init__()

    def encode(self, x: List[str]) -> List[List[int]]:
        tokens = []
        for sentence in x:
            tokens.append([
                self.vocab_en[word] if word in self.vocab_en else self.default_en
                for word in sentence.lower().split(" ")
            ])
        return tokens

    def decode(self, x: List[List[int]]) -> List[str]:
        sentenses = []
        for tokens in x:
            words = [
                self.vocab_fr[token] if token in self.vocab_fr else self.default_fr
                for token in tokens
            ]
            sentenses.append(" ".join(words))
        return sentenses


def load_toy_model(hidden_size: int) -> ToyModel:
    return ToyModel(hidden_size)


def load_reducer(*args: Any, **kwargs: Any) -> Reducer:
    return Reducer()


def load_translator(*args: Any, **kwargs: Any) -> Translator:
    return Translator()


def load_tokenizer(*args: Any, **kwargs: Any) -> Tokenizer:
    return Tokenizer()
