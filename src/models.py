from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DNA_RC_CHANNELS = (3, 2, 1, 0)


def reverse_complement_one_hot(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a (batch, channel, position) one-hot tensor."""
    channel_index = torch.as_tensor(DNA_RC_CHANNELS, device=x.device)
    return torch.flip(x, dims=(-1,)).index_select(1, channel_index)


def reverse_complement_feature_pairs(x: torch.Tensor) -> torch.Tensor:
    """Reverse positions and swap the tied first/second halves of RCPS features."""
    if x.ndim != 3 or x.shape[1] % 2:
        raise ValueError("RCPS features must have shape (batch, even_channels, position)")
    half = x.shape[1] // 2
    channel_index = torch.cat(
        (
            torch.arange(half, x.shape[1], device=x.device),
            torch.arange(0, half, device=x.device),
        )
    )
    return torch.flip(x, dims=(-1,)).index_select(1, channel_index)


class DNAConvNet(nn.Module):
    """Input: (batch, 4, length); output: one logit per sequence."""

    def __init__(self, config: dict):
        super().__init__()
        c1 = int(config["conv_channels"])
        layers: list[nn.Module] = [
            nn.Conv1d(4, c1, int(config["kernel_size"]), padding="same"), nn.ReLU()
        ]
        final_channels = c1
        if bool(config.get("second_conv", False)):
            c2 = int(config["second_conv_channels"])
            layers += [nn.Conv1d(c1, c2, int(config["second_kernel_size"]), padding="same"), nn.ReLU()]
            final_channels = c2
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(final_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).squeeze(-1)).squeeze(-1)


class PostHocConjoined(nn.Module):
    """Make any classifier exactly RC invariant by averaging both orientations."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self.backbone(x) + self.backbone(reverse_complement_one_hot(x))
        )


class RCPSConv1d(nn.Module):
    """Convolution whose output filters occur in tied RC partner pairs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        input_rc_permutation: tuple[int, ...],
    ):
        super().__init__()
        if out_channels % 2:
            raise ValueError("RCPSConv1d requires an even output width")
        if kernel_size % 2 == 0:
            raise ValueError("RCPSConv1d requires odd kernels for exact same-padding equivariance")
        if len(input_rc_permutation) != in_channels:
            raise ValueError("Invalid input RC permutation")
        half_out = out_channels // 2
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(half_out, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(half_out))
        self.register_buffer(
            "input_rc_permutation",
            torch.as_tensor(input_rc_permutation, dtype=torch.long),
            persistent=False,
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def full_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        partner = torch.flip(
            self.weight.index_select(1, self.input_rc_permutation), dims=(-1,)
        )
        return torch.cat((self.weight, partner)), torch.cat((self.bias, self.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluate both members of every RC pair with the identical half-width
        # convolution.  Constructing a concatenated partner-weight tensor is
        # algebraically equivalent, but different float32 accumulation paths can
        # compound across layers and obscure the strict numerical guarantee.
        first = F.conv1d(x, self.weight, self.bias, padding=self.padding)
        x_rc = torch.flip(x, dims=(-1,)).index_select(1, self.input_rc_permutation)
        partner = torch.flip(
            F.conv1d(x_rc, self.weight, self.bias, padding=self.padding), dims=(-1,)
        )
        return torch.cat((first, partner), dim=1)


class RCPSDNAConvNet(nn.Module):
    """Strictly RC-invariant CNN using equivariant feature pairs."""

    def __init__(self, config: dict):
        super().__init__()
        widths = [int(config["conv_channels"])]
        kernels = [int(config["kernel_size"])]
        if bool(config.get("second_conv", False)):
            widths.append(int(config["second_conv_channels"]))
            kernels.append(int(config["second_kernel_size"]))
        if any(width % 2 for width in widths):
            raise ValueError("Every RCPS convolution width must be even")

        layers: list[nn.Module] = []
        in_channels = 4
        input_permutation = DNA_RC_CHANNELS
        for width, kernel in zip(widths, kernels):
            layers += [RCPSConv1d(in_channels, width, kernel, input_permutation), nn.ReLU()]
            half = width // 2
            input_permutation = tuple(range(half, width)) + tuple(range(half))
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.pair_classifier = nn.Linear(widths[-1] // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.features(x)).squeeze(-1)
        half = pooled.shape[1] // 2
        invariant = 0.5 * (pooled[:, :half] + pooled[:, half:])
        return self.pair_classifier(invariant).squeeze(-1)


class DNASequenceTransformerBlock(nn.Module):
    """Small pre-norm self-attention block with optional signed relative bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
        n_patches: int,
        relative_position: bool,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("Transformer d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, d_model)
        self.projection_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )
        if relative_position:
            self.relative_position_bias = nn.Parameter(
                torch.zeros(n_heads, 2 * n_patches - 1)
            )
            coordinates = torch.arange(n_patches)
            relative_index = coordinates[:, None] - coordinates[None, :] + n_patches - 1
            self.register_buffer("relative_position_index", relative_index, persistent=False)
        else:
            self.relative_position_bias = None
            self.register_buffer("relative_position_index", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        normalized = self.norm1(x)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.relative_position_bias is not None:
            bias = self.relative_position_bias[:, self.relative_position_index]
            scores = scores + bias.unsqueeze(0)
        attention = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.projection_dropout(self.projection(attended))
        return x + self.feedforward(self.norm2(x))


class DNASequenceTransformer(nn.Module):
    """CPU-sized DNA Transformer with explicit positional-encoding ablations."""

    def __init__(self, config: dict, position_encoding: str):
        super().__init__()
        if position_encoding not in {"none", "absolute", "relative"}:
            raise ValueError(f"Unknown position encoding: {position_encoding}")
        sequence_length = int(config["sequence_length"])
        patch_size = int(config["patch_size"])
        if sequence_length % patch_size:
            raise ValueError("sequence_length must be divisible by patch_size")
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.n_patches = sequence_length // patch_size
        self.position_encoding = position_encoding
        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        self.patch_embedding = nn.Conv1d(
            4, d_model, kernel_size=patch_size, stride=patch_size
        )
        if position_encoding == "absolute":
            self.absolute_position = nn.Parameter(
                torch.zeros(1, self.n_patches, d_model)
            )
        else:
            self.register_parameter("absolute_position", None)
        self.blocks = nn.ModuleList(
            [
                DNASequenceTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    feedforward_dim=int(config["feedforward_dim"]),
                    dropout=float(config.get("dropout", 0.0)),
                    n_patches=self.n_patches,
                    relative_position=position_encoding == "relative",
                )
                for _ in range(int(config["num_layers"]))
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 4 or x.shape[2] != self.sequence_length:
            raise ValueError(
                f"Expected input shape (batch, 4, {self.sequence_length}), got {tuple(x.shape)}"
            )
        tokens = self.patch_embedding(x).transpose(1, 2)
        if self.absolute_position is not None:
            tokens = tokens + self.absolute_position
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.final_norm(tokens).mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(config: dict, architecture: str = "standard") -> nn.Module:
    if architecture == "standard":
        return DNAConvNet(config)
    if architecture == "rcps":
        return RCPSDNAConvNet(config)
    if architecture.startswith("transformer_"):
        position_encoding = architecture.removeprefix("transformer_")
        return DNASequenceTransformer(config, position_encoding=position_encoding)
    raise ValueError(f"Unknown architecture: {architecture}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
