"""Validated training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


DataProfile = Literal["fast", "balanced", "extensive", "full"]
DATA_PROFILES: dict[DataProfile, dict[str, Any]] = {
    "fast": {
        "target_purchases": 25_000,
        "label": "Fast",
        "description": "Small deterministic sample for quick experiments.",
    },
    "balanced": {
        "target_purchases": 100_000,
        "label": "Balanced",
        "description": "Default sample balancing runtime and coverage.",
    },
    "extensive": {
        "target_purchases": 250_000,
        "label": "Extensive",
        "description": "Larger sample for broader user and item coverage.",
    },
    "full": {
        "target_purchases": None,
        "label": "Full",
        "description": "All source purchases without sampling.",
    },
}


@dataclass(frozen=True)
class TrainingConfig:
    data_profile: DataProfile = "balanced"
    embedding_dim: int = 32
    dense_1: int = 64
    dense_2: int = 32
    batch_size: int = 4096
    epochs: int = 3
    negative_ratio: int = 2
    learning_rate: float = 0.001
    shuffle_buffer: int = 100_000
    evaluation_negatives: int = 100
    evaluation_batch_size: int = 8192
    random_seed: int = 42
    probe_customer_key: int | None = None
    probe_article_key: int | None = None

    def __post_init__(self) -> None:
        if self.data_profile not in DATA_PROFILES:
            raise ValueError(f"data_profile must be one of {list(DATA_PROFILES)}")
        positive_ints = {
            "embedding_dim": self.embedding_dim,
            "dense_1": self.dense_1,
            "dense_2": self.dense_2,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "shuffle_buffer": self.shuffle_buffer,
            "evaluation_batch_size": self.evaluation_batch_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.negative_ratio < 0:
            raise ValueError("negative_ratio cannot be negative")
        if self.evaluation_negatives < 10:
            raise ValueError("evaluation_negatives must be at least 10")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (self.probe_customer_key is None) != (self.probe_article_key is None):
            raise ValueError("probe_customer_key and probe_article_key must be set together")
        if self.probe_customer_key is not None and min(
            self.probe_customer_key, self.probe_article_key  # type: ignore[arg-type]
        ) < 0:
            raise ValueError("Probe IDs cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})

    @classmethod
    def from_settings(cls) -> "TrainingConfig":
        """Read project settings lazily so pure unit tests need no app config."""
        from app.config import get_settings

        defaults = cls()
        settings = get_settings()
        values: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            values[field_name] = getattr(
                settings, field_name.upper(), getattr(settings, field_name, getattr(defaults, field_name))
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
