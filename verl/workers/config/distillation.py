# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from dataclasses_json import Undefined, dataclass_json

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.trainer.distillation.losses import DISTILL_LOSS_MAP

from .engine import FSDPEngineConfig
from .optimizer import FSDPOptimizerConfig

__all__ = [
    "DistillationConfig",
    "DistillStudentConfig",
    "DistillTeacherConfig",
]


@dataclass
class DistillStudentConfig(BaseConfig):
    """Student-side configuration for distillation training.

    This mirrors the style of `ActorConfig`/`CriticConfig` with FSDP engine and optimizer blocks.

    Args:
        model_path: HF model path or local checkpoint for the student model.
        dtype: Trainer precision string, e.g. "bf16", "f16", or "32-true".
        fsdp_config: FSDP engine configuration for wrapping the student model.
        optim_config: Optimizer configuration for student updates.
        checkpoint_config: Checkpoint saving/loading configuration for the student model.
    """

    model_path: str = ""
    dtype: str = "bf16"
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    optim_config: FSDPOptimizerConfig = field(default_factory=FSDPOptimizerConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        if not self.model_path:
            raise ValueError("student.model_path must be specified")
        if self.dtype not in {"bf16", "f16", "16-mixed", "bf16-mixed", "32-true"}:
            # Keep permissive; DistillationWorker maps this to torch dtype.
            raise ValueError(
                f"Unsupported student dtype '{self.dtype}'. Expected one of bf16/f16/16-mixed/bf16-mixed/32-true."
            )


@dataclass
class DistillTeacherConfig(BaseConfig):
    """Teacher-side configuration for distillation training.

    The `engine` block follows the rollout engine style (e.g., SGLang/vLLM),
    similar to `trainer/config/rollout/rollout.yaml`.

    Args:
        model_path: HF model path or local checkpoint for the teacher model.
        engine: Inference engine config (e.g., SGLang) used for sampling/generation.
    """

    model_path: str = ""
    engine: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model_path:
            raise ValueError("teacher.model_path must be specified")
        if not isinstance(self.engine, dict):
            raise TypeError("teacher.engine must be a dict-like config")


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DistillationConfig(BaseConfig):
    """Top-level distillation hyperparameters and options.

    This config complements the student/teacher blocks. It encapsulates
    loss selection, temperature, and sampling strategy knobs that are consumed
    by the distillation trainer/model.

    Note: Validation ensures values are within reasonable ranges, similar to PPO configs.
    """

    enable_distill: bool = field(default=False)
    teacher_model_path: str = field(default="")
    temperature: float = field(default=1.0)
    distillation_loss_ratio: float = field(default=0.9)
    distill_loss: str = field(default="forward_kl")
    compile_distill_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to `torch.compile` the per-token distillation loss. This is especially "
            "helpful in reducing GPU memory footprint. Default to False."
        },
    )
    forward_ratio: float = field(
        default=0.5,
        metadata={"help": "Forward loss ratio for combined KL. Ignored when using other divergence metrics."},
    )

    # [begin] Divergence loss specific configurations
    js_beta: float = field(
        default=0.5,
        metadata={"help": "Beta for Jensen-Shannon divergence. Ignored when using other divergence metrics."},
    )
    skl_alpha: float = field(
        default=0.1,
        metadata={"help": "Alpha in Skew KL divergence. Ignored when using other divergence metrics."},
    )
    tvd_log_scale: bool = field(
        default=False,
        metadata={"help": "Whether to use log scale for the input to the TVD distillation loss."},
    )
    # [end] Divergence loss specific configurations

    val_include_distill_loss: bool = field(default=False)

    # [begin] sampling and generation configs
    sample_method: Literal["supervised", "on-policy", "sequence-level"] = field(default="supervised")
    sample_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of batches whose responses are sampled from student (on-policy) distribution "
            "or teacher (sequence-evel)  distribution rather than using the original responses, "
            "same as the huggingface GKD trainer (parameter self.lmbda)."
            " https://huggingface.co/docs/trl/gkd_trainer#trl.GKDConfig "
            "e.g., 0.4 means 40% of batches are using the responses sampled "
            "from student/teacher model, with 60% using original data "
            "Ignored when using supervised methods (ground-truth tokens)."
        },
    )
    max_new_tokens: int = field(
        default=100,
        metadata={
            "help": "Maximum number of tokens to generate for each response "
            "during on-policy or sequence-level sampling."
        },
    )
    sample_temperature: float = field(
        default=0.8,
        metadata={
            "help": "Sample temperature used for on-policy or sequence-level response token generation. "
            "The higher the temperature, the more random the completions."
        },
    )
    # [end] sampling and generation configs

    include_prompt_loss: bool = field(
        default=False,
        metadata={"help": "Whether to include prompt token loss in the distillation loss."},
    )

    exclude_cot_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to exclude the chain-of-thought (CoT) token loss from the distillation loss. "
            "This is useful when the CoT loss is not applicable or desired in the distillation process."
        },
    )

    eot_token: Optional[int] = field(
        default=None,
        metadata={"help": "End-of-think token used to exclude cot tokens during distillation loss calculation."},
    )

    # misc options used by DistillationWorker._build_model_optimizer
    use_liger: bool = field(default=False)

    def __post_init__(self):
        if self.enable_distill:
            assert self.teacher_model_path, "Teacher model path is required for distillation training."

            assert self.distill_loss in DISTILL_LOSS_MAP, (
                f"Only {', '.join(list(DISTILL_LOSS_MAP.keys()))} are supported for distillation training."
            )
            assert 0.0 <= self.distillation_loss_ratio <= 1.0, "Distillation loss ratio for KL should be in [0, 1]."
            assert self.temperature > 0.0, "Temperature should be positive."

            if self.distill_loss == "combined_kl":
                assert 0.0 < self.forward_ratio < 1.0, "Forward loss ratio for combined KL should be in (0, 1)."
            if self.distill_loss == "js":
                assert 0.0 <= self.js_beta <= 1.0, "Beta for Jensen-Shannon divergence should be in [0, 1]."

            assert self.sample_method in {
                "supervised",
                "on-policy",
                "sequence-level",
            }, f"Unsupported sample method: {self.sample_method}"

            assert 0.0 <= self.sample_fraction <= 1.0, (
                f"Sample ratio should be in [0, 1], got sample_fraction={self.sample_fraction}."
            )
            assert self.max_new_tokens > 0, f"max_new_tokens to generate should be positive, got {self.max_new_tokens}."
            assert self.sample_temperature > 0.0, (
                f"Sample temperature should be positive, got sample_temperature={self.sample_temperature}."
            )

            if self.exclude_cot_loss:
                assert self.eot_token is not None, (
                    "When excluding CoT loss, eot_token must be specified to identify the end of CoT tokens."
                )

    @property
    def distill_loss_kwargs(self):
        return {
            "temperature": self.temperature,
            "distillation_loss_ratio": self.distillation_loss_ratio,
            "forward_ratio": self.forward_ratio,
            "beta": self.js_beta,
            "alpha": self.skl_alpha,
            "log_scale": self.tvd_log_scale,
        }
