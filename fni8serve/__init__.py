# SPDX-License-Identifier: MIT
"""fni8-serve — W8A8 dp4a inference server for Volta / CMP 100-210, over `fni8`."""

from .batch import BatchRequest, BatchResult, generate_batch
from .config import ServeConfig
from .convert import convert_hf_to_fni8, quantize_state_dict
from .engine import LLMEngine, SamplingParams
from .loader import checkpoint_info, load_fni8_checkpoint, load_fni8_state_dict
from .models import (
    ModelConfig,
    ModelRunner,
    build_model,
    is_supported,
    list_models,
)
from .scheduling import (
    ParallelismStrategy,
    estimate_throughput,
    plan_parallelism,
    validate_parallel_config,
)

__all__ = [
    "ServeConfig",
    "load_fni8_checkpoint",
    "load_fni8_state_dict",
    "checkpoint_info",
    "convert_hf_to_fni8",
    "quantize_state_dict",
    "LLMEngine",
    "SamplingParams",
    "ModelConfig",
    "ModelRunner",
    "build_model",
    "is_supported",
    "list_models",
    "BatchRequest",
    "BatchResult",
    "generate_batch",
    "ParallelismStrategy",
    "plan_parallelism",
    "estimate_throughput",
    "validate_parallel_config",
]
__version__ = "0.0.1"
