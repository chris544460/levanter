# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Script to import HuggingFace models and save them as Levanter Tensorstore checkpoints.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp
import levanter
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import RayConfig, TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


@dataclass
class ImportHfConfig:
    """Configuration for importing HuggingFace models to Levanter checkpoints."""

    # Source HF checkpoint
    hf_checkpoint: Union[str, RepoRef]
    """HF repository reference or path (e.g., 'meta-llama/Llama-2-7b-hf' or 'gs://bucket/path')"""

    # Target Levanter checkpoint
    output_path: str
    """Output path for Levanter checkpoint (local or cloud, e.g., 'gs://bucket/checkpoints/model')"""

    # Model configuration
    model: LmConfig = field(default_factory=LlamaConfig)
    """Levanter model configuration"""

    use_hf_model_config: bool = True
    """If True, use the model configuration from the HF checkpoint instead of the provided model config"""

    # Processing options
    tokenizer: Optional[str] = None
    """Override tokenizer path/name. If None, will use tokenizer from HF checkpoint"""

    dtype: Optional[str] = "bfloat16"
    """Target dtype for the saved checkpoint (e.g., 'float32', 'bfloat16', 'float16')"""

    resize_vocab_to_match_tokenizer: bool = True
    """If True, resize model vocab to match tokenizer vocab size"""

    # Device/mesh configuration
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            wandb=WandbConfig(mode="disabled"), ray=RayConfig(auto_start_cluster=False)
        )
    )
    """Trainer configuration for device mesh and axis mappings"""


def _coerce_to_repo_ref(checkpoint: Union[str, RepoRef]) -> RepoRef:
    """Convert string or RepoRef to RepoRef."""
    if isinstance(checkpoint, str):
        return RepoRef.from_string(checkpoint)
    return checkpoint


def main(config: ImportHfConfig):
    """Main function to import HF model and save as Levanter checkpoint."""
    logger.setLevel(logging.INFO)

    start_time = time.time()

    logger.info("Starting HF to Levanter conversion")
    logger.info(f"Source: {config.hf_checkpoint}")
    logger.info(f"Target: {config.output_path}")
    logger.info(f"Model type: {config.model.__class__.__name__}")
    logger.info(f"Use HF config: {config.use_hf_model_config}")
    logger.info(f"Target dtype: {config.dtype}")

    # Initialize Levanter
    logger.info("Initializing Levanter")
    levanter.initialize(config)

    # Coerce checkpoint to RepoRef
    hf_checkpoint = _coerce_to_repo_ref(config.hf_checkpoint)

    tokenizer_path = config.tokenizer
    if tokenizer_path is None:
        tokenizer_path = hf_checkpoint.model_name_or_path

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    # Set up device mesh and axis mappings for model operations
    logger.info("Setting up device mesh and axis mappings")
    # Get HF checkpoint converter
    logger.info("Setting up HF checkpoint converter")
    converter = config.model.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)

    logger.info("Loading HF model...")
    with use_cpu_device():
        model = converter.load_pretrained(
            config.model.model_type,
            config=config.model if not config.use_hf_model_config else None,
            axis_mapping=config.trainer.parameter_axis_mapping,
            dtype=getattr(jnp, config.dtype) if config.dtype else None,
            resize_vocab_to_match_tokenizer=config.resize_vocab_to_match_tokenizer,
        )

    # Ensure output directory exists for local paths
    if not config.output_path.startswith(("gs://", "s3://", "az://", "http://", "https://")):
        # Local path - create directory
        os.makedirs(config.output_path, exist_ok=True)
        logger.info(f"Created output directory: {config.output_path}")

    logger.info(f"Saving checkpoint to Tensorstore format: {config.output_path}")

    # Save as Tensorstore checkpoint
    manager = GlobalAsyncCheckpointManager()

    def commit_callback():
        elapsed = time.time() - start_time
        logger.info(f"Checkpoint committed to Tensorstore successfully! Total time: {elapsed:.2f}s")

    save_checkpoint(
        tree=model,
        checkpoint_path=config.output_path,
        manager=manager,
        commit_callback=commit_callback,
        step=0,
    )
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    levanter.config.main(main)()
