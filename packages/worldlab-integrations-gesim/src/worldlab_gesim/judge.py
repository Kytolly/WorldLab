"""Local VLM World Judge abstraction and Qwen2.5-VL implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .base import RewardClient, RewardResult


class WorldJudge(RewardClient):
    """Qwen2.5-VL-3B-Instruct skeleton for a trained Judge head."""
    def __init__(
        self,
        model_path: str | Path = "model_zoo/qwen/Qwen2.5-VL-3B-Instruct",
        *,
        success_head_path: str | Path | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        frame_token: str = "<|world_judge_frame|>",
        head_hidden_dim: int = 1024,
        load_in_constructor: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.success_head_path = (
            Path(success_head_path) if success_head_path is not None else None
        )
        self.device = device
        self.dtype = dtype
        self.frame_token = frame_token
        self.head_hidden_dim = head_hidden_dim
        self._model: Any = None
        self._processor: Any = None
        self._success_head: Any = None
        if load_in_constructor:
            self.load()

    def load(self) -> None:
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Qwen model directory does not exist: {self.model_path}. "
                "Download Qwen/Qwen2.5-VL-3B-Instruct into model_zoo first."
            )
        if self.success_head_path is None or not self.success_head_path.is_file():
            raise FileNotFoundError(
                "a trained World Judge success head is required; pass success_head_path"
            )
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise ImportError(
                "install local Judge dependencies with "
                '"worldlab-integrations-gesim[judge]"'
            ) from error
        torch_dtype = getattr(torch, self.dtype, None)
        if torch_dtype is None:
            raise ValueError(f"unsupported torch dtype: {self.dtype}")
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "output_hidden_states": True,
        }
        if self.device is not None:
            kwargs["device_map"] = self.device
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_path), **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(str(self.model_path))
        added_tokens = self._processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.frame_token]}
        )
        if added_tokens:
            self._model.resize_token_embeddings(len(self._processor.tokenizer))
        hidden_size = int(self._model.config.hidden_size)
        self._success_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.head_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.head_hidden_dim, 1),
        )
        if self.success_head_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(self.success_head_path), device="cpu")
        else:
            state = torch.load(self.success_head_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._success_head.load_state_dict(state)
        model_device = next(self._model.parameters()).device
        self._success_head.to(device=model_device, dtype=torch_dtype)
        self._success_head.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        for parameter in self._success_head.parameters():
            parameter.requires_grad_(False)

    def evaluate(self, head_frames: np.ndarray, task: str) -> RewardResult:
        frames = _validate_head_frames(head_frames)
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if self._model is None or self._processor is None:
            self.load()
        logits = self._predict_logits(frames, task)
        success = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        return RewardResult(success=success, progress=None)
    
    def _predict_logits(self, frames: np.ndarray, task: str) -> np.ndarray:
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise ImportError(
                "install local Judge dependencies with "
                '"worldlab-integrations-gesim[judge]"'
            ) from error
        logits: list[float] = []
        for frame in frames:
            image = Image.fromarray(frame, mode="RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{task}\n{self.frame_token}"},
            ]}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self._processor(text=[text], images=[image], return_tensors="pt")
            if hasattr(self._model, "device"):
                inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self._model(**inputs, output_hidden_states=True)
            hidden = output.hidden_states[-1]
            token_id = self._processor.tokenizer.convert_tokens_to_ids(self.frame_token)
            positions = (inputs["input_ids"] == token_id).nonzero(as_tuple=False)
            if positions.shape[0] != 1:
                raise RuntimeError(
                    f"expected one frame token {self.frame_token!r}, found {positions.shape[0]}"
                )
            frame_hidden = hidden[0, int(positions[0, 1])]
            logits.append(float(self._success_head(frame_hidden.unsqueeze(0)).squeeze().item()))
        return np.asarray(logits, dtype=np.float32)


def _validate_head_frames(value: np.ndarray) -> np.ndarray:
    frames = np.asarray(value)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"head_frames must have shape (T, H, W, 3), got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"head_frames must be uint8, got {frames.dtype}")
    if frames.shape[0] == 0:
        raise ValueError("head_frames must not be empty")
    return frames.copy()


# Descriptive name used by the integration API; ``WorldJudge`` remains the
# concise author-facing name used in the GE-Sim paper and examples.
Qwen25VLWorldJudge = WorldJudge
