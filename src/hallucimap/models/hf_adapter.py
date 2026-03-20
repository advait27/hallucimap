"""HuggingFace adapter for hallucimap (local model inference).

Loads a HuggingFace causal language model locally and runs inference in
a thread pool to keep the async interface non-blocking.

Examples
--------
>>> import asyncio
>>> from hallucimap.models.hf_adapter import HFAdapter
>>>
>>> adapter = HFAdapter(model="meta-llama/Llama-3-8B-Instruct", device="cuda")
>>> text = asyncio.run(adapter.complete("What is the capital of France?"))
>>> print(text)
"""

from __future__ import annotations

import asyncio
from functools import partial

from hallucimap.models.base import BaseLLMAdapter, CompletionRequest, CompletionResponse

# Imports are deferred to _ensure_loaded() so that importing this module
# does not fail when torch/transformers are not installed.
_TRANSFORMERS_AVAILABLE: bool | None = None


class HFAdapter(BaseLLMAdapter):
    """Adapter for local HuggingFace causal LMs.

    Model is loaded lazily on first call to avoid import-time GPU allocation.

    Parameters
    ----------
    model : str
        HuggingFace model name or local path.
    device : str
        PyTorch device string: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    hf_token : str | None
        HuggingFace access token for private/gated models.
        Falls back to ``HF_TOKEN`` environment variable.
    load_in_8bit : bool
        Enable bitsandbytes 8-bit quantization (requires ``bitsandbytes``).

    Examples
    --------
    >>> adapter = HFAdapter(model="gpt2")  # small model for testing
    >>> text = asyncio.run(adapter.complete("Once upon a time"))
    """

    def __init__(
        self,
        model: str = "gpt2",
        device: str = "cpu",
        hf_token: str | None = None,
        load_in_8bit: bool = False,
    ) -> None:
        import os

        super().__init__(model_id=model)
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.load_in_8bit = load_in_8bit
        self._pipeline: object | None = None  # lazy

    def _ensure_loaded(self) -> None:
        """Load the model pipeline on first call."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for HFAdapter: "
                "pip install transformers torch"
            ) from e

        # TODO Phase 2: support chat-template formatting for instruct models
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
        )
        model_kwargs: dict = {}
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        lm = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=self.hf_token,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
            **model_kwargs,
        )
        self._pipeline = pipeline(
            "text-generation",
            model=lm,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def _sync_complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Blocking completion — called in a thread pool."""
        self._ensure_loaded()
        assert self._pipeline is not None
        outputs = self._pipeline(  # type: ignore[operator]
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,  # type: ignore[union-attr]
        )
        generated: str = outputs[0]["generated_text"]  # type: ignore[index]
        # Strip the prompt prefix
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a completion using a local HuggingFace model.

        Parameters
        ----------
        prompt : str
            Input prompt.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum new tokens.
        system_prompt : str | None
            If provided, prepended to the prompt as ``[SYSTEM] ... [/SYSTEM]``.

        Returns
        -------
        str
            Generated text.

        Examples
        --------
        >>> adapter = HFAdapter(model="gpt2")
        >>> text = asyncio.run(adapter.complete("The quick brown fox"))
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"[SYSTEM] {system_prompt} [/SYSTEM]\n{prompt}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._sync_complete, full_prompt, temperature, max_tokens),
        )

    async def complete_structured(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a structured completion.

        Parameters
        ----------
        request : CompletionRequest
            Fully specified request.

        Returns
        -------
        CompletionResponse
            Response (token counts unavailable for local models).

        Examples
        --------
        >>> req = CompletionRequest(prompt="Hello world")
        >>> resp = asyncio.run(adapter.complete_structured(req))
        """
        text = await self.complete(
            request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
        )
        return CompletionResponse(
            text=text,
            model_id=self.model_id,
            finish_reason="stop",
        )
