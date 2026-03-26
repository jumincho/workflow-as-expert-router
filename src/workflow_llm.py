"""Workflow-level LLM wrappers used by WaE-Router pilot."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from openai import AsyncOpenAI, BadRequestError, OpenAI

from MAR.LLM.llm import LLM
from MAR.LLM.price import cost_count

from .safe_exec import run_python_tests


Message = Dict[str, str]


_RUNTIME_TELEMETRY: Dict[str, int] = {
    "endpoint_calls": 0,
    "overflow_retries": 0,
    "context_trim_retries": 0,
}


def _bump_runtime_counter(key: str, delta: int = 1) -> None:
    _RUNTIME_TELEMETRY[key] = int(_RUNTIME_TELEMETRY.get(key, 0)) + int(delta)


def reset_runtime_telemetry() -> None:
    _RUNTIME_TELEMETRY["endpoint_calls"] = 0
    _RUNTIME_TELEMETRY["overflow_retries"] = 0
    _RUNTIME_TELEMETRY["context_trim_retries"] = 0


def snapshot_runtime_telemetry() -> Dict[str, int]:
    return {
        "endpoint_calls": int(_RUNTIME_TELEMETRY.get("endpoint_calls", 0)),
        "overflow_retries": int(_RUNTIME_TELEMETRY.get("overflow_retries", 0)),
        "context_trim_retries": int(_RUNTIME_TELEMETRY.get("context_trim_retries", 0)),
    }


def _normalize_messages(messages: Union[List[Message], str]) -> List[Message]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def _short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _messages_hash(messages: List[Message]) -> str:
    return _short_hash(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def _messages_to_text(messages: List[Message]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


def extract_python_code(text: str) -> str:
    match = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_inline_tests(text: str) -> List[str]:
    """Extract MBPP-style inline test lines from a prompt body."""
    tests, _, _ = extract_inline_tests_with_meta(text)
    return tests


def extract_inline_tests_with_meta(text: str) -> Tuple[List[str], str, str]:
    """Extract inline tests and return (tests, source, parse_error)."""
    parse_error = ""
    match = re.search(
        r"Your code should pass these tests:\s*```python(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        body = match.group(1).strip()
        lines = [x.strip() for x in body.splitlines() if x.strip()]
        if lines:
            return lines, "dataset_provided", ""
        return [], "dataset_provided", "empty_python_test_block"

    # Secondary path: any fenced python block that contains assert lines.
    blocks = re.findall(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        assert_lines = [x for x in lines if x.startswith("assert ")]
        if assert_lines:
            return assert_lines, "extracted_fenced", ""

    # Last fallback: inline assert extraction.
    assert_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("assert "):
            assert_lines.append(line)
    if assert_lines:
        return assert_lines, "extracted_inline_assert", ""

    parse_error = "no_assert_or_test_block_found"
    return [], "unavailable", parse_error


def is_python_syntax_valid(text: str) -> bool:
    code = extract_python_code(text)
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


@dataclass
class EndpointSpec:
    name: str
    model_id: str
    base_url: str
    api_key: str
    timeout: float = 120.0
    price_name: Optional[str] = None


class EndpointLLM(LLM):
    """OpenAI-compatible client for local vLLM endpoints."""

    def __init__(self, spec: EndpointSpec):
        self.spec = spec
        self.model_name = spec.name
        self.client = OpenAI(
            base_url=spec.base_url,
            api_key=spec.api_key,
            timeout=spec.timeout,
        )
        self.async_client = AsyncOpenAI(
            base_url=spec.base_url,
            api_key=spec.api_key,
            timeout=spec.timeout,
        )
        seed_raw = os.environ.get("WAE_REQUEST_SEED", "").strip()
        self.default_seed: Optional[int] = int(seed_raw) if seed_raw else None
        self.default_temperature = float(
            os.environ.get("WAE_DEFAULT_TEMPERATURE", str(self.DEFAULT_TEMPERATURE))
        )
        self.default_top_p = float(os.environ.get("WAE_DEFAULT_TOP_P", "1.0"))
        self._last_retry_meta: Dict[str, Union[str, int]] = {
            "retries": 0,
            "overflow_retries": 0,
            "context_trim_retries": 0,
            "initial_max_tokens": 0,
            "final_max_tokens": 0,
            "last_error": "",
        }
        self.last_call_meta: Dict[str, Union[str, int, float]] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries": 0,
            "overflow_retries": 0,
            "context_trim_retries": 0,
            "initial_max_tokens": 0,
            "final_max_tokens": 0,
            "temperature": float(self.default_temperature),
            "top_p": float(self.default_top_p),
            "prompt_hash": "",
            "output_hash": "",
            "endpoint_name": self.spec.name,
            "model_id": self.spec.model_id,
        }

    @staticmethod
    def _next_max_tokens_on_context_error(error_text: str, current_max_tokens: int) -> Optional[int]:
        # Example:
        # "... model's maximum context length is 3072 tokens and your request has 2472 input tokens ..."
        ctx_match = re.search(
            r"maximum context length is\s*(\d+)\s*tokens.*?request has\s*(\d+)\s*input tokens",
            error_text,
            flags=re.IGNORECASE,
        )
        if ctx_match:
            max_ctx = int(ctx_match.group(1))
            input_tokens = int(ctx_match.group(2))
            budget = max_ctx - input_tokens - 32
            if budget < 32:
                return None
            return min(current_max_tokens - 1, budget)

        # Fallback when the server does not provide token counts.
        if current_max_tokens > 256:
            return max(128, current_max_tokens // 2)
        if current_max_tokens > 64:
            return 64
        return None

    @staticmethod
    def _truncate_messages_on_context_error(
        messages: List[Message], error_text: str
    ) -> Optional[List[Message]]:
        ctx_match = re.search(
            r"maximum context length is\s*(\d+)\s*tokens.*?request has\s*(\d+)\s*input tokens",
            error_text,
            flags=re.IGNORECASE,
        )
        if not ctx_match:
            return None
        max_ctx = int(ctx_match.group(1))
        input_tokens = int(ctx_match.group(2))
        if input_tokens <= max_ctx:
            return None

        trimmed = [dict(m) for m in messages]
        marker = "Your code should pass these tests:"
        for i in range(len(trimmed) - 1, -1, -1):
            content = str(trimmed[i].get("content", ""))
            if marker in content:
                head = content.split(marker, 1)[0].rstrip()
                trimmed[i]["content"] = head + "\n\n[Tests omitted due context limit]\n"
                return trimmed

        if not trimmed:
            return None
        idx = max(range(len(trimmed)), key=lambda j: len(str(trimmed[j].get("content", ""))))
        content = str(trimmed[idx].get("content", ""))
        if len(content) < 256:
            return None
        keep = max(192, int(len(content) * 0.7))
        trimmed[idx]["content"] = content[:keep].rstrip() + "\n\n[Truncated due context limit]\n"
        return trimmed

    def _chat_create_with_retry(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_comps: int,
    ):
        cur_messages = [dict(m) for m in messages]
        cur_max_tokens = max_tokens
        retries = 0
        overflow_retries = 0
        context_trim_retries = 0
        last_error = ""
        for _ in range(6):
            req = {
                "model": self.spec.model_id,
                "messages": cur_messages,
                "max_tokens": cur_max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": num_comps,
            }
            if self.default_seed is not None:
                req["seed"] = self.default_seed
            try:
                response = self.client.chat.completions.create(**req)
                self._last_retry_meta = {
                    "retries": int(retries),
                    "overflow_retries": int(overflow_retries),
                    "context_trim_retries": int(context_trim_retries),
                    "initial_max_tokens": int(max_tokens),
                    "final_max_tokens": int(cur_max_tokens),
                    "last_error": str(last_error),
                }
                return response
            except BadRequestError as e:
                err = str(e)
                last_error = err
                next_max = self._next_max_tokens_on_context_error(err, cur_max_tokens)
                if next_max is not None and next_max < cur_max_tokens:
                    retries += 1
                    overflow_retries += 1
                    _bump_runtime_counter("overflow_retries", 1)
                    cur_max_tokens = next_max
                    continue
                next_messages = self._truncate_messages_on_context_error(cur_messages, err)
                if next_messages is not None and next_messages != cur_messages:
                    retries += 1
                    context_trim_retries += 1
                    _bump_runtime_counter("context_trim_retries", 1)
                    cur_messages = next_messages
                    continue
                raise
        raise RuntimeError("retry loop exhausted")

    async def _achat_create_with_retry(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_comps: int,
    ):
        cur_messages = [dict(m) for m in messages]
        cur_max_tokens = max_tokens
        retries = 0
        overflow_retries = 0
        context_trim_retries = 0
        last_error = ""
        for _ in range(6):
            req = {
                "model": self.spec.model_id,
                "messages": cur_messages,
                "max_tokens": cur_max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": num_comps,
            }
            if self.default_seed is not None:
                req["seed"] = self.default_seed
            try:
                response = await self.async_client.chat.completions.create(**req)
                self._last_retry_meta = {
                    "retries": int(retries),
                    "overflow_retries": int(overflow_retries),
                    "context_trim_retries": int(context_trim_retries),
                    "initial_max_tokens": int(max_tokens),
                    "final_max_tokens": int(cur_max_tokens),
                    "last_error": str(last_error),
                }
                return response
            except BadRequestError as e:
                err = str(e)
                last_error = err
                next_max = self._next_max_tokens_on_context_error(err, cur_max_tokens)
                if next_max is not None and next_max < cur_max_tokens:
                    retries += 1
                    overflow_retries += 1
                    _bump_runtime_counter("overflow_retries", 1)
                    cur_max_tokens = next_max
                    continue
                next_messages = self._truncate_messages_on_context_error(cur_messages, err)
                if next_messages is not None and next_messages != cur_messages:
                    retries += 1
                    context_trim_retries += 1
                    _bump_runtime_counter("context_trim_retries", 1)
                    cur_messages = next_messages
                    continue
                raise
        raise RuntimeError("async retry loop exhausted")

    def gen(
        self,
        messages: Union[List[Message], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        messages = _normalize_messages(messages)
        if max_tokens is None:
            max_tokens = 1024
        if temperature is None:
            temperature = self.default_temperature
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        response = self._chat_create_with_retry(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.default_top_p,
            num_comps=num_comps,
        )
        texts = [choice.message.content or "" for choice in response.choices]
        prompt = "".join(m["content"] for m in messages)
        prompt_hash = _messages_hash(messages)
        first_output_hash = _short_hash(texts[0] if texts else "")
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        _bump_runtime_counter("endpoint_calls", 1)
        self.last_call_meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "retries": int(self._last_retry_meta.get("retries", 0)),
            "overflow_retries": int(self._last_retry_meta.get("overflow_retries", 0)),
            "context_trim_retries": int(
                self._last_retry_meta.get("context_trim_retries", 0)
            ),
            "initial_max_tokens": int(self._last_retry_meta.get("initial_max_tokens", max_tokens)),
            "final_max_tokens": int(self._last_retry_meta.get("final_max_tokens", max_tokens)),
            "temperature": float(temperature),
            "top_p": float(self.default_top_p),
            "prompt_hash": prompt_hash,
            "output_hash": first_output_hash,
            "endpoint_name": self.spec.name,
            "model_id": self.spec.model_id,
        }
        price_name = self.spec.price_name or self.spec.name
        for text in texts:
            cost_count(prompt, text, price_name)
        if num_comps == 1:
            return texts[0]
        return texts

    async def agen(
        self,
        messages: Union[List[Message], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        messages = _normalize_messages(messages)
        if max_tokens is None:
            max_tokens = 1024
        if temperature is None:
            temperature = self.default_temperature
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        response = await self._achat_create_with_retry(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.default_top_p,
            num_comps=num_comps,
        )
        texts = [choice.message.content or "" for choice in response.choices]
        prompt = "".join(m["content"] for m in messages)
        prompt_hash = _messages_hash(messages)
        first_output_hash = _short_hash(texts[0] if texts else "")
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        _bump_runtime_counter("endpoint_calls", 1)
        self.last_call_meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "retries": int(self._last_retry_meta.get("retries", 0)),
            "overflow_retries": int(self._last_retry_meta.get("overflow_retries", 0)),
            "context_trim_retries": int(
                self._last_retry_meta.get("context_trim_retries", 0)
            ),
            "initial_max_tokens": int(self._last_retry_meta.get("initial_max_tokens", max_tokens)),
            "final_max_tokens": int(self._last_retry_meta.get("final_max_tokens", max_tokens)),
            "temperature": float(temperature),
            "top_p": float(self.default_top_p),
            "prompt_hash": prompt_hash,
            "output_hash": first_output_hash,
            "endpoint_name": self.spec.name,
            "model_id": self.spec.model_id,
        }
        price_name = self.spec.price_name or self.spec.name
        for text in texts:
            cost_count(prompt, text, price_name)
        if num_comps == 1:
            return texts[0]
        return texts


class WorkflowLLM(LLM):
    """Wrap a workflow as a single LLM-like callable."""

    def __init__(
        self,
        workflow_id: str,
        workflow_def: Dict[str, object],
        endpoint_resolver: Callable[[str], EndpointLLM],
    ):
        self.workflow_id = workflow_id
        self.workflow_def = workflow_def
        self.model_name = f"wf::{workflow_id}"
        self._resolve_endpoint = endpoint_resolver
        self.call_count = 0
        self.last_latency_s = 0.0
        self._premium_debug_log = os.environ.get("WAE_PREMIUM_DEBUG_LOG", "").strip()
        self.last_trace: Dict[str, Union[str, int, float, List[Dict[str, object]]]] = {}

    @property
    def method(self) -> str:
        return str(self.workflow_def.get("method", "io"))

    @property
    def base_model(self) -> str:
        return str(self.workflow_def["base_model"])

    def _endpoint(self) -> EndpointLLM:
        return self._resolve_endpoint(self.base_model)

    def _as_text_prompt(self, messages: Union[List[Message], str]) -> List[Message]:
        msgs = _normalize_messages(messages)
        text = _messages_to_text(msgs)
        return [{"role": "user", "content": text}]

    def _pick_consistent_candidate(self, candidates: List[str]) -> str:
        for cand in candidates:
            if is_python_syntax_valid(cand):
                return cand
        return max(candidates, key=len) if candidates else ""

    def _reset_trace(self) -> None:
        self.last_trace = {
            "workflow_id": self.workflow_id,
            "base_model": self.base_model,
            "method": self.method,
            "endpoint_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "overflow_retries": 0,
            "context_trim_retries": 0,
            "retry_calls": 0,
            "prompt_hashes": [],
            "output_hashes": [],
            "endpoint_names": [],
            "model_ids": [],
            "invoked_workflow_ids": [self.model_name],
        }

    def _consume_endpoint_meta(self, endpoint: EndpointLLM) -> None:
        meta = endpoint.last_call_meta
        self.last_trace["endpoint_calls"] = int(self.last_trace["endpoint_calls"]) + 1
        self.last_trace["prompt_tokens"] = int(self.last_trace["prompt_tokens"]) + int(
            meta.get("prompt_tokens", 0)
        )
        self.last_trace["completion_tokens"] = int(
            self.last_trace["completion_tokens"]
        ) + int(meta.get("completion_tokens", 0))
        self.last_trace["overflow_retries"] = int(
            self.last_trace["overflow_retries"]
        ) + int(meta.get("overflow_retries", 0))
        self.last_trace["context_trim_retries"] = int(
            self.last_trace.get("context_trim_retries", 0)
        ) + int(meta.get("context_trim_retries", 0))
        if int(meta.get("retries", 0)) > 0:
            self.last_trace["retry_calls"] = int(self.last_trace["retry_calls"]) + 1
        prompt_hash = str(meta.get("prompt_hash", "") or "")
        output_hash = str(meta.get("output_hash", "") or "")
        endpoint_name = str(meta.get("endpoint_name", "") or "")
        model_id = str(meta.get("model_id", "") or "")
        if prompt_hash:
            self.last_trace["prompt_hashes"].append(prompt_hash)
        if output_hash:
            self.last_trace["output_hashes"].append(output_hash)
        if endpoint_name:
            self.last_trace["endpoint_names"].append(endpoint_name)
        if model_id:
            self.last_trace["model_ids"].append(model_id)

    def _call_endpoint(self, messages: Union[List[Message], str]) -> str:
        endpoint = self._endpoint()
        out = str(endpoint.gen(messages))
        self.call_count += 1
        self._consume_endpoint_meta(endpoint)
        return out

    def _run_io(self, messages: Union[List[Message], str]) -> str:
        return self._call_endpoint(messages)

    def _run_refine2(self, messages: Union[List[Message], str]) -> str:
        text_prompt = self._as_text_prompt(messages)
        draft = self._call_endpoint(text_prompt)
        refine_prompt = [
            {
                "role": "user",
                "content": (
                    "You wrote the following draft answer.\n\n"
                    f"{draft}\n\n"
                    "Revise it to improve correctness and robustness. "
                    "If code is required, return code in ```python``` blocks."
                ),
            }
        ]
        revised = self._call_endpoint(refine_prompt)
        return revised

    def _run_self_consistency3(self, messages: Union[List[Message], str]) -> str:
        text_prompt = self._as_text_prompt(messages)
        k = int(self.workflow_def.get("params", {}).get("k", 3))
        cands: List[str] = []
        for _ in range(k):
            out = self._call_endpoint(text_prompt)
            cands.append(out)
        return self._pick_consistent_candidate(cands)

    def _run_critique_refine(self, messages: Union[List[Message], str]) -> str:
        text_prompt = self._as_text_prompt(messages)
        draft = self._call_endpoint(text_prompt)
        critique = str(
            self._call_endpoint(
                [
                    {
                        "role": "user",
                        "content": (
                            "Critique the following answer. Focus on logic errors, "
                            "missing edge cases, and testability.\n\n"
                            f"{draft}"
                        ),
                    }
                ]
            )
        )
        refined = str(
            self._call_endpoint(
                [
                    {
                        "role": "user",
                        "content": (
                            "Improve the draft using the critique.\n\n"
                            f"Draft:\n{draft}\n\nCritique:\n{critique}\n\n"
                            "Return only the improved final answer."
                        ),
                    }
                ]
            )
        )
        return refined

    def _repair_python_code_once(self, prompt_blob: str, candidate: str) -> str:
        repair_prompt = [
            {
                "role": "user",
                "content": (
                    "The following Python answer has syntax/format issues.\n"
                    "Fix it and return only valid Python code.\n"
                    "Do not use markdown fences.\n"
                    "Preserve the intended function signatures.\n\n"
                    f"Original task prompt:\n{prompt_blob}\n\n"
                    f"Candidate:\n{candidate}"
                ),
            }
        ]
        return self._call_endpoint(repair_prompt)

    def _run_gen_test_select3(self, messages: Union[List[Message], str]) -> str:
        text_prompt = self._as_text_prompt(messages)
        prompt_blob = _messages_to_text(text_prompt)
        tests, tests_source, tests_parse_error = extract_inline_tests_with_meta(prompt_blob)
        params = self.workflow_def.get("params", {})
        k = int(params.get("k", 3))
        timeout_s = int(params.get("exec_timeout_s", 8))
        enable_repair = bool(int(os.environ.get("WAE_PREMIUM_SYNTAX_REPAIR", "1")))
        query_hash = hashlib.md5(prompt_blob.encode("utf-8")).hexdigest()[:10]
        phase = str(os.environ.get("WAE_PREMIUM_PHASE", "unknown"))
        split = str(os.environ.get("WAE_PREMIUM_SPLIT", "unknown"))
        require_tests = bool(int(os.environ.get("WAE_PREMIUM_REQUIRE_TESTS", "0")))

        if require_tests and not tests:
            fallback = self._run_io(messages)
            self._log_premium_debug(
                {
                    "workflow_id": self.workflow_id,
                    "base_model": self.base_model,
                    "phase": phase,
                    "split": split,
                    "query_hash": query_hash,
                    "k": k,
                    "tests_present": False,
                    "tests_count": 0,
                    "tests_source": tests_source,
                    "tests_parse_error": tests_parse_error,
                    "early_stop": False,
                    "early_stop_at": None,
                    "candidates": [],
                    "final_selected_idx": None,
                    "final_selected_pass": None,
                    "final_selection_reason": "no_tests_fallback_io",
                }
            )
            return fallback

        candidates: List[str] = []
        cand_meta: List[Dict[str, object]] = []
        early_stop_at: Optional[int] = None
        for _ in range(k):
            out = self._call_endpoint(text_prompt)
            repaired = False
            syntax_ok = bool(is_python_syntax_valid(out))
            if enable_repair and not syntax_ok:
                out = self._repair_python_code_once(prompt_blob, out)
                repaired = True
                syntax_ok = bool(is_python_syntax_valid(out))
            candidates.append(out)
            meta = {
                "idx": len(candidates),
                "syntax_ok": syntax_ok,
                "repaired_once": repaired,
                "test_pass": None,
                "test_pass_count": 0,
                "test_total": len(tests),
            }
            if tests:
                code = extract_python_code(out)
                pass_count = 0
                for test_line in tests:
                    ok_single, _ = run_python_tests(
                        code, [test_line], timeout_s=timeout_s
                    )
                    pass_count += int(bool(ok_single))
                ok = pass_count >= len(tests)
                meta["test_pass_count"] = int(pass_count)
                meta["test_pass"] = bool(ok)
                cand_meta.append(meta)
                if ok:
                    early_stop_at = len(candidates)
                    self._log_premium_debug(
                        {
                            "workflow_id": self.workflow_id,
                            "base_model": self.base_model,
                            "phase": phase,
                            "split": split,
                            "query_hash": query_hash,
                            "k": k,
                            "tests_present": True,
                            "tests_count": len(tests),
                            "tests_source": tests_source,
                            "tests_parse_error": tests_parse_error,
                            "syntax_repair_enabled": bool(enable_repair),
                            "early_stop": True,
                            "early_stop_at": early_stop_at,
                            "candidates": cand_meta,
                            "final_selected_idx": early_stop_at,
                            "final_selected_pass": True,
                            "final_selection_reason": "first_test_pass",
                        }
                    )
                    return out
            else:
                cand_meta.append(meta)

        final_selection_reason = "syntax_filter_or_length"
        final_idx: Optional[int] = None
        if tests and cand_meta:
            best_idx = max(
                range(len(cand_meta)),
                key=lambda i: (
                    int(cand_meta[i].get("test_pass_count", 0)),
                    int(bool(cand_meta[i].get("syntax_ok", False))),
                    -int(cand_meta[i].get("idx", i + 1)),
                ),
            )
            final = candidates[best_idx]
            final_idx = best_idx + 1
            final_selection_reason = "max_test_pass_count"
        else:
            final = self._pick_consistent_candidate(candidates)
            for i, cand in enumerate(candidates):
                if cand == final:
                    final_idx = i + 1
                    break

        final_pass: Optional[bool] = None
        if tests:
            if final_idx is not None and 0 <= final_idx - 1 < len(cand_meta):
                final_pass = cand_meta[final_idx - 1]["test_pass"]  # type: ignore[assignment]
            else:
                code = extract_python_code(final)
                ok, _ = run_python_tests(code, tests, timeout_s=timeout_s)
                final_pass = bool(ok)

        self._log_premium_debug(
            {
                "workflow_id": self.workflow_id,
                "base_model": self.base_model,
                "phase": phase,
                "split": split,
                "query_hash": query_hash,
                "k": k,
                "tests_present": bool(tests),
                "tests_count": len(tests),
                "tests_source": tests_source,
                "tests_parse_error": tests_parse_error,
                "syntax_repair_enabled": bool(enable_repair),
                "early_stop": False,
                "early_stop_at": None,
                "candidates": cand_meta,
                "final_selected_idx": final_idx,
                "final_selected_pass": final_pass,
                "final_selection_reason": final_selection_reason,
            }
        )
        return final

    def _log_premium_debug(self, payload: Dict[str, object]) -> None:
        if not self._premium_debug_log:
            return
        event = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": self.method,
            **payload,
        }
        try:
            with open(self._premium_debug_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Logging must not break main inference.
            pass

    def gen(
        self,
        messages: Union[List[Message], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        start = time.time()
        self._reset_trace()
        if self.method == "io":
            out = self._run_io(messages)
        elif self.method == "refine2":
            out = self._run_refine2(messages)
        elif self.method == "self_consistency3":
            out = self._run_self_consistency3(messages)
        elif self.method == "critique_refine":
            out = self._run_critique_refine(messages)
        elif self.method == "gen_test_select3":
            out = self._run_gen_test_select3(messages)
        else:
            out = self._run_io(messages)
        self.last_latency_s = time.time() - start
        self.last_trace["latency_s"] = float(self.last_latency_s)
        p_hashes = list(self.last_trace.get("prompt_hashes", []))
        if p_hashes:
            self.last_trace["prompt_template_hash"] = p_hashes[0]
        else:
            self.last_trace["prompt_template_hash"] = ""
        self.last_trace["output_hash"] = _short_hash(str(out))
        return out

    async def agen(
        self,
        messages: Union[List[Message], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # Pilot keeps synchronous execution path for determinism.
        return self.gen(messages, max_tokens=max_tokens, temperature=temperature, num_comps=num_comps)
