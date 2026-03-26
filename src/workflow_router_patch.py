"""Runtime patching utilities and WaE router implementation."""

from __future__ import annotations

import json
import os
import time
import hashlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from MAR.Graph.graph import Graph
from MAR.LLM.llm_registry import LLMRegistry
from MAR.LLM.price import MODEL_PRICE
from MAR.MasRouter.mas_router import MasRouter, VAE, vae_loss_function
from MAR.Utils.globals import CompletionTokens, Cost, PromptTokens
from MAR.Utils.utils import get_kwargs
from loguru import logger

from .workflow_llm import (
    EndpointLLM,
    EndpointSpec,
    WorkflowLLM,
    snapshot_runtime_telemetry,
)
from .workflow_profile import WORKFLOW_PROFILE, workflow_map


class EndpointManager:
    """Load endpoint definitions and serve LLM instances."""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._endpoints: Dict[str, EndpointLLM] = {}
        self._specs: Dict[str, Dict[str, object]] = {}
        models = raw.get("models", {})
        for name, spec in models.items():
            self._specs[name] = dict(spec)
            api_key = spec.get("api_key", "EMPTY")
            api_key_env = spec.get("api_key_env")
            if api_key_env:
                api_key = os.environ.get(api_key_env, api_key)
            endpoint_spec = EndpointSpec(
                name=name,
                model_id=spec["model_id"],
                base_url=spec["base_url"],
                api_key=api_key,
                timeout=float(spec.get("timeout", 120.0)),
                price_name=name,
            )
            self._endpoints[name] = EndpointLLM(endpoint_spec)
            price_cfg = spec.get("price", {"input": 0.2, "output": 0.2})
            MODEL_PRICE[name] = {
                "input": float(price_cfg["input"]),
                "output": float(price_cfg["output"]),
            }

    @property
    def endpoint_names(self) -> List[str]:
        return list(self._endpoints.keys())

    @property
    def endpoint_specs(self) -> Dict[str, Dict[str, object]]:
        return dict(self._specs)

    def get(self, name: str) -> EndpointLLM:
        if name not in self._endpoints:
            raise KeyError(f"Unknown endpoint name: {name}")
        return self._endpoints[name]

    def verify_workflow_mappings(self, workflow_candidates: List[Dict[str, object]]) -> List[str]:
        missing = []
        for wf in workflow_candidates:
            base_model = str(wf.get("base_model", ""))
            if base_model not in self._endpoints:
                missing.append(str(wf.get("id")))
        return missing

    def heterogeneity_summary(self, required_names: Optional[List[str]] = None) -> Dict[str, object]:
        names = required_names or self.endpoint_names
        model_ids = [str(self._specs[n]["model_id"]) for n in names if n in self._specs]
        base_urls = [str(self._specs[n]["base_url"]) for n in names if n in self._specs]
        unique_model_ids = sorted(set(model_ids))
        unique_base_urls = sorted(set(base_urls))
        return {
            "required_endpoints": names,
            "unique_model_ids": unique_model_ids,
            "unique_base_urls": unique_base_urls,
            "num_required": len(names),
            "num_unique_model_ids": len(unique_model_ids),
            "num_unique_base_urls": len(unique_base_urls),
            "is_heterogeneous": len(unique_model_ids) >= 2 and len(unique_base_urls) >= 2,
        }

    def check_ready(
        self,
        required_names: Optional[List[str]] = None,
        retries: int = 6,
        interval_s: float = 10.0,
        warmup: bool = True,
    ) -> Dict[str, object]:
        names = required_names or self.endpoint_names
        status: Dict[str, Dict[str, object]] = {
            name: {"ready": False, "attempts": 0, "last_error": ""} for name in names
        }
        for _ in range(max(retries, 1)):
            all_ready = True
            for name in names:
                item = status[name]
                if item["ready"]:
                    continue
                item["attempts"] = int(item["attempts"]) + 1
                try:
                    llm = self.get(name)
                    llm.client.models.list()
                    if warmup:
                        llm.client.chat.completions.create(
                            model=llm.spec.model_id,
                            messages=[{"role": "user", "content": "Reply with: READY"}],
                            max_tokens=8,
                            temperature=0.0,
                            n=1,
                        )
                    item["ready"] = True
                    item["last_error"] = ""
                except Exception as e:
                    item["last_error"] = str(e)
                    all_ready = False
            if all_ready:
                break
            time.sleep(interval_s)
        return {
            "required_endpoints": names,
            "all_ready": all(status[n]["ready"] for n in names),
            "status": status,
        }

    def as_llm_profile(self) -> List[Dict[str, str]]:
        prof = []
        for name in self.endpoint_names:
            prof.append(
                {
                    "Name": name,
                    "Description": f"Local vLLM endpoint for {name}.",
                }
            )
        return prof


class LLMRuntimePatch:
    """Monkey-patch LLMRegistry.get for local endpoints and wf:: IDs."""

    def __init__(self, endpoint_manager: EndpointManager):
        self.endpoint_manager = endpoint_manager
        self.workflow_index = workflow_map()
        self._orig_get = LLMRegistry.get
        self._cache: Dict[str, object] = {}
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        endpoint_manager = self.endpoint_manager
        workflow_index = self.workflow_index
        cache = self._cache
        orig_get = self._orig_get

        def _patched_get(cls, model_name: Optional[str] = None):
            key = model_name or endpoint_manager.endpoint_names[0]
            if key in cache:
                return cache[key]
            if key.startswith("wf::"):
                wf_id = key.split("wf::", 1)[1]
                if wf_id not in workflow_index:
                    raise KeyError(f"Unknown workflow id: {wf_id}")
                llm = WorkflowLLM(
                    workflow_id=wf_id,
                    workflow_def=workflow_index[wf_id],
                    endpoint_resolver=endpoint_manager.get,
                )
                cache[key] = llm
                return llm
            if key in endpoint_manager.endpoint_names:
                llm = endpoint_manager.get(key)
                cache[key] = llm
                return llm
            return orig_get(key)

        LLMRegistry.get = classmethod(_patched_get)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        LLMRegistry.get = self._orig_get
        self._installed = False


class WaERouter(MasRouter):
    """MasRouter variant where LLM routing is replaced by workflow routing."""

    def __init__(
        self,
        workflow_candidates: List[Dict[str, object]],
        in_dim: int = 384,
        hidden_dim: int = 64,
        max_agent: int = 6,
        temp: float = 0.5,
        workflow_prior: Optional[Dict[str, float]] = None,
        workflow_prior_beta: float = 0.0,
        device=None,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            max_agent=max_agent,
            temp=temp,
            device=device,
        )
        self.workflow_candidates = workflow_candidates
        self.workflow_encoder = VAE(in_dim, hidden_dim, hidden_dim)
        self.workflow_context_encoder = nn.Linear(hidden_dim * 3, hidden_dim)
        self.workflow_temp = 1.0
        self.workflow_prior = dict(workflow_prior or {})
        self.workflow_prior_beta = float(workflow_prior_beta)
        self.greedy_routing = bool(int(os.environ.get("WAE_ROUTER_GREEDY", "0")))
        self.greedy_router_components = bool(
            int(os.environ.get("WAE_ROUTER_GREEDY_ALL", "0"))
        )
        self.last_workflow_entropy = torch.tensor(0.0, device=self.device)

    def _encode_workflows(self) -> torch.Tensor:
        workflow_text = [
            f"{wf['name']} : {wf['description']} : method={wf['method']} : base={wf['base_model']}"
            for wf in self.workflow_candidates
        ]
        emb = self.text_encoder(workflow_text)
        wf_hat, wf_z, wf_mu, wf_log_var = self.workflow_encoder(emb)
        wf_loss = vae_loss_function(wf_hat, emb, wf_mu, wf_log_var)
        wf_z = F.normalize(wf_z, p=2, dim=1)
        return wf_z, wf_loss

    def _route_workflows_by_role(
        self,
        workflow_emb: torch.Tensor,
        selected_roles: List[List[Dict[str, str]]],
        query_context: torch.Tensor,
        collab_context: torch.Tensor,
        role_context: torch.Tensor,
        force_workflow_id: Optional[str] = None,
    ) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        selected_ids: List[List[int]] = [[] for _ in selected_roles]
        log_probs = torch.zeros([len(selected_roles), 1], device=self.device)
        entropies: List[torch.Tensor] = []
        contexts = self.workflow_context_encoder(
            torch.concat([query_context, collab_context, role_context], dim=-1)
        )
        contexts = F.normalize(contexts, p=2, dim=1)

        force_idx = None
        if force_workflow_id:
            for idx, wf in enumerate(self.workflow_candidates):
                if wf["id"] == force_workflow_id:
                    force_idx = idx
                    break
            if force_idx is None:
                raise ValueError(f"force_workflow_id not found: {force_workflow_id}")

        for i, roles in enumerate(selected_roles):
            for role in roles:
                if force_idx is not None:
                    selected_ids[i].append(force_idx)
                    continue
                role_name = role["Name"]
                allow = []
                for idx, wf in enumerate(self.workflow_candidates):
                    if role_name in wf["allowed_roles"]:
                        allow.append(idx)
                if not allow:
                    allow = list(range(len(self.workflow_candidates)))
                logits = torch.matmul(
                    contexts[i].unsqueeze(0), workflow_emb[allow].T
                ).squeeze(0)  # [n_allow]
                if self.workflow_prior_beta > 0.0 and self.workflow_prior:
                    prior_logits = []
                    for idx in allow:
                        wf_id = str(self.workflow_candidates[idx]["id"])
                        prior = float(self.workflow_prior.get(wf_id, 0.5))
                        prior = max(1e-4, min(1.0 - 1e-4, prior))
                        prior_logits.append(float(torch.logit(torch.tensor(prior))))
                    logits = logits + self.workflow_prior_beta * torch.tensor(
                        prior_logits, device=self.device, dtype=logits.dtype
                    )
                scores = torch.softmax(logits / self.workflow_temp, dim=0)
                entropies.append(-(scores * torch.log(scores + 1e-12)).sum())
                if self.greedy_routing:
                    local_idx = int(torch.argmax(scores, dim=0).item())
                else:
                    cumsum = torch.cumsum(scores, dim=0)
                    rand = torch.rand([1], device=self.device)
                    local_idx = int((cumsum > rand).float().argmax(dim=0).item())
                global_idx = allow[local_idx]
                selected_ids[i].append(global_idx)
                log_probs[i][0] = log_probs[i][0] + torch.log(scores[local_idx] + 1e-12)
        entropy = (
            torch.stack(entropies).mean()
            if entropies
            else torch.tensor(0.0, device=self.device)
        )
        return selected_ids, log_probs, entropy

    def _deterministic_collab_forward(
        self, collabs_embedding: torch.Tensor, queries_embedding: torch.Tensor
    ):
        collab_hat, collab_z, collab_mu, collab_logvar = (
            self.collab_determiner.collab_encoder(collabs_embedding)
        )
        collab_z = F.normalize(collab_z, p=2, dim=1)

        context_hat, context_z, context_mu, context_logvar = (
            self.collab_determiner.context_encoder(queries_embedding)
        )
        context_z = F.normalize(context_z, p=2, dim=1)

        scores = torch.matmul(context_z, collab_z.T)
        scores = torch.softmax(scores / self.collab_determiner.temp, dim=1)
        selected_index = torch.argmax(scores, dim=1)
        log_probs = torch.log(
            scores[torch.arange(scores.size(0)), selected_index] + 1e-12
        ).unsqueeze(1)
        collab_embedding = collab_z[selected_index]
        vae_loss1 = vae_loss_function(collab_hat, collabs_embedding, collab_mu, collab_logvar)
        vae_loss2 = vae_loss_function(context_hat, queries_embedding, context_mu, context_logvar)
        vae_loss = vae_loss1 + vae_loss2
        return selected_index, log_probs, collab_embedding, vae_loss

    def _deterministic_role_forward(
        self,
        roles_list: List[torch.Tensor],
        contexts: torch.Tensor,
        agent_num_int: torch.Tensor,
    ):
        role_alloc = self.role_allocation
        selected_roles_idx: List[List[torch.Tensor]] = []
        log_probs = torch.zeros([contexts.size(0), 1], device=self.device)
        summary_role_list: List[torch.Tensor] = []
        vae_loss = None
        for i, roles in enumerate(roles_list):
            selected_roles_idx.append([])
            role_hat, role_z, role_mu, role_log_var = role_alloc.role_encoder(roles)
            role_embedding = F.normalize(role_z, p=2, dim=1)

            one_loss = vae_loss_function(role_hat, roles, role_mu, role_log_var)
            vae_loss = one_loss if vae_loss is None else vae_loss + one_loss

            current_role_embedding = role_alloc.init_role_embedding
            history_role_embedding = role_alloc.init_role_embedding
            for _ in range(int(agent_num_int[i].item())):
                history_role_embedding = history_role_embedding + current_role_embedding
                history_role_embedding = F.layer_norm(
                    history_role_embedding, history_role_embedding.shape[1:]
                )
                contexts_embedding = role_alloc.context_encoder(
                    torch.cat([contexts[i].unsqueeze(0), history_role_embedding], dim=1)
                )
                contexts_embedding = F.normalize(contexts_embedding, p=2, dim=1)
                scores = torch.matmul(contexts_embedding, role_embedding.T)
                scores = torch.softmax(scores / role_alloc.temp, dim=1)
                selected_index = torch.argmax(scores, dim=1)
                log_probs[i][0] = log_probs[i][0] + torch.log(
                    scores[torch.arange(scores.size(0)), selected_index] + 1e-12
                ).unsqueeze(1)
                current_role_embedding = role_embedding[selected_index]
                selected_roles_idx[-1].append(selected_index)
            summary_role_list.append(history_role_embedding)

        summary_role = torch.cat(summary_role_list, dim=0)
        avg_vae_loss = vae_loss / max(len(roles_list), 1)
        return selected_roles_idx, log_probs, summary_role, avg_vae_loss

    def _workflows_from_indices(
        self, idx_lists: List[List[int]]
    ) -> List[List[Dict[str, str]]]:
        out: List[List[Dict[str, str]]] = []
        for idxs in idx_lists:
            out.append(
                [
                    {
                        "Name": f"wf::{self.workflow_candidates[idx]['id']}",
                        "Description": self.workflow_candidates[idx]["description"],
                    }
                    for idx in idxs
                ]
            )
        return out

    def forward(
        self,
        queries: List[str],
        tasks: List[Dict[str, str]],
        collabs: List[Dict[str, str]],
        given_task: Optional[List[int]] = None,
        prompt_file: str = "MAR/Roles/FinalNode/mbpp.json",
        force_workflow_id: Optional[str] = None,
    ):
        tasks_list = self._preprocess_data(tasks)
        collabs_list = self._preprocess_data(collabs)
        task_role_database, task_role_emb = self.encoder_roles()

        queries_embedding = self.text_encoder(queries)
        tasks_embedding = self.text_encoder(tasks_list)
        collabs_embedding = self.text_encoder(collabs_list)

        selected_tasks_idx, tasks_probs, query_context = self.task_classifier(
            queries_embedding, tasks_embedding
        )
        selected_tasks = (
            [tasks[idx] for idx in selected_tasks_idx]
            if given_task is None
            else [tasks[idx] for idx in given_task]
        )
        tasks_role_list = [task_role_database[task["Name"]] for task in selected_tasks]
        tasks_role_emb_list = [task_role_emb[task["Name"]] for task in selected_tasks]

        if self.greedy_router_components:
            selected_collabs_idx, collab_log_probs, collab_context, collab_vae_loss = (
                self._deterministic_collab_forward(
                    collabs_embedding=collabs_embedding,
                    queries_embedding=queries_embedding,
                )
            )
        else:
            selected_collabs_idx, collab_log_probs, collab_context, collab_vae_loss = (
                self.collab_determiner(collabs_embedding, queries_embedding)
            )
        selected_collabs = [collabs[idx] for idx in selected_collabs_idx]

        agent_num_int, agent_num_float, num_vae_loss = self.num_determiner(
            queries_embedding
        )
        if self.greedy_router_components:
            selected_roles_idx, role_log_probs, role_context, role_vae_loss = (
                self._deterministic_role_forward(
                    roles_list=tasks_role_emb_list,
                    contexts=torch.concat([query_context, collab_context], dim=-1),
                    agent_num_int=agent_num_int,
                )
            )
        else:
            selected_roles_idx, role_log_probs, role_context, role_vae_loss = (
                self.role_allocation(
                    tasks_role_emb_list,
                    torch.concat([query_context, collab_context], dim=-1),
                    agent_num_int,
                )
            )
        selected_roles = [
            [task_roles[s_idx.item()] for s_idx in selected_roles_id_list]
            for task_roles, selected_roles_id_list in zip(tasks_role_list, selected_roles_idx)
        ]

        workflow_emb, workflow_vae_loss = self._encode_workflows()
        selected_workflow_idx, workflow_log_probs, workflow_entropy = self._route_workflows_by_role(
            workflow_emb=workflow_emb,
            selected_roles=selected_roles,
            query_context=query_context,
            collab_context=collab_context,
            role_context=role_context,
            force_workflow_id=force_workflow_id,
        )
        self.last_workflow_entropy = workflow_entropy
        selected_workflows = self._workflows_from_indices(selected_workflow_idx)

        log_probs = workflow_log_probs + role_log_probs + collab_log_probs
        vae_loss = collab_vae_loss + num_vae_loss + role_vae_loss + workflow_vae_loss

        final_result = []
        costs = []
        selected_workflow_ids = []
        selected_role_names = []
        sample_debug = []
        for query, task, wfs, collab, roles in zip(
            queries, selected_tasks, selected_workflows, selected_collabs, selected_roles
        ):
            previous_cost = Cost.instance().value
            previous_prompt_tokens = PromptTokens.instance().value
            previous_completion_tokens = CompletionTokens.instance().value
            previous_rt = snapshot_runtime_telemetry()
            kwargs = get_kwargs(collab["Name"], len(wfs))
            llm_names = [wf["Name"] for wf in wfs]
            role_names = [role["Name"] for role in roles]
            selected_workflow_ids.append(llm_names)
            selected_role_names.append(role_names)
            fallback_template_hash = hashlib.md5(
                (
                    f"{prompt_file}|{task['Name']}|{collab['Name']}|"
                    + "|".join(role_names)
                    + "|"
                    + "|".join(llm_names)
                ).encode("utf-8")
            ).hexdigest()[:12]
            logger.info(f"Query: {query}")
            logger.info(f"Task: {task['Name']}")
            logger.info(f"Workflows: {llm_names}")
            logger.info(f"Reasoning: {collab['Name']}")
            logger.info(f"Roles: {role_names}")
            logger.info("-----------------------------------")
            graph = Graph(
                domain=task["Name"],
                llm_names=llm_names,
                agent_names=role_names,
                decision_method="FinalRefer",
                prompt_file=prompt_file,
                reasoning_name=collab["Name"],
                **kwargs,
            )
            self.g = graph
            answer = graph.run(inputs={"query": query}, num_rounds=kwargs["num_rounds"])[0][0]
            final_result.append(answer)
            costs.append(Cost.instance().value - previous_cost)
            cur_rt = snapshot_runtime_telemetry()

            selected_endpoint_names = [
                str(self.workflow_candidates[idx]["base_model"])
                for idx in selected_workflow_idx[len(sample_debug)]
            ]
            selected_model_ids = [
                str(self.workflow_candidates[idx].get("base_model_id", "unknown"))
                for idx in selected_workflow_idx[len(sample_debug)]
            ]
            invoked_workflow_ids = list(llm_names)
            invoked_prompt_hashes: List[str] = []
            invoked_output_hashes: List[str] = []
            invoked_endpoint_names: List[str] = []
            invoked_model_ids: List[str] = []
            llm_infer_latency_s = 0.0
            for wf_name in llm_names:
                try:
                    wf_llm = LLMRegistry.get(wf_name)
                    tr = getattr(wf_llm, "last_trace", {}) or {}
                    llm_infer_latency_s += float(tr.get("latency_s", 0.0) or 0.0)
                    invoked_workflow_ids.extend(
                        list(tr.get("invoked_workflow_ids", []))
                    )
                    invoked_prompt_hashes.extend(list(tr.get("prompt_hashes", [])))
                    invoked_output_hashes.extend(list(tr.get("output_hashes", [])))
                    invoked_endpoint_names.extend(list(tr.get("endpoint_names", [])))
                    invoked_model_ids.extend(list(tr.get("model_ids", [])))
                except Exception:
                    continue

            sample_debug.append(
                {
                    "query_hash": hashlib.md5(query.encode("utf-8")).hexdigest()[:10],
                    "task": task["Name"],
                    "collab": collab["Name"],
                    "selected_workflow_ids": llm_names,
                    "invoked_workflow_ids": list(dict.fromkeys(invoked_workflow_ids)),
                    "selected_roles": role_names,
                    "selected_base_models": selected_endpoint_names,
                    "selected_endpoint_names": selected_endpoint_names,
                    "selected_model_ids": selected_model_ids,
                    "invoked_endpoint_names": list(dict.fromkeys(invoked_endpoint_names)),
                    "invoked_model_ids": list(dict.fromkeys(invoked_model_ids)),
                    "prompt_hashes": invoked_prompt_hashes,
                    "output_hashes": invoked_output_hashes,
                    "prompt_template_hash": (
                        invoked_prompt_hashes[0]
                        if invoked_prompt_hashes
                        else fallback_template_hash
                    ),
                    "output_hash": hashlib.md5(str(answer).encode("utf-8")).hexdigest()[:12],
                    "prompt_tokens": int(PromptTokens.instance().value - previous_prompt_tokens),
                    "completion_tokens": int(
                        CompletionTokens.instance().value - previous_completion_tokens
                    ),
                    "llm_infer_latency_s": float(llm_infer_latency_s),
                    "overflow_retries": int(
                        cur_rt["overflow_retries"] - previous_rt["overflow_retries"]
                    ),
                    "endpoint_calls": int(
                        cur_rt["endpoint_calls"] - previous_rt["endpoint_calls"]
                    ),
                }
            )

        return (
            final_result,
            costs,
            log_probs,
            tasks_probs,
            vae_loss,
            agent_num_float,
            selected_workflow_ids,
            selected_role_names,
            sample_debug,
        )


def dump_selected_workflows(path: str, selected_workflow_ids: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(selected_workflow_ids, f, ensure_ascii=False, indent=2)


def default_workflow_candidates() -> List[Dict[str, object]]:
    return list(WORKFLOW_PROFILE)
