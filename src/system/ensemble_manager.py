# src/system/ensemble_manager.py

import gc
import logging
import torch
from typing import Dict, Any, Optional, Tuple, List

# Core SymRAG components needed to build an SCM instance on the fly
from ..system.system_control_manager import SystemControlManager
from ..integrators.hybrid_integrator import HybridIntegrator
from ..reasoners.neural_retriever import NeuralRetriever
from ..reasoners.networkx_symbolic_reasoner_drop import GraphSymbolicReasonerDrop
from ..reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner
from ..utils.ensemble_helpers import EnsembleFuser
from ..utils.dimension_manager import DimensionalityManager
from ..queries.query_expander import QueryExpander
from ..system.response_aggregator import UnifiedResponseAggregator
from ..utils.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class EnsembleManager:
    """
    Orchestrates a multi-model ensemble. Supports:
      1) Per-query sequential mode (existing behavior).
      2) Batched per-model mode (load each model once for all queries, then fuse).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        resource_manager,
        dim_manager: DimensionalityManager,
        query_expander: QueryExpander,
        response_aggregator: UnifiedResponseAggregator
    ):
        self.config = config
        self.ensemble_config = config.get('ensemble', {})

        # Shared components that are model-agnostic
        self.resource_manager = resource_manager
        self.dim_manager = dim_manager
        self.query_expander = query_expander
        self.response_aggregator = response_aggregator

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _initialize_scm_for_model(self, model_key: str, dataset_type: str) -> SystemControlManager:
        """
        Creates a complete, single-model SystemControlManager instance on demand.
        This logic mirrors your ablation builder and allows per-model overrides.
        """
        all_model_cfgs = self.config.get('model_configs', {})
        if model_key not in all_model_cfgs:
            raise ValueError(
                f"Missing 'model_configs.{model_key}' in config. "
                f"Available keys: {list(all_model_cfgs.keys())}"
            )
        model_cfg = all_model_cfgs.get(model_key, {})
        model_name = model_cfg.get('model_name')
        if not model_name:
            raise ValueError(f"Could not find model_name for model_key '{model_key}' in config.yaml")

        ds = (dataset_type or "").strip().lower()
        logger.info(f"Initializing SCM stack for model_key='{model_key}' model_name='{model_name}' dataset='{ds}'")

        # 1) Neural Retriever (per-model; allow per-model overrides)
        neural_retriever = NeuralRetriever(
            model_name=model_name,
            use_quantization=model_cfg.get('use_quantization', self.config.get('neural_use_quantization', True)),
            device=self.dim_manager.device,
            few_shot_examples_path=(model_cfg.get('few_shot_examples_path') if ds == 'drop' else None)
        )

        # 2) Symbolic Reasoner (dataset-specific, allow per-model rules override)
        rules_file = (
            model_cfg.get('rules_file') or
            (self.config.get('hotpotqa_rules_file') if ds == 'hotpotqa' else
             self.config.get('drop_rules_dynamic_file') or
             self.config.get('drop_rules_file') or
             "data/rules_drop.json")
        )

        if ds == 'drop':
            symbolic_reasoner = GraphSymbolicReasonerDrop(rules_file=rules_file, dim_manager=self.dim_manager)
        else:
            symbolic_reasoner = GraphSymbolicReasoner(rules_file=rules_file, dim_manager=self.dim_manager)

        # 3) Hybrid Integrator
        hybrid_integrator = HybridIntegrator(
            symbolic_reasoner=symbolic_reasoner,
            neural_retriever=neural_retriever,
            query_expander=self.query_expander,
            dim_manager=self.dim_manager,
            dataset_type=ds
        )

        # 4) System Control Manager (separate metrics dir per model & dataset)
        metrics_collector = MetricsCollector(metrics_dir=f"logs/ensemble_runs/{ds}/{model_key}")

        scm = SystemControlManager(
            hybrid_integrator=hybrid_integrator,
            resource_manager=self.resource_manager,
            aggregator=self.response_aggregator,
            metrics_collector=metrics_collector
        )

        # --- Provenance logging: ensure we really loaded what we think we loaded ---
        try:
            nr = scm.hybrid_integrator.neural_retriever
            actual_model_obj = getattr(nr, "model", None)
            actual_tok_obj = getattr(nr, "tokenizer", None)
            actual_name = None
            if actual_model_obj is not None:
                # HF models commonly expose _name_or_path in config
                cfg = getattr(actual_model_obj, "config", None)
                actual_name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
            logger.error("=== MODEL CHECK ===")
            logger.error(f"Config model_key: {model_key}")
            logger.error(f"Config requested name: {model_name}")
            logger.error(f"Actual loaded model name: {actual_name}")
            logger.error(f"Model object id: {id(actual_model_obj)} | Tokenizer id: {id(actual_tok_obj)}")
            logger.error("==================")
        except Exception as prov_err:
            logger.warning(f"Model provenance logging failed for '{model_key}': {prov_err}")

        return scm

    def _unload_model(self, scm: Optional[SystemControlManager]) -> None:
        """
        Centralized safe unload to avoid OOM and reduce code duplication.
        """
        if not scm:
            return
        try:
            if hasattr(scm, "hybrid_integrator"):
                nr = getattr(scm.hybrid_integrator, "neural_retriever", None)
                # Free heavy components when present
                if nr is not None:
                    for attr in ("model", "tokenizer"):
                        try:
                            if getattr(nr, attr, None) is not None:
                                delattr(nr, attr)
                        except Exception:
                            pass
                # Drop references
                try:
                    del scm.hybrid_integrator.neural_retriever
                except Exception:
                    pass
                try:
                    del scm.hybrid_integrator
                except Exception:
                    pass
            # Finally drop SCM
            del scm
        except Exception as unload_err:
            logger.warning(f"Partial unload: {unload_err}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _extract_response_details(self, scm_response: Any) -> Dict[str, Any]:
        """
        Safely extracts the answer payload and confidence from SCM output.
        Accepts either (resp, path) or resp-only variants.
        Made more permissive to reduce 'no_valid_results' due to shape mismatches.
        """
        try:
            logger.debug(f"Raw SCM response type: {type(scm_response)}")
            # Unwrap (resp, path) if present
            if isinstance(scm_response, (tuple, list)) and len(scm_response) >= 1:
                response_obj = scm_response[0]
                path = scm_response[1] if len(scm_response) > 1 else "unknown"
            else:
                response_obj = scm_response
                path = "unknown"

            if not isinstance(response_obj, dict):
                logger.error(f"Unexpected SCM response (not dict): {str(response_obj)[:400]}")
                return {'answer': None, 'confidence': 0.0, 'reasoning_path': path}

            # Try common keys, in priority order
            payload = None
            for k in ("result", "prediction", "answer", "output", "final"):
                if k in response_obj:
                    payload = response_obj[k]
                    break

            # Sometimes wrapped inside 'data' or 'payload'
            if payload is None:
                data_like = response_obj.get("data") or response_obj.get("payload")
                if isinstance(data_like, dict):
                    for k in ("result", "prediction", "answer", "output", "final"):
                        if k in data_like:
                            payload = data_like[k]
                            break

            # Fallback: if the response itself is already a DROP-like dict
            if payload is None and all(x in response_obj for x in ("number", "spans", "date")):
                payload = {
                    "number": response_obj.get("number", ""),
                    "spans": response_obj.get("spans", []),
                    "date": response_obj.get("date", {"day": "", "month": "", "year": ""}),
                    "status": response_obj.get("status", "success"),
                    "confidence": response_obj.get("confidence", 0.0),
                    "type": response_obj.get("type", None),
                }

            confidence = 0.0
            if isinstance(payload, dict):
                # Extract confidence if present; otherwise 0.0
                try:
                    confidence = float(payload.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0

            if payload is None:
                logger.error(f"Failed to extract payload from SCM response: {str(response_obj)[:400]}")

            return {
                'answer': payload,
                'confidence': confidence,
                'reasoning_path': path
            }
        except Exception as e:
            logger.exception(f"_extract_response_details failed: {e}")
            return {'answer': None, 'confidence': 0.0, 'reasoning_path': "unknown"}

    # -----------------------------
    # Existing per-query sequential API (kept)
    # -----------------------------
    def process_query_ensemble_sequential(
        self,
        query: str,
        context: str,
        query_id: str,
        dataset_type: str,
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Loads each model sequentially for THIS query only (slow for big runs),
        runs the query, unloads to free VRAM, then fuses results.
        """
        model_results: Dict[str, Dict[str, Any]] = {}
        model_keys = self.ensemble_config.get('models', [])
        ds = (dataset_type or "").strip().lower()

        for model_key in model_keys:
            scm: Optional[SystemControlManager] = None
            try:
                # 1) Build SCM for this model only
                scm = self._initialize_scm_for_model(model_key, ds)

                # --- provenance snapshot for this run ---
                try:
                    nr = scm.hybrid_integrator.neural_retriever
                    actual_model_obj = getattr(nr, "model", None)
                    actual_name = None
                    if actual_model_obj is not None:
                        cfg = getattr(actual_model_obj, "config", None)
                        actual_name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
                except Exception:
                    actual_name = None

                # 2) Run the query
                scm_response = scm.process_query_with_fallback(
                    query=query,
                    context=context,
                    query_id=f"{query_id}_{model_key}",
                    dataset_type=ds,
                    **kwargs
                )

                # 3) Extract & store
                extracted = self._extract_response_details(scm_response)
                extracted["source_model"] = model_key
                extracted["source_model_name"] = actual_name
                extracted["source_reasoner_path"] = extracted.get("reasoning_path", "unknown")
                model_results[model_key] = extracted

            except Exception as e:
                logger.error(f"Ensemble step failed for model {model_key}: {e}", exc_info=True)
                model_results[model_key] = {
                    'answer': None, 'confidence': 0.0, 'error': str(e),
                    'source_model': model_key, 'source_model_name': None, 'source_reasoner_path': 'error'
                }

            finally:
                # 4) Unload to avoid OOM
                self._unload_model(scm)

        # 5) Fuse
        fusion_strategy = self.ensemble_config.get('fusion_strategy', 'error_aware')
        fuser = EnsembleFuser(fusion_strategy=fusion_strategy)

        # Helpful log before fusion
        try:
            for mk, res in model_results.items():
                logger.info(f"[SEQ FUSION INPUT] model={mk} name={res.get('source_model_name')} "
                            f"conf={res.get('confidence')} type={type(res.get('answer'))}")
        except Exception:
            pass

        fused_result = fuser.fuse(query, model_results)

        return fused_result, "ensemble_sequential"

    # -----------------------------
    # Batched per-model API (fast path for whole datasets)
    # -----------------------------
    def process_dataset_ensemble_batched(
        self,
        *,
        queries: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
        query_ids: Optional[List[str]] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        dataset_type: str,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process ALL samples with Model 1, then ALL with Model 2, then ALL with Model 3.
        Finally fuse per sample.

        Supports TWO input styles (keyword-only to avoid ambiguity):
          - (queries, contexts, query_ids): lists of equal length (preferred; matches main.py)
          - samples: list of dicts with keys 'query_id', 'query', 'context' (legacy support)

        Args:
            queries: List[str] of query texts.
            contexts: List[str] of contexts aligned with queries.
            query_ids: List[str] of IDs aligned with queries.
            samples: Optional legacy input: list of dicts.
            dataset_type: 'drop' or 'hotpotqa'
            **kwargs: forwarded to process_query_with_fallback

        Returns:
            Dict[str, Dict]: mapping query_id -> fused_result_dict
        """
        ds = (dataset_type or "").strip().lower()
        model_keys = self.ensemble_config.get('models', [])

        # Normalize inputs into a list of sample dicts
        normalized_samples: List[Dict[str, Any]] = []

        if samples and isinstance(samples, list):
            # Legacy path: samples already provided
            for s in samples:
                qid = s.get("query_id")
                q = s.get("query")
                c = s.get("context", "")
                if not qid or not q:
                    logger.warning(f"Skipping sample without query_id or query: {s}")
                    continue
                normalized_samples.append({"query_id": qid, "query": q, "context": c})
        else:
            # New preferred path: queries/contexts/query_ids lists
            queries = queries or []
            contexts = contexts or []
            query_ids = query_ids or []
            if not (len(queries) == len(contexts) == len(query_ids)):
                raise ValueError("queries, contexts, and query_ids must be provided and have the same length.")
            for qid, q, c in zip(query_ids, queries, contexts):
                if not qid or not q:
                    logger.warning(f"Skipping sample without query_id or query: qid={qid}, query={q}")
                    continue
                normalized_samples.append({"query_id": qid, "query": q, "context": c})

        # Prepare containers
        by_qid_results: Dict[str, Dict[str, Any]] = {}  # qid -> {model_key -> {answer,..}}
        by_qid_query: Dict[str, str] = {}               # qid -> query string
        order: List[str] = []

        # Initialize per-sample slots
        for s in normalized_samples:
            qid = s["query_id"]
            order.append(qid)
            by_qid_results[qid] = {}
            by_qid_query[qid] = s["query"]

        # Run all samples through each model once
        for model_key in model_keys:
            scm: Optional[SystemControlManager] = None
            actual_name = None
            try:
                logger.info(f"[BATCHED] Loading model '{model_key}' for full dataset of {len(order)} samples...")
                scm = self._initialize_scm_for_model(model_key, ds)

                # provenance snapshot
                try:
                    nr = scm.hybrid_integrator.neural_retriever
                    actual_model_obj = getattr(nr, "model", None)
                    if actual_model_obj is not None:
                        cfg = getattr(actual_model_obj, "config", None)
                        actual_name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
                except Exception:
                    actual_name = None

                for s in normalized_samples:
                    qid = s["query_id"]
                    q = s["query"]
                    c = s.get("context", "")
                    try:
                        scm_response = scm.process_query_with_fallback(
                            query=q,
                            context=c,
                            query_id=f"{qid}_{model_key}",
                            dataset_type=ds,
                            **kwargs
                        )
                        extracted = self._extract_response_details(scm_response)
                        extracted["source_model"] = model_key
                        extracted["source_model_name"] = actual_name
                        extracted["source_reasoner_path"] = extracted.get("reasoning_path", "unknown")
                        by_qid_results[qid][model_key] = extracted
                    except Exception as e:
                        logger.error(f"[BATCHED] Model {model_key} failed on qid={qid}: {e}", exc_info=True)
                        by_qid_results[qid][model_key] = {
                            'answer': None, 'confidence': 0.0, 'error': str(e),
                            'source_model': model_key, 'source_model_name': actual_name,
                            'source_reasoner_path': 'error'
                        }

            finally:
                # Unload model after finishing all samples
                self._unload_model(scm)

        # Fuse per sample
        fusion_strategy = self.ensemble_config.get('fusion_strategy', 'error_aware')
        fuser = EnsembleFuser(fusion_strategy=fusion_strategy)

        fused_outputs: Dict[str, Dict[str, Any]] = {}
        for qid in order:
            model_results_for_qid = by_qid_results.get(qid, {})
            # Helpful log before fusion
            try:
                for mk, res in model_results_for_qid.items():
                    logger.info(f"[BATCH FUSION INPUT] qid={qid} model={mk} name={res.get('source_model_name')} "
                                f"conf={res.get('confidence')} type={type(res.get('answer'))}")
            except Exception:
                pass

            fused = fuser.fuse(by_qid_query.get(qid, ""), model_results_for_qid)
            fused_outputs[qid] = fused

        return fused_outputs
