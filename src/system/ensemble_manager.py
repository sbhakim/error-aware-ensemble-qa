# src/system/ensemble_manager.py

import gc
import logging
import torch
import os
import json
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

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
from ..utils.ensemble_helpers import canonical_model_key

logger = logging.getLogger(__name__)


# ============================================================================
# NEW: Meta-Learning for Adaptive Fusion Weights
# ============================================================================

class MetaLearningWeights:
    """
    Learns optimal per-feature, per-model weights from validation performance.

    Uses simple online updates (exponential moving average) to adapt fusion weights
    based on observed success rates. This allows the ensemble to discover nuanced
    interactions between query features and model capabilities.

    Example:
        If Llama-3.2-3B consistently fails on temporal queries with "before halftime"
        but succeeds on "earlier in the game", the meta-learner will down-weight
        Llama specifically for the temporal_ambiguity feature.
    """

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize meta-learning component.

        Args:
            learning_rate: EMA alpha for weight updates (0.1 = 10% new, 90% old)
        """
        self.lr = learning_rate
        # (feature_name, model_key) -> weight multiplier
        self.feature_model_weights = defaultdict(lambda: 1.0)
        # Track statistics for analysis
        self.performance_history = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.logger = logging.getLogger(__name__)

    def get_weight(self, features: dict, model_key: str) -> float:
        """
        Returns multiplicative weight for this model given query features.

        Weight < 1.0: down-weight model (likely to fail on this feature)
        Weight > 1.0: up-weight model (likely to succeed on this feature)

        Args:
            features: Dict of boolean features from QueryFeatureExtractor
            model_key: Model identifier (e.g., "llama-3.2-3b")

        Returns:
            Combined weight (clamped to [0.2, 2.0] for stability)
        """
        weight = 1.0
        model_key_canon = canonical_model_key(model_key)

        for feature_name, is_present in features.items():
            if is_present:  # Only consider active features
                key = (feature_name, model_key_canon)
                weight *= self.feature_model_weights[key]

        # Clamp to reasonable range to prevent extreme values
        return max(0.2, min(2.0, weight))

    def update(self, features: dict, model_key: str, was_correct: bool):
        """
        Online update after observing a result.

        If model succeeded on a query with feature F, increase weight.
        If model failed, decrease weight.

        Args:
            features: Dict of boolean features
            model_key: Model identifier
            was_correct: Whether this model's answer was correct
        """
        model_key_canon = canonical_model_key(model_key)

        for feature_name, is_present in features.items():
            if is_present:
                key = (feature_name, model_key_canon)

                # Update statistics
                self.performance_history[key]['total'] += 1
                if was_correct:
                    self.performance_history[key]['correct'] += 1

                # Calculate empirical success rate for this feature-model combo
                stats = self.performance_history[key]
                empirical_rate = stats['correct'] / stats['total']

                # Target weight: boost if success rate > 0.5, down-weight if < 0.5
                # Maps [0, 1] success rate to [0.5, 1.5] weight
                target_weight = 0.5 + (empirical_rate - 0.5) * 2

                # Exponential moving average update
                current = self.feature_model_weights[key]
                self.feature_model_weights[key] = (1 - self.lr) * current + self.lr * target_weight

                self.logger.debug(
                    f"Meta-learning update: ({feature_name}, {model_key_canon}) "
                    f"rate={empirical_rate:.2f} -> weight={self.feature_model_weights[key]:.3f}"
                )

    def save_weights(self, path: str):
        """Persist learned weights for future runs."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            data = {
                'weights': {f"{k[0]}::{k[1]}": v for k, v in self.feature_model_weights.items()},
                'stats': {
                    f"{k[0]}::{k[1]}": v for k, v in self.performance_history.items()
                }
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Meta-learned weights saved to {path}")
        except Exception as e:
            self.logger.warning(f"Failed to save meta-learning weights: {e}")

    def load_weights(self, path: str):
        """Load previously learned weights from file."""
        if not os.path.exists(path):
            self.logger.info(f"No existing weights file at {path}. Starting fresh.")
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Restore weights
            for key_str, weight in data.get('weights', {}).items():
                feature, model = key_str.split('::')
                self.feature_model_weights[(feature, model)] = weight

            # Restore statistics
            for key_str, stats in data.get('stats', {}).items():
                feature, model = key_str.split('::')
                self.performance_history[(feature, model)] = stats

            self.logger.info(
                f"Loaded {len(self.feature_model_weights)} meta-learned weights from {path}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load meta-learning weights: {e}")

    def get_summary(self) -> dict:
        """Returns summary of learned weights for analysis."""
        summary = {
            'total_feature_model_pairs': len(self.feature_model_weights),
            'weights': {},
            'statistics': {}
        }

        for (feature, model), weight in sorted(self.feature_model_weights.items()):
            key_str = f"{feature}::{model}"
            summary['weights'][key_str] = round(weight, 3)

            stats = self.performance_history.get((feature, model), {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                summary['statistics'][key_str] = {
                    'success_rate': round(stats['correct'] / stats['total'], 3),
                    'total_observations': stats['total']
                }

        return summary


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

        # NEW: Initialize meta-learning for adaptive fusion weights
        self.meta_weights = MetaLearningWeights(learning_rate=0.1)
        self.weights_save_path = os.path.join("logs", "meta_learned_weights.json")
        self.meta_weights.load_weights(self.weights_save_path)  # Load from previous runs if available
        logger.info(f"Meta-learning initialized. Weights will be saved to: {self.weights_save_path}")

    def _resolve_rules_file(self, model_cfg: Dict[str, Any], dataset_type: str) -> str:
        """
        Resolve rules file with explicit fallback order and existence checks.
        DROP order: model override -> dynamic rules -> static rules -> default static path.
        Hotpot order: model override -> configured hotpot rules -> default hotpot path.
        """
        ds = (dataset_type or "").strip().lower()
        model_override = model_cfg.get('rules_file')

        if ds == 'drop':
            candidates = [
                model_override,
                self.config.get('drop_rules_dynamic_file'),
                self.config.get('drop_rules_file'),
                "data/rules_drop.json"
            ]
        else:
            candidates = [
                model_override,
                self.config.get('hotpotqa_rules_file'),
                "data/rules_hotpotqa.json"
            ]

        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.exists(path):
                if ds == 'drop' and path != candidates[1] and candidates[1]:
                    logger.warning(
                        f"DROP dynamic rules missing or unavailable; using fallback rules file: {path}"
                    )
                return path

        raise FileNotFoundError(
            f"No valid rules file found for dataset='{ds}'. Tried: "
            f"{[p for p in candidates if p]}"
        )

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
        rules_file = self._resolve_rules_file(model_cfg=model_cfg, dataset_type=ds)

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
                        except Exception as e:
                            logger.debug(f"Failed to delete neural_retriever.{attr}: {e}")
                # Drop references
                try:
                    del scm.hybrid_integrator.neural_retriever
                except Exception as e:
                    logger.debug(f"Failed to delete scm.hybrid_integrator.neural_retriever: {e}")
                try:
                    del scm.hybrid_integrator
                except Exception as e:
                    logger.debug(f"Failed to delete scm.hybrid_integrator: {e}")
            # Finally drop SCM
            del scm
        except Exception as unload_err:
            logger.warning(f"Partial unload: {unload_err}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"torch.cuda.empty_cache failed: {e}")

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
                except Exception as e:
                    logger.debug(f"Could not capture provenance snapshot for model '{model_key}': {e}")
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
        except Exception as e:
            logger.debug(f"Failed to log sequential fusion inputs: {e}")

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
                except Exception as e:
                    logger.debug(f"Could not capture batched provenance for model '{model_key}': {e}")
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
            query_text = by_qid_query.get(qid, "")

            # Helpful log before fusion
            try:
                for mk, res in model_results_for_qid.items():
                    logger.info(f"[BATCH FUSION INPUT] qid={qid} model={mk} name={res.get('source_model_name')} "
                                f"conf={res.get('confidence')} type={type(res.get('answer'))}")
            except Exception as e:
                logger.debug(f"Failed to log batched fusion inputs for qid={qid}: {e}")

            # NEW: Apply meta-learned weights to adjust confidences before fusion
            adjusted_results = self.apply_meta_weights_to_results(query_text, model_results_for_qid)

            # Fuse with adjusted confidences
            fused = fuser.fuse(query_text, adjusted_results)
            fused_outputs[qid] = fused

        return fused_outputs

    def apply_meta_weights_to_results(self, query: str, model_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Apply meta-learned weights to adjust model confidences before fusion.

        Args:
            query: Query string for feature extraction
            model_results: Dict[model_key -> result_dict with 'confidence']

        Returns:
            Updated model_results with adjusted confidences
        """
        from ..utils.ensemble_helpers import QueryFeatureExtractor

        # Extract features from query
        extractor = QueryFeatureExtractor()
        features = extractor.extract_features(query)

        # Apply meta-learned weights to each model's confidence
        adjusted_results = {}
        for model_key, result in model_results.items():
            meta_weight = self.meta_weights.get_weight(features, model_key)
            original_conf = result.get('confidence', 0.0)

            # Create copy and adjust confidence by meta-learned weight
            adjusted_result = result.copy()
            adjusted_result['confidence'] = min(0.99, original_conf * meta_weight)
            adjusted_result['meta_weight_applied'] = meta_weight
            adjusted_result['original_confidence'] = original_conf

            adjusted_results[model_key] = adjusted_result

            logger.debug(
                f"[Meta-Weight] {model_key}: conf {original_conf:.3f} * weight {meta_weight:.3f} "
                f"= {adjusted_result['confidence']:.3f}"
            )

        return adjusted_results

    def update_meta_weights(self, query: str, model_results: Dict[str, Dict],
                           ground_truth: Optional[Dict] = None):
        """
        Update meta-learning weights based on observed performance.

        Should be called after ground truth is available to provide learning signal.

        Args:
            query: Query string
            model_results: Dict[model_key -> result with 'answer']
            ground_truth: Correct answer for evaluation
        """
        if not ground_truth:
            return

        from ..utils.ensemble_helpers import QueryFeatureExtractor, are_drop_values_equivalent

        # Extract features
        extractor = QueryFeatureExtractor()
        features = extractor.extract_features(query)

        # Determine answer type from ground truth
        answer_type = self._get_answer_type(ground_truth)

        # Update weights for each model based on correctness
        for model_key, result in model_results.items():
            prediction = result.get('answer')
            if prediction:
                try:
                    was_correct = are_drop_values_equivalent(prediction, ground_truth, answer_type)
                    self.meta_weights.update(features, model_key, was_correct)
                except Exception as e:
                    logger.warning(f"Failed to update meta-weights for {model_key}: {e}")

    def _get_answer_type(self, answer: dict) -> str:
        """Determine answer type (number/spans/date) from DROP answer dict."""
        if not isinstance(answer, dict):
            return 'unknown'

        num = answer.get('number', '')
        if num and str(num).strip():
            return 'number'

        spans = answer.get('spans', [])
        if spans and any(str(s).strip() for s in spans):
            return 'spans'

        date = answer.get('date', {})
        if isinstance(date, dict) and any(str(date.get(k, '')).strip() for k in ['day', 'month', 'year']):
            return 'date'

        return 'unknown'
