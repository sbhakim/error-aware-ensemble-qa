# main.py

import os
import sys
import json
import time
import warnings
import argparse
import logging
import urllib3
import yaml
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Transformers library not found. Please install it: pip install transformers")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    print("Sentence-transformers library not found. Please install it: pip install sentence-transformers")
    sys.exit(1)

# SymRAG component imports
try:
    from src.reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner
    from src.reasoners.networkx_symbolic_reasoner_drop import GraphSymbolicReasonerDrop
    from src.reasoners.neural_retriever import NeuralRetriever
    from src.integrators.hybrid_integrator import HybridIntegrator
    from src.utils.dimension_manager import DimensionalityManager
    from src.utils.rule_extractor import RuleExtractor
    from src.queries.query_logger import QueryLogger
    from src.resources.resource_manager import ResourceManager
    from src.config.config_loader import ConfigLoader
    from src.queries.query_expander import QueryExpander
    from src.utils.evaluation import Evaluation
    # CHANGE: import SystemControlManager / UnifiedResponseAggregator from their modules (avoid package re-export reliance)
    from src.system.system_control_manager import SystemControlManager  # NEW: direct import
    from src.system.response_aggregator import UnifiedResponseAggregator  # NEW: direct import
    from src.utils.metrics_collector import MetricsCollector
    from src.utils.sample_debugger import SampleDebugger
    from src.utils.device_manager import DeviceManager
    from src.utils.progress import tqdm, ProgressManager
    from src.utils.output_capture import capture_output
    from src.ablation_study import setup_and_orchestrate_ablation
    from src.utils.data_loaders import load_hotpotqa, load_drop_dataset
    # NEW: EnsembleManager entry point (contains batched dataset processing)
    from src.system.ensemble_manager import EnsembleManager  # NEW
except ImportError as e:
    print(f"Error importing SymRAG components: {e}")
    print("Please ensure main.py is run from the project root directory or PYTHONPATH is set correctly.")
    sys.exit(1)

urllib3.disable_warnings()  # type: ignore
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_symrag_system(
    samples: int = 200,
    dataset_type: str = 'hotpotqa',
    args: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    """
    Main execution function for the SymRAG system for standard evaluation runs.
    Decides between single-model and ensemble mode based on config.
    """
    # CHANGE: load configuration up front to decide path
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "src", "config", "config.yaml")
        if not os.path.exists(config_path):
            logger.error(f"Main configuration file not found at: {config_path}")
            return {"error": f"Main configuration file not found: {config_path}"}
        config = ConfigLoader.load_config(config_path)
    except Exception as e:
        logger.exception(f"Failed to load configuration: {e}")
        return {"error": f"Configuration loading failed: {e}"}

    # ---- OPTIONAL ENSEMBLE OVERRIDES (minimal hook) ----
    def _deep_update(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(dst, dict) or not isinstance(src, dict):
            return dst
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    ov_path = os.getenv("ENSEMBLE_OVERRIDES")
    if ov_path and os.path.exists(ov_path):
        try:
            with open(ov_path, "r", encoding="utf-8") as f:
                ov_cfg = yaml.safe_load(f) or {}
            _deep_update(config, ov_cfg)
            logger.error(f"[ENSEMBLE_OVERRIDES] applied from {ov_path}")
        except Exception as e:
            logger.error(f"[ENSEMBLE_OVERRIDES] failed to load {ov_path}: {e}")
    # ----------------------------------------------------

    # CHANGE: branch to ensemble if enabled in config
    if config.get('ensemble', {}).get('enabled'):
        return run_ensemble_system(samples=samples, dataset_type=dataset_type, args=args, config=config)
    else:
        return run_single_model_system(samples=samples, dataset_type=dataset_type, args=args, config=config)


def run_single_model_system(
    samples: int,
    dataset_type: str,
    args: Optional[argparse.Namespace],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Contains the original logic for running the system with a single model.
    Minimal changes: fixed imports, guarded paths, consistent few-shot handling.
    """
    print(f"\n=== Initializing SymRAG System in SINGLE-MODEL MODE for Dataset: {dataset_type.upper()} ===")

    # Configure library logging levels
    for lib_name in ['transformers', 'sentence_transformers', 'urllib3.connectionpool', 'h5py', 'numexpr', 'spacy']:
        logging.getLogger(lib_name).setLevel(logging.WARNING)

    # Configure project-specific logger levels
    log_level = logging.DEBUG if args and args.debug else logging.INFO
    for logger_name in [
        'src.utils.dimension_manager', 'src.integrators.hybrid_integrator',
        'src.reasoners.networkx_symbolic_reasoner_base', 'src.reasoners.networkx_symbolic_reasoner_drop',
        'src.reasoners.neural_retriever', 'src.system.system_control_manager',
        'src.system.response_aggregator', 'src.system.system_logic_helpers'
    ]:
        logging.getLogger(logger_name).setLevel(log_level)

    # Use the pre-loaded config
    model_name = config.get("model_name")
    if not model_name:
        logger.error("model_name not found in configuration.")
        return {"error": "model_name missing from configuration."}

    # 2. DeviceManager
    device = DeviceManager.get_device()
    print(f"Using device: {device}")

    # 3. ResourceManager
    print("Initializing Resource Manager...")
    resource_config_path = os.path.join(os.path.dirname(__file__), "src", "config", "resource_config.yaml")
    resource_manager = ResourceManager(
        config_path=resource_config_path,
        enable_performance_tracking=True,
        history_window_size=100
    )

    # 4. DimensionalityManager
    print("Initializing Dimensionality Manager...")
    alignment_config = config.get('alignment', {})
    target_dim = alignment_config.get('target_dim', 768)
    dimensionality_manager = DimensionalityManager(target_dim=target_dim, device=device)

    # 5. Load Dataset and Set up Paths
    print(f"Loading evaluation dataset: {dataset_type.upper()}...")
    test_queries: List[Dict[str, Any]] = []
    ground_truths: Dict[str, Any] = {}

    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    ds = (dataset_type or "").strip().lower()
    if ds == 'drop':
        drop_path = config.get("drop_dataset_path", os.path.join(default_data_dir, "drop_dataset_dev.json"))
        if not os.path.exists(drop_path):
            return {"error": f"DROP dataset not found at {drop_path}"}
        test_queries = load_drop_dataset(drop_path, max_samples=samples)
        for s in test_queries:
            if "query_id" in s and "answer" in s:
                ground_truths[s["query_id"]] = s["answer"]
    elif ds == 'hotpotqa':
        hotpotqa_path = config.get("hotpotqa_dataset_path", os.path.join(default_data_dir, "hotpot_dev_distractor_v1.json"))
        if not os.path.exists(hotpotqa_path):
            return {"error": f"HotpotQA dataset not found at {hotpotqa_path}"}
        test_queries = load_hotpotqa(hotpotqa_path, max_samples=samples)
        for s in test_queries:
            if "query_id" in s and "answer" in s:
                ground_truths[s["query_id"]] = s["answer"]
    else:
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    if not test_queries:
        return {"error": "No test queries loaded"}

    # 6. Dynamic Rule Extraction for DROP
    current_rules_path = ""
    if ds == 'drop':
        rule_extractor = RuleExtractor()
        questions_for_rules = [qa.get('query') for qa in test_queries if qa.get('query')]
        passages_for_rules = [qa.get('context') for qa in test_queries if qa.get('context')]
        try:
            print("Extracting DROP-specific rules dynamically...")
            dynamic_rules = rule_extractor.extract_rules_from_drop(
                drop_json_path=config.get("drop_dataset_path"),
                questions=questions_for_rules,
                passages=passages_for_rules,
                min_support=config.get('drop_rule_min_support', 5)
            )
            dynamic_rules_path = config.get("drop_rules_dynamic_file", os.path.join(default_data_dir, "rules_drop_dynamic.json"))
            os.makedirs(os.path.dirname(dynamic_rules_path), exist_ok=True)
            with open(dynamic_rules_path, "w", encoding="utf-8") as f:
                json.dump(dynamic_rules, f, indent=2)
            current_rules_path = dynamic_rules_path
            logger.info(f"Switched to use dynamically extracted rules: {current_rules_path}")
        except Exception as e:
            logger.error(f"Failed to extract dynamic DROP rules: {e}. Using fallback.", exc_info=True)
            current_rules_path = config.get("drop_rules_file", os.path.join(default_data_dir, "rules_drop.json"))
    else:
        current_rules_path = config.get("hotpotqa_rules_file", os.path.join(default_data_dir, "rules_hotpotqa.json"))

    # NEW: rules file existence guard to prevent reasoner init failure
    if not os.path.exists(current_rules_path or ""):
        current_rules_path = (
            config.get("drop_rules_file", os.path.join(default_data_dir, "rules_drop.json"))
            if ds == 'drop' else
            config.get("hotpotqa_rules_file", os.path.join(default_data_dir, "rules_hotpotqa.json"))
        )

    print(f"Loaded {len(test_queries)} test queries. Using rules from: {current_rules_path}")

    # 7. Initialize Components (Symbolic/Neural Reasoners, Integrator, SCM)
    if ds == 'drop':
        symbolic_reasoner: Union[GraphSymbolicReasonerDrop, GraphSymbolicReasoner] = GraphSymbolicReasonerDrop(
            rules_file=current_rules_path, dim_manager=dimensionality_manager
        )
    else:
        symbolic_reasoner = GraphSymbolicReasoner(
            rules_file=current_rules_path, dim_manager=dimensionality_manager
        )

    # NEW: consistent few-shot handling (prefer per-model override, fallback to global)
    few_shots_path: Optional[str] = None
    if ds == 'drop' and config.get("use_drop_few_shots"):
        model_cfg_key = next(
            (k for k, v in config.get("model_configs", {}).items() if v.get("model_name") == model_name),
            None
        )
        per_model_cfg = config.get("model_configs", {}).get(model_cfg_key or "", {})
        few_shots_path = per_model_cfg.get("few_shot_examples_path", config.get("drop_few_shot_examples_path"))
    if few_shots_path:
        logger.info(f"[Single-Model] Few-shot path resolved: {few_shots_path} | exists={os.path.exists(few_shots_path)}")
    else:
        logger.info("[Single-Model] Few-shot path not set (disabled or missing).")

    neural_retriever = NeuralRetriever(
        model_name,
        use_quantization=config.get('neural_use_quantization', True),
        device=device,
        few_shot_examples_path=few_shots_path
    )

    # Query expander (optional complexity config)
    complexity_cfg_path = os.path.join(os.path.dirname(__file__), "src", "config", "complexity_rules.yaml")
    query_expander = QueryExpander(complexity_config=complexity_cfg_path if os.path.exists(complexity_cfg_path) else None)

    evaluator = Evaluation(dataset_type=dataset_type)

    hybrid_integrator = HybridIntegrator(
        symbolic_reasoner=symbolic_reasoner,
        neural_retriever=neural_retriever,
        query_expander=query_expander,
        dim_manager=dimensionality_manager,
        dataset_type=dataset_type
    )

    response_aggregator = UnifiedResponseAggregator(include_explanations=True)

    # NEW: guard log_dir when args is None
    log_dir = args.log_dir if args and getattr(args, "log_dir", None) else "logs"
    metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=os.path.join(log_dir, dataset_type, "metrics_collection")
    )

    system_manager = SystemControlManager(
        hybrid_integrator=hybrid_integrator,
        resource_manager=resource_manager,
        aggregator=response_aggregator,
        metrics_collector=metrics_collector,
        error_retry_limit=config.get('error_retry_limit', 2),
        max_query_time=config.get('max_query_time', 30.0)
    )

    sample_debugger = SampleDebugger(num_samples_to_print=2)
    sample_debugger.select_random_query_ids([q.get("query_id") for q in test_queries if q.get("query_id")])

    print(f"\n=== Testing System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")
    results_list: List[Dict[str, Any]] = []

    # Build iterator (respect --show-progress)
    query_iterator = tqdm(
        test_queries,
        desc="Processing Queries",
        unit="query",
        disable=not ProgressManager.SHOW_PROGRESS
    )

    # 8. Main Query Processing Loop (single-model mode)
    for q_info in query_iterator:
        query_id_val: Optional[str] = q_info.get("query_id")
        query_text_val: Optional[str] = q_info.get("query")
        context_val: str = q_info.get("context", "")

        if not query_id_val or not query_text_val:
            logger.warning(f"Skipping query due to missing ID or text: {q_info}")
            continue

        start_t = time.time()
        try:
            # Run the full pipeline with fallback logic
            response_obj, reasoning_path = system_manager.process_query_with_fallback(
                query=query_text_val,
                context=context_val,
                query_id=query_id_val,
                dataset_type=dataset_type
            )
            elapsed = time.time() - start_t

            # Extract predicted answer payload (DROP: structured; Hotpot: string/structured)
            prediction_payload = response_obj.get("result") if isinstance(response_obj, dict) else None

            # Optional debug sampling (mirrors your current behavior)
            sample_debugger.print_debug_if_selected(
                query_id=query_id_val,
                query_text=query_text_val,
                ground_truth_answer=ground_truths.get(query_id_val),
                system_prediction_value=prediction_payload,
                actual_reasoning_path=reasoning_path,
                eval_metrics=None,
                dataset_type=args.dataset
            )

            # Evaluate per-query (guarded)
            if prediction_payload is not None and query_id_val in ground_truths:
                try:
                    eval_metrics = evaluator.evaluate(
                        predictions={query_id_val: prediction_payload},
                        ground_truths={query_id_val: ground_truths[query_id_val]}
                    )
                    # Lightweight, readable per-item print (keeps your current logging style)
                    print("\n" + "-" * 10 + f" Results for QID: {query_id_val} " + "-" * 10)
                    print(f"  Reasoning Path: {reasoning_path}")
                    print(f"  Overall Processing Time: {elapsed:.3f}s")
                    if isinstance(prediction_payload, dict):
                        print(f"  Prediction: {prediction_payload}")
                    else:
                        print(f"  Prediction: {str(prediction_payload)[:300]}")
                    print("  Evaluation Metrics:")
                    print(f"    Exact Match: {eval_metrics.get('average_exact_match', 0.0):.3f}")
                    print(f"    F1 Score: {eval_metrics.get('average_f1', 0.0):.3f}")
                    if dataset_type.lower() != 'drop':
                        print(f"    ROUGE-L: {eval_metrics.get('average_rougeL', 0.0):.3f}")
                except Exception as ev_e:
                    logger.warning(f"Evaluation failed for QID {query_id_val}: {ev_e}")
            else:
                print("\n" + "-" * 10 + f" Results for QID: {query_id_val} " + "-" * 10)
                print(f"  Reasoning Path: {reasoning_path}")
                print(f"  Overall Processing Time: {elapsed:.3f}s")
                print("  Prediction: <none or empty>")
                print("  Evaluation skipped (missing prediction or ground truth).")

            results_list.append({
                "query_id": query_id_val,
                "reasoning_path": reasoning_path,
                "prediction": prediction_payload,
                "elapsed": elapsed
            })

        except Exception as loop_e:
            logger.exception(f"Critical error while processing QID {query_id_val}: {loop_e}")

    # 9. Final Report / Summary (guarded; uses components if present)
    print("\n" + "=" * 20 + " System Performance Summary (from SystemControlManager) " + "=" * 20)
    try:
        # If the SCM prints/aggregates internally, this ensures final write-out
        if hasattr(system_manager, "finalize_run") and callable(system_manager.finalize_run):
            system_manager.finalize_run()
    except Exception as sum_e:
        logger.warning(f"finalize_run failed: {sum_e}")

    try:
        # MetricsCollector optional finalization (safe if implemented)
        if hasattr(metrics_collector, "finalize") and callable(metrics_collector.finalize):
            metrics_collector.finalize()
    except Exception as mc_e:
        logger.warning(f"MetricsCollector.finalize failed: {mc_e}")

    return {"status": "Single-model run completed.", "num_queries": len(results_list)}


def run_ensemble_system(
    samples: int,
    dataset_type: str,
    args: Optional[argparse.Namespace],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Initializes and runs the system in sequential multi-model ensemble mode.
    Supports a batched path (load-once-per-model, process all queries) when available.
    """
    print(f"\n=== Initializing SymRAG System in ENSEMBLE MODE for Dataset: {dataset_type.upper()} ===")

    # 1) Initialize Shared Components (device, managers)
    device = DeviceManager.get_device()
    resource_manager = ResourceManager(
        config_path=os.path.join(os.path.dirname(__file__), "src", "config", "resource_config.yaml")
    )
    dimensionality_manager = DimensionalityManager(
        target_dim=config.get('alignment', {}).get('target_dim', 768),
        device=device
    )
    response_aggregator = UnifiedResponseAggregator(include_explanations=True)

    complexity_config_path = os.path.join(os.path.dirname(__file__), "src", "config", "complexity_rules.yaml")
    query_expander = QueryExpander(
        complexity_config=complexity_config_path if os.path.exists(complexity_config_path) else None
    )
    evaluator = Evaluation(dataset_type=dataset_type)

    # 2) Load Dataset
    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    ds = (dataset_type or "").strip().lower()
    if ds == 'drop':
        drop_path = config.get("drop_dataset_path", os.path.join(default_data_dir, "drop_dataset_dev.json"))
        if not os.path.exists(drop_path):
            return {"error": f"DROP dataset not found at {drop_path}"}
        test_queries = load_drop_dataset(drop_path, max_samples=samples)
    elif ds == 'hotpotqa':
        hotpotqa_path = config.get("hotpotqa_dataset_path", os.path.join(default_data_dir, "hotpot_dev_distractor_v1.json"))
        if not os.path.exists(hotpotqa_path):
            return {"error": f"HotpotQA dataset not found at {hotpotqa_path}"}
        test_queries = load_hotpotqa(hotpotqa_path, max_samples=samples)
    else:
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    ground_truths = {
        q['query_id']: q['answer'] for q in test_queries if 'query_id' in q and 'answer' in q
    }

    # 3) Initialize EnsembleManager with shared components
    ensemble_manager = EnsembleManager(
        config=config,
        resource_manager=resource_manager,
        dim_manager=dimensionality_manager,
        query_expander=query_expander,
        response_aggregator=response_aggregator
    )

    # 4) Main Ensemble Processing
    print(f"\n=== Testing Ensemble System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")

    # Attempt batched model-loading path when the manager provides it; fallback to per-query
    results_list: List[Dict[str, Any]] = []
    fused_em_total = 0.0
    fused_f1_total = 0.0
    fused_count = 0

    # Prepare sequences for the batched path
    queries_seq: List[str] = [q.get("query", "") for q in test_queries]
    contexts_seq: List[str] = [q.get("context", "") for q in test_queries]
    ids_seq: List[str] = [q.get("query_id", "") for q in test_queries]

    if hasattr(ensemble_manager, "process_dataset_ensemble_batched") and callable(getattr(ensemble_manager, "process_dataset_ensemble_batched")):
        print("Running ENSEMBLE in **BATCHED** mode (load-once-per-model), as requested.")
        # Run once per model over all queries; returns fused results for each qid
        start_all = time.time()
        try:
            fused_results_per_qid: Dict[str, Dict[str, Any]] = ensemble_manager.process_dataset_ensemble_batched(
                queries=queries_seq,
                contexts=contexts_seq,
                query_ids=ids_seq,
                dataset_type=ds
            )
        except Exception as e:
            logger.exception(f"Batched ensemble execution failed; falling back to per-query path: {e}")
            fused_results_per_qid = {}

        # If batched failed (empty), fall back to per-query
        if not fused_results_per_qid:
            logger.warning("Batched path did not return results; switching to per-query ensemble execution.")
            fused_results_per_qid = {}

            query_iterator = tqdm(
                test_queries,
                desc="Processing Queries (Ensemble)",
                unit="query",
                disable=not ProgressManager.SHOW_PROGRESS
            )

            for q_info in query_iterator:
                qid = q_info.get("query_id")
                qtext = q_info.get("query")
                ctxt = q_info.get("context", "")
                if not qid or not qtext:
                    logger.warning(f"Skipping query due to missing ID or text: {q_info}")
                    continue

                start_time = time.time()
                try:
                    fused_result_obj, reasoning_path = ensemble_manager.process_query_ensemble_sequential(
                        query=qtext,
                        context=ctxt,
                        query_id=qid,
                        dataset_type=dataset_type,
                        supporting_facts=q_info.get("supporting_facts")
                    )
                    processing_time = time.time() - start_time

                    fused_results_per_qid[qid] = fused_result_obj

                    print("\n" + "-" * 10 + f" Fused Result for QID: {qid} " + "-" * 10)
                    print(f"  Final Fused Answer: {fused_result_obj.get('answer')}")
                    print(f"  Fusion Strategy: {fused_result_obj.get('fusion_type')}")
                    if fused_result_obj.get('routing_reason'):
                        print(f"  Routing Reason: {fused_result_obj.get('routing_reason')}")
                    print(f"  Final Confidence: {fused_result_obj.get('confidence', 0.0):.3f}")
                    print(f"  Total Ensemble Time: {processing_time:.3f}s")
                    print("-" * (30 + len(str(qid))))

                    # Evaluation
                    gt_answer = ground_truths.get(qid)
                    if gt_answer and fused_result_obj.get('status') == 'success':
                        try:
                            eval_metrics = evaluator.evaluate(
                                predictions={qid: fused_result_obj.get('answer')},
                                ground_truths={qid: gt_answer}
                            )
                            em = float(eval_metrics.get('average_exact_match', 0.0))
                            f1 = float(eval_metrics.get('average_f1', 0.0))
                            fused_em_total += em
                            fused_f1_total += f1
                            fused_count += 1

                            print("  Evaluation Metrics (Fused):")
                            print(f"    Exact Match: {em:.3f}")
                            print(f"    F1 Score: {f1:.3f}")
                            if ds != 'drop':
                                print(f"    ROUGE-L: {float(eval_metrics.get('average_rougeL', 0.0)):.3f}")
                        except Exception as ev_e:
                            logger.warning(f"Evaluation failed for fused QID {qid}: {ev_e}")
                    elif not gt_answer:
                        logger.warning(f"No ground truth for QID {qid} to evaluate fused result.")

                    results_list.append({
                        "query_id": qid,
                        "fused_result": fused_result_obj,
                        "elapsed": processing_time
                    })

                except Exception as e:
                    logger.exception(f"Critical error in ensemble processing loop for query {qid}: {e}")
        else:
            # Batched path succeeded: iterate in natural order and print/evaluate
            total_elapsed = time.time() - start_all
            for q_info in test_queries:
                qid = q_info.get("query_id")
                fused_result_obj = fused_results_per_qid.get(qid, {})
                fused_answer = fused_result_obj.get('answer')

                print("\n" + "-" * 10 + f" Fused Result for QID: {qid} " + "-" * 10)
                print(f"  Final Fused Answer: {fused_answer}")
                print(f"  Fusion Strategy: {fused_result_obj.get('fusion_type')}")
                if fused_result_obj.get('routing_reason'):
                    print(f"  Routing Reason: {fused_result_obj.get('routing_reason')}")
                print(f"  Final Confidence: {fused_result_obj.get('confidence', 0.0):.3f}")
                # Per-item elapsed isn't tracked in batched method; print overall time once at the end
                print("-" * (30 + len(str(qid))))

                # Evaluate fused result
                gt_answer = ground_truths.get(qid)
                if gt_answer and fused_result_obj.get('status') == 'success':
                    try:
                        eval_metrics = evaluator.evaluate(
                            predictions={qid: fused_answer},
                            ground_truths={qid: gt_answer}
                        )
                        em = float(eval_metrics.get('average_exact_match', 0.0))
                        f1 = float(eval_metrics.get('average_f1', 0.0))
                        fused_em_total += em
                        fused_f1_total += f1
                        fused_count += 1

                        print("  Evaluation Metrics (Fused):")
                        print(f"    Exact Match: {em:.3f}")
                        print(f"    F1 Score: {f1:.3f}")
                        if ds != 'drop':
                            print(f"    ROUGE-L: {float(eval_metrics.get('average_rougeL', 0.0)):.3f}")
                    except Exception as ev_e:
                        logger.warning(f"Evaluation failed for fused QID {qid}: {ev_e}")
                elif not gt_answer:
                    logger.warning(f"No ground truth for QID {qid} to evaluate fused result.")

                # Store a minimal result record (no per-query elapsed in batched mode)
                results_list.append({
                    "query_id": qid,
                    "fused_result": fused_result_obj,
                    "elapsed": None
                })

            # Print a single consolidated timing note for batched mode
            print(f"\n[Batched ensemble completed in {total_elapsed:.3f}s for {len(test_queries)} queries across all models]")

    else:
        # No batched method available -> standard per-query sequential ensemble
        print("Running ENSEMBLE in per-query sequential mode (fallback: no batched method available).")
        query_iterator = tqdm(
            test_queries,
            desc="Processing Queries (Ensemble)",
            unit="query",
            disable=not ProgressManager.SHOW_PROGRESS
        )

        for q_info in query_iterator:
            query_id_val = q_info.get("query_id")
            query_text_val = q_info.get("query")
            local_context_val = q_info.get("context", "")

            if not query_id_val or not query_text_val:
                logger.warning(f"Skipping query due to missing ID or text: {q_info}")
                continue

            start_time = time.time()
            try:
                fused_result_obj, reasoning_path = ensemble_manager.process_query_ensemble_sequential(
                    query=query_text_val,
                    context=local_context_val,
                    query_id=query_id_val,
                    dataset_type=dataset_type,
                    # CHANGE: Safe kwargs passthrough if present
                    supporting_facts=q_info.get("supporting_facts")
                )
                processing_time = time.time() - start_time

                fused_answer = fused_result_obj.get('answer')
                print("\n" + "-" * 10 + f" Fused Result for QID: {query_id_val} " + "-" * 10)
                print(f"  Final Fused Answer: {fused_answer}")
                print(f"  Fusion Strategy: {fused_result_obj.get('fusion_type')}")
                if fused_result_obj.get('routing_reason'):
                    print(f"  Routing Reason: {fused_result_obj.get('routing_reason')}")
                print(f"  Final Confidence: {fused_result_obj.get('confidence', 0.0):.3f}")
                print(f"  Total Ensemble Time: {processing_time:.3f}s")
                print("-" * (30 + len(str(query_id_val))))

                # Evaluate fused result
                gt_answer = ground_truths.get(query_id_val)
                if gt_answer and fused_result_obj.get('status') == 'success':
                    try:
                        eval_metrics = evaluator.evaluate(
                            predictions={query_id_val: fused_answer},
                            ground_truths={query_id_val: gt_answer}
                        )
                        em = float(eval_metrics.get('average_exact_match', 0.0))
                        f1 = float(eval_metrics.get('average_f1', 0.0))
                        fused_em_total += em
                        fused_f1_total += f1
                        fused_count += 1

                        print("  Evaluation Metrics (Fused):")
                        print(f"    Exact Match: {em:.3f}")
                        print(f"    F1 Score: {f1:.3f}")
                        if ds != 'drop':
                            print(f"    ROUGE-L: {float(eval_metrics.get('average_rougeL', 0.0)):.3f}")
                    except Exception as ev_e:
                        logger.warning(f"Evaluation failed for fused QID {query_id_val}: {ev_e}")
                elif not gt_answer:
                    logger.warning(f"No ground truth for QID {query_id_val} to evaluate fused result.")

                results_list.append({
                    "query_id": query_id_val,
                    "fused_result": fused_result_obj,
                    "elapsed": processing_time
                })

            except Exception as e:
                logger.exception(f"Critical error in ensemble processing loop for query {query_id_val}: {e}")

    # Lightweight final fused summary
    print("\n" + "=" * 20 + " Ensemble Run Completed " + "=" * 20)
    if fused_count > 0:
        print(f"Fused Results over {fused_count} evaluated queries:")
        print(f"  Avg EM: {fused_em_total / fused_count:.3f}")
        print(f"  Avg F1: {fused_f1_total / fused_count:.3f}")

    return {"status": "Ensemble run completed.", "num_queries": len(results_list)}


def execute_ablation_study(args: argparse.Namespace):
    """Wrapper to call the ablation orchestration."""
    print(f"\n--- main.py: execute_ablation_study for Dataset: {args.dataset.upper()} ---")
    try:
        setup_and_orchestrate_ablation(args)
    except Exception as e:
        logger.error(f"Call to setup_and_orchestrate_ablation failed: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run SymRAG system with output capture and optional ablation study.'
    )
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'drop'],
                        help='Dataset to use for evaluation (hotpotqa or drop)')
    parser.add_argument('--log-dir', default='logs', help='Directory to save log files')
    parser.add_argument('--no-output-capture', action='store_true', help='Disable output capture to file')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to process (for standard run or ablation)')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG level logging for main and src components')
    parser.add_argument('--show-progress', action='store_true', help='Show tqdm progress bars')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run the ablation study instead of a standard evaluation.')
    parser.add_argument('--ablation-config', type=str, default='src/config/ablation_config.yaml',
                        help='Path to the YAML file defining ablation configurations.')
    parser.add_argument('--ablation-name', type=str, default=None,
                        help='Name of a specific ablation configuration to run. If not specified, runs all.')

    parsed_args = parser.parse_args()

    if parsed_args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('src').setLevel(logging.DEBUG)
        print("--- DEBUG Logging Enabled for main.py and 'src' package ---")
    else:
        logger.setLevel(logging.INFO)

    # Respect --show-progress flag
    ProgressManager.SHOW_PROGRESS = parsed_args.show_progress

    dataset_log_dir = os.path.join(parsed_args.log_dir, parsed_args.dataset)
    os.makedirs(dataset_log_dir, exist_ok=True)

    if parsed_args.run_ablation:
        print(f"--- Starting Ablation Study for {parsed_args.dataset.upper()} ---")
        if parsed_args.no_output_capture:
            execute_ablation_study(parsed_args)
        else:
            ablation_output_dir = os.path.join(dataset_log_dir, "ablation_run_logs")
            os.makedirs(ablation_output_dir, exist_ok=True)
            try:
                with capture_output(output_dir=ablation_output_dir) as output_path:
                    print(f"Ablation study output is being captured to: {output_path}")
                    execute_ablation_study(parsed_args)
            except Exception as e:
                print(f"ERROR: Failed to set up output capture or run ablation study: {e}", file=sys.stderr)
                logger.error(f"Failed to set up output capture or run ablation study: {e}", exc_info=True)
    else:
        # Dispatch to the correct mode (single vs ensemble) internally
        if parsed_args.no_output_capture:
            run_symrag_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
        else:
            try:
                with capture_output(output_dir=dataset_log_dir) as output_path:
                    print(f"Output from this run is being captured to: {output_path}")
                    run_symrag_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
            except Exception as e:
                print(f"ERROR: Failed to set up output capture or run system: {e}", file=sys.stderr)
                logger.error(f"Failed to set up output capture or run system: {e}", exc_info=True)
