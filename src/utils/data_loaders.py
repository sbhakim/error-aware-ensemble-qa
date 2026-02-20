# src/utils/data_loaders.py

import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def load_hotpotqa(hotpotqa_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a portion of the HotpotQA dataset.
    Each sample includes a query, ground-truth answer,
    combined context, and a 'type' = 'ground_truth_available_hotpotqa'.
    """
    if not os.path.exists(hotpotqa_path):
        logger.error(f"HotpotQA dataset file not found at: {hotpotqa_path}")
        return []
    dataset: List[Dict[str, Any]] = []
    try:
        with open(hotpotqa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse HotpotQA JSON from {hotpotqa_path}: {e}")
        return []

    count = 0
    for example in data:
        if not all(k in example for k in ['question', 'answer', 'supporting_facts', 'context', '_id']):
            logger.warning(f"Skipping invalid HotpotQA example (missing keys): {example.get('_id', 'Unknown ID')}")
            continue

        question = example['question']
        answer = example['answer']
        supporting_facts = example['supporting_facts']

        # Create set of (title, sent_idx) tuples for quick lookup
        supporting_set = set((title, idx) for title, idx in supporting_facts)

        # Separate supporting and non-supporting documents
        supporting_docs = []
        other_docs = []

        for title, sents in example.get('context', []):
            if not isinstance(title, str) or not isinstance(sents, list):
                continue

            # Check if this document has any supporting sentences
            has_supporting = any((title, i) in supporting_set for i in range(len(sents)))

            # Format sentences, marking supporting ones
            formatted_sents = []
            for i, sent in enumerate(sents):
                if (title, i) in supporting_set:
                    formatted_sents.append(f"[RELEVANT] {sent}")
                else:
                    formatted_sents.append(str(sent))

            doc_str = f"{title}: {' '.join(formatted_sents)}"

            if has_supporting:
                supporting_docs.append(doc_str)
            else:
                other_docs.append(doc_str)

        # Put supporting documents first, then others
        context_str_parts = supporting_docs + other_docs
        context_str = "\n\n".join(context_str_parts)

        dataset.append({
            "query_id": example['_id'],
            "query": question,
            "answer": answer,
            "context": context_str,
            "type": "ground_truth_available_hotpotqa",
            "supporting_facts": supporting_facts
        })
        count += 1
        if max_samples and count >= max_samples:
            logger.info(f"Loaded {count} HotpotQA samples (max requested: {max_samples}).")
            break
    logger.info(f"Finished loading HotpotQA. Total samples: {len(dataset)}.")
    return dataset


def load_drop_dataset(drop_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a portion of the DROP dataset.
    """
    if not os.path.exists(drop_path):
        logger.error(f"DROP dataset file not found at: {drop_path}")
        return []
    dataset: List[Dict[str, Any]] = []
    try:
        with open(drop_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse DROP JSON from {drop_path}: {e}")
        return []

    total_loaded_qas = 0
    for passage_id, passage_content in data.items():
        if not isinstance(passage_content,
                          dict) or 'passage' not in passage_content or 'qa_pairs' not in passage_content:
            logger.warning(f"Skipping invalid passage structure for passage_id: {passage_id}")
            continue

        passage_text = passage_content['passage']
        if not isinstance(passage_content['qa_pairs'], list):
            logger.warning(f"Invalid qa_pairs format (not a list) for passage_id: {passage_id}")
            continue

        for qa_pair_idx, qa_pair in enumerate(passage_content['qa_pairs']):
            if not isinstance(qa_pair, dict) or 'question' not in qa_pair or 'answer' not in qa_pair:
                logger.warning(f"Skipping invalid qa_pair structure in passage_id {passage_id}, index {qa_pair_idx}")
                continue

            question = qa_pair['question']
            answer_obj = qa_pair['answer']
            query_id = qa_pair.get("query_id", f"{passage_id}-{qa_pair_idx}")

            dataset.append({
                "query_id": query_id,
                "query": question,
                "context": passage_text,
                "answer": answer_obj,
                "type": "ground_truth_available_drop"
            })
            total_loaded_qas += 1
            if max_samples and total_loaded_qas >= max_samples:
                logger.info(f"Loaded {total_loaded_qas} DROP samples (max requested: {max_samples}).")
                return dataset

        if max_samples and total_loaded_qas >= max_samples:
            break

    logger.info(f"Finished loading DROP. Total samples: {len(dataset)}.")
    return dataset


def load_squad_data(squad_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a portion of the SQuAD 2.0 dataset.
    Each sample includes a query, ground-truth answer (or empty for unanswerable),
    context passage, and a 'type' = 'ground_truth_available_squad'.

    SQuAD 2.0 includes both answerable and unanswerable questions.
    For unanswerable questions (is_impossible=true), the answer should be empty string.
    """
    if not os.path.exists(squad_path):
        logger.error(f"SQuAD dataset file not found at: {squad_path}")
        return []

    dataset: List[Dict[str, Any]] = []
    try:
        with open(squad_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse SQuAD JSON from {squad_path}: {e}")
        return []

    if 'data' not in data:
        logger.error(f"Invalid SQuAD format: missing 'data' key in {squad_path}")
        return []

    count = 0
    total_unanswerable = 0

    for article in data['data']:
        if 'title' not in article or 'paragraphs' not in article:
            logger.warning(f"Skipping invalid article (missing title or paragraphs)")
            continue

        title = article['title']

        for para in article['paragraphs']:
            if 'context' not in para or 'qas' not in para:
                logger.warning(f"Skipping invalid paragraph in article '{title}'")
                continue

            context = para['context']

            for qa in para['qas']:
                if 'question' not in qa or 'id' not in qa:
                    logger.warning(f"Skipping invalid QA (missing question or id)")
                    continue

                question = qa['question']
                qid = qa['id']
                is_impossible = qa.get('is_impossible', False)

                # Handle answer extraction
                if is_impossible:
                    # For unanswerable questions, answer should be empty
                    answer = ""
                    total_unanswerable += 1
                else:
                    # For answerable questions, extract the first answer text
                    answers = qa.get('answers', [])
                    if answers and len(answers) > 0:
                        answer = answers[0].get('text', '')
                    else:
                        # No answer provided but not marked impossible - treat as empty
                        answer = ""
                        logger.warning(f"Question {qid} has no answers but is_impossible=false")

                dataset.append({
                    "query_id": qid,
                    "query": question,
                    "answer": answer,
                    "context": context,
                    "type": "ground_truth_available_squad",
                    "is_impossible": is_impossible,
                    "title": title
                })
                count += 1

                if max_samples and count >= max_samples:
                    logger.info(f"Loaded {count} SQuAD samples (max requested: {max_samples}). "
                              f"Unanswerable: {total_unanswerable}, Answerable: {count - total_unanswerable}.")
                    return dataset

    logger.info(f"Finished loading SQuAD. Total samples: {len(dataset)}. "
                f"Unanswerable: {total_unanswerable}, Answerable: {len(dataset) - total_unanswerable}.")
    return dataset