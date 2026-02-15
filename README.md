# Error-Aware Ensemble QA: Exploiting Model Diversity for Robust Question Answering

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A hierarchical ensemble architecture that exploits complementary failure modes across diverse language models for robust hybrid question answering. This system combines neuro-symbolic reasoning with error-aware fusion to achieve improved accuracy on structured information extraction tasks.

## 🎯 Key Features

- **Error-Aware Ensemble Fusion**: Three-stage cascade (unanimous → majority → feature-based routing) that outperforms naive voting
- **Model Diversity Exploitation**: Each model contributes unique successes; ensemble achieves 48% EM vs 42% best single model (+6pp)
- **Hybrid Neuro-Symbolic Architecture**: Adaptive routing between symbolic pattern matching and neural generation
- **Dynamic Rule Extraction**: Automatic mining of symbolic reasoning patterns from training data
- **Multi-Model Support**: Llama-3.2-3B, Mistral-7B-Instruct-v0.3, Gemma-1.1-7B-it with 8-bit quantization

## 📊 Performance Highlights

On the challenging DROP dataset (50-query validation):

| Configuration | EM (%) | Unique Contributions |
|---------------|--------|---------------------|
| Llama-3.2-3B | 36.0 | 5 queries |
| Mistral-7B | 40.0 | 7 queries |
| Gemma-1.1-7B | 42.0 | 10 queries |
| **Error-Aware Ensemble** | **48.0** | **+6pp improvement** |
| Theoretical Maximum | 76.0 | 38/50 recoverable |

**Fusion Strategy Distribution:**
- 26% queries: Unanimous agreement (all models align)
- 36% queries: Majority voting (2/3 models agree)
- 38% queries: Error-aware routing (feature-based model selection)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Ensemble Orchestration                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Llama-3B   │  │ Mistral-7B  │  │  Gemma-7B   │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                 │                 │           │
│         └─────────────────┴─────────────────┘           │
│                         ↓                                │
│              ┌─────────────────────┐                    │
│              │  Error-Aware Fusion │                    │
│              │  • Unanimous (26%)  │                    │
│              │  • Majority (36%)   │                    │
│              │  • Routing (38%)    │                    │
│              └─────────┬───────────┘                    │
└────────────────────────┼─────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │   Adaptive Control Module     │
         │   • Query complexity analysis │
         │   • Resource monitoring       │
         │   • Pathway routing           │
         └───────────┬───────────────────┘
                     ↓
     ┌───────────────┴───────────────┐
     ↓                                ↓
┌─────────────┐              ┌─────────────┐
│  Symbolic   │              │   Neural    │
│  Reasoning  │ ←──Hybrid──→ │  Retriever  │
│  • Rules    │              │  • LLM      │
│  • Graphs   │              │  • Few-shot │
└─────────────┘              └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 32GB system RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/error-aware-ensemble-qa.git
cd error-aware-ensemble-qa

# Create virtual environment
conda create -n ensemble_qa python=3.9
conda activate ensemble_qa

# Install dependencies
pip install -r requirements.txt

# Download SpaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

### Download Datasets

Due to size constraints, datasets are not included in the repository. Download them separately:

```bash
# Create data directory
mkdir -p data

# Download DROP dataset
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset_dev.json -O data/drop_dataset_dev.json

# Download HotpotQA dataset
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/hotpot_dev_distractor_v1.json
```

Small reference files (few-shot examples, rules) are included in `data/`.

### Basic Usage

**Single-Model Mode:**
```bash
python main.py --dataset drop --samples 50
```

**Ensemble Mode:**
```bash
# Modify src/config/config.yaml:
# ensemble.enabled: true

python main.py --dataset drop --samples 50 --show-progress
```

**Run Ablation Studies:**
```bash
python main.py --dataset drop --samples 50 --run-ablation
```

## 📁 Project Structure

```
error-aware-ensemble-qa/
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
│
├── src/
│   ├── config/
│   │   ├── config.yaml          # Main configuration
│   │   ├── ablation_config.yaml # Ablation study configs
│   │   └── resource_config.yaml # Resource thresholds
│   │
│   ├── system/
│   │   ├── ensemble_manager.py       # Multi-model orchestration
│   │   ├── system_control_manager.py # Adaptive routing
│   │   └── response_aggregator.py    # Response formatting
│   │
│   ├── reasoners/
│   │   ├── neural_retriever.py                # LLM inference
│   │   ├── networkx_symbolic_reasoner_base.py # Base symbolic engine
│   │   └── networkx_symbolic_reasoner_drop.py # DROP-specific logic
│   │
│   ├── integrators/
│   │   └── hybrid_integrator.py      # Symbolic-neural fusion
│   │
│   ├── utils/
│   │   ├── ensemble_helpers.py       # Error-aware fusion logic
│   │   ├── rule_extractor.py         # Dynamic rule mining
│   │   ├── evaluation.py             # Metrics computation
│   │   ├── metrics_collector.py      # Performance tracking
│   │   ├── device_manager.py         # GPU/CPU management
│   │   └── dimension_manager.py      # Embedding alignment
│   │
│   └── queries/
│       ├── query_expander.py         # Query complexity analysis
│       └── query_logger.py           # Query logging
│
├── data/                        # Small reference files only
│   ├── drop_few_shot_examples.json
│   ├── rules_drop.json
│   ├── rules_hotpotqa.json
│   └── empty_rules.json
│
├── scripts/
│   └── analyze_model_diversity.py   # Validation analysis tools
│
└── Manuscript/                  # Research paper (LaTeX)
    └── main.tex
```

## 🔧 Configuration

Key configuration options in `src/config/config.yaml`:

```yaml
# Ensemble mode
ensemble:
  enabled: true              # Set to false for single-model
  batched: true              # Process all queries per model
  models:
    - "llama-3.2-3b"
    - "mistral-7b"
    - "gemma-1.1-7b"
  fusion_strategy: "error_aware"  # Options: error_aware, confidence, majority_vote

# Model-specific settings
model_configs:
  "llama-3.2-3b":
    model_name: "meta-llama/Llama-3.2-3B"
    few_shot_examples_path: "data/drop_few_shot_examples.json"

# Feature flags
use_drop_few_shots: 1        # Enable few-shot learning for DROP
```

## 📈 Validation Results

Comprehensive validation on 50 DROP queries demonstrates:

**Model Diversity:**
- 38/50 queries solvable by at least one model (76% theoretical maximum)
- Each model contributes 5-10 unique correct answers
- Low error correlation validates complementary strengths

**Ensemble Efficiency:**
- Achieves 24/50 correct (48% EM) vs 21/50 best single (42%)
- 63.2% efficiency in recovering model diversity (24/38)
- 2 ensemble-only successes (all singles failed)
- 16 missed opportunities (room for improvement)

**Fusion Cascade Contribution:**
- Unanimous: 13/50 (26%) - All models agree
- Majority: 18/50 (36%) - 2/3 models agree
- Error-aware: 19/50 (38%) - Feature-based routing required

## 🧪 Running Experiments

**Reproduce Validation Results:**
```bash
# Step 1: Run each model individually
python main.py --dataset drop --samples 50 > logs/llama_validation.txt
# (Edit config.yaml to switch models)

# Step 2: Run ensemble
python main.py --dataset drop --samples 50 > logs/ensemble_validation.txt

# Step 3: Analyze diversity
python scripts/analyze_model_diversity.py
```

**Ablation Studies:**
```bash
# Compare fusion strategies
python main.py --dataset drop --samples 50 --run-ablation

# Test without few-shot learning
# (Set use_drop_few_shots: 0 in config.yaml)
python main.py --dataset drop --samples 50
```

## 📊 Evaluation Metrics

- **Exact Match (EM)**: Percentage of predictions matching ground truth exactly
- **F1 Score**: Token-level overlap between prediction and ground truth
- **Fusion Efficiency**: (Ensemble correct) / (Theoretical maximum)
- **Recovery Rate**: Queries where ensemble succeeds but at least one single model fails

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Confidence Calibration**: Improve model probability alignment
2. **Feature Engineering**: Add entity density, question complexity signals
3. **Memory Optimization**: Enable efficient deployment of multiple 7B models
4. **Meta-Learning**: Learn fusion weights from validation performance

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{hakim2025error,
  title={Exploiting Model Diversity Through Error-Aware Ensemble Fusion: A Hierarchical Architecture for Robust Hybrid Question Answering},
  author={Hakim, Safayat Bin and Song, Houbing Herbert},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DROP dataset: Dua et al., "DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs"
- HotpotQA dataset: Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"
- Model providers: Meta (Llama), Mistral AI (Mistral), Google (Gemma)

## 📞 Contact

For questions or collaborations:
📧 safayat [dot] b [dot] hakim [at] gmail [dot] com

---

**Status**: Research code - Validated on DROP dataset with 50-query validation set. Ensemble achieves +6pp improvement over best single model.
