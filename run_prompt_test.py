#!/usr/bin/env python3
"""
Prompt Variant A/B Testing Script
Tests top 3 prompt variants against baseline on DROP samples.
"""
import json
import sys
import os
import subprocess
import shutil
from datetime import datetime

# Backup and restore neural_retriever.py for testing
NEURAL_RETRIEVER_PATH = 'src/reasoners/neural_retriever.py'
BACKUP_PATH = 'src/reasoners/neural_retriever.py.backup'

def backup_neural_retriever():
    """Backup the original neural_retriever.py"""
    shutil.copy(NEURAL_RETRIEVER_PATH, BACKUP_PATH)
    print(f"✓ Backed up {NEURAL_RETRIEVER_PATH}")

def restore_neural_retriever():
    """Restore the original neural_retriever.py"""
    shutil.copy(BACKUP_PATH, NEURAL_RETRIEVER_PATH)
    print(f"✓ Restored {NEURAL_RETRIEVER_PATH}")
    os.remove(BACKUP_PATH)

def patch_neural_retriever_with_variant(variant_instructions):
    """
    Patch neural_retriever.py with variant instructions.
    This is a simple string replacement approach.
    """
    with open(NEURAL_RETRIEVER_PATH, 'r') as f:
        content = f.read()

    # Replace instructions for each type
    # Number instruction
    old_num = 'instruction = "Your task is to provide ONLY the single, final numerical answer. Do NOT include units (like \'yards\' or \'points\') unless the question explicitly asks for the unit as part of the answer. Do NOT include any reasoning, explanation, or introductory phrases. For example, if the answer is 7, respond with \'7\'."'
    new_num = f'instruction = "{variant_instructions["number"]}"'
    content = content.replace(old_num, new_num)

    # Span instruction
    old_span = 'instruction = "Your task is to provide ONLY the name(s) of the player(s), team(s), or specific entity(ies) requested. If multiple distinct entities are requested by the question, separate them with a comma. Do NOT provide explanations or introductory phrases. The answer should be a direct span from the text if possible."'
    new_span = f'instruction = "{variant_instructions["span"]}"'
    content = content.replace(old_span, new_span)

    # Date instruction
    old_date = 'instruction = "Your task is to provide ONLY the date. If the answer is a full date, format it as MM/DD/YYYY. If only a year, provide just the YYYY. If only month and year, format as MM/YYYY. Do NOT provide explanations or introductory phrases."'
    new_date = f'instruction = "{variant_instructions["date"]}"'
    content = content.replace(old_date, new_date)

    # Default instruction
    old_default = 'instruction = "Carefully analyze the question and context. Provide the single, most precise answer. This could be a number (output only the number), a short text span (like a name or specific phrase from the text), or a date (MM/DD/YYYY or YYYY). Do NOT include explanations or units unless they are explicitly part of the answer span itself."'
    new_default = f'instruction = "{variant_instructions["default"]}"'
    content = content.replace(old_default, new_default)

    with open(NEURAL_RETRIEVER_PATH, 'w') as f:
        f.write(content)

    print(f"✓ Patched neural_retriever.py with variant instructions")

def run_test(variant_name, samples=10):
    """Run main.py with the current variant"""
    print(f"\n{'='*60}")
    print(f"Running test: {variant_name} ({samples} samples)")
    print(f"{'='*60}")

    log_file = f"/tmp/prompt_test_{variant_name.replace(' ', '_').lower()}_{datetime.now().strftime('%H%M%S')}.log"

    cmd = f"CONDA_NO_PLUGINS=true conda run -n hysym python main.py --dataset drop --samples {samples} --no-output-capture 2>&1 | tee {log_file}"

    result = subprocess.run(cmd, shell=True, capture_output=False)

    # Extract metrics from log
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()

        # Find EM and F1
        for line in log_content.split('\n'):
            if 'Avg EM:' in line:
                em = float(line.split('Avg EM:')[1].strip())
            if 'Avg F1:' in line:
                f1 = float(line.split('Avg F1:')[1].strip())

        return {
            'variant': variant_name,
            'em': em,
            'f1': f1,
            'samples': samples,
            'log_file': log_file
        }
    except Exception as e:
        print(f"⚠️ Error extracting metrics: {e}")
        return {
            'variant': variant_name,
            'em': 0.0,
            'f1': 0.0,
            'samples': samples,
            'log_file': log_file,
            'error': str(e)
        }

def main():
    # Load variants
    with open('prompt_variants.json', 'r') as f:
        data = json.load(f)

    # Select top 3 variants
    variants_to_test = [
        ('Baseline', data['baseline']),
        ('Variant 5: Hybrid', data['variants'][4]),
        ('Variant 2: Grounding', data['variants'][1]),
        ('Variant 3: Structured', data['variants'][2])
    ]

    print("="*60)
    print("PROMPT VARIANT A/B TESTING")
    print("="*60)
    print(f"Variants to test: {len(variants_to_test)}")
    print(f"Samples per variant: 10")
    print(f"Estimated time: ~40-50 minutes total")
    print("="*60)

    # Backup original
    backup_neural_retriever()

    results = []

    try:
        for i, (name, variant_data) in enumerate(variants_to_test, 1):
            print(f"\n\n### Test {i}/{len(variants_to_test)}: {name} ###\n")

            # Patch with variant instructions
            if name != 'Baseline':
                patch_neural_retriever_with_variant(variant_data['instructions'])
            # Baseline uses original (already backed up)

            # Run test
            result = run_test(name, samples=10)
            results.append(result)

            # Restore for next variant
            restore_neural_retriever()
            backup_neural_retriever()

        # Final restore
        restore_neural_retriever()

        # Print results
        print("\n\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)

        for result in results:
            print(f"\n{result['variant']}:")
            print(f"  EM: {result['em']:.1%}")
            print(f"  F1: {result['f1']:.1%}")
            print(f"  Log: {result['log_file']}")

        # Find best
        best = max(results, key=lambda x: x['em'])
        print(f"\n{'='*60}")
        print(f"🏆 WINNER: {best['variant']}")
        print(f"   EM: {best['em']:.1%}, F1: {best['f1']:.1%}")
        print(f"{'='*60}")

        # Save results
        with open('prompt_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to prompt_test_results.json")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        restore_neural_retriever()
        raise

if __name__ == "__main__":
    main()
