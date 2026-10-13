#!/usr/bin/env python3
"""
Compute accuracy from judge_api.py output: the share of entries whose o3 verdict starts with "Yes".
Prints accuracy for all scenarios, for real and synthetic clips, and for each source.

Usage: python compute_accuracy.py <judge_results1.json> [judge_results2.json] ...
"""

import json
import re
import sys
from collections import defaultdict

REAL_SOURCES = ['MMAU', 'UnusualData']
SYNTHETIC_SOURCES = ['VEO3', 'GTA']
VERDICT_PATTERN = re.compile(r'\W*(yes|no)\b', re.IGNORECASE)


def get_verdict(entry):
    """Return True for Yes, False for No, None if the judge response has no Yes/No verdict"""
    try:
        text = entry['judge_response']['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        return None
    match = VERDICT_PATTERN.match(text or '')
    return None if match is None else match.group(1).lower() == 'yes'


def accuracy_line(label, verdicts):
    if not verdicts:
        return f"  {label:<14} {'-':>8}"
    correct = sum(1 for v in verdicts if v)
    return f"  {label:<14} {100 * correct / len(verdicts):7.2f}%  ({correct}/{len(verdicts)})"


def report(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    by_source = defaultdict(list)
    unparsed = []
    for entry in data:
        verdict = get_verdict(entry)
        if verdict is None:
            unparsed.append(entry.get('ID', 'Unknown'))
        # An entry without a Yes/No verdict counts as incorrect
        by_source[entry.get('ID', '').split('_')[0]].append(bool(verdict))

    real = [v for s in REAL_SOURCES for v in by_source.get(s, [])]
    synthetic = [v for s in SYNTHETIC_SOURCES for v in by_source.get(s, [])]

    print(f"\n{file_path}")
    print(accuracy_line("All", [v for vs in by_source.values() for v in vs]))
    print(accuracy_line("Real", real))
    print(accuracy_line("Synthetic", synthetic))
    for source in REAL_SOURCES + SYNTHETIC_SOURCES + sorted(set(by_source) - set(REAL_SOURCES + SYNTHETIC_SOURCES)):
        if by_source.get(source):
            print(accuracy_line(source, by_source[source]))
    if unparsed:
        print(f"  ⚠️  {len(unparsed)} entries had no Yes/No verdict and count as incorrect: {', '.join(unparsed[:5])}"
              + (" ..." if len(unparsed) > 5 else ""))


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_accuracy.py <judge_results1.json> [judge_results2.json] ...")
        return
    for file_path in sys.argv[1:]:
        report(file_path)


if __name__ == "__main__":
    main()
