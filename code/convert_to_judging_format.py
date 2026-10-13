#!/usr/bin/env python3
"""
Convert to Judging Format Script
Extracts the response text from model outputs (any supported provider) and pairs it with the
ground truth in the o3 judge prompt, saving a `_for_judging.json` file for judge_api.py.

Usage: python convert_to_judging_format.py <results1.json> [results2.json] ...
"""

import json
import os
import sys
from typing import Any, Dict

# Judge prompt used for the paper's results (kept verbatim, including its original wording)
JUDGE_PROMPT_TEMPLATE = """Your task is to decide whether the model-generated description captures the same core unusual or dangerous road event as the ground truth.
Focus specifically on whether the model matches the core cause, effect, and timing of the danger. Descriptions should agree on what unusual thing happened, when it happened, and why it was dangerous.
If the model omits the suddenness of the action, a major agent involved, or misrepresents who caused the danger, mark it as a mismatch.
You may ignore small differences in phrasing or object types, but do not accept answers that are vague, incomplete, or shift the core focus of the event.
First analyze, then report the final answer in either of Yes or No followed by the detailed explanation.
Examples:
Yes (if the model captures the same unusual or dangerous event)
---detailed explanation---
No (if it misses or distorts the key causual or temporal element)
---detailed explanation---

Model response: {extracted_results}

Ground truth: {ground_truth}"""


def get_response_text(api_response: Dict[str, Any]) -> str:
    """Extract the response text, detecting the provider format from the response structure"""
    if not isinstance(api_response, dict):
        return ""

    # OpenAI batch result: {"response": {"body": <chat completion>}}
    response = api_response.get('response')
    if isinstance(response, dict) and isinstance(response.get('body'), dict):
        api_response = response['body']

    # Claude batch result: {"result": {"message": <message>}}
    result = api_response.get('result')
    if isinstance(result, dict) and isinstance(result.get('message'), dict):
        api_response = result['message']

    # OpenAI chat completion (also vLLM and other OpenAI-compatible servers)
    choices = api_response.get('choices')
    if isinstance(choices, list) and choices:
        return (choices[0].get('message') or {}).get('content') or ""

    # Claude message: content is a list of blocks
    content = api_response.get('content')
    if isinstance(content, list):
        return "".join(block.get('text', '') for block in content
                       if isinstance(block, dict) and block.get('type') == 'text')

    # Gemini: {"text": ...}
    if isinstance(api_response.get('text'), str):
        return api_response['text']

    return ""


def create_judge_prompt(extracted_results: str, ground_truth: str) -> str:
    """Create the judging prompt"""
    return JUDGE_PROMPT_TEMPLATE.format(extracted_results=extracted_results, ground_truth=ground_truth)


def convert_file(file_path: str):
    """Convert one results file and save it with a `_for_judging.json` suffix"""
    print(f"📄 Processing file: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    empty_ids = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        response_text = get_response_text(entry.get('api_response', {}))
        if not response_text:
            empty_ids.append(entry.get('ID', 'Unknown'))
        entry['extracted_results'] = response_text
        # judge_api.py sends the `prompt` field to the judge
        entry['prompt'] = create_judge_prompt(response_text, entry.get('ground_truth', ''))

    print(f"✅ Converted {len(data)} entries")
    if empty_ids:
        print(f"⚠️  {len(empty_ids)} entries have no response text and will be judged on an empty answer:")
        for item_id in empty_ids[:5]:
            print(f"   - {item_id}")
        if len(empty_ids) > 5:
            print(f"   ... and {len(empty_ids) - 5} more")

    new_file_path = f"{os.path.splitext(file_path)[0]}_for_judging.json"
    with open(new_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Judging file saved: {new_file_path}")


def main():
    """Main function to process multiple files"""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_judging_format.py <file1.json> [file2.json] ...")
        print("Example: python convert_to_judging_format.py results/16_frames/gpt-4.1/16_frames_gpt-4.1.json")
        return

    for file_path in sys.argv[1:]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        convert_file(file_path)


if __name__ == "__main__":
    main()
