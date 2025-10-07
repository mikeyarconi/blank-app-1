import json
from typing import Any, Dict, Optional


def parse_extraction_response(
    response_string: str,
    category_definitions: Dict[str, str],
    structured_data_definition: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(response_string)
        expected_keys = list(category_definitions.keys())
        if structured_data_definition and "output_key_for_structured" in structured_data_definition:
            expected_keys.append(structured_data_definition["output_key_for_structured"])
        for key in expected_keys:
            if key not in data:
                print(f"Warning: Expected key '{key}' not found in LLM response.")
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def parse_processed_list_response(response_string: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(response_string)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def parse_comparison_response(response_string: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(response_string)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
