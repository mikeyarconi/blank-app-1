from __future__ import annotations

from typing import Any, Dict, List, Optional


# Prompt Template 1: Entity Extraction Plaintext

def build_p1_entity_extraction_prompt(
    input_text_variable_name: str,
    input_text: str,
    category_definitions_variable_name: str,
    category_definitions_json: Dict[str, str],
    input_metadata_variable_name: Optional[str] = None,
    input_metadata: Optional[str] = None,
    structured_data_definition_variable_name: Optional[str] = None,
    structured_data_definition_json: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds the P1 prompt string.

    The final output must be a single JSON object per the template instructions.
    """
    lines: List[str] = []
    lines.append("TASK: Please analyze the provided -{{input_text_variable_name}} and any accompanying -{{input_metadata_variable_name}}.\n")
    lines.append(
        "Your goal is to extract specific types of information and structure it as a JSON object.\n"
    )
    lines.append(
        "Using the definitions provided in -{{category_definitions_variable_name}}, identify and list items for each defined category. The keys in your output JSON for these lists should match the 'output_key_for_list_X' values in -{{category_definitions_variable_name}}. Each list should contain unique string values.\n"
    )
    lines.append(
        "Additionally, if -{{structured_data_definition_variable_name}} is provided, extract detailed structured information matching its definition. The key in your output JSON for this structured data should match the 'output_key_for_structured' in -{{structured_data_definition_variable_name}}. The value should be a list of objects, where each object's structure is guided by 'fields_description'.\n"
    )
    lines.append("The final output must be a single JSON object.\n\n")

    # Inject dynamic variables
    lines.append(f"-{input_text_variable_name}:\n<{input_text}>\n")

    if input_metadata_variable_name:
        meta = input_metadata or ""
        lines.append(f"-{input_metadata_variable_name}: (if applicable)\n<{meta}>\n")

    lines.append(f"-{category_definitions_variable_name}:\n<{category_definitions_json}>\n")

    if structured_data_definition_variable_name and structured_data_definition_json is not None:
        lines.append(
            f"-{structured_data_definition_variable_name}: (if applicable)\n<{structured_data_definition_json}>\n"
        )

    return "".join(lines)


# Prompt Template 2: Processing Extracted List Plaintext

def build_p2_process_list_prompt(
    input_list_variable_name: str,
    input_list: List[str],
    list_item_category_name: str,
    processing_task_description: str,
    output_format_description: str,
) -> str:
    lines: List[str] = []
    lines.append(
        f"TASK: You have received a list of items identified as -{{list_item_category_name}}:\n\n"
    )
    lines.append(f"-{input_list_variable_name}:\n<{input_list}>\n\n")
    lines.append("Your task is to perform the following operation as described in -{{processing_task_description}}:\n\n")
    lines.append(f"<{processing_task_description}>\n\n")
    lines.append("Please provide the output in the format described in -{{output_format_description}}:\n\n")
    lines.append(f"<{output_format_description}>\n")
    return "".join(lines)


# Prompt Template 3: Comparing Two Sets of Extracted Entities

def build_p3_compare_sets_prompt(
    entity_set_A_variable_name: str,
    entity_set_A: Dict[str, Any],
    entity_set_B_variable_name: str,
    entity_set_B: Dict[str, Any],
    categories_to_compare_variable_name: str,
    categories_to_compare: List[str],
    comparison_criteria_variable_name: str,
    comparison_criteria: str,
    output_format_description_comparison: str,
) -> str:
    lines: List[str] = []
    lines.append("TASK: You are provided with two sets of extracted entities:\n\n")
    lines.append(f"Set A (-{{{entity_set_A_variable_name}}}):\n<{entity_set_A}>\n\n")
    lines.append(f"Set B (-{{{entity_set_B_variable_name}}}):\n<{entity_set_B}>\n\n")
    lines.append(
        "Your objective is to compare these two sets. Focus your comparison on the categories specified in -{{categories_to_compare_variable_name}}:\n\n"
    )
    lines.append(f"<{categories_to_compare}>\n\n")
    lines.append(
        "Follow the comparison instructions outlined in -{{comparison_criteria_variable_name}}:\n\n"
    )
    lines.append(f"<{comparison_criteria}>\n\n")
    lines.append(
        "Present your comparison results in the JSON format described in -{{output_format_description_comparison}}:\n\n"
    )
    lines.append(f"<{output_format_description_comparison}>\n")
    return "".join(lines)
