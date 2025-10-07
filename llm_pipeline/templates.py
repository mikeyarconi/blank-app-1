from typing import Optional, Dict, Any, List


def render_prompt_p1_entity_extraction(
    input_text_variable_name: str,
    input_metadata_variable_name: Optional[str],
    category_definitions_variable_name: str,
    structured_data_definition_variable_name: Optional[str],
    actual_content_for_input_text: str,
    actual_content_for_input_metadata: Optional[str],
    actual_category_definitions_json: str,
    actual_structured_data_definition_json: Optional[str],
    prompt_index: int = 1,
    step_index: int = 1,
    prompt_name_suffix: str = "ExtractCategorizedEntities",
    predecessor_index: int = 0,
    connector_index: int = 0,
) -> str:
    """
    Renders the Prompt Template 1 per the provided spec. Returns the plaintext prompt.
    All "-{{var}}" prefixed placeholders are embedded as described.
    """
    header = (
        f"p_{prompt_index}_parallel_{step_index}: P_{prompt_index}_parallel_{step_index}_{prompt_name_suffix}_{predecessor_index}_{connector_index}\n\n"
        "Precedes no other prompts (or specify if it connects to a different next step)\n\n"
        "Connects to prompt index 2 (or as per your pipeline design)\n\n"
        "Can run in parallel with other prompts at step 1\n\n"
        "-------------------------------------------------------------------------------------------\n\n"
        "Prompt Notes:\n\n"
        "This prompt accepts a source text document as user input. It instructs the LLM to identify and create N separate lists of distinct items/entities based on predefined categories, and optionally, a list of structured data points found in the text.\n\n"
    )

    body = (
        "TASK: Please analyze the provided -{{input_text_variable_name}} and any accompanying -{{input_metadata_variable_name}}.\n\n"
        "Your goal is to extract specific types of information and structure it as a JSON object.\n\n"
        "Using the definitions provided in -{{category_definitions_variable_name}}, identify and list items for each defined category. The keys in your output JSON for these lists should match the 'output_key_for_list_X' values in -{{category_definitions_variable_name}}. Each list should contain unique string values.\n\n"
        "Additionally, if -{{structured_data_definition_variable_name}} is provided, extract detailed structured information matching its definition. The key in your output JSON for this structured data should match the 'output_key_for_structured' in -{{structured_data_definition_variable_name}}. The value should be a list of objects, where each object's structure is guided by 'fields_description'.\n\n"
        "The final output must be a single JSON object.\n\n"
        f"-{{input_text_variable_name}}:\n\n<{{actual_content_for_input_text_variable_name_goes_here}}>\n\n"
        f"-{{input_metadata_variable_name}}: (if applicable)\n\n<{{actual_content_for_input_metadata_variable_name_goes_here}}>\n\n"
        f"-{{category_definitions_variable_name}}:\n\n<{{actual_json_object_for_category_definitions_goes_here}}>\n\n"
        f"-{{structured_data_definition_variable_name}}: (if applicable)\n\n<{{actual_json_object_for_structured_data_definition_goes_here}}>\n\n"
    )

    # Replace variable labels with the provided names to make the prompt self-contained
    body = body.replace("{{input_text_variable_name}}", input_text_variable_name)
    body = body.replace("{{input_metadata_variable_name}}", input_metadata_variable_name or "input_metadata")
    body = body.replace("{{category_definitions_variable_name}}", category_definitions_variable_name)
    body = body.replace(
        "{{structured_data_definition_variable_name}}",
        structured_data_definition_variable_name or "structured_data_definition",
    )

    # Inject actual contents
    body = body.replace("{{actual_content_for_input_text_variable_name_goes_here}}", actual_content_for_input_text)
    body = body.replace(
        "{{actual_content_for_input_metadata_variable_name_goes_here}}",
        actual_content_for_input_metadata or "",
    )
    body = body.replace(
        "{{actual_json_object_for_category_definitions_goes_here}}",
        actual_category_definitions_json,
    )
    body = body.replace(
        "{{actual_json_object_for_structured_data_definition_goes_here}}",
        actual_structured_data_definition_json or "",
    )

    return header + body


def render_prompt_p2_process_list(
    input_list_variable_name: str,
    list_item_category_name: str,
    processing_task_description: str,
    output_format_description: str,
    actual_list_json: str,
    prompt_index: int = 2,
    step_index: int = 2,
    prompt_name_suffix: str = "ProcessExtractedList",
    predecessor_index: int = 1,
    connector_index: int = 3,
) -> str:
    header = (
        f"p_{prompt_index}_parallel_{step_index}: P_{prompt_index}_parallel_{step_index}_{prompt_name_suffix}_{predecessor_index}_{connector_index}\n\n"
        "Precedes Prompt 3 (or 0 if terminal step in this path)\n\n"
        "Connects from Prompt 1 (specifically, takes one of the lists output by P1)\n\n"
        "Can run in parallel with other prompts at step 2\n\n"
        "-------------------------------------------------------------------------------------------\n\n"
        "Prompt Notes:\n\n"
        "This prompt accepts a list of items (e.g., a list of 'Unique_Subjects' or 'Identified_Objects' extracted by a previous prompt) and performs a specified operation on them. The operation could be generating descriptions, finding relationships, classifying items, summarizing the list, etc., as defined by the TASK parameter.\n\n"
    )

    body = (
        "TASK: You have received a list of items identified as -{{list_item_category_name}}:\n\n"
        f"-{{input_list_variable_name}}:\n\n<{{actual_list_content_goes_here}}>\n\n"
        "Your task is to perform the following operation as described in -{{processing_task_description}}:\n\n"
        "<{{actual_task_description_content_goes_here}}>\n\n"
        "Please provide the output in the format described in -{{output_format_description}}:\n\n"
        "<{{actual_output_format_description_content_goes_here}}>\n\n"
    )

    body = body.replace("{{input_list_variable_name}}", input_list_variable_name)
    body = body.replace("{{list_item_category_name}}", list_item_category_name)
    body = body.replace("{{processing_task_description}}", "processing_task_description")
    body = body.replace("{{output_format_description}}", "output_format_description")

    body = body.replace("{{actual_list_content_goes_here}}", actual_list_json)
    body = body.replace("{{actual_task_description_content_goes_here}}", processing_task_description)
    body = body.replace("{{actual_output_format_description_content_goes_here}}", output_format_description)

    return header + body


def render_prompt_p3_compare_sets(
    entity_set_A_variable_name: str,
    entity_set_B_variable_name: str,
    categories_to_compare_variable_name: str,
    comparison_criteria_variable_name: str,
    output_format_description_comparison: str,
    actual_json_A: str,
    actual_json_B: str,
    actual_categories_to_compare: str,
    actual_comparison_criteria: str,
    actual_output_format_description: str,
    prompt_index: int = 3,
    step_index: int = 3,
    prompt_name_suffix: str = "CompareExtractedEntitySets",
    predecessor_indices_label: str = "1Aand1B",
    connector_index: int = 0,
) -> str:
    header = (
        f"p_{prompt_index}_parallel_{step_index}: P_{prompt_index}_parallel_{step_index}_{prompt_name_suffix}_{predecessor_indices_label}_{connector_index}\n\n"
        "Precedes no other prompts (or specify if it connects further)\n\n"
        "Connects from two previous 'ExtractCategorizedEntities' prompts (e.g., output of P1_DocAlpha and P1_DocBeta)\n\n"
        "Can run in parallel with other prompts at step 3 (or a later step depending on pipeline)\n\n"
        "-------------------------------------------------------------------------------------------\n\n"
        "Prompt Notes:\n\n"
        "This prompt accepts two sets of categorized entities (each set being a JSON object, presumably extracted from two different source documents like Source_Text_One and Source_Text_Two). The task is to compare these sets based on specified criteria, such as finding overlaps, differences, or evaluating alignment for one or more common categories.\n\n"
    )

    body = (
        "TASK: You are provided with two sets of extracted entities:\n\n"
        f"Set A (-{{entity_set_A_variable_name}}):\n\n<{{actual_json_content_for_entity_set_A_goes_here}}>\n\n"
        f"Set B (-{{entity_set_B_variable_name}}):\n\n<{{actual_json_content_for_entity_set_B_goes_here}}>\n\n"
        "Your objective is to compare these two sets. Focus your comparison on the categories specified in -{{categories_to_compare_variable_name}}:\n\n"
        "<{{actual_list_of_category_keys_goes_here}}>\n\n"
        "Follow the comparison instructions outlined in -{{comparison_criteria_variable_name}}:\n\n"
        "<{{actual_comparison_criteria_description_goes_here}}>\n\n"
        "Present your comparison results in the JSON format described in -{{output_format_description_comparison}}:\n\n"
        "<{{actual_output_format_description_for_comparison_goes_here}}>\n\n"
    )

    body = body.replace("{{entity_set_A_variable_name}}", entity_set_A_variable_name)
    body = body.replace("{{entity_set_B_variable_name}}", entity_set_B_variable_name)
    body = body.replace("{{categories_to_compare_variable_name}}", categories_to_compare_variable_name)
    body = body.replace("{{comparison_criteria_variable_name}}", comparison_criteria_variable_name)
    body = body.replace(
        "{{output_format_description_comparison}}", output_format_description_comparison
    )

    body = body.replace("{{actual_json_content_for_entity_set_A_goes_here}}", actual_json_A)
    body = body.replace("{{actual_json_content_for_entity_set_B_goes_here}}", actual_json_B)
    body = body.replace("{{actual_list_of_category_keys_goes_here}}", actual_categories_to_compare)
    body = body.replace("{{actual_comparison_criteria_description_goes_here}}", actual_comparison_criteria)
    body = body.replace(
        "{{actual_output_format_description_for_comparison_goes_here}}",
        actual_output_format_description,
    )

    return header + body
