import os
import json
from llm_pipeline import (
    PerplexityClient,
    render_prompt_p1_entity_extraction,
    render_prompt_p2_process_list,
    render_prompt_p3_compare_sets,
    parse_extraction_response,
    parse_processed_list_response,
    parse_comparison_response,
)


def main() -> None:
    # Example inputs
    input_text = "Paris is the capital of France. The Eiffel Tower is a landmark."
    input_metadata = "Source: demo text"
    category_defs = {
        "Unique_Subjects": "Distinct subjects or primary topics mentioned.",
        "Unique_Elements": "Specific components or individual items identified.",
    }
    structured_def = {
        "output_key_for_structured": "Detailed_Observations",
        "fields_description": {
            "item_name": "The observed entity",
            "attribute_type": "e.g., color, size, quantity",
            "extracted_value": "The specific value noted",
        },
    }

    prompt_p1 = render_prompt_p1_entity_extraction(
        input_text_variable_name="input_text",
        input_metadata_variable_name="input_metadata",
        category_definitions_variable_name="category_definitions",
        structured_data_definition_variable_name="structured_definition",
        actual_content_for_input_text=input_text,
        actual_content_for_input_metadata=input_metadata,
        actual_category_definitions_json=json.dumps(category_defs),
        actual_structured_data_definition_json=json.dumps(structured_def),
    )

    client = PerplexityClient()
    p1_response = client.complete(prompt_p1, temperature=0.0, max_tokens=1000)
    print("P1 Raw Response:\n", p1_response)

    parsed_p1 = parse_extraction_response(p1_response, category_defs, structured_def)
    if not parsed_p1:
        print("Failed to parse P1 response; exiting.")
        return

    # Use one list for P2
    input_list = parsed_p1.get("Unique_Subjects", [])
    if not input_list:
        print("No Unique_Subjects found; using Unique_Elements instead.")
        input_list = parsed_p1.get("Unique_Elements", [])

    prompt_p2 = render_prompt_p2_process_list(
        input_list_variable_name="input_list",
        list_item_category_name="Unique_Subjects",
        processing_task_description=(
            "For each item in the list, provide a one-sentence explanation."
        ),
        output_format_description=(
            "Return a JSON object with key 'described_items' containing a list of objects with 'item' and 'description'."
        ),
        actual_list_json=json.dumps(input_list),
    )

    p2_response = client.complete(prompt_p2, temperature=0.2, max_tokens=800)
    print("P2 Raw Response:\n", p2_response)
    parsed_p2 = parse_processed_list_response(p2_response)

    # P3 demonstration: compare P1 result with a mocked second set
    second_set = {
        "Unique_Subjects": ["Paris", "Louvre Museum"],
        "Unique_Elements": ["Eiffel Tower", "Seine River"],
    }

    prompt_p3 = render_prompt_p3_compare_sets(
        entity_set_A_variable_name="entity_set_A",
        entity_set_B_variable_name="entity_set_B",
        categories_to_compare_variable_name="categories_to_compare",
        comparison_criteria_variable_name="comparison_criteria",
        output_format_description_comparison="format_description",
        actual_json_A=json.dumps(parsed_p1),
        actual_json_B=json.dumps(second_set),
        actual_categories_to_compare=json.dumps(["Unique_Subjects", "Unique_Elements"]),
        actual_comparison_criteria=(
            "For each category, list common items, items only in A, items only in B, and a Jaccard similarity score."
        ),
        actual_output_format_description=(
            "Return a JSON object keyed by category with 'common_items', 'items_only_in_A', 'items_only_in_B', 'similarity_jaccard_index'."
        ),
    )

    p3_response = client.complete(prompt_p3, temperature=0.0, max_tokens=800)
    print("P3 Raw Response:\n", p3_response)
    parsed_p3 = parse_comparison_response(p3_response)

    print("\nParsed Outputs:")
    print("P1 Parsed:", json.dumps(parsed_p1, indent=2))
    print("P2 Parsed:", json.dumps(parsed_p2, indent=2) if parsed_p2 else None)
    print("P3 Parsed:", json.dumps(parsed_p3, indent=2) if parsed_p3 else None)


if __name__ == "__main__":
    main()
