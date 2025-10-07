import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st

from llm.perplexity_client import PerplexityClient
from llm.prompts import (
    build_p1_entity_extraction_prompt,
    build_p2_process_list_prompt,
    build_p3_compare_sets_prompt,
)
from llm.parsers import (
    parse_extraction_response,
    parse_processed_list_response,
    parse_comparison_response,
)


st.set_page_config(page_title="OPF Prompt Runner", page_icon="🧩", layout="wide")
st.title("🧩 Official Prompt Framework (OPF) — Prompt Runner")


@st.cache_data
def get_default_category_definitions() -> Dict[str, str]:
    return {
        "Unique_Subjects": "Distinct subjects or primary topics mentioned.",
        "Unique_Elements": "Specific components or individual items identified.",
    }


def ensure_client() -> Optional[PerplexityClient]:
    try:
        return PerplexityClient()
    except Exception as e:
        st.warning(
            "Perplexity client is not configured. Set env var PERPLEXITY_API_KEY to enable API calls."
        )
        st.caption(str(e))
        return None


with st.sidebar:
    st.header("Perplexity Settings")
    pplx_key = st.text_input(
        "PERPLEXITY_API_KEY",
        value=os.getenv("PERPLEXITY_API_KEY", ""),
        type="password",
        help="Not stored. Used only for this session.",
    )
    model = st.text_input("Model", value="sonar-pro")
    api_url = st.text_input(
        "API URL", value="https://api.perplexity.ai/chat/completions"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_p = st.slider("top_p", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.number_input("max_tokens (optional)", min_value=0, value=0)

    enable_calls = st.checkbox("Enable API calls", value=False)


tab1, tab2, tab3 = st.tabs([
    "P1 — Entity Extraction",
    "P2 — Process List",
    "P3 — Compare Sets",
])


with tab1:
    st.subheader("P1 — Entity Extraction Plaintext")
    input_text_variable_name = st.text_input(
        "Input text variable name", value="input_text"
    )
    input_text = st.text_area("Input text", height=180)

    input_metadata_variable_name = st.text_input(
        "Input metadata variable name (optional)", value="input_metadata"
    )
    input_metadata = st.text_area("Input metadata (optional)", height=100)

    category_definitions_variable_name = st.text_input(
        "Category definitions variable name", value="category_definitions"
    )
    category_defs_default = json.dumps(get_default_category_definitions())
    category_definitions_json_text = st.text_area(
        "Category definitions JSON", value=category_defs_default, height=160
    )

    structured_data_definition_variable_name = st.text_input(
        "Structured data definition variable name (optional)",
        value="structured_data_definition",
    )
    structured_data_definition_json_text = st.text_area(
        "Structured data definition JSON (optional)", value="", height=140
    )

    gen_p1 = st.button("Build P1 prompt")
    if gen_p1:
        try:
            category_definitions_json = json.loads(category_definitions_json_text or "{}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid category definitions JSON: {e}")
            category_definitions_json = {}

        structured_def_json = None
        if structured_data_definition_json_text.strip():
            try:
                structured_def_json = json.loads(structured_data_definition_json_text)
            except json.JSONDecodeError as e:
                st.error(f"Invalid structured definition JSON: {e}")

        prompt_text = build_p1_entity_extraction_prompt(
            input_text_variable_name=input_text_variable_name,
            input_text=input_text,
            category_definitions_variable_name=category_definitions_variable_name,
            category_definitions_json=category_definitions_json,
            input_metadata_variable_name=(
                input_metadata_variable_name or None
            ),
            input_metadata=(input_metadata or None),
            structured_data_definition_variable_name=(
                structured_data_definition_variable_name or None
            ),
            structured_data_definition_json=structured_def_json,
        )
        st.code(prompt_text)

    st.divider()
    st.caption("Call Perplexity (expects JSON-only output)")
    call_p1 = st.button("Run P1 via Perplexity")
    if call_p1 and enable_calls:
        os.environ["PERPLEXITY_API_KEY"] = pplx_key
        try:
            category_definitions_json = json.loads(category_definitions_json_text or "{}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid category definitions JSON: {e}")
            category_definitions_json = {}
        structured_def_json = None
        if structured_data_definition_json_text.strip():
            try:
                structured_def_json = json.loads(structured_data_definition_json_text)
            except json.JSONDecodeError as e:
                st.error(f"Invalid structured definition JSON: {e}")

        prompt_text = build_p1_entity_extraction_prompt(
            input_text_variable_name=input_text_variable_name,
            input_text=input_text,
            category_definitions_variable_name=category_definitions_variable_name,
            category_definitions_json=category_definitions_json,
            input_metadata_variable_name=(
                input_metadata_variable_name or None
            ),
            input_metadata=(input_metadata or None),
            structured_data_definition_variable_name=(
                structured_data_definition_variable_name or None
            ),
            structured_data_definition_json=structured_def_json,
        )
        client = PerplexityClient(api_key=pplx_key, api_url=api_url, model=model)
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns JSON only."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or None,
            response_format={"type": "json_object"},
        )
        text = client.extract_text(response)
        st.text_area("Raw response", value=text, height=200)
        parsed = parse_extraction_response(
            text, category_definitions=category_definitions_json, structured_data_definition=structured_def_json
        )
        st.json(parsed or {})


with tab2:
    st.subheader("P2 — Process Extracted List")
    list_item_category_name = st.text_input(
        "List item category name", value="Unique_Subjects"
    )
    input_list_variable_name = st.text_input("Input list variable name", value="input_list")
    input_list_text = st.text_area(
        "Input list JSON (e.g. [\"A\", \"B\"])", value="[]", height=120
    )
    processing_task_description = st.text_area(
        "Processing task description", value="Describe each item in one sentence.", height=100
    )
    output_format_description = st.text_area(
        "Output format description",
        value='{"described_items":[{"item": "...","description":"..."}]}',
        height=100,
    )
    gen_p2 = st.button("Build P2 prompt")
    if gen_p2:
        try:
            input_list = json.loads(input_list_text)
            prompt_text = build_p2_process_list_prompt(
                input_list_variable_name=input_list_variable_name,
                input_list=input_list,
                list_item_category_name=list_item_category_name,
                processing_task_description=processing_task_description,
                output_format_description=output_format_description,
            )
            st.code(prompt_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid input list JSON: {e}")

    st.divider()
    call_p2 = st.button("Run P2 via Perplexity")
    if call_p2 and enable_calls:
        os.environ["PERPLEXITY_API_KEY"] = pplx_key
        try:
            input_list = json.loads(input_list_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid input list JSON: {e}")
            input_list = []

        prompt_text = build_p2_process_list_prompt(
            input_list_variable_name=input_list_variable_name,
            input_list=input_list,
            list_item_category_name=list_item_category_name,
            processing_task_description=processing_task_description,
            output_format_description=output_format_description,
        )
        client = PerplexityClient(api_key=pplx_key, api_url=api_url, model=model)
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns JSON only."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or None,
            response_format={"type": "json_object"},
        )
        text = client.extract_text(response)
        st.text_area("Raw response", value=text, height=200)
        parsed = parse_processed_list_response(text)
        st.json(parsed or {})


with tab3:
    st.subheader("P3 — Compare Entity Sets")
    entity_set_A_variable_name = st.text_input(
        "Entity set A variable name", value="entity_set_A"
    )
    entity_set_A_text = st.text_area("Entity set A JSON", value="{}", height=140)
    entity_set_B_variable_name = st.text_input(
        "Entity set B variable name", value="entity_set_B"
    )
    entity_set_B_text = st.text_area("Entity set B JSON", value="{}", height=140)
    categories_to_compare_variable_name = st.text_input(
        "Categories to compare variable name", value="categories_to_compare"
    )
    categories_to_compare_text = st.text_input(
        "Categories to compare JSON list", value='["Unique_Subjects", "Unique_Elements"]'
    )
    comparison_criteria_variable_name = st.text_input(
        "Comparison criteria variable name", value="comparison_criteria"
    )
    comparison_criteria = st.text_area(
        "Comparison criteria",
        value=(
            "For each category, identify common items, items only in A, items only in B, "
            "and compute a Jaccard similarity index if applicable."
        ),
        height=120,
    )
    output_format_description_comparison = st.text_area(
        "Output format description",
        value=(
            '{"<category>": {"common_items": [], "items_only_in_A": [], "items_only_in_B": [], "similarity_jaccard_index": 0.0}}'
        ),
        height=120,
    )
    gen_p3 = st.button("Build P3 prompt")
    if gen_p3:
        try:
            entity_set_A = json.loads(entity_set_A_text or "{}")
            entity_set_B = json.loads(entity_set_B_text or "{}")
            categories_to_compare = json.loads(categories_to_compare_text or "[]")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            prompt_text = build_p3_compare_sets_prompt(
                entity_set_A_variable_name=entity_set_A_variable_name,
                entity_set_A=entity_set_A,
                entity_set_B_variable_name=entity_set_B_variable_name,
                entity_set_B=entity_set_B,
                categories_to_compare_variable_name=categories_to_compare_variable_name,
                categories_to_compare=categories_to_compare,
                comparison_criteria_variable_name=comparison_criteria_variable_name,
                comparison_criteria=comparison_criteria,
                output_format_description_comparison=output_format_description_comparison,
            )
            st.code(prompt_text)

    st.divider()
    call_p3 = st.button("Run P3 via Perplexity")
    if call_p3 and enable_calls:
        os.environ["PERPLEXITY_API_KEY"] = pplx_key
        try:
            entity_set_A = json.loads(entity_set_A_text or "{}")
            entity_set_B = json.loads(entity_set_B_text or "{}")
            categories_to_compare = json.loads(categories_to_compare_text or "[]")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            prompt_text = build_p3_compare_sets_prompt(
                entity_set_A_variable_name=entity_set_A_variable_name,
                entity_set_A=entity_set_A,
                entity_set_B_variable_name=entity_set_B_variable_name,
                entity_set_B=entity_set_B,
                categories_to_compare_variable_name=categories_to_compare_variable_name,
                categories_to_compare=categories_to_compare,
                comparison_criteria_variable_name=comparison_criteria_variable_name,
                comparison_criteria=comparison_criteria,
                output_format_description_comparison=output_format_description_comparison,
            )
            client = PerplexityClient(api_key=pplx_key, api_url=api_url, model=model)
            response = client.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that returns JSON only."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or None,
                response_format={"type": "json_object"},
            )
            text = PerplexityClient.extract_text(response)
            st.text_area("Raw response", value=text, height=200)
            parsed = parse_comparison_response(text)
            st.json(parsed or {})

