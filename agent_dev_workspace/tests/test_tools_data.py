import os
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import matplotlib.figure
from pandas.testing import assert_frame_equal
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools
from tools_dtm import make_tools as make_dtm_tools


def test_tokenize_tool_known_answer_10():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    # Summary checks
    assert result["n_documents"] == 8
    assert result["total_tokens"] == 27
    assert result["avg_tokens_per_doc"] == 3.375

    # Known Answer 10 expected token lists across all 8 documents
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    # Verify storage location 1: session.dataframe["unigrams"]
    assert "unigrams" in session.dataframe.columns
    assert list(session.dataframe["unigrams"]) == expected_unigrams

    # Verify storage location 2: session.artifacts["unigrams"]
    assert "unigrams" in session.artifacts
    assert session.artifacts["unigrams"] == expected_unigrams


def test_tokenize_tool_uninitialized_dataframe():
    session = SessionState()
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    with pytest.raises(ValueError) as exc_info:
        tokenize_tool.invoke({})
    assert "session.dataframe is missing or does not contain a 'text' column" in str(exc_info.value)


def test_tokenize_tool_missing_text_column():
    session = SessionState()
    session.dataframe = pd.DataFrame({"other_col": [1, 2, 3]})
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    with pytest.raises(ValueError) as exc_info:
        tokenize_tool.invoke({})
    assert "session.dataframe is missing or does not contain a 'text' column" in str(exc_info.value)


def test_check_missing_tool_known_answer_2():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})

    # Known Answer 2: 8 documents, 1 missing at position 7, missing_pct=12.5
    assert result["n_documents"] == 8
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]
    assert np.isclose(result["missing_pct"], 12.5, atol=1e-3)


def test_check_missing_tool_various_missing_types():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["valid text", None, np.nan, "", "   ", "another valid"]
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})

    assert result["n_documents"] == 6
    assert result["missing_count"] == 4
    assert result["missing_indices"] == [1, 2, 3, 4]
    assert np.isclose(result["missing_pct"], 66.6667, atol=1e-3)


def test_check_missing_tool_empty_dataframe():
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": []})

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})

    assert result["n_documents"] == 0
    assert result["missing_count"] == 0
    assert result["missing_indices"] == []
    assert result["missing_pct"] == 0.0


def test_check_missing_tool_uninitialized_dataframe():
    session = SessionState()
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    with pytest.raises(ValueError) as exc_info:
        check_missing_tool.invoke({})
    assert "session.dataframe is missing or does not contain a 'text' column" in str(exc_info.value)


def test_check_missing_tool_missing_text_column():
    session = SessionState()
    session.dataframe = pd.DataFrame({"other_col": [1, 2, 3]})
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    with pytest.raises(ValueError) as exc_info:
        check_missing_tool.invoke({})
    assert "session.dataframe is missing or does not contain a 'text' column" in str(exc_info.value)


def test_check_duplicates_tool_known_answer_3():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_dup_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Part 1: drop=False reports 1 duplicate without changing session state
    result_check = check_dup_tool.invoke({"drop": False})
    assert result_check["initial_documents"] == 8
    assert result_check["duplicate_count"] == 1
    assert result_check["dropped_count"] == 0
    assert result_check["remaining_documents"] == 8
    assert len(session.dataframe) == 8

    # Part 2: drop=True removes both duplicate copies (keep=False) leaving 6 rows
    result_drop = check_dup_tool.invoke({"drop": True})
    assert result_drop["initial_documents"] == 8
    assert result_drop["duplicate_count"] == 1
    assert result_drop["dropped_count"] == 2
    assert result_drop["remaining_documents"] == 6
    assert len(session.dataframe) == 6

    # Verify rows 5 and 6 ("always gamma delta") were both removed
    remaining_texts = list(session.dataframe["text"])
    assert "always gamma delta" not in remaining_texts


def test_check_duplicates_tool_tokenization_unigrams_and_stale_dtm_cleanup():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    check_dup_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 1. Tokenize dataset
    tokenize_tool.invoke({})
    original_unigrams = list(session.dataframe["unigrams"])
    assert len(original_unigrams) == 8

    # 2. Build DTM
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    # Save deep copy of session.dataframe and sparse matrix copy of feature_matrix before drop=False
    df_before = session.dataframe.copy(deep=True)
    dtm_before_copy = session.feature_matrix.copy()

    # 3. Test drop=False leaves dataframe and built DTM completely unchanged
    check_dup_tool.invoke({"drop": False})

    assert_frame_equal(session.dataframe, df_before)
    assert np.array_equal(session.feature_matrix.toarray(), dtm_before_copy.toarray())

    # 4. Test drop=True updates unigrams and clears stale DTM/labels
    check_dup_tool.invoke({"drop": True})

    expected_unigrams_subset = [original_unigrams[i] for i in [0, 1, 2, 3, 4, 7]]

    # Verify unigrams equal the original rows at positions [0,1,2,3,4,7] in both storage locations
    assert list(session.dataframe["unigrams"]) == expected_unigrams_subset
    assert session.artifacts["unigrams"] == expected_unigrams_subset

    # Assert stale DTM, labels, and vectorizer cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.labels is None
    assert session.categories == []
    assert "count_vectorizer" not in session.artifacts


def test_sample_data_tool_known_answer_13():
    # Fresh session 1
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")

    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools1 = make_tools(session1)
    tokenize_tool1 = next(t for t in data_tools1 if t.name == "tokenize_tool")
    sample_tool1 = next(t for t in data_tools1 if t.name == "sample_data_tool")

    tokenize_tool1.invoke({})
    original_df1 = session1.dataframe.copy(deep=True)
    original_unigrams1 = list(session1.dataframe["unigrams"])

    # Invoke sample_data_tool with n=4, random_state=42
    result1 = sample_tool1.invoke({"n": 4, "random_state": 42})

    # Assert Known Answer 13: sampled_indices == [1, 5, 0, 7]
    assert result1["initial_documents"] == 8
    assert result1["sampled_documents"] == 4
    assert result1["sampled_indices"] == [1, 5, 0, 7]
    assert result1["random_state"] == 42

    # Verify matching row contents and unigrams for indices [1, 5, 0, 7]
    expected_texts = [original_df1.loc[i, "text"] for i in [1, 5, 0, 7]]
    expected_unigrams = [original_unigrams1[i] for i in [1, 5, 0, 7]]

    assert list(session1.dataframe["text"]) == expected_texts
    assert list(session1.dataframe["unigrams"]) == expected_unigrams
    assert session1.artifacts["unigrams"] == expected_unigrams

    # Fresh session 2 for reproducibility check
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")

    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools2 = make_tools(session2)
    tokenize_tool2 = next(t for t in data_tools2 if t.name == "tokenize_tool")
    sample_tool2 = next(t for t in data_tools2 if t.name == "sample_data_tool")

    tokenize_tool2.invoke({})
    result2 = sample_tool2.invoke({"n": 4, "random_state": 42})

    # Assert reproducibility across fresh session
    assert result2["sampled_indices"] == [1, 5, 0, 7]
    assert_frame_equal(session1.dataframe, session2.dataframe)


def test_sample_data_tool_stale_dtm_cleanup_and_invalid_n():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Build DTM
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})
    assert session.feature_matrix is not None

    df_before = session.dataframe.copy(deep=True)

    # Invalid n=0 check (should raise ValueError without modifying state)
    with pytest.raises(ValueError) as exc_info:
        sample_tool.invoke({"n": 0})
    assert "n must be a positive integer" in str(exc_info.value)
    assert_frame_equal(session.dataframe, df_before)

    # Invalid n>8 check (should raise ValueError without modifying state)
    with pytest.raises(ValueError) as exc_info:
        sample_tool.invoke({"n": 10})
    assert "exceeds total available documents" in str(exc_info.value)
    assert_frame_equal(session.dataframe, df_before)

    # Valid sampling clears stale DTM/labels
    sample_tool.invoke({"n": 3, "random_state": 42})
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.labels is None
    assert session.categories == []
    assert "count_vectorizer" not in session.artifacts


def test_describe_data_tool_known_answer_18():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_tool.invoke({})

    # 1. Check exact text_length series on session.dataframe
    expected_text_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert list(session.dataframe["text_length"]) == expected_text_lengths

    # 2. Overall stats (Known Answer 18)
    overall = result["text_length_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert np.isclose(overall["min"], 0.0, atol=1e-3)
    assert np.isclose(overall["25%"], 17.75, atol=1e-3)
    assert np.isclose(overall["50%"], 20.5, atol=1e-3)
    assert np.isclose(overall["75%"], 25.25, atol=1e-3)
    assert np.isclose(overall["max"], 30.0, atol=1e-3)

    # 3. Per-category stats (Known Answer 18)
    cat_stats = result["category_stats"]
    assert "catA" in cat_stats
    assert "catB" in cat_stats

    catA = cat_stats["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5, atol=1e-3)
    assert np.isclose(catA["min"], 0.0, atol=1e-3)
    assert np.isclose(catA["25%"], 12.75, atol=1e-3)
    assert np.isclose(catA["50%"], 20.0, atol=1e-3)
    assert np.isclose(catA["75%"], 24.5, atol=1e-3)
    assert np.isclose(catA["max"], 29.0, atol=1e-3)

    catB = cat_stats["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert np.isclose(catB["min"], 18.0, atol=1e-3)
    assert np.isclose(catB["25%"], 18.0, atol=1e-3)
    assert np.isclose(catB["50%"], 21.0, atol=1e-3)
    assert np.isclose(catB["75%"], 25.5, atol=1e-3)
    assert np.isclose(catB["max"], 30.0, atol=1e-3)

    # 4. Figure check
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)


def test_describe_data_tool_edge_cases():
    # Edge case 1: empty dataframe raises ValueError
    session1 = SessionState()
    session1.dataframe = pd.DataFrame({"text": [], "category_name": []})
    tools1 = make_tools(session1)
    describe_tool1 = next(t for t in tools1 if t.name == "describe_data_tool")

    with pytest.raises(ValueError) as exc_info:
        describe_tool1.invoke({})
    assert "session.dataframe is empty" in str(exc_info.value)

    # Edge case 2: single-row dataframe returns None for std
    session2 = SessionState()
    session2.dataframe = pd.DataFrame({"text": ["hello world"], "category_name": ["cat1"]})
    tools2 = make_tools(session2)
    describe_tool2 = next(t for t in tools2 if t.name == "describe_data_tool")

    res2 = describe_tool2.invoke({})
    assert res2["text_length_stats"]["count"] == 1
    assert res2["text_length_stats"]["std"] is None
    assert res2["category_stats"]["cat1"]["count"] == 1
    assert res2["category_stats"]["cat1"]["std"] is None


def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Known Answer 11: subdirectory="newdataset" returns directories: [], files: ["Reddit-stock-sentiment.csv"]
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_from_workspace_dir(monkeypatch):
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Explicitly determine workspace_dir and assert it exists as a directory
    cwd = Path.cwd().resolve()
    repo_root = cwd.parent if cwd.name == "agent_dev_workspace" else cwd
    workspace_dir = repo_root / "agent_dev_workspace"

    assert workspace_dir.is_dir()

    # Always change directory to workspace_dir
    monkeypatch.chdir(workspace_dir)

    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_sorting_and_error_handling():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Test alphabetical sorting on a known directory
    res = list_files_tool.invoke({"subdirectory": "."})
    assert res["directories"] == sorted(res["directories"])
    assert res["files"] == sorted(res["files"])

    # Test nonexistent subdirectory raises ValueError
    with pytest.raises(ValueError) as exc_info1:
        list_files_tool.invoke({"subdirectory": "nonexistent_folder_xyz"})
    assert "does not exist" in str(exc_info1.value)

    # Test pointing to a file instead of a directory raises ValueError
    with pytest.raises(ValueError) as exc_info2:
        list_files_tool.invoke({"subdirectory": "newdataset/Reddit-stock-sentiment.csv"})
    assert "is not a directory" in str(exc_info2.value)


def test_inspect_data_tool_functionality_and_serialization():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Tokenize to populate unigrams list column
    tokenize_tool.invoke({})

    df_before = session.dataframe.copy(deep=True)

    # 1. Default n_rows=5 inspection
    result5 = inspect_tool.invoke({"n_rows": 5})

    assert result5["total_rows"] == 8
    assert "text" in result5["columns"]
    assert result5["n_rows_inspected"] == 5
    assert len(result5["preview"]) == 5

    # Check unigrams list column is preserved and present in preview records
    assert isinstance(result5["preview"][0]["unigrams"], list)
    assert result5["preview"][0]["unigrams"] == ["always", "alpha", "alpha", "alpha", "beta"]

    # Strict JSON serialization assertion with allow_nan=False
    json.dumps(result5, allow_nan=False)

    # 2. n_rows larger than dataset (n_rows=10 on 8-row dataset)
    result10 = inspect_tool.invoke({"n_rows": 10})
    assert result10["n_rows_inspected"] == 8
    assert len(result10["preview"]) == 8

    # Row 7 in fixture is loaded as empty string "" -> verify exact string ""
    assert result10["preview"][7]["text"] == ""
    assert result10["preview"][7]["unigrams"] == []

    # Strict JSON serialization check on full 8-row preview
    json.dumps(result10, allow_nan=False)

    # 3. Verify session.dataframe is completely unchanged after inspect_data_tool calls
    assert_frame_equal(session.dataframe, df_before)


def test_inspect_data_tool_null_conversion_and_empty_df():
    session = SessionState()
    data_tools = make_tools(session)
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Test small dataframe with None, np.nan, and list containing np.nan
    session.dataframe = pd.DataFrame({
        "text": ["valid", None, np.nan],
        "category_name": ["catA", "catB", np.nan],
        "tags": [["a", "b"], [np.nan, "c"], [None]]
    })

    result = inspect_tool.invoke({"n_rows": 5})
    assert result["total_rows"] == 3
    assert result["n_rows_inspected"] == 3

    # Row 0: valid values
    assert result["preview"][0]["text"] == "valid"
    assert result["preview"][0]["tags"] == ["a", "b"]

    # Row 1: None text and list containing None
    assert result["preview"][1]["text"] is None
    assert result["preview"][1]["tags"] == [None, "c"]

    # Row 2: np.nan mapped to None
    assert result["preview"][2]["text"] is None
    assert result["preview"][2]["category_name"] is None
    assert result["preview"][2]["tags"] == [None]

    # Verify strict JSON dumps with allow_nan=False
    json.dumps(result, allow_nan=False)

    # Test empty dataframe case
    session_empty = SessionState()
    session_empty.dataframe = pd.DataFrame({"text": [], "category_name": []})
    tools_empty = make_tools(session_empty)
    inspect_empty_tool = next(t for t in tools_empty if t.name == "inspect_data_tool")

    res_empty = inspect_empty_tool.invoke({"n_rows": 5})
    assert res_empty["total_rows"] == 0
    assert res_empty["n_rows_inspected"] == 0
    assert res_empty["preview"] == []
    json.dumps(res_empty, allow_nan=False)


def test_inspect_data_tool_uninitialized_and_invalid_inputs():
    session = SessionState()
    data_tools = make_tools(session)
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Uninitialized session.dataframe
    with pytest.raises(ValueError) as exc1:
        inspect_tool.invoke({"n_rows": 5})
    assert "session.dataframe is missing or uninitialized" in str(exc1.value)

    # Invalid n_rows <= 0
    session.dataframe = pd.DataFrame({"text": ["a", "b"]})
    with pytest.raises(ValueError) as exc2:
        inspect_tool.invoke({"n_rows": 0})
    assert "n_rows must be a positive integer" in str(exc2.value)
