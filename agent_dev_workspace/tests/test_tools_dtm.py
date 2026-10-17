import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_fixture():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load test fixture into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load build_dtm_tool from student workspace
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool with default settings and integer min_df=1
    result = build_dtm_tool.invoke({
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0
    })

    # Assert Known Answer 4: 8x5 matrix, 21 non-zero, 40 total elements, 47.5% sparsity
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)
    assert "result_id" in result

    # Check session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_build_dtm_tool_invalid_ngram_range():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invalid ngram_range cases that should raise ValueError
    invalid_ranges = [
        [1],          # length 1
        [1, 2, 3],    # length 3
        [0, 1],       # non-positive integer
        [2, 1],       # min > max
    ]

    for invalid_range in invalid_ranges:
        with pytest.raises(ValueError) as exc_info:
            build_dtm_tool.invoke({"ngram_range": invalid_range})
        assert "ngram_range" in str(exc_info.value)


def test_term_frequency_tool_known_answer_14():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_freq_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    # Test default top_n=20
    result = term_freq_tool.invoke({"top_n": 20})

    assert result["total_terms"] == 5
    assert result["total_word_count"] == 27

    # Known Answer 14 expected counts:
    # always: 7, gamma: 7, alpha: 6, delta: 4, beta: 3
    expected_counts = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }
    assert result["top_terms"] == expected_counts

    # Verify full_report DataFrame contains all 5 terms and order is strictly: always, gamma, alpha, delta, beta
    result_id = result["result_id"]
    report_df = session.results_store[result_id]["full_report"]
    assert list(report_df["feature"]) == ["always", "gamma", "alpha", "delta", "beta"]
    assert list(report_df["frequency"]) == [7, 7, 6, 4, 3]


def test_term_frequency_tool_top_n_slicing():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_freq_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    # Invoke with top_n=2
    result = term_freq_tool.invoke({"top_n": 2})

    # Summary top_terms must only contain top 2 terms ("always": 7 and "gamma": 7)
    assert len(result["top_terms"]) == 2
    assert result["top_terms"] == {"always": 7, "gamma": 7}

    # full_report MUST still contain all 5 terms
    result_id = result["result_id"]
    report_df = session.results_store[result_id]["full_report"]
    assert len(report_df) == 5
    assert list(report_df["feature"]) == ["always", "gamma", "alpha", "delta", "beta"]


def test_term_frequency_tool_invalid_top_n():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_freq_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    with pytest.raises(ValueError) as exc_info:
        term_freq_tool.invoke({"top_n": -5})
    assert "top_n must be a positive integer" in str(exc_info.value)


def test_term_frequency_tool_uninitialized_dtm():
    session = SessionState()
    dtm_tools = make_tools(session)
    term_freq_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    with pytest.raises(ValueError) as exc_info:
        term_freq_tool.invoke({"top_n": 20})
    assert "session.feature_matrix and session.feature_names must be initialized first" in str(exc_info.value)


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    # Full matrix slice test (defaults n_terms=20, n_documents=20)
    result_full = heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})
    assert result_full["n_terms"] == 5
    assert result_full["n_documents"] == 8
    assert result_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result_full["doc_labels"] == [f"Doc {i}" for i in range(8)]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert result_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # Corner slice test (n_terms=2, n_documents=3)
    result_corner = heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert result_corner["n_terms"] == 2
    assert result_corner["n_documents"] == 3
    assert result_corner["terms"] == ["alpha", "always"]
    assert result_corner["doc_labels"] == ["Doc 0", "Doc 1", "Doc 2"]

    expected_corner_matrix = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
    assert result_corner["matrix_slice"] == expected_corner_matrix

    # Invalid dimensions test
    for invalid_n in [-1, 0]:
        with pytest.raises(ValueError) as exc_info:
            heatmap_tool.invoke({"n_terms": invalid_n, "n_documents": 5})
        assert "n_terms must be a positive integer" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            heatmap_tool.invoke({"n_terms": 5, "n_documents": invalid_n})
        assert "n_documents must be a positive integer" in str(exc_info.value)
