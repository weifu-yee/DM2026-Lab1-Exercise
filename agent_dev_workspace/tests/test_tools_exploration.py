import pytest
import pandas as pd
import numpy as np
import matplotlib.figure
from unittest.mock import MagicMock
from scipy.sparse import csr_matrix
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_known_answers():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load test fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM to populate session.artifacts["count_vectorizer"]
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known answer checks from TEST_FIXTURE.md #16
    # 1. Doc 0 vs Doc 1 (both catA, near-identical text) -> 0.9847
    res_0_1 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 2. Doc 0 vs Doc 3 (catA vs catB, share only "always") -> 0.0909
    res_0_3 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    # 3. Doc 3 vs Doc 4 (both catB, near-identical text) -> 0.9847
    res_3_4 = cosine_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 4. Doc 0 vs Doc 7 (catA vs empty-text row) -> 0.0
    res_0_7 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert res_0_7["similarity"] == 0.0

    # Identical document check (Doc 0 vs Doc 0) -> 1.0
    res_0_0 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 0})
    assert res_0_0["similarity"] == pytest.approx(1.0, abs=1e-4)


def test_cosine_similarity_null_missing_na_text():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Manually introduce None, np.nan, and pd.NA into dataframe rows while vectorizer remains fitted
    session.dataframe.loc[0, "text"] = None
    session.dataframe.loc[1, "text"] = np.nan
    session.dataframe.loc[2, "text"] = pd.NA

    # Comparing any null text document against a valid text document (e.g. Doc 3) yields similarity 0.0
    res_none = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert res_none["similarity"] == 0.0

    res_nan = cosine_tool.invoke({"doc_idx1": 1, "doc_idx2": 3})
    assert res_nan["similarity"] == 0.0

    res_na = cosine_tool.invoke({"doc_idx1": 2, "doc_idx2": 3})
    assert res_na["similarity"] == 0.0


def test_cosine_similarity_vectorizer_not_refitted():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Mock fit method on count_vectorizer to verify it is never called
    vectorizer = session.artifacts["count_vectorizer"]
    vectorizer.fit = MagicMock()
    vectorizer.fit_transform = MagicMock()

    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})

    vectorizer.fit.assert_not_called()
    vectorizer.fit_transform.assert_not_called()


def test_cosine_similarity_validation_and_missing():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Out of bounds indices
    with pytest.raises(ValueError, match="out of bounds"):
        cosine_tool.invoke({"doc_idx1": -1, "doc_idx2": 1})

    with pytest.raises(ValueError, match="out of bounds"):
        cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 10})

    # Missing vectorizer error
    session.artifacts.pop("count_vectorizer", None)
    with pytest.raises(ValueError, match="No 'count_vectorizer' found"):
        cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})


def test_feature_correlation_matrix_known_answer_19():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feat_corr_tool.invoke({})

    # 1. Exact term order by variance descending (TEST_FIXTURE.md #19)
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # 2. Known correlation matrix values
    expected_corr = np.array([
        [1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718, 1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882, 0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ])
    matrix_arr = np.array(res["correlation_matrix"])
    assert matrix_arr.shape == (5, 5)
    np.testing.assert_allclose(matrix_arr, expected_corr, atol=1e-3)

    # 3. Check symmetry and diagonal ones
    np.testing.assert_allclose(matrix_arr, matrix_arr.T, atol=1e-5)
    np.testing.assert_allclose(np.diag(matrix_arr), np.ones(5), atol=1e-5)

    # 4. Check pending figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # 5. Check full_report
    stored_res = session.results_store[res["result_id"]]
    full_report = stored_res["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == expected_terms
    assert list(full_report.index) == expected_terms


def test_feature_correlation_matrix_edge_cases_and_validations():
    session = SessionState()
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Missing feature_matrix
    with pytest.raises(ValueError, match="No feature_matrix or feature_names"):
        feat_corr_tool.invoke({})

    # Populate session with sample data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Invalid top_n <= 0
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        feat_corr_tool.invoke({"top_n": 0})

    # Test top_n = 1
    res_1 = feat_corr_tool.invoke({"top_n": 1})
    assert res_1["terms"] == ["alpha"]
    assert res_1["correlation_matrix"] == [[1.0]]

    # Insufficient documents (< 2)
    session.feature_matrix = session.feature_matrix[:1, :]
    with pytest.raises(ValueError, match="Insufficient documents"):
        feat_corr_tool.invoke({})

    # Zero-variance (constant column) term raising ValueError
    session.feature_matrix = csr_matrix(np.ones((4, 3)))
    session.feature_names = ["const1", "const2", "const3"]
    with pytest.raises(ValueError, match="zero variance"):
        feat_corr_tool.invoke({})
