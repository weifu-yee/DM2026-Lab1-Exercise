import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_fixture():
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

    # Build DTM (8x5 matrix)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    # Load variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Invoke variance_filter_tool with threshold 0.15 according to Known Answer 5
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert Known Answer 5 summary results
    assert result["total_features"] == 5
    assert result["kept_features_count"] == 4
    assert result["removed_features_count"] == 1
    assert result["all_zero_variance"] is False
    assert "result_id" in result

    # Verify stored full_report DataFrame contents and variances
    stored_entry = session.results_store[result["result_id"]]
    report_df = stored_entry["full_report"]

    assert isinstance(report_df, pd.DataFrame)
    assert list(report_df.columns[:2]) == ["feature", "variance"]

    # Check that full_report contains all 5 unique terms without duplicates
    assert len(report_df) == 5
    assert len(set(report_df["feature"])) == 5
    assert set(report_df["feature"]) == {"alpha", "always", "beta", "delta", "gamma"}

    # Known variances per term from TEST_FIXTURE.md #5:
    # alpha: 1.1875, gamma: 1.109375, delta: 0.25, beta: 0.234375, always: 0.109375
    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }

    for _, row in report_df.iterrows():
        feature = row["feature"]
        var_val = row["variance"]
        assert feature in expected_variances
        assert np.isclose(var_val, expected_variances[feature], atol=1e-3)

    # Check sorting descending order
    variances_list = report_df["variance"].tolist()
    assert variances_list == sorted(variances_list, reverse=True)


def test_variance_filter_exact_threshold_equality():
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
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # delta has variance exactly 0.25. Setting threshold=0.25 removes delta (> rule)
    result = variance_filter_tool.invoke({"threshold": 0.25})
    # alpha (1.1875) and gamma (1.109375) have variance > 0.25. delta (0.25), beta (0.234375), always (0.109375) are removed.
    assert result["kept_features_count"] == 2
    assert result["removed_features_count"] == 3


def test_variance_filter_all_zero_variance():
    session = SessionState()
    # Mock a DTM where all entries are 1 (zero variance across rows)
    session.feature_matrix = csr_matrix([[1, 1], [1, 1], [1, 1]])
    session.feature_names = ["term1", "term2"]

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    result = variance_filter_tool.invoke({"threshold": 0.0})
    assert result["all_zero_variance"] is True
    assert result["kept_features_count"] == 0
    assert result["removed_features_count"] == 2


def test_variance_filter_zero_documents():
    session = SessionState()
    session.feature_matrix = csr_matrix((0, 5))
    session.feature_names = ["t1", "t2", "t3", "t4", "t5"]

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    with pytest.raises(ValueError) as exc_info:
        variance_filter_tool.invoke({"threshold": 0.0})
    assert "0 documents" in str(exc_info.value)


def test_variance_filter_uninitialized_dtm():
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    with pytest.raises(ValueError) as exc_info:
        variance_filter_tool.invoke({"threshold": 0.15})
    assert "session.feature_matrix" in str(exc_info.value)


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using tools_dtm's build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    # Initialize filtering tools from tools_filtering
    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Execute pearson_filter_tool for target_class="catB" with threshold=0.0
    result = pearson_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert "result_id" in result
    assert result["target_class"] == "catB"
    assert result["total_features"] == 5

    # Retrieve full report DataFrame from results_store
    result_id = result["result_id"]
    stored_entry = session.results_store[result_id]
    report_df = stored_entry["full_report"]

    # Convert report to dict mapping feature name -> pearson_r
    pearson_dict = dict(zip(report_df["feature"], report_df["pearson_r"]))

    # Known Answer 6 expected values for catB (tolerance 1e-3)
    expected_correlations = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for feature, expected_r in expected_correlations.items():
        assert feature in pearson_dict
        assert pearson_dict[feature] == pytest.approx(expected_r, abs=1e-3)

    # With threshold = 0.0, positive correlations (delta, gamma, always) should be kept -> 3 kept
    assert result["kept_features_count"] == 3
    assert result["removed_features_count"] == 2


def test_pearson_filter_invalid_target_class():
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
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    with pytest.raises(ValueError) as exc_info:
        pearson_tool.invoke({"target_class": "nonexistent_class", "threshold": 0.0})
    assert "not found in dataset labels" in str(exc_info.value)


def test_pearson_filter_constant_term_exclusion():
    session = SessionState()

    # Create mock DTM where column 0 is non-constant, column 1 is constant (all 1s)
    session.feature_matrix = csr_matrix([[1, 1], [0, 1], [1, 1], [0, 1]])
    session.feature_names = ["varying_term", "constant_term"]
    session.set_labels(["catA", "catA", "catB", "catB"])

    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run with threshold=0.0. constant_term gets r=0.0 but is constant, so it MUST be excluded.
    result = pearson_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["total_features"] == 2
    # varying_term correlation with catB: [1, 0, 1, 0] vs [0, 0, 1, 1] => r = 0.0, std > 0 => kept at threshold=0.0
    # constant_term: std == 0 => constant => excluded
    assert result["kept_features_count"] == 1
    assert result["removed_features_count"] == 1


def test_pearson_filter_labels_length_mismatch():
    session = SessionState()

    # DTM has 4 documents
    session.feature_matrix = csr_matrix([[1, 0], [0, 1], [1, 1], [0, 1]])
    session.feature_names = ["term1", "term2"]
    # Labels only have 3 elements (mismatch with 4 DTM rows)
    session.labels = ["catA", "catA", "catB"]

    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    with pytest.raises(ValueError) as exc_info:
        pearson_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert "Mismatch between labels count" in str(exc_info.value)


def test_spearman_filter_tool_known_answer_6():
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
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    filtering_tools = make_filtering_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result = spearman_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert "result_id" in result
    assert result["target_class"] == "catB"
    assert result["total_features"] == 5

    result_id = result["result_id"]
    stored_entry = session.results_store[result_id]
    report_df = stored_entry["full_report"]

    assert list(report_df.columns[:2]) == ["feature", "spearman_r"]

    # Check full_report contains all 5 unique terms without duplicates
    assert len(report_df) == 5
    assert len(set(report_df["feature"])) == 5
    assert set(report_df["feature"]) == {"alpha", "always", "beta", "delta", "gamma"}

    # Check sorting descending order
    spearman_list = report_df["spearman_r"].tolist()
    assert spearman_list == sorted(spearman_list, reverse=True)

    spearman_dict = dict(zip(report_df["feature"], report_df["spearman_r"]))

    # Known Answer 6 expected Spearman values for catB (tolerance 1e-3)
    expected_correlations = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for feature, expected_r in expected_correlations.items():
        assert feature in spearman_dict
        assert spearman_dict[feature] == pytest.approx(expected_r, abs=1e-3)

    assert result["kept_features_count"] == 3
    assert result["removed_features_count"] == 2


def test_spearman_filter_invalid_target_class():
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
    build_dtm_tool.invoke({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})

    filtering_tools = make_filtering_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    with pytest.raises(ValueError) as exc_info:
        spearman_tool.invoke({"target_class": "nonexistent_class", "threshold": 0.0})
    assert "not found in dataset labels" in str(exc_info.value)


def test_spearman_filter_constant_term_exclusion():
    session = SessionState()

    # Create mock DTM where column 0 is non-constant, column 1 is constant (all 1s)
    session.feature_matrix = csr_matrix([[1, 1], [0, 1], [1, 1], [0, 1]])
    session.feature_names = ["varying_term", "constant_term"]
    session.set_labels(["catA", "catA", "catB", "catB"])

    filtering_tools = make_filtering_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result = spearman_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["total_features"] == 2
    assert result["kept_features_count"] == 1
    assert result["removed_features_count"] == 1


def test_spearman_filter_labels_length_mismatch():
    session = SessionState()

    session.feature_matrix = csr_matrix([[1, 0], [0, 1], [1, 1], [0, 1]])
    session.feature_names = ["term1", "term2"]
    session.labels = ["catA", "catA", "catB"]

    filtering_tools = make_filtering_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    with pytest.raises(ValueError) as exc_info:
        spearman_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert "Mismatch between labels count" in str(exc_info.value)
