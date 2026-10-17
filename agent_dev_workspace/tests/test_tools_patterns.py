import pytest
import pandas as pd
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session


def test_mine_patterns_fpgrowth_ka17():
    """Tests Known Answer 17: FPGrowth with tfidf, variance, and term_frequency filters."""
    session = _setup_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. tfidf filter - catA (min_sup=3 -> alpha: 3, min_sup=4 -> 0 patterns)
    res_a3 = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 3})
    assert res_a3["patterns"] == [{"pattern": ["alpha"], "support": 3}]

    res_a4 = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 4})
    assert res_a4["patterns"] == []

    # 2. tfidf filter - catB (min_sup=4 -> gamma: 4)
    res_b4 = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 4})
    assert res_b4["patterns"] == [{"pattern": ["gamma"], "support": 4}]

    # 3. Zero-surviving term filters (variance & term_frequency)
    for filt in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            res = mine_patterns_tool.invoke({"category_name": cat, "filtering_method": filt, "algorithm": "fpgrowth", "min_sup": 1})
            assert res["patterns"] == []
            assert res["note"] == "no terms survived filtering"

            # Check storage & artifacts for zero-surviving terms
            artifact_key = f"patterns_{cat}_{filt}_fpgrowth"
            assert artifact_key in session.artifacts
            assert session.artifacts[artifact_key] == []

            result_id = res["result_id"]
            stored = session.results_store[result_id]
            assert isinstance(stored["full_report"], pd.DataFrame)
            assert list(stored["full_report"].columns) == ["pattern", "support"]


def test_mine_patterns_topk_maxfpgrowth_ka20():
    """Tests Known Answer 20: Top-K FAE and MaxFPGrowth algorithms."""
    session = _setup_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. TopK algorithm with k=1, k=2, k=3
    for k_val in [1, 2, 3]:
        res_a = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val})
        assert res_a["patterns"] == [{"pattern": ["alpha"], "support": 3}]

        res_b = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val})
        assert res_b["patterns"] == [{"pattern": ["gamma"], "support": 4}]

    # 2. MaxFPGrowth algorithm (returns 0 patterns even at min_sup=1)
    for sup in [1, 4]:
        res_max_a = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": sup})
        assert res_max_a["patterns"] == []

        res_max_b = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": sup})
        assert res_max_b["patterns"] == []


def test_mine_patterns_unicode_terms_regression():
    """Regression test: verifies that Unicode non-ASCII terms like 'café' and 'naïve'
    are successfully mined and mapped back without triggering Windows cp950 encoding errors in PAMI DF2DB.
    """
    session = SessionState()
    unicode_df = pd.DataFrame([
        {"text": "café naïve espresso café naïve", "label": "catUnicode", "category_name": "catUnicode"},
        {"text": "café naïve latte café", "label": "catUnicode", "category_name": "catUnicode"},
        {"text": "café espresso naïve", "label": "catUnicode", "category_name": "catUnicode"},
    ])
    session.dataframe = unicode_df

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({
        "category_name": "catUnicode",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })

    # 'café' and 'naïve' both appear in all 3 documents
    mined_patterns = res["patterns"]
    pattern_items = [item for p in mined_patterns for item in p["pattern"]]

    assert "café" in pattern_items
    assert "naïve" in pattern_items


def test_mine_patterns_invalid_inputs():
    """Tests handling of invalid parameters and missing inputs."""
    session = _setup_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Non-existent category
    with pytest.raises(ValueError, match="not found"):
        mine_patterns_tool.invoke({"category_name": "non_existent_cat", "min_sup": 1})

    # Missing min_sup for fpgrowth
    with pytest.raises(ValueError, match="min_sup"):
        mine_patterns_tool.invoke({"category_name": "catA", "algorithm": "fpgrowth"})

    # Invalid min_sup <= 0
    with pytest.raises(ValueError, match="min_sup"):
        mine_patterns_tool.invoke({"category_name": "catA", "algorithm": "fpgrowth", "min_sup": 0})

    # Missing k for topk
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_patterns_tool.invoke({"category_name": "catA", "algorithm": "topk"})

    # Invalid algorithm name (Pydantic validation error due to Literal type)
    with pytest.raises(ValidationError):
        mine_patterns_tool.invoke({"category_name": "catA", "algorithm": "apriori", "min_sup": 1})

    # Invalid filtering method (Pydantic validation error due to Literal type)
    with pytest.raises(ValidationError):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "random_filter", "min_sup": 1})


def test_mine_patterns_all_stopwords_category():
    """Tests a category where all text consists of English stopwords."""
    session = _setup_session()
    # Add a row to dataframe with category 'catStopwords' and text containing only stopwords
    new_row = pd.DataFrame([{"text": "the is an a in on", "label": "catStopwords", "category_name": "catStopwords"}])
    session.dataframe = pd.concat([session.dataframe, new_row], ignore_index=True)

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({"category_name": "catStopwords", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})

    assert res["patterns"] == []
    assert "empty vocabulary" in res["note"].lower()

    artifact_key = "patterns_catStopwords_tfidf_fpgrowth"
    assert artifact_key in session.artifacts
    assert session.artifacts[artifact_key] == []

    result_id = res["result_id"]
    stored = session.results_store[result_id]
    assert isinstance(stored["full_report"], pd.DataFrame)
    assert stored["full_report"].empty
    assert list(stored["full_report"].columns) == ["pattern", "support"]


def test_mine_patterns_artifacts_and_full_report():
    """Verifies that results are stored in session artifacts and results_store with full_report."""
    session = _setup_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 3})
    result_id = res["result_id"]
    artifact_key = "patterns_catA_tfidf_fpgrowth"

    # Artifact check
    assert artifact_key in session.artifacts
    assert session.artifacts[artifact_key] == [{"pattern": ["alpha"], "support": 3}]

    # Results store & full_report check
    assert result_id in session.results_store
    stored_entry = session.results_store[result_id]
    assert "full_report" in stored_entry
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert not full_report.empty
    assert list(full_report.columns) == ["pattern", "support"]
