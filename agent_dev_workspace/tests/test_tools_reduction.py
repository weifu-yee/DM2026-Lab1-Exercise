import pytest
import pandas as pd
import numpy as np
import matplotlib.figure
from scipy.sparse import csr_matrix
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_binarize_labels_binary():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["doc1", "doc2", "doc3", "doc4"],
        "category_name": ["catA", "catB", "catA", "catB"]
    })

    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res["shape"] == [4, 2]
    assert res["classes"] == ["catA", "catB"]
    assert res["column_names"] == ["catA", "catB"]

    # Check artifact storage
    artifact_df = session.artifacts["binarized_labels"]
    assert isinstance(artifact_df, pd.DataFrame)
    assert artifact_df.shape == (4, 2)
    assert list(artifact_df.columns) == ["catA", "catB"]

    # Check one-hot values
    expected_values = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])
    np.testing.assert_array_equal(artifact_df.values, expected_values)

    # Check session synchronization
    assert session.categories == ["catA", "catB"]
    assert list(session.labels) == ["catA", "catB", "catA", "catB"]

    # Check stored result full_report
    stored_result = session.results_store[res["result_id"]]
    pd.testing.assert_frame_equal(stored_result["full_report"], artifact_df)


def test_binarize_labels_multiclass():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "category_name": ["catA", "catC", "catB", "catA", "catC"]
    })

    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res["shape"] == [5, 3]
    assert res["classes"] == ["catA", "catB", "catC"]

    artifact_df = session.artifacts["binarized_labels"]
    assert list(artifact_df.columns) == ["catA", "catB", "catC"]

    expected_values = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    np.testing.assert_array_equal(artifact_df.values, expected_values)


def test_binarize_labels_validation_errors():
    # 1. Missing dataframe
    session1 = SessionState()
    tools1 = make_reduction_tools(session1)
    binarize_tool1 = next(t for t in tools1 if t.name == "binarize_labels_tool")
    with pytest.raises(ValueError, match="No dataframe loaded in session"):
        binarize_tool1.invoke({})

    # 2. Empty dataframe
    session2 = SessionState()
    session2.dataframe = pd.DataFrame()
    tools2 = make_reduction_tools(session2)
    binarize_tool2 = next(t for t in tools2 if t.name == "binarize_labels_tool")
    with pytest.raises(ValueError, match="Session dataframe is empty"):
        binarize_tool2.invoke({})

    # 3. Missing category column
    session3 = SessionState()
    session3.dataframe = pd.DataFrame({"text": ["doc1", "doc2"]})
    tools3 = make_reduction_tools(session3)
    binarize_tool3 = next(t for t in tools3 if t.name == "binarize_labels_tool")
    with pytest.raises(ValueError, match="No 'category_name' or 'category' column found"):
        binarize_tool3.invoke({})

    # 4. NaN category label
    session4 = SessionState()
    session4.dataframe = pd.DataFrame({"category_name": ["catA", np.nan]})
    tools4 = make_reduction_tools(session4)
    binarize_tool4 = next(t for t in tools4 if t.name == "binarize_labels_tool")
    with pytest.raises(ValueError, match="Dataset contains missing \\(NaN\\) category labels"):
        binarize_tool4.invoke({})

    # 5. Blank/empty string category label
    session5 = SessionState()
    session5.dataframe = pd.DataFrame({"category_name": ["catA", "   "]})
    tools5 = make_reduction_tools(session5)
    binarize_tool5 = next(t for t in tools5 if t.name == "binarize_labels_tool")
    with pytest.raises(ValueError, match="Dataset contains blank/empty string category labels"):
        binarize_tool5.invoke({})


def test_reduce_dimensions_pca_known_answer_7():
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

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #7: PCA
    res_pca = reduce_tool.invoke({"method": "pca"})
    expected_pca_coords = np.array([
        [ 2.3669,  0.8871],
        [ 1.7039,  0.2642],
        [ 1.0408, -0.3587],
        [-2.0759,  1.0129],
        [-1.4554,  0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [ 0.0894, -1.4822]
    ])
    np.testing.assert_allclose(res_pca["coordinates"], expected_pca_coords, atol=1e-3)
    assert res_pca["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    
    # Verify PCA plot legend contains catA and catB
    ax_pca = session.pending_figure.axes[0]
    legend_texts_pca = [t.get_text() for t in ax_pca.get_legend().get_texts()]
    assert "catA" in legend_texts_pca and "catB" in legend_texts_pca


def test_reduce_dimensions_tsne_known_answer_8():
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

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #8: t-SNE (perplexity=2, random_state=42)
    res_tsne = reduce_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    expected_tsne_coords = np.array([
        [ 1330.9991,    32.1039],
        [  951.2459,   393.5588],
        [  530.8209,   682.9791],
        [-1558.6448,  -314.1625],
        [ -915.5153,  -346.3907],
        [ -280.0313, -1261.2115],
        [ -280.0313, -1261.2115],
        [ -701.3674,  1465.3973]
    ])
    np.testing.assert_allclose(res_tsne["coordinates"], expected_tsne_coords, atol=1e-3)
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    ax_tsne = session.pending_figure.axes[0]
    legend_texts_tsne = [t.get_text() for t in ax_tsne.get_legend().get_texts()]
    assert "catA" in legend_texts_tsne and "catB" in legend_texts_tsne


def test_reduce_dimensions_umap_known_answer_9():
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

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #9: UMAP (n_neighbors=3, random_state=42)
    res_umap = reduce_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    expected_umap_coords = np.array([
        [ 6.6811, -6.8218],
        [ 6.9907, -7.2245],
        [ 7.7022, -7.3712],
        [10.2228, -6.2310],
        [ 9.8759, -5.6537],
        [ 9.3831, -6.5748],
        [ 9.2153, -5.8297],
        [ 8.5860, -7.2622]
    ])
    np.testing.assert_allclose(res_umap["coordinates"], expected_umap_coords, atol=1e-3)
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    ax_umap = session.pending_figure.axes[0]
    legend_texts_umap = [t.get_text() for t in ax_umap.get_legend().get_texts()]
    assert "catA" in legend_texts_umap and "catB" in legend_texts_umap

    # Check full report and artifacts
    assert session.artifacts["reduced_coordinates"].shape == (8, 2)
    stored_result = session.results_store[res_umap["result_id"]]
    assert len(stored_result["full_report"]) == 8


def test_reduce_dimensions_summary_truncation_and_validations():
    session = SessionState()
    
    # Create synthetic dataset with 30 documents (> 20 rows limit)
    n_docs = 30
    rng = np.random.RandomState(42)
    synth_matrix = rng.randn(n_docs, 10)
    synth_cats = ["ClassA" if i % 2 == 0 else "ClassB" for i in range(n_docs)]

    session.dataframe = pd.DataFrame({"text": [f"doc_{i}" for i in range(n_docs)], "category_name": synth_cats})
    session.feature_matrix = synth_matrix
    session.feature_names = [f"term_{i}" for i in range(10)]

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Run PCA on 30 documents
    res = reduce_tool.invoke({"method": "pca"})

    # Verify summary coordinates are truncated to top 20 rows
    assert len(res["coordinates"]) == 20

    # Verify full report and artifacts retain all 30 rows
    assert len(session.artifacts["reduced_coordinates"]) == 30
    stored = session.results_store[res["result_id"]]
    assert len(stored["full_report"]) == 30

    # Validation Error 1: n_components != 2
    with pytest.raises(ValueError, match="n_components must be exactly 2"):
        reduce_tool.invoke({"n_components": 3})

    # Validation Error 2: perplexity >= n_samples (30)
    with pytest.raises(ValueError, match="perplexity.*must be less than n_samples"):
        reduce_tool.invoke({"method": "tsne", "perplexity": 30})

    # Validation Error 3: n_neighbors >= n_samples (30)
    with pytest.raises(ValueError, match="n_neighbors.*must be strictly less than n_samples"):
        reduce_tool.invoke({"method": "umap", "n_neighbors": 30})

    # Validation Error 4: Dataframe and DTM row count mismatch
    session.dataframe = session.dataframe.iloc[:25]
    with pytest.raises(ValueError, match="Mismatch: dataframe has 25 rows, but feature_matrix has 30 rows"):
        reduce_tool.invoke({"method": "pca"})
