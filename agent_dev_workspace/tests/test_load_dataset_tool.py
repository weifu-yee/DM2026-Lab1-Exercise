from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool_fixture():
    # Instantiate session state
    session = SessionState()

    # Load premade tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Invoke tool on test fixture
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Assertions on returned dictionary according to Known Answer 1
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
    assert "result_id" in result

    # Assertions on dataframe stored in session
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
    assert "category_name" in session.dataframe.columns
    assert set(session.dataframe["category_name"]) == {"catA", "catB"}
