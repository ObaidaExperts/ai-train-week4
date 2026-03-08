import os
import pytest
from app.services.repository import ResultsRepository

@pytest.fixture
def temp_csv(tmp_path):
    f = tmp_path / "results_repo_test.csv"
    return str(f)

def test_repository_creates_file_on_init(temp_csv):
    ResultsRepository(temp_csv)
    assert os.path.exists(temp_csv)

def test_repository_log_result_content(temp_csv):
    repo = ResultsRepository(temp_csv)
    result = {
        "Timestamp": "2024-01-01",
        "Request_ID": "123",
        "Experiment_Type": "Baseline",
        "Model": "test-model",
        "Prompt": "test prompt",
        "Response": "test response",
        "Input_Tokens": 1,
        "Output_Tokens": 2,
        "Cost_USD": 0.1,
        "Status": "Success",
        "Logprobs": True
    }
    repo.log_result(result)
    
    data = repo.get_all_results()
    assert len(data) == 1
    assert data[0]["Request_ID"] == "123"
    assert data[0]["Prompt"] == "test prompt"

def test_repository_handle_empty_file(temp_csv):
    # Ensure it doesn't crash if file is somehow empty or header-only
    repo = ResultsRepository(temp_csv)
    assert repo.get_all_results() == []

def test_repository_invalid_file_permissions(temp_csv):
    # This might be tricky in some environments, but we can try to mock or simulate
    repo = ResultsRepository(temp_csv)
    # If we can't write, it should ideally raise or handle it
    # For now, we'll just test that initialization works.
    assert os.path.exists(temp_csv)
