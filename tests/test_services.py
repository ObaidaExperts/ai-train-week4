import os
import pytest
from unittest.mock import MagicMock
from app.services.experiment_service import ExperimentService
from app.services.repository import ResultsRepository
from app.core.models import AIModel

@pytest.fixture
def temp_results_file(tmp_path):
    f = tmp_path / "test_results.csv"
    return str(f)

@pytest.fixture
def repository(temp_results_file):
    return ResultsRepository(temp_results_file)

def test_repository_initialization(temp_results_file):
    repo = ResultsRepository(temp_results_file)
    assert os.path.exists(temp_results_file)
    with open(temp_results_file, 'r') as f:
        header = f.readline().strip()
        assert "Timestamp" in header

def test_repository_log_and_get(repository):
    result = {
        "Timestamp": "2024-01-01",
        "Experiment_Type": "Test",
        "Model": "gpt-4o",
        "Input_Tokens": 10,
        "Output_Tokens": 20,
        "Cost_USD": 0.001,
        "Status": "Success"
    }
    repository.log_result(result)
    results = repository.get_all_results()
    assert len(results) == 1
    assert results[0]["Model"] == "gpt-4o"

def test_experiment_service_calculate_cost():
    service = ExperimentService(repository=MagicMock())
    # gpt-4o: input 2.5, output 10.0 per 1M
    # 1M input = 2.5 USD
    # 1M output = 10.0 USD
    cost = service.calculate_cost(1_000_000, 1_000_000, AIModel.GPT_4O)
    assert cost == 12.5

def test_experiment_service_analyze_text_mocked():
    mock_repo = MagicMock()
    mock_openai = MagicMock()
    mock_anthropic = MagicMock()
    service = ExperimentService(repository=mock_repo, openai_client=mock_openai, anthropic_client=mock_anthropic)
    
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Mocked response"
    mock_response.usage.completion_tokens = 50
    mock_openai.chat.completions.create.return_value = mock_response
    
    result = service.analyze_text("Hello", AIModel.GPT_4O)
    
    assert result["response"] == "Mocked response"
    assert result["log_analysis"]["input_tokens"] > 0
    assert mock_repo.log_result.called

def test_experiment_service_analyze_claude_mocked():
    mock_repo = MagicMock()
    mock_openai = MagicMock()
    mock_anthropic = MagicMock()
    service = ExperimentService(repository=mock_repo, openai_client=mock_openai, anthropic_client=mock_anthropic)
    
    # Setup mock response
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Claude response")]
    mock_message.usage.input_tokens = 10
    mock_message.usage.output_tokens = 20
    mock_anthropic.messages.create.return_value = mock_message
    
    result = service.analyze_text("Hello Claude", AIModel.CLAUDE_3_5_SONNET)
    
    assert result["response"] == "Claude response"
    assert result["log_analysis"]["input_tokens"] == 10
    assert result["log_analysis"]["output_tokens"] == 20
    assert mock_repo.log_result.called

def test_experiment_service_api_error():
    mock_repo = MagicMock()
    mock_openai = MagicMock()
    service = ExperimentService(repository=mock_repo, openai_client=mock_openai)
    
    # Simulate an API error
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI API Down")
    
    with pytest.raises(Exception) as excinfo:
        service.analyze_text("Hello", AIModel.GPT_4O)
    
    assert "OpenAI API Down" in str(excinfo.value)
    # The error is raised, ensuring middleware can catch it

def test_experiment_service_repository_failure():
    mock_repo = MagicMock()
    mock_openai = MagicMock()
    service = ExperimentService(repository=mock_repo, openai_client=mock_openai)
    
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"
    mock_response.usage.completion_tokens = 10
    mock_openai.chat.completions.create.return_value = mock_response
    
    # Simulate repository failure
    mock_repo.log_result.side_effect = Exception("Disk Full")
    
    with pytest.raises(Exception) as excinfo:
        service.analyze_text("Hello", AIModel.GPT_4O)
    
    assert "Disk Full" in str(excinfo.value)
