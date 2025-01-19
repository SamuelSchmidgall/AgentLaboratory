import pytest
from unittest.mock import Mock, patch
from ..models import TokenCounter, ModelConfig, OpenAIStrategy, AnthropicStrategy
from ..factory import LLMStrategyFactory
from ..query_manager import LLMQueryManager


def test_token_counter():
    counter = TokenCounter()
    counter.update_counts("gpt-4o", 100, 50)
    counter.update_counts("gpt-4o", 150, 75)

    assert counter.tokens_in["gpt-4o"] == 250
    assert counter.tokens_out["gpt-4o"] == 125

    model_configs = {
        "gpt-4o": ModelConfig("gpt-4o", 2.50, 10.00, "openai")
    }

    expected_cost = (250 * 2.50 + 125 * 10.00) / 1_000_000
    assert counter.calculate_cost(model_configs) == expected_cost


def test_openai_strategy_no_api_key():
    config = ModelConfig("gpt-4o", 2.50, 10.00, "openai")
    strategy = OpenAIStrategy(config)

    with pytest.raises(ValueError, match="No API key provided for OpenAI API"):
        strategy.query("test prompt", "test system prompt")


def test_openai_strategy_default_key():
    config = ModelConfig("gpt-4o", 2.50, 10.00, "openai")
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="test response"))]
    mock_client.chat.completions.create.return_value = mock_completion

    with patch('llm_client.models.OpenAI', return_value=mock_client) as mock_openai:
        strategy = OpenAIStrategy(config, "default-key")
        response = strategy.query("test prompt", "test system prompt")

        assert response == "test response"
        mock_openai.assert_called_once_with(api_key="default-key")


def test_openai_strategy_override_key():
    config = ModelConfig("gpt-4o", 2.50, 10.00, "openai")
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="test response"))]
    mock_client.chat.completions.create.return_value = mock_completion

    with patch('llm_client.models.OpenAI', return_value=mock_client) as mock_openai:
        strategy = OpenAIStrategy(config, "default-key")
        response = strategy.query("test prompt", "test system prompt", "override-key")

        assert response == "test response"
        mock_openai.assert_called_once_with(api_key="override-key")


def test_anthropic_strategy_no_api_key():
    config = ModelConfig("claude-3-5-sonnet", 3.00, 12.00, "anthropic")
    strategy = AnthropicStrategy(config)

    with pytest.raises(ValueError, match="No API key provided for Anthropic API"):
        strategy.query("test prompt", "test system prompt")


def test_strategy_factory_with_api_keys():
    api_keys = {
        "openai": "test-openai-key",
        "anthropic": "test-anthropic-key",
        "deepseek": "test-deepseek-key"
    }
    factory = LLMStrategyFactory(api_keys)

    # Test OpenAI strategy creation with default key
    strategy = factory.create_strategy("gpt-4o")
    assert isinstance(strategy, OpenAIStrategy)
    assert strategy.default_api_key == "test-openai-key"

    # Test Anthropic strategy creation with default key
    strategy = factory.create_strategy("claude-3-5-sonnet")
    assert isinstance(strategy, AnthropicStrategy)
    assert strategy.default_api_key == "test-anthropic-key"


@patch('llm_client.factory.LLMStrategyFactory.create_strategy')
def test_query_manager_integration(mock_create_strategy):
    api_keys = {"openai": "test-openai-key"}
    mock_strategy = Mock()
    mock_strategy.query.return_value = "test response"
    mock_strategy.count_tokens.return_value = 10
    mock_strategy.config = ModelConfig("test-model", 1.0, 1.0, "openai")
    mock_create_strategy.return_value = mock_strategy

    manager = LLMQueryManager(api_keys)

    # Test with default key
    response = manager.query_model(
        "gpt-4o",
        "test prompt",
        "test system prompt",
        print_cost=False
    )
    mock_strategy.query.assert_called_with(
        "test prompt",
        "test system prompt",
        None,  # Using default key
        None,
    )

    # Test with override key
    response = manager.query_model(
        "gpt-4o",
        "test prompt",
        "test system prompt",
        api_key="override-key",
        print_cost=False
    )
    mock_strategy.query.assert_called_with(
        "test prompt",
        "test system prompt",
        "override-key",
        None,
    )


def test_query_manager_no_api_key():
    manager = LLMQueryManager()  # No API keys provided

    with pytest.raises(ValueError, match="No API key provided. Please provide an API key."):
        manager.query_model(
            "gpt-4o",
            "test prompt",
            "test system prompt",
            print_cost=False
        )