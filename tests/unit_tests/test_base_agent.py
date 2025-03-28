import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import from agents module
from agents import BaseAgent

class TestBaseAgent:
    """Tests for the BaseAgent class."""
    
    def test_base_agent_initialization(self):
        """Test BaseAgent initialization."""
        agent = BaseAgent(
            model="gpt-4o-mini",
            notes=[{"phases": ["test phase"], "note": "Test note"}],
            max_steps=50,
            openai_api_key="fake-api-key"
        )
        
        # Check initialization values
        assert agent.model == "gpt-4o-mini"
        assert agent.notes == [{"phases": ["test phase"], "note": "Test note"}]
        assert agent.max_steps == 50
        assert agent.openai_api_key == "fake-api-key"
        assert agent.history == []
        assert agent.prev_comm == ""
    
    def test_base_agent_reset(self):
        """Test BaseAgent reset method."""
        agent = BaseAgent(model="gpt-4o-mini")
        
        # Add some history
        agent.history = ["entry1", "entry2"]
        agent.prev_comm = "previous command"
        
        # Reset agent
        agent.reset()
        
        # Check that history is cleared
        assert agent.history == []
        assert agent.prev_comm == ""
    
    @patch('agents.query_model')
    def test_base_agent_inference(self, mock_query_model):
        """Test BaseAgent inference method."""
        # Setup mock response
        mock_query_model.return_value = "Test response"
        
        # Create agent and override abstract methods
        class TestAgent(BaseAgent):
            def context(self, phase):
                return "Test context"
            
            def phase_prompt(self, phase):
                return "Test phase prompt"
            
            def role_description(self):
                return "Test role description"
            
            def command_descriptions(self, phase):
                return "Test command descriptions"
            
            def example_command(self, phase):
                return "Test example command"
        
        agent = TestAgent(model="gpt-4o-mini")
        
        # Call inference method
        result = agent.inference(
            research_topic="Test topic",
            phase="test phase",
            step=1,
            feedback="Test feedback"
        )
        
        # Check that query_model was called
        mock_query_model.assert_called_once()
        
        # Check that the result is correct
        assert result == "Test response"
        
        # Check that history was updated
        assert len(agent.history) == 1
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        agent = BaseAgent(model="gpt-4o-mini")
        
        with pytest.raises(NotImplementedError):
            agent.context("test phase")
        
        with pytest.raises(NotImplementedError):
            agent.phase_prompt("test phase")
        
        with pytest.raises(NotImplementedError):
            agent.role_description()
        
        with pytest.raises(NotImplementedError):
            agent.command_descriptions("test phase")
        
        with pytest.raises(NotImplementedError):
            agent.example_command("test phase")