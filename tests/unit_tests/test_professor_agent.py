import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import from agents module
from agents import ProfessorAgent

class TestProfessorAgent:
    """Tests for the ProfessorAgent class."""
    
    def test_professor_agent_initialization(self):
        """Test ProfessorAgent initialization."""
        agent = ProfessorAgent(
            model="gpt-4o-mini",
            notes=[{"phases": ["report writing"], "note": "Test note"}],
            max_steps=50,
            openai_api_key="fake-api-key"
        )
        
        # Check initialization values
        assert agent.model == "gpt-4o-mini"
        assert agent.notes == [{"phases": ["report writing"], "note": "Test note"}]
        assert agent.max_steps == 50
        assert agent.openai_api_key == "fake-api-key"
        assert agent.phases == ["report writing"]
    
    def test_professor_role_description(self):
        """Test ProfessorAgent role description."""
        agent = ProfessorAgent(model="gpt-4o-mini")
        
        # Check role description
        assert agent.role_description() == "a computer science professor at a top university."
    
    def test_professor_command_descriptions(self):
        """Test ProfessorAgent command descriptions."""
        agent = ProfessorAgent(model="gpt-4o-mini")
        
        # Get command descriptions for report writing
        cmd_desc = agent.command_descriptions("report writing")
        
        # Check that it contains key information
        assert "LATEX" in cmd_desc
        assert "report here" in cmd_desc
    
    def test_professor_phase_prompt(self):
        """Test ProfessorAgent phase prompt."""
        agent = ProfessorAgent(model="gpt-4o-mini")
        
        # Get phase prompt for report writing
        phase_prompt = agent.phase_prompt("report writing")
        
        # Check that it contains key information
        assert "directing a PhD student" in phase_prompt
        assert "write a report in latex" in phase_prompt
    
    def test_invalid_phase(self):
        """Test ProfessorAgent with invalid phase."""
        agent = ProfessorAgent(model="gpt-4o-mini")
        
        # Check that using an invalid phase raises an exception
        with pytest.raises(Exception) as excinfo:
            agent.phase_prompt("invalid phase")
        
        assert "Invalid phase" in str(excinfo.value)
        
        with pytest.raises(Exception) as excinfo:
            agent.command_descriptions("invalid phase")
        
        assert "Invalid phase" in str(excinfo.value)
    
    @patch('agents.query_model')
    def test_generate_readme(self, mock_query_model):
        """Test ProfessorAgent generate_readme method."""
        # Setup mock response
        mock_response = "```markdown\n# Test README\n\nThis is a test README.\n```"
        mock_query_model.return_value = mock_response
        
        # Create agent
        agent = ProfessorAgent(model="gpt-4o-mini")
        agent.report = "Test report"
        
        # Call generate_readme method
        result = agent.generate_readme()
        
        # Check that query_model was called
        mock_query_model.assert_called_once()
        
        # Check that the result is correct - should remove "```markdown" but keep the trailing "```"
        expected_result = mock_response.replace("```markdown", "")
        assert result == expected_result