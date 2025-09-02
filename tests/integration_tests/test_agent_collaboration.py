import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import agent classes
from agents import ProfessorAgent, PhDStudentAgent

class TestAgentCollaboration:
    """Tests for collaboration between different agent types."""
    
    @patch('agents.query_model')
    def test_professor_phd_collaboration(self, mock_query_model):
        """Test collaboration between ProfessorAgent and PhDStudentAgent."""
        # Setup mock responses for different agent calls
        def mock_query_side_effect(*args, **kwargs):
            # Extract the system prompt to determine which agent is calling
            system_prompt = args[1] if len(args) > 1 else kwargs.get('system_prompt', '')
            
            if "professor" in system_prompt.lower():
                return "```LATEX\nTest report from professor-phd collaboration\n```"
            elif "phd student" in system_prompt.lower():
                return "```DIALOGUE\nLet's work on this report together.\n```"
            else:
                return "Default response"
        
        mock_query_model.side_effect = mock_query_side_effect
        
        # Create agents
        professor = ProfessorAgent(model="gpt-4o-mini")
        phd_student = PhDStudentAgent(model="gpt-4o-mini")
        
        # Simulate report writing collaboration
        # Step 1: PhD student proposes initial ideas
        phd_response = phd_student.inference(
            research_topic="Test research topic",
            phase="report writing",
            step=1
        )
        
        assert "DIALOGUE" in phd_response
        assert "Let's work on this report together" in phd_response
        
        # Step 2: Professor reviews and finalizes the report
        professor_response = professor.inference(
            research_topic="Test research topic",
            phase="report writing",
            step=2,
            feedback=phd_response
        )
        
        assert "LATEX" in professor_response
        assert "Test report from professor-phd collaboration" in professor_response
        
        # Check that professor saves the report
        professor.report = professor_response.replace("```LATEX\n", "").replace("\n```", "")
        assert professor.report == "Test report from professor-phd collaboration"