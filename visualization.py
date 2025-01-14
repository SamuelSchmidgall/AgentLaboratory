import json
import os
from datetime import datetime

class AgentLogger:
    def __init__(self, agent_name, log_dir="agent_logs"):
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs = []

        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log_step(self, step_data):
        """Log a single step of agent behavior"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': len(self.logs),
            **step_data
        }
        self.logs.append(log_entry)

    def save_logs(self):
        """Save current logs to file"""
        filename = f"{self.log_dir}/{self.agent_name}_{self.current_session}.json"
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        return filename

    def get_logs(self):
        """Return current logs"""
        return self.logs

    def clear_logs(self):
        """Clear current logs"""
        self.logs = []

class VisualizationManager:
    def __init__(self):
        self.loggers = {}

    def get_logger(self, agent_name):
        """Get or create a logger for an agent"""
        if agent_name not in self.loggers:
            self.loggers[agent_name] = AgentLogger(agent_name)
        return self.loggers[agent_name]

    def get_all_logs(self):
        """Get logs from all agents"""
        return {name: logger.get_logs() for name, logger in self.loggers.items()}

    def save_all_logs(self):
        """Save logs from all agents"""
        return {name: logger.save_logs() for name, logger in self.loggers.items()}