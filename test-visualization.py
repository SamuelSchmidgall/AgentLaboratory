import os
import json
import random
from datetime import datetime

def generate_test_logs(num_entries=50):
    """Generate test log entries"""
    logs = []
    start_time = datetime.now()

    for i in range(num_entries):
        log_entry = {
            "timestamp": start_time.isoformat(),
            "step": i,
            "command": "EDIT" if i % 2 == 0 else "REPLACE",
            "code_lines": [f"print('Step {i}')"],
            "score": 0.4 + random.random() * 0.2,
            "model_response": f"Test response for step {i}"
        }
        logs.append(log_entry)

    return logs

def save_logs(logs, directory="agent_logs"):
    """Save logs to a JSON file"""
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mle_solver_{timestamp}.json"
    filepath = os.path.join(directory, filename)

    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=2)

    # Create or update latest symlink
    latest_link = os.path.join(directory, "mle_solver_latest.json")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    with open(latest_link, 'w') as f:
        json.dump(logs, f, indent=2)

    return filepath

def create_visualization_html(log_file_path):
    """Create the visualization HTML file"""
    if not os.path.exists("research_dir"):
        os.makedirs("research_dir")

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Agent Lab Visualization</title>
    <meta charset="utf-8">
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
    <script src="https://unpkg.com/recharts/umd/Recharts.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, sans-serif;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect } = React;
        const { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } = Recharts;

        function App() {
            const [data, setData] = useState([]);
            const [selectedStep, setSelectedStep] = useState(null);

            useEffect(() => {
                fetch('../agent_logs/mle_solver_latest.json')
                    .then(res => res.json())
                    .then(data => setData(data))
                    .catch(err => console.error('Error loading data:', err));
            }, []);

            const formatScore = (score) => (score * 100).toFixed(1) + '%';

            return (
                <div className="w-full max-w-7xl mx-auto p-4">
                    <h1 className="text-3xl font-bold mb-6">Agent Lab Progress</h1>

                    {/* Chart */}
                    <div className="bg-white p-4 rounded-lg shadow mb-6">
                        <h2 className="text-xl font-bold mb-4">Score Progress</h2>
                        <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={data}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="step" />
                                    <YAxis domain={[0, 1]} tickFormatter={formatScore} />
                                    <Tooltip formatter={formatScore} />
                                    <Legend />
                                    <Line type="monotone" dataKey="score" stroke="#2563eb" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Command List */}
                        <div className="bg-white p-4 rounded-lg shadow">
                            <h2 className="text-xl font-bold mb-4">Commands</h2>
                            <div className="h-96 overflow-y-auto">
                                {data.map((entry, idx) => (
                                    <div
                                        key={idx}
                                        onClick={() => setSelectedStep(idx)}
                                        className={\`p-3 rounded cursor-pointer \${
                                            selectedStep === idx ? 'bg-blue-100' : 'hover:bg-gray-100'
                                        }\`}
                                    >
                                        <div className="flex justify-between">
                                            <span>Step {entry.step}</span>
                                            <span className="text-gray-500">{entry.command}</span>
                                        </div>
                                        <div className="text-sm text-gray-600">
                                            Score: {formatScore(entry.score)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Details Panel */}
                        <div className="bg-white p-4 rounded-lg shadow">
                            <h2 className="text-xl font-bold mb-4">Step Details</h2>
                            {selectedStep !== null && data[selectedStep] ? (
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="font-medium">Code Changes</h3>
                                        <pre className="bg-gray-100 p-3 rounded mt-2 overflow-x-auto">
                                            <code>{data[selectedStep].code_lines.join('\\n')}</code>
                                        </pre>
                                    </div>
                                    <div>
                                        <h3 className="font-medium">Model Response</h3>
                                        <p className="text-gray-600 mt-2">
                                            {data[selectedStep].model_response}
                                        </p>
                                    </div>
                                    <div className="text-sm text-gray-500">
                                        {new Date(data[selectedStep].timestamp).toLocaleString()}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center text-gray-500">
                                    Select a step to view details
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>"""

    output_path = os.path.join("research_dir", "visualization.html")
    with open(output_path, 'w') as f:
        f.write(html_content)

    return output_path

def main():
    """Main function to run the visualization test"""
    print("Starting visualization test...")

    # Generate and save test logs
    logs = generate_test_logs()
    log_file_path = save_logs(logs)
    print(f"Generated test logs at: {log_file_path}")

    # Create visualization HTML
    viz_path = create_visualization_html(log_file_path)
    print(f"Created visualization interface at: {viz_path}")

    # Print some information about the logs
    print(f"Log file contains {len(logs)} entries")
    print("Sample log entry:")
    print(json.dumps(logs[0], indent=2))

    print("\nVisualization test complete!")
    print("\nTo view the visualization:")
    print("1. Start a local server: python -m http.server")
    print("2. Open http://localhost:8000/research_dir/visualization.html in your browser")

if __name__ == "__main__":
    main()