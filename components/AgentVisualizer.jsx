import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Card components
const Card = ({ children, className = '' }) => (
    <div className={`bg-white p-6 rounded-lg shadow-md ${className}`}>{children}</div>
);

const CardHeader = ({ children }) => (
    <div className="mb-4">{children}</div>
);

const CardContent = ({ children }) => (
    <div>{children}</div>
);

const CardTitle = ({ children }) => (
    <h2 className="text-xl font-bold text-gray-700">{children}</h2>
);

const AgentLabVisualization = () => {
    const [logData, setLogData] = useState([]);
    const [selectedStep, setSelectedStep] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadData = async () => {
            try {
                const response = await window.fs.readFile('agent_logs/mle_solver_latest.json');
                const data = JSON.parse(new TextDecoder().decode(response));
                setLogData(data);
            } catch (err) {
                console.error('Error loading log data:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, []);

    const formatScore = (score) => (score * 100).toFixed(1) + '%';

    if (loading) {
        return (
            <div className="w-full h-screen flex items-center justify-center">
                <div className="text-lg text-gray-600">Loading data...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="w-full h-screen flex items-center justify-center">
                <div className="text-lg text-red-600">Error: {error}</div>
            </div>
        );
    }

    return (
        <div className="w-full mx-auto p-4 bg-gray-50 min-h-screen">
            <h1 className="text-3xl font-bold mb-6 text-gray-800">Agent Lab Progress</h1>

            <Card className="mb-6">
                <CardHeader>
                    <CardTitle>Score Progress Over Time</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={logData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                <XAxis
                                    dataKey="step"
                                    tick={{ fill: '#4b5563' }}
                                    stroke="#9ca3af"
                                />
                                <YAxis
                                    domain={[0, 1]}
                                    tickFormatter={formatScore}
                                    tick={{ fill: '#4b5563' }}
                                    stroke="#9ca3af"
                                />
                                <Tooltip
                                    formatter={formatScore}
                                    contentStyle={{
                                        backgroundColor: 'white',
                                        border: '1px solid #e5e7eb',
                                        borderRadius: '0.375rem'
                                    }}
                                />
                                <Legend />
                                <Line
                                    type="monotone"
                                    dataKey="score"
                                    stroke="#2563eb"
                                    strokeWidth={2}
                                    dot={{ stroke: '#2563eb', strokeWidth: 2 }}
                                    activeDot={{ r: 6 }}
                                    name="Score"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Commands</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-96 overflow-y-auto">
                            {logData.map((entry, idx) => (
                                <div
                                    key={idx}
                                    onClick={() => setSelectedStep(idx)}
                                    className={`p-4 rounded-md cursor-pointer transition duration-200 ${
                                        selectedStep === idx
                                            ? 'bg-blue-50 border border-blue-200'
                                            : 'hover:bg-gray-50 border border-transparent'
                                    }`}
                                >
                                    <div className="flex justify-between items-center">
                                        <span className="font-medium text-gray-700">Step {entry.step}</span>
                                        <span className="text-sm text-gray-500">{entry.command}</span>
                                    </div>
                                    <div className="text-sm text-gray-600 mt-1">
                                        Score: {formatScore(entry.score)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Step Details</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {selectedStep !== null && logData[selectedStep] ? (
                            <div className="space-y-6">
                                <div>
                                    <h3 className="font-medium text-gray-700">Code Changes</h3>
                                    <pre className="bg-gray-50 p-4 rounded-md mt-2 overflow-x-auto border border-gray-200">
                                        <code className="text-sm">{logData[selectedStep].code_lines.join('\n')}</code>
                                    </pre>
                                </div>
                                <div>
                                    <h3 className="font-medium text-gray-700">Model Response</h3>
                                    <p className="text-gray-600 mt-2 leading-relaxed">
                                        {logData[selectedStep].model_response}
                                    </p>
                                </div>
                                <div className="text-sm text-gray-500">
                                    {new Date(logData[selectedStep].timestamp).toLocaleString()}
                                </div>
                            </div>
                        ) : (
                            <div className="text-center text-gray-500 py-8">
                                Select a step to view details
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default AgentLabVisualization;