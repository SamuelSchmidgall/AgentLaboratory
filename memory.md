# Personalized Research with Memory in Agent Laboratory

## Overview

The Memory Agent integration with Mem0 enhances Agent Laboratory by providing a persistent memory layer that remembers researcher preferences, past research topics, and insights across multiple research sessions. This enables a more personalized and contextually aware research experience.

## Key Benefits

1. **Researcher Continuity**: The system remembers your previous research topics, preferences, and insights, creating a continuous research experience across multiple sessions.

2. **Personalized Guidance**: Agents adapt their recommendations based on your past preferences and research interests.

3. **Context Awareness**: The system maintains context across research phases and across separate research sessions.

4. **Knowledge Accumulation**: As you conduct more research, the system becomes increasingly tailored to your specific research style and interests.

5. **Cost Efficiency**: By remembering important context, the system reduces redundant LLM calls and token usage.

## How It Works

The Memory Agent integrates with [Mem0](https://mem0.ai/), a specialized memory service for AI applications. It stores and retrieves key research information at each phase:

1. **Research Topic Context**: Stores the research topic and domain to establish context.

2. **Literature Review Insights**: Remembers important papers and key findings from literature reviews.

3. **Research Plan Preferences**: Stores methodological preferences and experimental approaches.

4. **Dataset Preferences**: Remembers your preferred datasets and data preparation techniques.

5. **Experimental Results**: Maintains a history of experiment outcomes and insights.

6. **Results Interpretation**: Stores analytical approaches and conclusions.

7. **Research Reports**: Maintains summaries of completed research projects.

## Setup Requirements

1. **Mem0 API Key**: Sign up at [mem0.ai](https://mem0.ai/) to obtain an API key.

2. **Install Mem0 Package**: 
   ```bash
   pip install mem0ai
   ```

3. **Researcher ID**: Create a unique identifier for yourself to maintain personalized memory.

## Usage

To enable the memory features, provide your Mem0 API key and researcher ID when running Agent Laboratory:

```bash
python ai_lab_repo.py --api-key "YOUR_API_KEY" --llm-backend "o1-mini" \
  --research-topic "YOUR_RESEARCH_IDEA" --domain "computer_vision" \
  --mem0-api-key "YOUR_MEM0_API_KEY" --researcher-id "YOUR_UNIQUE_ID"
```

You can also set the MEM0_API_KEY as an environment variable:

```bash
export MEM0_API_KEY="your-mem0-api-key"
python ai_lab_repo.py --api-key "YOUR_API_KEY" --research-topic "YOUR_IDEA" --researcher-id "YOUR_ID"
```

## Example Use Cases

1. **Research Series**: Conduct a series of related research projects that build upon each other, with the system remembering insights from previous projects.

2. **Methodology Development**: Refine experimental methodologies over time, with the system learning your preferred approaches.

3. **Domain Expertise Building**: As you conduct more research in a specific domain, the system builds a personalized knowledge base in that area.

4. **Literature Tracking**: The system remembers which papers you've already reviewed and can avoid recommending the same papers repeatedly.

## Privacy and Data Control

- All your research data is stored in your Mem0 account, which you control.
- You can delete memories at any time through the Mem0 dashboard.
- Your API key and researcher ID are required to access your memories, ensuring privacy.

## Limitations

- Memory features are only available when both Mem0 API key and researcher ID are provided.
- The system works best with consistent researcher IDs across sessions.
- Very large research artifacts (like full code or reports) may be summarized for efficient storage.