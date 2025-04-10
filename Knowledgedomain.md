# Knowledge Domain Agent for Agent Laboratory

## Overview

The Knowledge Domain is a new addition to the Agent Laboratory framework that provides specialized domain knowledge to guide research in specific fields. This agent works alongside the existing agents (PhD Student, Postdoc, Professor, ML Engineer, and Software Engineer) to provide domain-specific insights, methodologies, and evaluation criteria.

## Supported Domains

The Knowledge Domain Agent currently supports the following domains:

- **biopharmaceutical**: Specialized knowledge for drug discovery, biomarkers, clinical trials, and regulatory considerations.
- **semiconductor**: Expertise in chip design, fabrication processes, materials science, and electronic design automation.
- **optimization**: Knowledge of mathematical programming, constraint satisfaction, metaheuristics, and operational research.
- **reinforcement_learning**: Focus on Markov decision processes, policy gradients, Q-learning, and environment modeling.
- **computer_vision**: Guidance on image processing, convolutional neural networks, object detection, and segmentation.
- **natural_language_processing**: Expertise in language modeling, transformers, and language understanding.
- **generative_ai**: Knowledge of diffusion models, GANs, VAEs, and content generation techniques.
- **general**: Default domain with general machine learning knowledge.

## How It Works

The Knowledge Domain Agent provides specialized insights at each phase of the research process:

1. **Literature Review**: Suggests key concepts, seminal papers, and important authors specific to the domain.
2. **Plan Formulation**: Recommends appropriate methodologies, experimental designs, and evaluation metrics.
3. **Data Preparation**: Provides domain-specific preprocessing techniques and feature engineering approaches.
4. **Running Experiments**: Suggests domain-specific hyperparameters, model architectures, and training strategies.
5. **Results Interpretation**: Offers domain-specific metrics, analysis techniques, and contextual interpretations.

## Usage

To use the Knowledge Domain Agent, simply specify the domain when running Agent Laboratory:

```bash
python ai_lab_repo.py --api-key "YOUR_API_KEY" --llm-backend "o1-mini" --research-topic "YOUR_RESEARCH_IDEA" --domain "biopharmaceutical"
```

Available domain options:
- general (default)
- biopharmaceutical
- semiconductor
- optimization
- reinforcement_learning
- computer_vision
- natural_language_processing
- generative_ai

## Examples

Here are some example research topics that could benefit from specific domains:

- **Biopharmaceutical**: "Predicting drug-target interactions using graph neural networks"
- **Semiconductor**: "Defect detection in semiconductor wafer maps using deep learning"
- **Optimization**: "Multi-objective optimization for resource allocation in cloud computing"
- **Reinforcement Learning**: "Sample-efficient reinforcement learning for robotic control"
- **Computer Vision**: "Real-time semantic segmentation for autonomous driving"
- **Natural Language Processing**: "Low-resource language adaptation for large language models"
- **Generative AI**: "Controllable text-to-image generation for product design"

## Extending the Domain Knowledge

The Knowledge Domain Agent can be extended with additional domains or deeper knowledge by modifying the `load_domain_knowledge` method in the `KnowledgeDomainAgent` class. This method loads domain-specific prompts that guide the agent's responses throughout the research workflow.

To add a new domain, simply add a new entry to the `domain_prompts` dictionary in this method with appropriate guidance for that domain.