# Idea to Execution IP

A Machine Learning pipeline that help in tranforming academic research papers and ideas into executable code for Numin Data. This project uses LLMs to interpret research papers, generate methodology, create a high-level plan, and develop trainable models with error handling capabilities.

## Overview

This pipeline automates the process of converting research ideas (based on flat or sequential datatypes) into working ML implementations through a step-by-step approach:

1. Summarize research papers and user ideas
2. Analyze the summary and extract key points
3. Determine the appropriate model architecture
4. Generate methodology and high-level implementation plan
5. Create training and testing code with automatic error correction
6. Execute the generated code and produce comprehensive reports

## Getting Started

### Prerequisites

* Python 3.8+
* Required Python packages (install via `pip install -r requirements.txt`):
  * google-generativeai
  * openai
  * anthropic
  * pypdf
  * pyyaml
  * subprocess

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/idea-to-execution-ip.git
cd idea-to-execution-ip

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. Add your initial idea in `user-idea.txt`
2. Add reference research papers in the `papers/` folder
3. Update the `config.yaml` file with:
   * Your attempt name
   * LLM model choice and API key
   * File paths to your data directories
4. Run the pipeline:
   ```bash
   python final-pipeline.py
   ```
5. Check logs for your attempt in the `logs/` folder
6. Find the final training and testing code in the `final_code_files/` folder
7. Check the generated report in the `report/` folder

### Configuration File Example

```yaml
---
name: idea-to-execution-ip
version: 1.0

llm_model: "claude-3-sonnet-20240229"  # Choose your LLM model
api_key: "your-api-key-here"           # Your API key
attempt_name: "my_first_run"           # Name your attempt
user_idea_path: 'prompts/idea.txt'     # Path to your idea file
paper_paths:                           # List of research papers
  - 'papers/paper1.txt'
  - 'papers/paper2.txt'

file_paths:
  training_data: 'training_data/data.csv'
  output_directory: 'saved_models'
  temp_numin_results_path: 'API_submission_temp'
...
```

## Pipeline Steps

1. **Summarization** : Condense research papers and user ideas
2. **Analysis** : Extract key insights from the summarized content
3. **Model Selection** : Determine appropriate model architecture
4. **Methodology Generation** : Create a detailed methodology
5. **Planning** : Generate high-level implementation plan
6. **Code Generation** : Create training and testing code
7. **Error Handling** : Automatically debug and fix code errors
8. **Execution** : Run the finalized code and analyze performance
9. **Reporting** : Generate comprehensive technical report

## Supported LLM Models

* Google Gemini models (Gemini 2.0 Flash)
* OpenAI GPT models (GPT-3.5-turbo)
* Anthropic Claude models (Claude-3-Sonnet)

## Directory Structure

```
idea-to-execution-ip/
├── base_codes/            # Base code templates
├── final_code_files/      # Generated code output
├── logs/                  # Execution logs
├── papers/                # Research papers
├── prompts/               # Instruction prompts
│   └── idea.txt           # User idea input
├── prompts_final/         # System prompts
├── report/                # Generated reports
├── training_data/         # Training data
├── config.yaml            # Configuration file
├── final-pipeline.py      # Main pipeline script
├── functionalities.py     # Helper functions
└── README.md              # This file
```

## Contributors

* Mohammad Kaif
* Rohan Indora

## Advisors

* Dr. Gautam Shroff
