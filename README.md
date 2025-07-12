## llm-benchmark

Test and compare different large language models on various tasks.

### Tasks

+ Code Generation
+ Mathematical Reasoning
+ Creative Writing
+ Data Analysis
+ Logical Reasoning
+ Summarisation
+ Technical Explanation
+ Problem Solving

### Install Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following

```python
## Prerequisites
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python3 -m pip install --upgrade pip
```

### LLM Benchmark

```python
## Set Environment Variables
OPENAI_API_KEY="your_api_key"
ANTHROPIC_API_KEY="your_api_key"
XAI_API_KEY="your_api_key"
DEEPSEEK_TOKEN="your_api_key"
GOOGLE_API_KEY="your_api_key"

## View Setup Guide
python3 -m benchmark setup

## Execute The Benchmark
python3 -m benchmark

## Deactivate Virtual Environment
deactivate
```

### Results

Generated results on the 12th July 2025.

```sh
================================================================================
LLM BENCHMARK SUMMARY
================================================================================
                                    Avg Quality  Quality Std  Avg Latency  Med Latency  Avg Cost  Total Cost  Error Rate
model_name                                                                                                              
Anthropic-claude-sonnet-4-20250514       87.928       14.031       14.919       10.779     0.011       0.178         0.0
DeepSeek-deepseek-chat                   89.875       10.455       34.939       28.055     0.001       0.014         0.0
Google-gemini-2.5-pro                    78.654       20.606       22.060       22.571     0.002       0.025         0.0
OpenAI-gpt-4.1                           89.375       14.818        7.834        7.589     0.005       0.079         0.0
xAI-grok-4-0709                          92.926       13.368       21.227       20.889     0.013       0.202         0.0
================================================================================

Best Overall Quality: xAI-grok-4-0709
Fastest Response: OpenAI-gpt-4.1
Most Cost-Effective: DeepSeek-deepseek-chat

📊 Results saved to benchmark_report.json
📈 Visualisations saved to llm_benchmark_results.png and llm_benchmark_detailed.png
🏆 Best Overall Model: xAI-grok-4-0709
📈 Overall Average Quality: 87.8%
💰 Total Cost: $0.4977
⚡ Average Latency: 20.20s
```

#### Results Inconsistencies

The variability in model rankings during the benchmark tests for each run can be attributed to several factors:

+ Randomness in Model Responses: Many language models incorporate some level of randomness in their responses. It can lead to different outputs for the same input across multiple runs, affecting the quality scores.
+ Statistical Variability: The average quality and latency metrics are calculated from multiple runs. If the number of runs is small, the results can be influenced by outliers or variations in performance during those specific runs.
+ API Load and Latency: The performance of models accessed via API can be affected by server load, network latency, and other external factors. Variations in these conditions can lead to differences in response times and potentially in the quality of responses.

### License

This project is licensed under the [MIT License](./LICENSE).

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.