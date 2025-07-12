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
Anthropic-claude-sonnet-4-20250514       90.062       13.138       15.132       11.268     0.011       0.179        0.00
DeepSeek-deepseek-chat                   93.812        8.085       29.205       24.411     0.001       0.013        0.00
Google-gemini-2.5-pro                    69.622       28.552       21.624       21.594     0.001       0.021        6.25
OpenAI-gpt-4.1                           89.688       12.175        6.379        6.355     0.005       0.083        0.00
xAI-grok-4-0709                          89.518       11.205       24.384       22.662     0.012       0.194        0.00
================================================================================

Best Overall Quality: DeepSeek-deepseek-chat
Fastest Response: OpenAI-gpt-4.1
Most Cost-Effective: DeepSeek-deepseek-chat

📊 Results saved to benchmark_report.json
📈 Visualisations saved to llm_benchmark_results.png and llm_benchmark_detailed.png
🏆 Best Overall Model: DeepSeek-deepseek-chat
📈 Overall Average Quality: 86.5%
💰 Total Cost: $0.4894
⚡ Average Latency: 19.34s
```

### License

This project is licensed under the [MIT License](./LICENSE).

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.