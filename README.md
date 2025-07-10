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

## View Setup Guide
python3 -m benchmark setup

## Execute The Benchmark
python3 -m benchmark

## Deactivate Virtual Environment
deactivate
```

### Results

```sh
================================================================================
LLM BENCHMARK SUMMARY
================================================================================
                                    Avg Quality  Quality Std  Avg Latency  Med Latency  Avg Cost  Total Cost  Error Rate
model_name                                                                                                              
Anthropic-claude-sonnet-4-20250514       90.073       11.228       15.256       11.630     0.011       0.178         0.0
DeepSeek-deepseek-chat                   90.972        7.975       35.278       32.898     0.001       0.013         0.0
OpenAI-gpt-4.1                           90.938       11.287       10.828        9.581     0.005       0.082         0.0
xAI-grok-4-0709                          93.392        9.310       33.799       28.474     0.012       0.190         0.0
================================================================================

Best Overall Quality: xAI-grok-4-0709
Fastest Response: OpenAI-gpt-4.1
Most Cost-Effective: DeepSeek-deepseek-chat

🏆 Best Overall Model: xAI-grok-4-0709
📈 Overall Average Quality: 91.3%
💰 Total Cost: $0.4635
⚡ Average Latency: 23.79s
```

### License

This project is licensed under the [MIT License](./LICENSE).

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.