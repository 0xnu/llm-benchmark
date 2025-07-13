## llm-benchmark

[![Lint](https://github.com/0xnu/llm-benchmark/actions/workflows/lint.yaml/badge.svg)](https://github.com/0xnu/llm-benchmark/actions/workflows/lint.yaml)
[![Release](https://img.shields.io/github/release/0xnu/llm-benchmark.svg)](https://github.com/0xnu/llm-benchmark/releases/latest)
[![License](https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53)](/LICENSE)

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
MOONSHOT_API_KEY="your_api_key"

## View Setup Guide
python3 -m scripts.benchmark setup

## Execute The Benchmark
python3 -m scripts.benchmark

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
Anthropic-claude-sonnet-4-20250514       90.125       11.258       14.971       10.418     0.019       0.300        0.00
DeepSeek-deepseek-chat                   89.460       10.532       43.017       35.208     0.001       0.014        0.00
Google-gemini-2.5-pro                    74.023       25.900       23.143       22.912     0.001       0.023        6.25
Moonshot-moonshot-v1-8k                  90.069       14.777        8.926        8.761     0.001       0.014        0.00
OpenAI-gpt-4.1                           89.375       10.782        8.003        6.904     0.010       0.164        0.00
xAI-grok-4-0709                          91.003       11.831       21.290       21.010     0.024       0.380        0.00
================================================================================

Best Overall Quality: xAI-grok-4-0709
Fastest Response: OpenAI-gpt-4.1
Most Cost-Effective: Moonshot-moonshot-v1-8k

📊 Results saved to results folder
🏆 Best Overall Model: xAI-grok-4-0709
📈 Overall Average Quality: 87.3%
💰 Total Cost: $0.8949
⚡ Average Latency: 19.89s
```

#### Results Inconsistencies

The variability in model rankings during the benchmark tests for each run can be attributed to several factors:

+ Randomness in Model Responses: Many language models incorporate some level of randomness in their responses. It can lead to different outputs for the same input across multiple runs, affecting the quality scores.
+ Statistical Variability: The average quality and latency metrics are calculated from multiple runs. If the number of runs is small, the results can be influenced by outliers or variations in performance during those specific runs.
+ API Load and Latency: The performance of models accessed via API can be affected by server load, network latency, and other external factors. Variations in these conditions can lead to differences in response times and potentially in the quality of responses.

### License

This project is licensed under the [Modified MIT License](./LICENSE).

### Citation

```tex
@misc{llmbenchmark,
  author       = {Oketunji, A.F.},
  title        = {LLM Benchmark},
  year         = 2025,
  version      = {0.0.4},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15872839},
  url          = {https://doi.org/10.5281/zenodo.15872839}
}
```

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.
