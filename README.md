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
OPENROUTER_API_KEY="your_api_key"

## View Setup Guide
python3 -m scripts.benchmark setup

## Execute The Benchmark
python3 -m scripts.benchmark

## Deactivate Virtual Environment
deactivate
```

### Results

Generated results on the 13th July 2025.

```sh
================================================================================
LLM BENCHMARK SUMMARY
================================================================================
                                    Avg Quality  Quality Std  Avg Latency  Med Latency  Avg Cost  Total Cost  Error Rate
model_name                                                                                                              
Anthropic-claude-sonnet-4-20250514       90.977       11.390       15.438       11.114     0.019       0.298        0.00
DeepSeek-deepseek-chat                   89.502       11.510       34.283       28.485     0.001       0.013        0.00
Google-gemini-2.5-pro                    77.065       26.758       21.421       21.223     0.002       0.026        6.25
Moonshot-moonshot-v1-8k                  91.278       10.618        7.345        6.967     0.001       0.014        0.00
OpenAI-gpt-4.1                           90.628       10.935        7.645        5.598     0.010       0.156        0.00
Qwen-qwen3-32b                           48.991       45.993       33.438       37.540     0.000       0.000        0.00
xAI-grok-4-0709                          95.341        7.110       23.227       21.574     0.026       0.409        0.00
================================================================================

Best Overall Quality: xAI-grok-4-0709
Fastest Response: Moonshot-moonshot-v1-8k
Most Cost-Effective: Qwen-qwen3-32b

📁 All results saved to results directory
🏆 Best Overall Model: xAI-grok-4-0709
📈 Overall Average Quality: 83.4%
💰 Total Cost: $0.9164
⚡ Average Latency: 20.40s
```

### License

This project is licensed under the [Modified MIT License](./LICENSE).

### Citation

```tex
@misc{llmbenchmark,
  author       = {Oketunji, A.F.},
  title        = {LLM Benchmark},
  year         = 2025,
  version      = {0.0.5},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15875079},
  url          = {https://doi.org/10.5281/zenodo.15875079}
}
```

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.
