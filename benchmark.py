#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@name: benchmark.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Wednesday July 10 05:05:15 2025
@updated: Saturday July 12 02:32:25 2025
@desc: Test and compare different large language models on various tasks.
@run: python3 benchmark.py
"""

import asyncio
import json
import os
from typing import Dict, List, Any
from dataclasses import asdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    BenchmarkResult, BenchmarkTask, LLMInterface,
    OpenAIInterface, AnthropicInterface, DeepSeekInterface, XAIInterface, GeminiInterface,
    QualityEvaluator, APIKeyManager, setup_environment
)

class LLMBenchmark:
    def __init__(self):
        self.models: List[LLMInterface] = []
        self.tasks: List[BenchmarkTask] = []
        self.results: List[BenchmarkResult] = []
        
    def add_model(self, model: LLMInterface):
        """Add a model to benchmark"""
        self.models.append(model)
        
    def add_task(self, task: BenchmarkTask):
        """Add a benchmark task"""
        self.tasks.append(task)
        
    def create_standard_tasks(self):
        """Create standard benchmark tasks"""
        tasks = [
            BenchmarkTask(
                name="Code Generation",
                prompt="Write a Python function that implements binary search on a sorted list. Include proper error handling and documentation.",
                category="Programming",
                expected_length="medium",
                evaluation_criteria=["correctness", "efficiency", "documentation"]
            ),
            BenchmarkTask(
                name="Mathematical Reasoning",
                prompt="Solve this step by step: If a train travels 120 miles in 2 hours, and then increases its speed by 25% for the next 90 miles, what is the total travel time?",
                category="Mathematics",
                expected_length="medium",
                evaluation_criteria=["accuracy", "reasoning", "clarity"]
            ),
            BenchmarkTask(
                name="Creative Writing",
                prompt="Write a short story (200 words) about a robot who discovers emotions. Focus on the moment of discovery.",
                category="Creative",
                expected_length="medium",
                evaluation_criteria=["creativity", "coherence", "emotional depth"]
            ),
            BenchmarkTask(
                name="Data Analysis",
                prompt="Explain the key differences between correlation and causation. Provide practical examples and explain why this distinction matters in data science.",
                category="Analysis",
                expected_length="long",
                evaluation_criteria=["accuracy", "examples", "practical relevance"]
            ),
            BenchmarkTask(
                name="Logical Reasoning",
                prompt="All roses are flowers. Some flowers are red. Some red things are beautiful. Can we conclude that some roses are beautiful? Explain your reasoning.",
                category="Logic",
                expected_length="short",
                evaluation_criteria=["logical validity", "clarity", "completeness"]
            ),
            BenchmarkTask(
                name="Summarisation",
                prompt="Summarise the key principles of effective communication in professional settings. Focus on practical, actionable advice.",
                category="Communication",
                expected_length="medium",
                evaluation_criteria=["completeness", "practicality", "organisation"]
            ),
            BenchmarkTask(
                name="Technical Explanation",
                prompt="Explain how neural networks learn through backpropagation. Make it accessible to someone with basic programming knowledge but no machine learning background.",
                category="Education",
                expected_length="long",
                evaluation_criteria=["clarity", "accuracy", "accessibility"]
            ),
            BenchmarkTask(
                name="Problem Solving",
                prompt="You have 8 balls that look identical. One ball is slightly heavier than the others. Using a balance scale only twice, how can you identify the heavier ball?",
                category="Logic",
                expected_length="medium",
                evaluation_criteria=["logical approach", "efficiency", "clarity"]
            )
        ]
        
        for task in tasks:
            self.add_task(task)
    
    async def run_benchmark(self, iterations: int = 3) -> List[BenchmarkResult]:
        """Run benchmark across all models and tasks"""
        print(f"Running benchmark with {len(self.models)} models and {len(self.tasks)} tasks")
        print(f"Iterations per task: {iterations}")
        
        for model in self.models:
            print(f"\nTesting {model.get_model_name()}...")
            
            for task in self.tasks:
                print(f"  Task: {task.name}")
                
                # Run multiple iterations for each task
                for i in range(iterations):
                    response, metadata = await model.generate(task.prompt)
                    
                    # Calculate quality score
                    quality_score = QualityEvaluator.evaluate_response(task, response)
                    
                    result = BenchmarkResult(
                        model_name=model.get_model_name(),
                        task_name=task.name,
                        response=response,
                        latency=metadata["latency"],
                        input_tokens=int(metadata["input_tokens"]),
                        output_tokens=int(metadata["output_tokens"]),
                        cost_estimate=model.estimate_cost(
                            int(metadata["input_tokens"]), 
                            int(metadata["output_tokens"])
                        ),
                        timestamp=datetime.now(),
                        quality_score=quality_score,
                        error=metadata.get("error")
                    )
                    
                    self.results.append(result)
                    
                    # Brief pause between requests
                    await asyncio.sleep(0.1)
        
        return self.results
    
    def analyse_results(self) -> Dict[str, Any]:
        """Analyse benchmark results"""
        if not self.results:
            return {"error": "No results to analyse"}
        
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Calculate metrics by model
        model_metrics = {}
        
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            
            model_metrics[model] = {
                'avg_latency': model_data['latency'].mean(),
                'median_latency': model_data['latency'].median(),
                'avg_cost': model_data['cost_estimate'].mean(),
                'total_tokens': model_data['input_tokens'].sum() + model_data['output_tokens'].sum(),
                'error_rate': (model_data['error'].notna().sum() / len(model_data)) * 100,
                'avg_response_length': model_data['response'].str.len().mean(),
                'avg_quality_score': model_data['quality_score'].mean(),
                'quality_score_std': model_data['quality_score'].std(),
                'min_quality_score': model_data['quality_score'].min(),
                'max_quality_score': model_data['quality_score'].max()
            }
        
        # Task-specific analysis
        task_metrics = {}
        for task in df['task_name'].unique():
            task_data = df[df['task_name'] == task]
            
            task_metrics[task] = {
                'models_tested': task_data['model_name'].nunique(),
                'avg_latency_across_models': task_data['latency'].mean(),
                'response_length_variance': task_data['response'].str.len().var(),
                'avg_quality_score_across_models': task_data['quality_score'].mean(),
                'quality_score_range': task_data['quality_score'].max() - task_data['quality_score'].min(),
                'best_performing_model': task_data.loc[task_data['quality_score'].idxmax(), 'model_name'] if len(task_data) > 0 else None
            }
        
        return {
            'model_metrics': model_metrics,
            'task_metrics': task_metrics,
            'overall_stats': {
                'total_requests': len(df),
                'total_cost': df['cost_estimate'].sum(),
                'avg_latency': df['latency'].mean(),
                'overall_avg_quality_score': df['quality_score'].mean(),
                'quality_score_std': df['quality_score'].std(),
                'best_overall_model': df.groupby('model_name')['quality_score'].mean().idxmax() if len(df) > 0 else None
            }
        }
    
    def generate_report(self, output_file: str = "benchmark_report.json"):
        """Generate detailed benchmark report"""
        analysis = self.analyse_results()
        
        report = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'models_tested': len(self.models),
                'tasks_completed': len(self.tasks),
                'total_requests': len(self.results)
            },
            'results': analysis,
            'raw_data': [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {output_file}")
        return report
    
    def create_visualisations(self):
        """Create visualisations of benchmark results"""
        if not self.results:
            print("No results to visualise")
            return
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Quality Score comparison
        quality_by_model = df.groupby('model_name')['quality_score'].mean()
        axes[0, 0].bar(quality_by_model.index, quality_by_model.values, color='lightblue')
        axes[0, 0].set_title('Average Quality Score by Model')
        axes[0, 0].set_ylabel('Quality Score (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add score labels on bars
        for i, v in enumerate(quality_by_model.values):
            axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 2. Latency comparison
        axes[0, 1].boxplot([df[df['model_name'] == model]['latency'].values 
                           for model in df['model_name'].unique()],
                          labels=df['model_name'].unique())
        axes[0, 1].set_title('Response Latency by Model')
        axes[0, 1].set_ylabel('Latency (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Cost comparison
        cost_by_model = df.groupby('model_name')['cost_estimate'].mean()
        axes[0, 2].bar(cost_by_model.index, cost_by_model.values, color='lightcoral')
        axes[0, 2].set_title('Average Cost per Request by Model')
        axes[0, 2].set_ylabel('Cost ($)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Task performance heatmap (Quality Scores)
        pivot_quality = df.pivot_table(values='quality_score', 
                                      index='task_name', 
                                      columns='model_name', 
                                      aggfunc='mean')
        sns.heatmap(pivot_quality, annot=True, fmt='.1f', ax=axes[1, 0], cmap='RdYlGn', vmin=0, vmax=100)
        axes[1, 0].set_title('Quality Scores by Task and Model (%)')
        
        # 5. Quality vs Latency scatter
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            axes[1, 1].scatter(model_data['latency'], model_data['quality_score'], 
                             label=model, alpha=0.7, s=50)
        axes[1, 1].set_xlabel('Latency (seconds)')
        axes[1, 1].set_ylabel('Quality Score (%)')
        axes[1, 1].set_title('Quality vs Latency Trade-off')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Quality vs Cost scatter
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            axes[1, 2].scatter(model_data['cost_estimate'], model_data['quality_score'], 
                             label=model, alpha=0.7, s=50)
        axes[1, 2].set_xlabel('Cost ($)')
        axes[1, 2].set_ylabel('Quality Score (%)')
        axes[1, 2].set_title('Quality vs Cost Trade-off')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('llm_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed analysis charts
        self._create_detailed_charts(df)
    
    def _create_detailed_charts(self, df):
        """Create additional detailed analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cost efficiency (quality per dollar)
        model_efficiency = df.groupby('model_name').apply(
            lambda x: x['quality_score'].mean() / (x['cost_estimate'].mean() * 1000)  # Quality per $0.001
        ).sort_values(ascending=False)
        
        axes[0, 0].bar(model_efficiency.index, model_efficiency.values, color='lightgreen')
        axes[0, 0].set_title('Cost Efficiency (Quality Score per $0.001)')
        axes[0, 0].set_ylabel('Quality Score per $0.001')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add efficiency labels
        for i, v in enumerate(model_efficiency.values):
            axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        # 2. Speed efficiency (quality per second)
        speed_efficiency = df.groupby('model_name').apply(
            lambda x: x['quality_score'].mean() / x['latency'].mean()
        ).sort_values(ascending=False)
        
        axes[0, 1].bar(speed_efficiency.index, speed_efficiency.values, color='orange')
        axes[0, 1].set_title('Speed Efficiency (Quality Score per Second)')
        axes[0, 1].set_ylabel('Quality Score per Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Task category performance
        category_performance = df.merge(
            pd.DataFrame([(task.name, task.category) for task in self.tasks], 
                        columns=['task_name', 'category']),
            on='task_name'
        ).groupby(['model_name', 'category'])['quality_score'].mean().unstack(fill_value=0)
        
        category_performance.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Performance by Task Category')
        axes[1, 0].set_ylabel('Average Quality Score (%)')
        axes[1, 0].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Error rates
        error_rates = df.groupby('model_name').apply(
            lambda x: (x['error'].notna().sum() / len(x)) * 100
        )
        
        axes[1, 1].bar(error_rates.index, error_rates.values, color='red', alpha=0.7)
        axes[1, 1].set_title('Error Rate by Model')
        axes[1, 1].set_ylabel('Error Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('llm_benchmark_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary_table(self):
        """Print a formatted summary table of results"""
        if not self.results:
            print("No results to summarise")
            return
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Calculate summary metrics
        summary = df.groupby('model_name').agg({
            'quality_score': ['mean', 'std'],
            'latency': ['mean', 'median'],
            'cost_estimate': ['mean', 'sum'],
            'error': lambda x: (x.notna().sum() / len(x)) * 100
        }).round(3)
        
        summary.columns = ['Avg Quality', 'Quality Std', 'Avg Latency', 'Med Latency', 'Avg Cost', 'Total Cost', 'Error Rate']
        
        print("\n" + "="*80)
        print("LLM BENCHMARK SUMMARY")
        print("="*80)
        print(summary.to_string())
        print("="*80)
        
        # Print best performers in each category
        best_quality = df.groupby('model_name')['quality_score'].mean().idxmax()
        best_speed = df.groupby('model_name')['latency'].mean().idxmin()
        best_cost = df.groupby('model_name')['cost_estimate'].mean().idxmin()
        
        print(f"\nBest Overall Quality: {best_quality}")
        print(f"Fastest Response: {best_speed}")
        print(f"Most Cost-Effective: {best_cost}")

async def main():
    """Main function to run the benchmark"""
    print("🚀 LLM Benchmark Suite")
    print("="*50)
    
    # Load API keys
    print("Setting up API keys...")
    api_keys = APIKeyManager.load_from_env()
    
    # Fallback to .env file
    if not any(api_keys.values()):
        print("No environment variables found, trying .env file...")
        api_keys = APIKeyManager.load_from_file('.env')
    
    # Debug: Print which keys were found (without revealing the actual keys)
    print("API keys found:")
    for provider, key in api_keys.items():
        status = "✓" if key else "✗"
        print(f"  {status} {provider}: {'Available' if key else 'Not found'}")
    
    # Additional debugging for environment variables
    print("\nEnvironment variable check:")
    env_vars_to_check = [
        'DEEPSEEK_API_KEY', 'DEEPSEEK_TOKEN', 'DEEPSEEK_KEY', 'DS_API_KEY',
        'XAI_API_KEY', 'XAI_TOKEN', 'XAI_KEY', 'GROK_API_KEY',
        'GOOGLE_API_KEY', 'GEMINI_API_KEY'
    ]
    for var in env_vars_to_check:
        value = os.getenv(var)
        print(f"  {var}: {'Set' if value else 'Not set'}")
    
    # Initialise benchmark
    benchmark = LLMBenchmark()
    
    # Add models based on available API keys
    models_added = 0
    
    if api_keys.get('openai'):
        try:
            benchmark.add_model(OpenAIInterface(api_keys['openai'], "gpt-4.1"))
            print("✓ Added OpenAI GPT-4.1")
            models_added += 1
        except Exception as e:
            print(f"✗ Failed to add OpenAI: {e}")
    
    if api_keys.get('anthropic'):
        try:
            benchmark.add_model(AnthropicInterface(api_keys['anthropic'], "claude-sonnet-4-20250514"))
            print("✓ Added Anthropic Claude 4 Sonnet")
            models_added += 1
        except Exception as e:
            print(f"✗ Failed to add Anthropic: {e}")
    
    if api_keys.get('deepseek'):
        try:
            benchmark.add_model(DeepSeekInterface(api_keys['deepseek'], "deepseek-chat"))
            print("✓ Added DeepSeek V3")
            models_added += 1
        except Exception as e:
            print(f"✗ Failed to add DeepSeek: {e}")
    
    if api_keys.get('xai'):
        try:
            benchmark.add_model(XAIInterface(api_keys['xai'], "grok-4-0709"))
            print("✓ Added xAI Grok")
            models_added += 1
        except Exception as e:
            print(f"✗ Failed to add xAI: {e}")
    
    if api_keys.get('google'):
        try:
            benchmark.add_model(GeminiInterface(api_keys['google'], "gemini-2.5-pro"))
            print("✓ Added Google Gemini 2.5 Pro")
            models_added += 1
        except Exception as e:
            print(f"✗ Failed to add Google Gemini: {e}")
    
    if models_added == 0:
        print("\n❌ No models were successfully added!")
        print("\nTo fix this, set your API keys using one of these methods:")
        print("\n1. Environment variables:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        print("   export DEEPSEEK_API_KEY='sk-...'")
        print("   export XAI_API_KEY='xai-...'")
        print("   export GOOGLE_API_KEY='AIza...'")
        print("\n2. Create a .env file:")
        print("   OPENAI_API_KEY=sk-...")
        print("   ANTHROPIC_API_KEY=sk-ant-...")
        print("   DEEPSEEK_API_KEY=sk-...")
        print("   XAI_API_KEY=xai-...")
        print("   GOOGLE_API_KEY=AIza...")
        print("\n3. Set them directly in the code (testing only)")
        return
    
    print(f"\n🚀 Successfully configured {models_added} models")
    
    # Create standard benchmark tasks
    benchmark.create_standard_tasks()
    print(f"Created {len(benchmark.tasks)} benchmark tasks")
    
    # Run benchmark
    print("\nStarting benchmark run...")
    iterations = 2  # Adjust as needed
    await benchmark.run_benchmark(iterations=iterations)
    
    # Analyse and generate report
    print("\nAnalysing results...")
    analysis = benchmark.analyse_results()
    
    # Print summary table
    benchmark.print_summary_table()
    
    # Generate detailed report
    benchmark.generate_report()
    
    # Create visualisations
    print("\nCreating visualisations...")
    benchmark.create_visualisations()
    
    print("\n✅ Benchmark completed successfully!")
    print("📊 Results saved to benchmark_report.json")
    print("📈 Visualisations saved to llm_benchmark_results.png and llm_benchmark_detailed.png")
    
    # Print key insights
    if 'overall_stats' in analysis:
        stats = analysis['overall_stats']
        print(f"\n🏆 Best Overall Model: {stats['best_overall_model']}")
        print(f"📈 Overall Average Quality: {stats['overall_avg_quality_score']:.1f}%")
        print(f"💰 Total Cost: ${stats['total_cost']:.4f}")
        print(f"⚡ Average Latency: {stats['avg_latency']:.2f}s")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_environment()
    else:
        asyncio.run(main())
