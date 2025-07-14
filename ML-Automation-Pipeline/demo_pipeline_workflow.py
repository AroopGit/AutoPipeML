#!/usr/bin/env python3
"""
Demo: AutoML Pipeline Workflow Example
=====================================

Quick Start for Your Own Data (CLI):
------------------------------------
# 1. Place your CSV file in the project directory (e.g., my_data.csv)
# 2. Run this command in your terminal:
#
#   python -m ml_automation_pipeline.cli run --data my_data.csv --target target_column
#
# 3. The pipeline will be optimized, evaluated, and saved automatically.
#    - Check the outputs/ directory for the exported pipeline and results.
#
# 4. To use the pipeline for predictions later:
#
#   from ml_automation_pipeline.automl import AutoMLPipeline
#   pipeline = AutoMLPipeline.load('outputs/job_xxx_pipeline.pkl')
#   predictions = pipeline.predict(new_data)
#
# That's it! No code changes needed—just your CSV and a single CLI command.

This script demonstrates how the Automated ML Pipeline System works,
showing the complete workflow from data loading to pipeline optimization
and evaluation. Each step is thoroughly commented to explain the process.

Inspired by TPOT (Tree-based Pipeline Optimization Tool), this system uses
genetic programming to automatically find the best machine learning pipeline.
"""

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
import time

# Import our AutoML system
from ml_automation_pipeline.automl import AutoMLPipeline
from ml_automation_pipeline.utils import load_data, validate_data

console = Console()

def create_demo_data():
    """
    Create a realistic demo dataset for classification.
    
    This simulates a real-world scenario where we have:
    - Multiple features with different scales
    - Some missing values
    - Categorical target variable
    - Mixed data types
    """
    console.print(Panel("[bold blue]Step 1: Creating Demo Dataset[/bold blue]"))
    
    np.random.seed(42)
    n_samples = 200
    
    # Generate features with different characteristics
    feature1 = np.random.normal(0, 1, n_samples)  # Standard normal
    feature2 = np.random.uniform(0, 100, n_samples)  # Uniform 0-100
    feature3 = np.random.exponential(2, n_samples)  # Exponential
    feature4 = np.random.poisson(5, n_samples)  # Poisson
    feature5 = np.random.binomial(10, 0.3, n_samples)  # Binomial
    
    # Create target based on feature relationships (realistic pattern)
    target = ((feature1 > 0.5) & (feature2 > 50)) | (feature3 > 3)
    target = target.astype(int)
    
    # Add some missing values (realistic scenario)
    missing_mask = np.random.random(n_samples) < 0.1  # 10% missing
    feature1[missing_mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'feature5': feature5,
        'target': target
    })
    
    console.print(f"✅ Created dataset with {n_samples} samples and {len(df.columns)-1} features")
    console.print(f"📊 Target distribution: {df['target'].value_counts().to_dict()}")
    console.print(f"🔍 Missing values: {df.isnull().sum().sum()}")
    
    return df

def demonstrate_data_validation(df):
    """
    Demonstrate data validation and preprocessing checks.
    
    This step ensures our data is ready for AutoML by:
    - Checking for missing values
    - Validating data types
    - Detecting problem type (classification vs regression)
    - Identifying feature characteristics
    """
    console.print(Panel("[bold blue]Step 2: Data Validation & Analysis[/bold blue]"))
    
    # Validate the data
    validation_results = validate_data(df, 'target')
    
    console.print("🔍 Data Validation Results:")
    console.print(f"   - Valid: {validation_results['is_valid']}")
    console.print(f"   - Problem Type: {validation_results['problem_type']}")
    console.print(f"   - Numeric Features: {len(validation_results['numeric_columns'])}")
    console.print(f"   - Categorical Features: {len(validation_results['categorical_columns'])}")
    
    if validation_results['warnings']:
        console.print("⚠️  Warnings:")
        for warning in validation_results['warnings']:
            console.print(f"   - {warning}")
    
    # Show target information
    target_info = validation_results['target_info']
    console.print(f"🎯 Target Analysis:")
    console.print(f"   - Type: {target_info['type']}")
    if target_info['type'] == 'categorical':
        console.print(f"   - Classes: {target_info['unique_values']}")
        console.print(f"   - Distribution: {target_info['class_distribution']}")
    else:
        console.print(f"   - Range: {target_info['min']:.2f} to {target_info['max']:.2f}")
        console.print(f"   - Mean: {target_info['mean']:.2f}")
    
    return validation_results

def demonstrate_pipeline_optimization(df, validation_results):
    """
    Demonstrate the core AutoML pipeline optimization process.
    
    This is where the genetic programming magic happens:
    1. Initialize genetic algorithm parameters
    2. Create initial population of pipeline candidates
    3. Evaluate each pipeline using cross-validation
    4. Apply genetic operations (selection, crossover, mutation)
    5. Evolve the population over multiple generations
    6. Find the best performing pipeline
    """
    console.print(Panel("[bold blue]Step 3: Pipeline Optimization with Genetic Programming[/bold blue]"))
    
    # Prepare data
    X = df.drop(columns=['target'])
    y = df['target']
    
    console.print("🧬 Genetic Programming Configuration:")
    console.print("   - Generations: 5 (evolution cycles)")
    console.print("   - Population Size: 20 (pipeline candidates)")
    console.print("   - CV Folds: 3 (cross-validation)")
    console.print("   - Problem Type: Auto-detected")
    
    # Initialize AutoML pipeline
    automl = AutoMLPipeline(
        generations=5,           # Number of evolution cycles
        population_size=20,      # Number of pipeline candidates
        cv_folds=3,             # Cross-validation folds
        random_state=42,        # For reproducibility
        problem_type='auto'     # Auto-detect classification/regression
    )
    
    console.print("\n🚀 Starting Genetic Programming Optimization...")
    console.print("   This process will:")
    console.print("   1. Create 20 random pipeline candidates")
    console.print("   2. Evaluate each using 3-fold cross-validation")
    console.print("   3. Select best performers for breeding")
    console.print("   4. Create new pipelines through crossover/mutation")
    console.print("   5. Repeat for 5 generations")
    console.print("   6. Return the best pipeline found")
    
    # Run optimization with progress tracking
    with Progress() as progress:
        task = progress.add_task("Optimizing pipelines...", total=5)
        
        # Fit the pipeline (this runs the genetic algorithm)
        automl.fit(X, y)
        
        # Update progress for each generation
        for gen in range(5):
            progress.update(task, advance=1)
            time.sleep(0.5)  # Simulate processing time
    
    console.print("\n✅ Optimization Complete!")
    console.print(f"🏆 Best Score: {automl.best_score:.4f}")
    console.print(f"🔧 Pipeline Steps: {[step[0] for step in automl.best_pipeline.steps]}")
    
    return automl

def demonstrate_pipeline_analysis(automl, X, y):
    """
    Demonstrate detailed analysis of the optimized pipeline.
    
    This step shows:
    - Pipeline architecture and components
    - Performance metrics
    - Feature importance (if available)
    - Model interpretability insights
    """
    console.print(Panel("[bold blue]Step 4: Pipeline Analysis & Evaluation[/bold blue]"))
    
    # Get test predictions
    X_train, X_test, y_train, y_test = automl.X_train, automl.X_test, automl.y_train, automl.y_test
    
    # Evaluate on test set
    metrics = automl.evaluate(X_test, y_test)
    
    console.print("📊 Performance Metrics:")
    for metric, value in metrics.items():
        console.print(f"   - {metric.capitalize()}: {value:.4f}")
    
    # Analyze pipeline structure
    console.print("\n🔧 Pipeline Architecture:")
    for i, (name, component) in enumerate(automl.best_pipeline.steps):
        console.print(f"   {i+1}. {name}: {type(component).__name__}")
        
        # Show component details
        if hasattr(component, 'get_params'):
            params = component.get_params()
            if len(params) > 0:
                console.print(f"      Parameters: {params}")
    
    # Show feature importance if available
    try:
        if hasattr(automl.best_pipeline.named_steps['estimator'], 'feature_importances_'):
            importances = automl.best_pipeline.named_steps['estimator'].feature_importances_
            feature_names = X.columns
            
            console.print("\n🎯 Feature Importance:")
            importance_table = Table()
            importance_table.add_column("Feature", style="cyan")
            importance_table.add_column("Importance", style="green")
            
            for name, importance in zip(feature_names, importances):
                importance_table.add_row(name, f"{importance:.4f}")
            
            console.print(importance_table)
    except:
        console.print("\nℹ️  Feature importance not available for this estimator type")
    
    return metrics

def demonstrate_pipeline_export(automl):
    """
    Demonstrate exporting the optimized pipeline for production use.
    
    This shows how to:
    - Save the pipeline for later use
    - Export as Python code
    - Deploy the pipeline in production
    """
    console.print(Panel("[bold blue]Step 5: Pipeline Export & Deployment[/bold blue]"))
    
    # Export pipeline as Python code
    export_file = "demo_optimized_pipeline.py"
    automl.export(export_file)
    
    console.print(f"📝 Exported pipeline to: {export_file}")
    console.print("   This file contains the complete pipeline code that can be:")
    console.print("   - Imported into other Python scripts")
    console.print("   - Deployed in production systems")
    console.print("   - Version controlled and shared")
    
    # Save pipeline object
    pipeline_file = "demo_pipeline.pkl"
    automl.save(pipeline_file)
    
    console.print(f"💾 Saved pipeline object to: {pipeline_file}")
    console.print("   This can be loaded later for predictions:")
    console.print("   ```python")
    console.print("   from ml_automation_pipeline.automl import AutoMLPipeline")
    console.print("   pipeline = AutoMLPipeline.load('demo_pipeline.pkl')")
    console.print("   predictions = pipeline.predict(new_data)")
    console.print("   ```")
    
    return export_file, pipeline_file

def demonstrate_prediction_workflow(automl, X_sample):
    """
    Demonstrate how to use the optimized pipeline for predictions.
    
    This shows the complete prediction workflow:
    - Loading new data
    - Preprocessing (handled automatically by pipeline)
    - Making predictions
    - Interpreting results
    """
    console.print(Panel("[bold blue]Step 6: Making Predictions with Optimized Pipeline[/bold blue]"))
    
    # Create sample data for prediction
    console.print("📥 New data for prediction:")
    console.print(X_sample.head())
    
    # Make predictions
    predictions = automl.predict(X_sample)
    console.print(f"\n🎯 Predictions: {predictions}")
    
    # If classification, show probabilities
    if hasattr(automl, '_is_classification') and automl._is_classification:
        try:
            probabilities = automl.predict_proba(X_sample)
            console.print(f"📊 Prediction Probabilities:")
            console.print(probabilities)
        except:
            console.print("ℹ️  Probability predictions not available")
    
    # Show pipeline preprocessing steps
    console.print("\n⚙️  Pipeline automatically applied:")
    for step_name, step_transformer in automl.best_pipeline.steps[:-1]:  # Exclude estimator
        console.print(f"   - {step_name}: {type(step_transformer).__name__}")
    
    console.print("   - estimator: Final prediction")
    
    return predictions

def main():
    """
    Main demonstration function that runs the complete AutoML workflow.
    
    This demonstrates the entire process from data preparation to
    production-ready pipeline deployment.
    """
    console.print(Panel.fit(
        "[bold green]AutoML Pipeline System - Complete Workflow Demo[/bold green]\n"
        "This demo shows how genetic programming automatically finds the best ML pipeline",
        border_style="green"
    ))
    
    try:
        # Step 1: Create demo data
        df = create_demo_data()
        
        # Step 2: Validate data
        validation_results = demonstrate_data_validation(df)
        
        # Step 3: Optimize pipeline
        X = df.drop(columns=['target'])
        y = df['target']
        automl = demonstrate_pipeline_optimization(df, validation_results)
        
        # Step 4: Analyze results
        metrics = demonstrate_pipeline_analysis(automl, X, y)
        
        # Step 5: Export pipeline
        export_file, pipeline_file = demonstrate_pipeline_export(automl)
        
        # Step 6: Demonstrate predictions
        X_sample = X.head(3)  # Sample for prediction demo
        predictions = demonstrate_prediction_workflow(automl, X_sample)
        
        # Summary
        console.print(Panel("[bold green]🎉 Demo Complete![/bold green]"))
        console.print("✅ Successfully demonstrated:")
        console.print("   - Data preparation and validation")
        console.print("   - Genetic programming pipeline optimization")
        console.print("   - Performance evaluation and analysis")
        console.print("   - Pipeline export and deployment")
        console.print("   - Prediction workflow")
        
        console.print(f"\n📁 Generated files:")
        console.print(f"   - {export_file}: Exported pipeline code")
        console.print(f"   - {pipeline_file}: Saved pipeline object")
        
        console.print("\n🚀 Your AutoML pipeline is ready for production use!")
        
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main() 