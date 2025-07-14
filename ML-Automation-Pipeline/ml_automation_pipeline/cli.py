import click
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from pathlib import Path

from .automl import AutoMLPipeline
from .utils import load_data, log_event, monitor_job

console = Console()

# Global job tracking
JOBS_FILE = "automl_jobs.json"
jobs_data = {}

def load_jobs():
    """Load jobs from file."""
    global jobs_data
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, 'r') as f:
                jobs_data = json.load(f)
        except:
            jobs_data = {}
    else:
        jobs_data = {}

def save_jobs():
    """Save jobs to file."""
    with open(JOBS_FILE, 'w') as f:
        json.dump(jobs_data, f, indent=2)

def create_job_id():
    """Create a unique job ID."""
    return f"job_{int(time.time())}"

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def main(verbose):
    """Automated ML Pipeline CLI Tool - Production Ready AutoML System"""
    if verbose:
        console.print("[bold blue]Verbose mode enabled[/bold blue]")
    load_jobs()

@main.command()
@click.option('--data', required=True, help='Path to the input data CSV file.')
@click.option('--target', required=True, help='Name of the target column.')
@click.option('--generations', default=10, help='Number of generations for genetic algorithm.')
@click.option('--population-size', default=50, help='Population size for genetic algorithm.')
@click.option('--cv-folds', default=5, help='Number of cross-validation folds.')
@click.option('--max-time', default=10, help='Maximum time in minutes.')
@click.option('--problem-type', default='auto', 
              type=click.Choice(['auto', 'classification', 'regression']),
              help='Problem type (auto-detected if not specified).')
@click.option('--output-dir', default='outputs', help='Output directory for results.')
@click.option('--export-pipeline', is_flag=True, help='Export the best pipeline as Python code.')
def run(data, target, generations, population_size, cv_folds, max_time, problem_type, output_dir, export_pipeline):
    """Run an AutoML pipeline optimization job."""
    
    # Validate inputs
    if not os.path.exists(data):
        console.print(f"[bold red]Error: Data file '{data}' not found![/bold red]")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create job ID and record
    job_id = create_job_id()
    job_record = {
        'id': job_id,
        'status': 'running',
        'start_time': datetime.now().isoformat(),
        'data_file': data,
        'target': target,
        'generations': generations,
        'population_size': population_size,
        'cv_folds': cv_folds,
        'max_time': max_time,
        'problem_type': problem_type,
        'output_dir': output_dir
    }
    
    jobs_data[job_id] = job_record
    save_jobs()
    
    console.print(Panel(f"[bold green]Starting AutoML Job: {job_id}[/bold green]"))
    console.print(f"Data: {data}")
    console.print(f"Target: {target}")
    console.print(f"Problem Type: {problem_type}")
    console.print(f"Generations: {generations}")
    console.print(f"Population Size: {population_size}")
    
    try:
        # Load data
        console.print("\n[bold yellow]Loading data...[/bold yellow]")
        df = load_data(data)
        
        if target not in df.columns:
            console.print(f"[bold red]Error: Target column '{target}' not found in data![/bold red]")
            console.print(f"Available columns: {list(df.columns)}")
            return
        
        X = df.drop(columns=[target])
        y = df[target]
        
        console.print(f"Data shape: {X.shape}")
        console.print(f"Target distribution: {y.value_counts().to_dict() if len(y.unique()) <= 10 else 'Too many unique values'}")
        
        # Initialize AutoML pipeline
        automl = AutoMLPipeline(
            generations=generations,
            population_size=population_size,
            cv_folds=cv_folds,
            max_time_minutes=max_time,
            problem_type=problem_type,
            random_state=42
        )
        
        # Run optimization with progress tracking
        console.print("\n[bold green]Starting pipeline optimization...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Create a custom progress callback
            def progress_callback(generation, best_score):
                progress.update(
                    progress.add_task(f"Generation {generation}/{generations} (Best: {best_score:.4f})", total=generations),
                    completed=generation
                )
            
            # Run the optimization
            automl.fit(X, y)
        
        # Evaluate results
        console.print("\n[bold green]Evaluating best pipeline...[/bold green]")
        
        # Check if we have a valid best pipeline
        if automl.best_pipeline is None or automl.best_score == -np.inf:
            console.print("[bold red]No valid pipeline found during optimization![/bold red]")
            job_record['status'] = 'failed'
            job_record['error'] = 'No valid pipeline found during optimization'
            job_record['end_time'] = datetime.now().isoformat()
            save_jobs()
            return
        
        try:
            metrics = automl.evaluate()
            
            # Update job record
            job_record['status'] = 'completed'
            job_record['end_time'] = datetime.now().isoformat()
            job_record['best_score'] = automl.best_score
            job_record['metrics'] = metrics
            job_record['pipeline_steps'] = [step[0] for step in automl.best_pipeline.steps]
        except Exception as e:
            console.print(f"[bold red]Error during evaluation: {str(e)}[/bold red]")
            job_record['status'] = 'failed'
            job_record['error'] = f'Evaluation error: {str(e)}'
            job_record['end_time'] = datetime.now().isoformat()
            save_jobs()
            return
        
        # Save pipeline
        pipeline_file = os.path.join(output_dir, f"{job_id}_pipeline.pkl")
        automl.save(pipeline_file)
        job_record['pipeline_file'] = pipeline_file
        
        # Export pipeline code if requested
        if export_pipeline:
            export_file = os.path.join(output_dir, f"{job_id}_pipeline.py")
            automl.export(export_file)
            job_record['export_file'] = export_file
            console.print(f"[bold blue]Pipeline exported to: {export_file}[/bold blue]")
        
        # Display results
        console.print("\n[bold green]Optimization Complete![/bold green]")
        console.print(f"Best Score: {automl.best_score:.4f}")
        console.print(f"Test Metrics: {metrics}")
        console.print(f"Pipeline Steps: {job_record['pipeline_steps']}")
        console.print(f"Pipeline saved to: {pipeline_file}")
        
        save_jobs()
        
    except Exception as e:
        console.print(f"[bold red]Error during optimization: {str(e)}[/bold red]")
        job_record['status'] = 'failed'
        job_record['error'] = str(e)
        job_record['end_time'] = datetime.now().isoformat()
        save_jobs()

@main.command()
@click.option('--job-id', help='Monitor specific job ID')
@click.option('--live', is_flag=True, help='Live monitoring with auto-refresh')
def monitor(job_id, live):
    """Monitor the progress of running jobs."""
    
    if job_id:
        if job_id not in jobs_data:
            console.print(f"[bold red]Job {job_id} not found![/bold red]")
            return
        
        job = jobs_data[job_id]
        display_job_details(job)
    else:
        # Show all jobs
        if not jobs_data:
            console.print("[bold yellow]No jobs found.[/bold yellow]")
            return
        
        table = Table(title="AutoML Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Data File", style="blue")
        table.add_column("Target", style="blue")
        table.add_column("Start Time", style="yellow")
        table.add_column("Best Score", style="magenta")
        
        for jid, job in jobs_data.items():
            status_style = {
                'running': 'green',
                'completed': 'blue',
                'failed': 'red'
            }.get(job['status'], 'white')
            
            table.add_row(
                jid,
                f"[{status_style}]{job['status']}[/{status_style}]",
                os.path.basename(job.get('data_file', 'N/A')),
                job.get('target', 'N/A'),
                job.get('start_time', 'N/A')[:19],
                f"{job.get('best_score', 'N/A'):.4f}" if job.get('best_score') is not None else 'N/A'
            )
        
        console.print(table)

def display_job_details(job):
    """Display detailed information about a specific job."""
    console.print(Panel(f"[bold blue]Job Details: {job['id']}[/bold blue]"))
    
    details = Table(show_header=False)
    details.add_column("Property", style="cyan")
    details.add_column("Value", style="white")
    
    details.add_row("Status", job['status'])
    details.add_row("Data File", job['data_file'])
    details.add_row("Target", job['target'])
    details.add_row("Problem Type", job['problem_type'])
    details.add_row("Generations", str(job['generations']))
    details.add_row("Population Size", str(job['population_size']))
    details.add_row("Start Time", job['start_time'])
    
    if 'end_time' in job:
        details.add_row("End Time", job['end_time'])
    
    if 'best_score' in job and job['best_score'] is not None:
        details.add_row("Best Score", f"{job['best_score']:.4f}")
    
    if 'metrics' in job:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in job['metrics'].items()])
        details.add_row("Test Metrics", metrics_str)
    
    if 'pipeline_steps' in job:
        details.add_row("Pipeline Steps", " → ".join(job['pipeline_steps']))
    
    if 'error' in job:
        details.add_row("Error", f"[red]{job['error']}[/red]")
    
    console.print(details)

@main.command()
@click.option('--job-id', help='Check status of specific job ID')
def status(job_id):
    """Check the status of current or past jobs."""
    
    if job_id:
        if job_id not in jobs_data:
            console.print(f"[bold red]Job {job_id} not found![/bold red]")
            return
        
        job = jobs_data[job_id]
        status_text = Text()
        status_text.append(f"Job {job_id}: ", style="bold blue")
        
        if job['status'] == 'running':
            status_text.append("RUNNING", style="bold green")
        elif job['status'] == 'completed':
            status_text.append("COMPLETED", style="bold blue")
        elif job['status'] == 'failed':
            status_text.append("FAILED", style="bold red")
        
        console.print(status_text)
        
        if job['status'] == 'completed' and 'best_score' in job:
            console.print(f"Best Score: {job['best_score']:.4f}")
        elif job['status'] == 'failed' and 'error' in job:
            console.print(f"Error: {job['error']}")
    else:
        # Show summary
        total_jobs = len(jobs_data)
        running_jobs = sum(1 for job in jobs_data.values() if job['status'] == 'running')
        completed_jobs = sum(1 for job in jobs_data.values() if job['status'] == 'completed')
        failed_jobs = sum(1 for job in jobs_data.values() if job['status'] == 'failed')
        
        console.print(Panel(f"""
[bold]Job Summary:[/bold]
Total Jobs: {total_jobs}
Running: {running_jobs}
Completed: {completed_jobs}
Failed: {failed_jobs}
        """))

@main.command()
@click.option('--job-id', required=True, help='Job ID to delete')
def delete(job_id):
    """Delete a job and its associated files."""
    
    if job_id not in jobs_data:
        console.print(f"[bold red]Job {job_id} not found![/bold red]")
        return
    
    job = jobs_data[job_id]
    
    # Delete associated files
    if 'pipeline_file' in job and os.path.exists(job['pipeline_file']):
        os.remove(job['pipeline_file'])
        console.print(f"Deleted pipeline file: {job['pipeline_file']}")
    
    if 'export_file' in job and os.path.exists(job['export_file']):
        os.remove(job['export_file'])
        console.print(f"Deleted export file: {job['export_file']}")
    
    # Remove from jobs data
    del jobs_data[job_id]
    save_jobs()
    
    console.print(f"[bold green]Job {job_id} deleted successfully![/bold green]")

@main.command()
def list():
    """List all jobs with their current status."""
    monitor(None, False)

if __name__ == "__main__":
    main() 