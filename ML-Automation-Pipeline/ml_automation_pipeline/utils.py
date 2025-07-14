import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path

console = Console()

# Setup logging
def setup_logging(log_file: str = "automl.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from various file formats with error handling.
    
    Args:
        path: Path to the data file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    file_extension = Path(path).suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(path)
        elif file_extension == '.xlsx':
            df = pd.read_excel(path)
        elif file_extension == '.json':
            df = pd.read_json(path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Successfully loaded data from {path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {path}: {str(e)}")
        raise

def validate_data(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate data for AutoML pipeline.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check if target column exists
    if target_column not in df.columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Target column '{target_column}' not found")
        return validation_results
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        validation_results['warnings'].append(f"Found {missing_values.sum()} missing values")
    
    # Check data types
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    validation_results['numeric_columns'] = list(numeric_columns)
    validation_results['categorical_columns'] = list(categorical_columns)
    
    # Check target column properties
    target_series = df[target_column]
    unique_values = target_series.nunique()
    
    if target_series.dtype in ['object', 'category'] or unique_values <= 20:
        validation_results['problem_type'] = 'classification'
        validation_results['target_info'] = {
            'type': 'categorical',
            'unique_values': unique_values,
            'class_distribution': target_series.value_counts().to_dict()
        }
    else:
        validation_results['problem_type'] = 'regression'
        validation_results['target_info'] = {
            'type': 'continuous',
            'min': target_series.min(),
            'max': target_series.max(),
            'mean': target_series.mean(),
            'std': target_series.std()
        }
    
    return validation_results

def log_event(message: str, level: str = "info", job_id: Optional[str] = None):
    """
    Log an event with timestamp and optional job ID.
    
    Args:
        message: Log message
        level: Log level (info, warning, error, debug)
        job_id: Optional job ID for tracking
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_prefix = f"[{job_id}] " if job_id else ""
    full_message = f"{job_prefix}{message}"
    
    if level == "info":
        logger.info(full_message)
        console.print(f"[green]{timestamp}[/green] {full_message}")
    elif level == "warning":
        logger.warning(full_message)
        console.print(f"[yellow]{timestamp}[/yellow] {full_message}")
    elif level == "error":
        logger.error(full_message)
        console.print(f"[red]{timestamp}[/red] {full_message}")
    elif level == "debug":
        logger.debug(full_message)
        console.print(f"[blue]{timestamp}[/blue] {full_message}")

def monitor_job(job_id: str, jobs_file: str = "automl_jobs.json") -> Dict[str, Any]:
    """
    Monitor the progress of a specific job.
    
    Args:
        job_id: Job ID to monitor
        jobs_file: Path to jobs file
        
    Returns:
        Job status dictionary
    """
    if not os.path.exists(jobs_file):
        return {'error': 'Jobs file not found'}
    
    try:
        with open(jobs_file, 'r') as f:
            jobs_data = json.load(f)
        
        if job_id not in jobs_data:
            return {'error': f'Job {job_id} not found'}
        
        job = jobs_data[job_id]
        
        # Calculate duration if job is completed
        if 'start_time' in job and 'end_time' in job:
            start_time = datetime.fromisoformat(job['start_time'])
            end_time = datetime.fromisoformat(job['end_time'])
            duration = end_time - start_time
            job['duration'] = str(duration)
        
        return job
        
    except Exception as e:
        return {'error': f'Error reading job data: {str(e)}'}

def display_job_summary(jobs_data: Dict[str, Any]):
    """
    Display a summary of all jobs in a nice table format.
    
    Args:
        jobs_data: Dictionary of job data
    """
    if not jobs_data:
        console.print("[yellow]No jobs found.[/yellow]")
        return
    
    table = Table(title="AutoML Jobs Summary")
    table.add_column("Job ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Data File", style="blue")
    table.add_column("Target", style="blue")
    table.add_column("Problem Type", style="magenta")
    table.add_column("Best Score", style="yellow")
    table.add_column("Duration", style="white")
    
    for job_id, job in jobs_data.items():
        # Status with color coding
        status = job.get('status', 'unknown')
        status_style = {
            'running': 'green',
            'completed': 'blue',
            'failed': 'red',
            'unknown': 'white'
        }.get(status, 'white')
        
        # Best score formatting
        best_score = job.get('best_score')
        if best_score is not None:
            score_str = f"{best_score:.4f}"
        else:
            score_str = "N/A"
        
        # Duration formatting
        duration = job.get('duration', 'N/A')
        
        table.add_row(
            job_id,
            f"[{status_style}]{status}[/{status_style}]",
            os.path.basename(job.get('data_file', 'N/A')),
            job.get('target', 'N/A'),
            job.get('problem_type', 'auto'),
            score_str,
            duration
        )
    
    console.print(table)

def create_sample_data(output_path: str = "data/sample.csv", n_samples: int = 1000):
    """
    Create sample data for testing the AutoML pipeline.
    
    Args:
        output_path: Path to save the sample data
        n_samples: Number of samples to generate
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate sample data
    np.random.seed(42)
    
    # Features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)
    feature4 = np.random.normal(0, 1, n_samples)
    feature5 = np.random.normal(0, 1, n_samples)
    
    # Target (classification example)
    target = (feature1 + feature2 + feature3 > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'feature5': feature5,
        'target': target
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    console.print(f"[green]Sample data created: {output_path}[/green]")
    console.print(f"Shape: {df.shape}")
    console.print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and monitoring.
    
    Returns:
        Dictionary with system information
    """
    import psutil
    import platform
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        'disk_usage': f"{psutil.disk_usage('/').free / (1024**3):.2f} GB free"
    }

def cleanup_old_jobs(days_old: int = 30, jobs_file: str = "automl_jobs.json"):
    """
    Clean up old completed jobs and their files.
    
    Args:
        days_old: Remove jobs older than this many days
        jobs_file: Path to jobs file
    """
    if not os.path.exists(jobs_file):
        return
    
    try:
        with open(jobs_file, 'r') as f:
            jobs_data = json.load(f)
        
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        jobs_to_remove = []
        
        for job_id, job in jobs_data.items():
            if job.get('status') == 'completed' and 'start_time' in job:
                job_timestamp = datetime.fromisoformat(job['start_time']).timestamp()
                if job_timestamp < cutoff_date:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            job = jobs_data[job_id]
            
            # Remove associated files
            if 'pipeline_file' in job and os.path.exists(job['pipeline_file']):
                os.remove(job['pipeline_file'])
            
            if 'export_file' in job and os.path.exists(job['export_file']):
                os.remove(job['export_file'])
            
            # Remove from jobs data
            del jobs_data[job_id]
        
        # Save updated jobs data
        with open(jobs_file, 'w') as f:
            json.dump(jobs_data, f, indent=2)
        
        console.print(f"[green]Cleaned up {len(jobs_to_remove)} old jobs[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during cleanup: {str(e)}[/red]") 