import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from ml_automation_pipeline.automl import AutoMLPipeline
from ml_automation_pipeline.utils import load_data, validate_data, create_sample_data

class TestAutoMLPipeline:
    """Test cases for AutoMLPipeline class."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y_classification = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        self.y_regression = self.X[:, 0] + self.X[:, 1] + np.random.normal(0, 0.1, 100)
        
        self.df_classification = pd.DataFrame(
            self.X, columns=[f'feature_{i}' for i in range(5)]
        )
        self.df_classification['target'] = self.y_classification
        
        self.df_regression = pd.DataFrame(
            self.X, columns=[f'feature_{i}' for i in range(5)]
        )
        self.df_regression['target'] = self.y_regression
    
    def test_initialization(self):
        """Test AutoMLPipeline initialization."""
        automl = AutoMLPipeline(
            generations=5,
            population_size=20,
            random_state=42,
            cv_folds=3,
            max_time_minutes=5,
            problem_type='auto'
        )
        
        assert automl.generations == 5
        assert automl.population_size == 20
        assert automl.random_state == 42
        assert automl.cv_folds == 3
        assert automl.max_time_minutes == 5
        assert automl.problem_type == 'auto'
        assert not automl.is_fitted
    
    def test_problem_type_detection_classification(self):
        """Test automatic problem type detection for classification."""
        automl = AutoMLPipeline(problem_type='auto')
        problem_type = automl._detect_problem_type(self.y_classification)
        assert problem_type == 'classification'
    
    def test_problem_type_detection_regression(self):
        """Test automatic problem type detection for regression."""
        automl = AutoMLPipeline(problem_type='auto')
        problem_type = automl._detect_problem_type(self.y_regression)
        assert problem_type == 'regression'
    
    def test_manual_problem_type(self):
        """Test manual problem type specification."""
        automl = AutoMLPipeline(problem_type='classification')
        problem_type = automl._detect_problem_type(self.y_regression)
        assert problem_type == 'classification'
    
    def test_fit_classification(self):
        """Test fitting classification pipeline."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_classification)
        
        assert automl.is_fitted
        assert automl.best_pipeline is not None
        assert automl.best_score > -np.inf
        assert automl._is_classification is True
    
    def test_fit_regression(self):
        """Test fitting regression pipeline."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_regression)
        
        assert automl.is_fitted
        assert automl.best_pipeline is not None
        assert automl.best_score > -np.inf
        assert automl._is_classification is False
    
    def test_predict(self):
        """Test prediction functionality."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_classification)
        predictions = automl.predict(self.X[:10])
        
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
    
    def test_predict_proba_classification(self):
        """Test probability prediction for classification."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_classification)
        proba = automl.predict_proba(self.X[:10])
        
        assert proba.shape == (10, 2)  # Binary classification
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict_proba_regression_error(self):
        """Test that predict_proba raises error for regression."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_regression)
        
        with pytest.raises(ValueError, match="predict_proba is only available for classification"):
            automl.predict_proba(self.X[:10])
    
    def test_evaluate(self):
        """Test pipeline evaluation."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_classification)
        metrics = automl.evaluate()
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_regression(self):
        """Test regression pipeline evaluation."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_regression)
        metrics = automl.evaluate()
        
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
        assert metrics['r2'] <= 1
    
    def test_export_pipeline(self):
        """Test pipeline export functionality."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_classification)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            export_file = f.name
        
        try:
            automl.export(export_file)
            assert os.path.exists(export_file)
            
            # Check that the exported file contains expected content
            with open(export_file, 'r') as f:
                content = f.read()
                assert 'Pipeline' in content
                assert 'best_pipeline' in content
        finally:
            os.unlink(export_file)
    
    def test_save_load(self):
        """Test saving and loading pipeline."""
        automl = AutoMLPipeline(
            generations=2,
            population_size=5,
            random_state=42
        )
        
        automl.fit(self.X, self.y_classification)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_file = f.name
        
        try:
            automl.save(save_file)
            assert os.path.exists(save_file)
            
            # Load the pipeline
            loaded_automl = AutoMLPipeline.load(save_file)
            
            assert loaded_automl.is_fitted
            assert loaded_automl.best_score == automl.best_score
            assert len(loaded_automl.best_pipeline.steps) == len(automl.best_pipeline.steps)
        finally:
            os.unlink(save_file)
    
    def test_fit_without_data(self):
        """Test that fit raises error when called without data."""
        automl = AutoMLPipeline()
        
        with pytest.raises(ValueError):
            automl.fit(None, None)
    
    def test_predict_without_fit(self):
        """Test that predict raises error when called without fitting."""
        automl = AutoMLPipeline()
        
        with pytest.raises(ValueError, match="Pipeline must be fitted before making predictions"):
            automl.predict(self.X)
    
    def test_evaluate_without_fit(self):
        """Test that evaluate raises error when called without fitting."""
        automl = AutoMLPipeline()
        
        with pytest.raises(ValueError, match="Pipeline must be fitted before evaluation"):
            automl.evaluate()

class TestUtils:
    """Test cases for utility functions."""
    
    def test_load_data_csv(self):
        """Test loading CSV data."""
        # Create temporary CSV file
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            file_path = f.name
        
        try:
            loaded_df = load_data(file_path)
            assert loaded_df.shape == df.shape
            assert list(loaded_df.columns) == list(df.columns)
        finally:
            os.unlink(file_path)
    
    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")
    
    def test_validate_data(self):
        """Test data validation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, np.nan],
            'feature2': [4, 5, 6, 7],
            'target': [0, 1, 0, 1]
        })
        
        validation = validate_data(df, 'target')
        
        assert validation['is_valid'] is True
        assert 'warnings' in validation
        assert 'numeric_columns' in validation
        assert 'categorical_columns' in validation
        assert validation['problem_type'] == 'classification'
    
    def test_validate_data_missing_target(self):
        """Test validation with missing target column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        validation = validate_data(df, 'target')
        
        assert validation['is_valid'] is False
        assert len(validation['errors']) > 0
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "sample.csv")
            df = create_sample_data(output_path, n_samples=100)
            
            assert os.path.exists(output_path)
            assert df.shape == (100, 6)  # 5 features + 1 target
            assert 'target' in df.columns

class TestCLI:
    """Test cases for CLI functionality."""
    
    @patch('ml_automation_pipeline.cli.load_data')
    @patch('ml_automation_pipeline.automl.AutoMLPipeline')
    def test_cli_run_command(self, mock_automl_class, mock_load_data):
        """Test CLI run command."""
        from ml_automation_pipeline.cli import main
        
        # Mock data loading
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        mock_load_data.return_value = mock_df
        
        # Mock AutoML pipeline
        mock_automl = MagicMock()
        mock_automl_class.return_value = mock_automl
        mock_automl.evaluate.return_value = {'accuracy': 0.85}
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mock_df.to_csv(f.name, index=False)
            file_path = f.name
        
        try:
            # Test CLI run command
            with patch('sys.argv', ['cli.py', 'run', '--data', file_path, '--target', 'target']):
                main()
                
                # Verify AutoML was called
                mock_automl_class.assert_called_once()
                mock_automl.fit.assert_called_once()
                mock_automl.evaluate.assert_called_once()
        finally:
            os.unlink(file_path)
    
    def test_cli_monitor_command(self):
        """Test CLI monitor command."""
        from ml_automation_pipeline.cli import main
        
        # Create temporary jobs file
        jobs_data = {
            'job_123': {
                'id': 'job_123',
                'status': 'completed',
                'data_file': 'test.csv',
                'target': 'target',
                'start_time': '2023-01-01T00:00:00',
                'best_score': 0.85
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(jobs_data, f)
            jobs_file = f.name
        
        try:
            # Mock the jobs file path
            with patch('ml_automation_pipeline.cli.JOBS_FILE', jobs_file):
                with patch('sys.argv', ['cli.py', 'monitor']):
                    main()
        finally:
            os.unlink(jobs_file)

if __name__ == "__main__":
    pytest.main([__file__]) 