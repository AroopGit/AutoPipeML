import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import random
import pickle
import os
from typing import Tuple, List, Dict, Any, Optional
from deap import base, creator, tools, algorithms
from rich.console import Console
import warnings
warnings.filterwarnings('ignore')

console = Console()

class AutoMLPipeline:
    """
    Core class for the Automated ML Pipeline System.
    Optimizes ML pipelines using genetic programming (TPOT-inspired).
    """
    
    def __init__(self, 
                 generations: int = 10,
                 population_size: int = 50,
                 random_state: Optional[int] = None,
                 cv_folds: int = 5,
                 max_time_minutes: int = 10,
                 problem_type: str = 'auto'):
        """
        Initialize the AutoML Pipeline.
        
        Args:
            generations: Number of generations for genetic algorithm
            population_size: Size of the population
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            max_time_minutes: Maximum time to run optimization
            problem_type: 'classification' or 'regression' or 'auto'
        """
        self.generations = generations
        self.population_size = population_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.max_time_minutes = max_time_minutes
        self.problem_type = problem_type
        
        # Set random seeds
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Initialize genetic programming components
        self._setup_genetic_programming()
        
        # Store best pipeline and results
        self.best_pipeline = None
        self.best_score = -np.inf
        self.history = []
        self.is_fitted = False
        
        # Problem type detection
        self._is_classification = None
        
    def _setup_genetic_programming(self):
        """Setup DEAP genetic programming components."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_pipeline)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def _create_individual(self):
        """Create a random pipeline individual."""
        pipeline = []
        
        # Add preprocessing steps
        if random.random() < 0.7:  # 70% chance of having imputer
            imputer = random.choice(['mean', 'median', 'most_frequent'])
            pipeline.append(('imputer', SimpleImputer(strategy=imputer)))
        
        if random.random() < 0.6:  # 60% chance of having scaler
            scaler = random.choice(['standard', 'minmax', 'robust'])
            if scaler == 'standard':
                pipeline.append(('scaler', StandardScaler()))
            elif scaler == 'minmax':
                pipeline.append(('scaler', MinMaxScaler()))
            else:
                pipeline.append(('scaler', RobustScaler()))
        
        if random.random() < 0.3:  # 30% chance of having feature selection
            # Ensure k doesn't exceed number of features
            max_k = min(5, self.X_train.shape[1] if hasattr(self, 'X_train') else 5)
            k = random.randint(1, max_k)
            # Use a default for feature selection if problem type not set yet
            if hasattr(self, '_is_classification') and self._is_classification:
                pipeline.append(('feature_selection', SelectKBest(f_classif, k=k)))
            else:
                pipeline.append(('feature_selection', SelectKBest(f_regression, k=k)))
        
        if random.random() < 0.2:  # 20% chance of having PCA
            # Ensure n_components doesn't exceed number of features
            max_components = min(3, self.X_train.shape[1] if hasattr(self, 'X_train') else 3)
            n_components = random.randint(1, max_components)
            pipeline.append(('pca', PCA(n_components=n_components)))
        
        # Add estimator - use a default if problem type not set yet
        if hasattr(self, '_is_classification') and self._is_classification:
            estimator = self._get_random_classifier()
        else:
            estimator = self._get_random_regressor()
        
        pipeline.append(('estimator', estimator))
        
        return creator.Individual(pipeline)
    
    def _get_random_classifier(self):
        """Get a random classifier with random hyperparameters."""
        classifier_type = random.choice([
            'random_forest', 'logistic', 'svm', 'decision_tree', 'knn', 'naive_bayes'
        ])
        
        if classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=random.randint(50, 200),
                max_depth=random.choice([None, random.randint(3, 20)]),
                random_state=self.random_state
            )
        elif classifier_type == 'logistic':
            return LogisticRegression(random_state=self.random_state)
        elif classifier_type == 'svm':
            return SVC(random_state=self.random_state)
        elif classifier_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=random.randint(3, 15),
                random_state=self.random_state
            )
        elif classifier_type == 'knn':
            return KNeighborsClassifier(n_neighbors=random.randint(3, 15))
        else:  # naive_bayes
            return GaussianNB()
    
    def _get_random_regressor(self):
        """Get a random regressor with random hyperparameters."""
        regressor_type = random.choice([
            'random_forest', 'linear', 'svm', 'decision_tree', 'knn'
        ])
        
        if regressor_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=random.randint(50, 200),
                max_depth=random.choice([None, random.randint(3, 20)]),
                random_state=self.random_state
            )
        elif regressor_type == 'linear':
            return LinearRegression()
        elif regressor_type == 'svm':
            return SVR()
        elif regressor_type == 'decision_tree':
            return DecisionTreeRegressor(
                max_depth=random.randint(3, 15),
                random_state=self.random_state
            )
        else:  # knn
            return KNeighborsRegressor(n_neighbors=random.randint(3, 15))
    
    def _evaluate_pipeline(self, individual):
        """Evaluate a pipeline individual."""
        try:
            # Create pipeline from individual
            pipeline = Pipeline(individual)
            
            # Perform cross-validation
            if self._is_classification:
                scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                       cv=min(self.cv_folds, len(self.y_train)), 
                                       scoring='accuracy')
            else:
                scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                       cv=min(self.cv_folds, len(self.y_train)), 
                                       scoring='r2')
            
            # Return mean score
            score = np.mean(scores)
            if np.isnan(score):
                return (-np.inf,)
            return (score,)
        except Exception as e:
            # Return worst possible score for failed pipelines
            return (-np.inf,)
    
    def _crossover(self, ind1, ind2):
        """Crossover operation between two individuals."""
        if len(ind1) > 1 and len(ind2) > 1:
            # Randomly select crossover points
            cxpoint1 = random.randint(0, len(ind1))
            cxpoint2 = random.randint(0, len(ind2))
            
            # Swap parts of the pipelines
            ind1[cxpoint1:], ind2[cxpoint2:] = ind2[cxpoint2:], ind1[cxpoint1:]
        
        return ind1, ind2
    
    def _mutate(self, individual):
        """Mutation operation on an individual."""
        if random.random() < 0.3:  # 30% mutation probability
            # Randomly modify one step in the pipeline
            if len(individual) > 0:
                idx = random.randint(0, len(individual) - 1)
                if idx == len(individual) - 1:  # Last step (estimator)
                    if self._is_classification:
                        individual[idx] = ('estimator', self._get_random_classifier())
                    else:
                        individual[idx] = ('estimator', self._get_random_regressor())
                else:  # Preprocessing step
                    # Replace with a random preprocessing step
                    pass  # Simplified for brevity
        
        return (individual,)
    
    def _detect_problem_type(self, y):
        """Detect if the problem is classification or regression."""
        if self.problem_type == 'auto':
            # Check if target is categorical
            if len(np.unique(y)) <= 20 or y.dtype == 'object':
                return 'classification'
            else:
                return 'regression'
        else:
            return self.problem_type
    
    def fit(self, X, y):
        """
        Optimize pipeline on training data.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        # Detect problem type
        self._is_classification = (self._detect_problem_type(y) == 'classification')
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, 
            stratify=y if self._is_classification else None
        )
        
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Track best individual
        hof = tools.HallOfFame(1)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run genetic algorithm
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox, 
            cxpb=0.7,  # crossover probability
            mutpb=0.3,  # mutation probability
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        # Store best pipeline
        console.print(f"[blue]Hall of Fame length: {len(hof)}[/blue]")
        if len(hof) > 0:
            console.print(f"[blue]Best fitness: {hof[0].fitness.values[0]}[/blue]")
            
        if len(hof) > 0 and hof[0].fitness.values[0] > -np.inf:
            console.print("[green]Using best pipeline from genetic algorithm[/green]")
            self.best_pipeline = Pipeline(hof[0])
            self.best_score = hof[0].fitness.values[0]
            # Fit the best pipeline
            self.best_pipeline.fit(self.X_train, self.y_train)
        else:
            # Fallback: create a simple pipeline
            console.print("[yellow]No valid pipeline found, creating fallback pipeline...[/yellow]")
            if self._is_classification:
                fallback_pipeline = [('estimator', RandomForestClassifier(random_state=self.random_state))]
            else:
                fallback_pipeline = [('estimator', RandomForestRegressor(random_state=self.random_state))]
            self.best_pipeline = Pipeline(fallback_pipeline)
            self.best_pipeline.fit(self.X_train, self.y_train)
            self.best_score = 0.5  # Default score for fallback
        
        self.history = logbook
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make predictions using the best pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.best_pipeline.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions (for classification)."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        if not self._is_classification:
            raise ValueError("predict_proba is only available for classification")
        return self.best_pipeline.predict_proba(X)
    
    def evaluate(self, X=None, y=None):
        """
        Evaluate the best pipeline on test data.
        
        Args:
            X: Test features (uses internal test set if None)
            y: Test targets (uses internal test set if None)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        if self._is_classification:
            accuracy = accuracy_score(y, y_pred)
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return {'mse': mse, 'r2': r2}
    
    def export(self, output_file: str):
        """
        Export the best pipeline as Python code.
        
        Args:
            output_file: Path to save the exported pipeline
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before export")
        
        # Generate Python code
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
            "from sklearn.decomposition import PCA",
            "from sklearn.feature_selection import SelectKBest, f_classif, f_regression",
            "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
            "from sklearn.linear_model import LogisticRegression, LinearRegression",
            "from sklearn.svm import SVC, SVR",
            "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor",
            "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor",
            "from sklearn.naive_bayes import GaussianNB",
            "from sklearn.impute import SimpleImputer",
            "",
            "# Best pipeline found by AutoML",
            "best_pipeline = Pipeline(["
        ]
        
        # Add pipeline steps
        for i, (name, estimator) in enumerate(self.best_pipeline.steps):
            if i > 0:
                code_lines.append("    " + str((name, estimator)) + ",")
            else:
                code_lines.append("    " + str((name, estimator)) + ",")
        
        code_lines.append("])")
        code_lines.append("")
        code_lines.append("# Usage:")
        code_lines.append("# pipeline.fit(X_train, y_train)")
        code_lines.append("# predictions = pipeline.predict(X_test)")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(code_lines))
    
    def save(self, filepath: str):
        """Save the fitted pipeline to disk."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a fitted pipeline from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f) 