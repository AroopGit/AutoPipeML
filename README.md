# Automated ML Pipeline System

A high-quality, production-ready Automated Machine Learning (AutoML) pipeline system inspired by [TPOT](https://github.com/EpistasisLab/tpot). This tool uses genetic programming to optimize machine learning pipelines, with a clean codebase, comprehensive monitoring, and a powerful CLI.

---

##  Quick Start: Use AutoML on Your Own Data (CLI)

1. **Place your CSV file in the project directory** (e.g., `my_data.csv`).
2. **Run this command in your terminal:**
   ```bash
   python -m ml_automation_pipeline.cli run --data my_data.csv --target target_column
   ```
3. **The pipeline will be optimized, evaluated, and saved automatically.**
   - Check the `outputs/` directory for the exported pipeline and results.
4. **To use the pipeline for predictions later:**
   ```python
   from ml_automation_pipeline.automl import AutoMLPipeline
   pipeline = AutoMLPipeline.load('outputs/job_xxx_pipeline.pkl')
   predictions = pipeline.predict(new_data)
   ```

**That's it! No code changes needed—just your CSV and a single CLI command.**

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone <this-repo-url>
cd ML-Automation-Pipeline

# Install dependencies
pip install -r requirements.txt

# Install as a package (optional, for CLI command)
pip install -e .
```

---

##  Full Usage (Python & CLI)

### Python API (for integration in your code)
```python
from ml_automation_pipeline.automl import AutoMLPipeline
import pandas as pd

df = pd.read_csv('your_data.csv')
X = df.drop(columns=['target_column'])
y = df['target_column']

# Run AutoML
automl = AutoMLPipeline().fit(X, y)
metrics = automl.evaluate()
print('Test metrics:', metrics)

# Predict on new data
preds = automl.predict(X)

# Save and load pipeline
automl.save('my_pipeline.pkl')
pipeline = AutoMLPipeline.load('my_pipeline.pkl')
```

### CLI (for non-coders or quick experiments)
```bash
python -m ml_automation_pipeline.cli run --data your_data.csv --target target_column
```

---

##  CLI Commands

- `run`: Start AutoML pipeline optimization
- `monitor`: View real-time progress and logs
- `status`: Check the status of jobs
- `delete`: Delete a job and its files
- `list`: List all jobs

See `python -m ml_automation_pipeline.cli --help` for all options.

---

##  Configuration

- **Generations**: Number of genetic algorithm generations (default: 10)
- **Population Size**: Size of the population (default: 50)
- **CV Folds**: Cross-validation folds (default: 5)
- **Max Time**: Maximum optimization time in minutes (default: 10)
- **Problem Type**: Auto-detected or manually specified

---

##  Output Structure

```
outputs/
├── job_1234567890_pipeline.pkl    # Saved pipeline
├── job_1234567890_pipeline.py     # Exported Python code
└── automl_jobs.json               # Job tracking data
```

---

##  Testing & Examples

- See `demo_pipeline_workflow.py` for a full, commented example.
- Use the sample data in `data/sample.csv` to try the system immediately.

---

##  License

This project is licensed under the [LGPL-3.0 License](https://www.gnu.org/licenses/lgpl-3.0.html) - see the LICENSE file for details.

---

##  Acknowledgments
- Inspired by the [TPOT project](https://github.com/EpistasisLab/tpot)
- Built with [scikit-learn](https://scikit-learn.org/)
- Genetic programming powered by [DEAP](https://deap.readthedocs.io/)
- Beautiful CLI with [Rich](https://rich.readthedocs.io/)

---

**Happy AutoML-ing! 🚀** 
