from setuptools import setup, find_packages

setup(
    name='ml_automation_pipeline',
    version='0.1.0',
    description='A production-ready Automated ML Pipeline System inspired by TPOT',
    author='Aroop Rath',
    author_email='work.arooprath@gmail.com',
    url='https://github.com/yourusername/ML-Automation-Pipeline',
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.2.0',
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'deap>=1.3.3',
        'tqdm>=4.64.0',
        'click>=8.1.0',
        'rich>=13.0.0',
        'joblib>=1.2.0',
    ],
    entry_points={
        'console_scripts': [
            'ml-automl=ml_automation_pipeline.cli:main',
        ],
    },
    python_requires='>=3.8',
) 