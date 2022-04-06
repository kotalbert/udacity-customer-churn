# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity.

## Project Description

This project is intended to demonstrate clean code principles in Machine Learning project.

The code is providing functionalities for end to end ML Pipeline, including the following stages:

- data ingestion
- exploratory data analysis
- data transformation
- feature engineering
- model training
- model validation

The code is intended to provide modularity, reproducibility as well of maintainability of the solution.

The full project is available on github: <https://github.com/kotalbert/udacity-customer-churn>,
including `requirements.txt` as well as `Makefile` automation file.

## Running Files

To run the project files, UNIX based system is required (e.g. Linux, MacOs).

Project has suite of unit tests. The test can be run and the results are written in log file `logs/churn_library.log`.
The log can be used to debug possible problems with the code.

### Using Makefile

Running the file was automated, using `Makefile`. If `make` utility is installed on the system, it is possible to
restore virtual environment for the project and run unit tests.

Create virtual environment and install dependencies.

```shell
make project-setup
```



Start environment and run unit tests.

```shell
make run-tests 
```

### Using command line

If system does not have `make` utility installed, project can be set up using following command line commands.

Create virtual environment and install dependencies.

```shell
python3 -m venv venv
source ./venv/bin/activate 
pip install --upgrade pip
pip install -r requirements.txt 
```

Start project virtual environment.

```shell
source ./venv/bin/activate
```

After starting the environment, tests can be run with `pytest`, alternatively using pure python:

```shell
pytest churn_script_logging_and_tests.py
```

```shell
python churn_script_logging_and_tests.py
```
