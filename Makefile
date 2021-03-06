# Makefile to automate some basic tasks

project-setup:
	(python3 -m venv venv; \
	source ./venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt)


run-tests:
	(source venv/bin/activate; \
	pytest churn_script_logging_and_tests.py)
