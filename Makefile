HOST ?= 0.0.0.0
PORT ?= 8000
DB_PATH ?= data/happytorch.db
SESSION_COOKIE_SECURE ?= false

.PHONY: prepare web jupyter

prepare:
	python prepare_notebooks.py

web:
	HOST=$(HOST) PORT=$(PORT) HAPPYTORCH_DB_PATH=$(DB_PATH) SESSION_COOKIE_SECURE=$(SESSION_COOKIE_SECURE) python start_web.py

jupyter:
	python start_jupyter.py
