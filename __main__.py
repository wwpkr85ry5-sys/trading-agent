[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trading-agent"
version = "0.1.0"
description = "ML-based trading agent and backtesting utilities"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn",
]

[project.scripts]
trading-agent = "cli:main"

[tool.setuptools]
py-modules = ["trading_agent", "data_loader", "cli"]

[tool.pytest.ini_options]
testpaths = ["tests"]
