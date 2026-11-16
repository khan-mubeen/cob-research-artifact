"""v5 synthesis, built in parallel with v4 and never through it

nothing in this package imports, calls or modifies the v4 generators in
src/data_synthesis/*.py. the few v4 design constants it reuses are copied here
with the file and name they came from, and tests/test_v5_synthesis.py parses the
v4 files to prove the copies still match

  PYTHONPATH=. python -m src.data_synthesis.v5.build        writes data/synthetic/v5
  PYTHONPATH=. python -m src.data_synthesis.v5.diagnostics  writes reports/v5
"""
