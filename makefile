all:

.PHONY: clean

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.pvtu *.vtu
	@rm -rf .pytest_cache/ __pycache__/ htmlcov/ .coverage
	@rm -rf stokesextrude/__pycache__/ tests/__pycache__/ examples/__pycache__/
	@rm -rf result* tests/result* examples/result*
	@rm -rf stokesextrude.egg-info/
