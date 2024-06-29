all:

.PHONY: clean

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.pvtu *.vtu
	@rm -rf .pytest_cache/ stokesextruded/__pycache__/ tests/__pycache__/
	@rm -rf result* tests/result* examples/result*
	@rm -rf stokesextruded.egg-info/
