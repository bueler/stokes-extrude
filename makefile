all:

.PHONY: clean

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.pvtu *.vtu
	@rm -rf .pytest_cache/ stokesextruded/__pycache__/ tests/__pycache__/
	@rm -rf result/ result.pvd tests/result/ tests/result.pvd
	@rm -rf stokesextruded.egg-info/
