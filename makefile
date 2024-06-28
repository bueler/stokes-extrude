all:

.PHONY: clean

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.pvtu *.vtu
	@rm -rf .pytest_cache/ __pycache__/ tests/__pycache__/
	@rm -rf result/ result.pvd
