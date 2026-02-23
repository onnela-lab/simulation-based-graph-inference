.PHONY : docs doctests lint sync tests clean clean-docs cook-list

build : lint tests docs doctests cook-list

lint :
	ruff format --check .

tests :
	CI=true pytest tests -v --cov=simulation_based_graph_inference \
		--cov-report=term-missing --cov-report=html --durations=5

docs :
	rm -rf docs/_build/plot_directive
	sphinx-build . docs/_build

doctests :
	sphinx-build -b doctest . docs/_build

sync : requirements.txt
	pip-sync

requirements.txt : requirements.in setup.py test_requirements.txt
	pip-compile -v -o $@ $<

test_requirements.txt : test_requirements.in setup.py
	pip-compile -v -o $@ $<

clean-docs :
	rm -rf docs/_build
	${MAKE} docs

cook-list : recipe.py
	cook ls '*'
