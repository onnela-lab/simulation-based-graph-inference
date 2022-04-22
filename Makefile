.PHONY : docs doctests lint sync tests clean clean_docs

build : lint tests docs doctests stubs

lint :
	flake8

tests :
	pytest -v --cov=simulation_based_graph_inference --cov-fail-under=100 --cov-report=term-missing --cov-report=html

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

STUB_FILES = convert generators graph
STUB_TARGETS = $(addprefix simulation_based_graph_inference/,${STUB_FILES:=.pyi})
stubs : ${STUB_TARGETS}

${STUB_TARGETS} : simulation_based_graph_inference/%.pyi : \
		generate_stub.py simulation_based_graph_inference/%.pyx
	python $< simulation_based_graph_inference.$* > $@

clean_docs :
	rm -rf docs/_build

clean : clean_docs
	rm -f simulation_based_graph_inference/*.pyi
	rm -f simulation_based_graph_inference/*.so
	rm -f simulation_based_graph_inference/*.html
	rm -f simulation_based_graph_inference/*.cpp
