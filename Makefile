.PHONY : docs doctests lint sync tests clean clean-docs doit-list

build : lint tests docs doctests stubs doit-list

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

clean-docs :
	rm -rf docs/_build
	${MAKE} docs

clean :
	rm -rf docs/_build
	rm -f simulation_based_graph_inference/*.pyi
	rm -f simulation_based_graph_inference/*.so
	rm -f simulation_based_graph_inference/*.html
	rm -f simulation_based_graph_inference/*.cpp


GENERATORS = generate_duplication_mutation_complementation generate_duplication_mutation_random \
    generate_poisson_random_attachment generate_redirection
CPROFILE_TARGETS = $(addprefix workspace/profile/,${GENERATORS:=.prof})
LPROFILE_TARGETS = $(addprefix workspace/lprofile/,${GENERATORS:=.lprof})

workspace/profile : ${CPROFILE_TARGETS}
workspace/lprofile : ${LPROFILE_TARGETS}

${CPROFILE_TARGETS} : workspace/profile/%.prof : simulation_based_graph_inference/scripts/profile.py \
		simulation_based_graph_inference/generators.pyx
	mkdir -p $(dir $@)
	python -m cProfile -o $@ $< $*

${LPROFILE_TARGETS} : workspace/lprofile/%.lprof : simulation_based_graph_inference/scripts/profile.py \
		simulation_based_graph_inference/generators.pyx
	mkdir -p $(dir $@)
	kernprof -l -z -o $@.tmp $< $*
	python -m line_profiler $@.tmp > $@
	rm -rf $@.tmp

doit-list : dodo.py
	doit list
