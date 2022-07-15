.PHONY : docs doctests lint sync tests clean clean-docs doit-list

build : lint tests docs doctests doit-list

lint :
	flake8

tests :
	pytest tests -v --cov=simulation_based_graph_inference --cov-fail-under=100 --cov-report=term-missing --cov-report=html

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

GENERATORS = duplication_mutation duplication_complementation poisson_random_attachment \
	redirection geometric
CPROFILE_TARGETS = $(addprefix workspace/profile/,${GENERATORS:=.prof})
LPROFILE_TARGETS = $(addprefix workspace/lprofile/,${GENERATORS:=.lprof})

workspace/profile : ${CPROFILE_TARGETS}
workspace/lprofile : ${LPROFILE_TARGETS}

${CPROFILE_TARGETS} : workspace/profile/%.prof : simulation_based_graph_inference/scripts/profile.py
	mkdir -p $(dir $@)
	python -m cProfile -o $@ $< --generator=$*

${LPROFILE_TARGETS} : workspace/lprofile/%.lprof : simulation_based_graph_inference/scripts/profile.py
	mkdir -p $(dir $@)
	kernprof -l -z -o $@.tmp $< --generator=$*
	python -m line_profiler $@.tmp > $@
	rm -rf $@.tmp

doit-list : dodo.py
	doit list
