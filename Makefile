cython :
	python setup.py build_ext --inplace

clean :
	python setup.py clean --all

package :
	python -m build

clean_package :
	rm -R dist build prolothar_rule_mining.egg-info

publish :
	twine upload --skip-existing --verbose dist/*

test :
	python -m coverage erase
	python -m coverage run --branch --source=./prolothar_queue_mining -m unittest discover -v
	python -m coverage xml -i
