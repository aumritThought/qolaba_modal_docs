.PHONY: test

test:
	sh -c '. _virtualenv/bin/activate; py.test tests'

.PHONY: test-all

test-all:
	tox

.PHONY: upload

upload: test-all build-dist
	_virtualenv/bin/twine upload dist/*
	make clean

.PHONY: build-dist

build-dist: clean
	_virtualenv/bin/pyproject-build

.PHONY: clean

clean:
	rm -f MANIFEST
	rm -rf build dist

dev-canny:
		modal deploy /home/prakhar-pc/qolaba/Modal-Deployments/src/image_to_image/controlnet/Canny.py --env dev

dev-depth:
		modal deploy /home/prakhar-pc/qolaba/Modal-Deployments/src/image_to_image/controlnet/Depth.py --env dev

dev-normal:
		modal deploy /home/prakhar-pc/qolaba/Modal-Deployments/src/image_to_image/controlnet/normal_copy.py --env dev

format:
	black src

reqs:
	pip install -r requirements.txt
# .PHONY: bootstrap

# bootstrap: _virtualenv
# 	_virtualenv/bin/pip install -e .
# ifneq ($(wildcard test-requirements.txt),)
# 	_virtualenv/bin/pip install -r test-requirements.txt
# endif
# 	make clean

# _virtualenv:
# 	python3 -m venv _virtualenv
# 	_virtualenv/bin/pip install --upgrade pip
# 	_virtualenv/bin/pip install --upgrade setuptools
# 	_virtualenv/bin/pip install --upgrade wheel
# 	_virtualenv/bin/pip install --upgrade build twine