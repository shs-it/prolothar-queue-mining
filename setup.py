# -*- coding: utf-8 -*-

#import order is important!
import pathlib
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

with open(HERE / 'requirements.txt', 'r') as f:
    install_reqs = [
        s for s in [
            line.split('#', 1)[0].strip(' \t\n') for line in f
        ] if '=' in s or s == 'tensorflow'
    ]

with open(HERE / 'version.txt', 'r') as f:
    version = f.read().strip()

def make_extension_from_pyx(path_to_pyx: str, include_dirs = None) -> Extension:
    return Extension(
        path_to_pyx.replace('/', '.').replace('.pyx', ''),
        sources=[path_to_pyx], language='c++',
        include_dirs=include_dirs)

if os.path.exists('prolothar_queue_mining/inference/queue/nr_of_servers/corder.pyx'):
    extensions = [
        make_extension_from_pyx("prolothar_queue_mining/model/event.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/model/server/server.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/model/server/counting_server.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/model/server/list_recording_server.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/model/job/job.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/model/queue.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/model/environment.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/inference/queue/nr_of_servers/corder.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/inference/queue/nr_of_servers/corder_lcfs.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/inference/queue/nr_of_servers/nr_of_servers_estimator.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/inference/queue/naive_brute_force.pyx"),
        make_extension_from_pyx("prolothar_queue_mining/inference/queue/times.pyx"),
    ]
else:
    extensions = []

# This call to setup() does all the work
cython_profiling_activated = os.environ.get('CYTHON_PROFILING', 'False') == 'True'
setup(
    name="prolothar-queue-mining",
    version=version,
    description="algorithms for queue mining from event logs",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://gitlab.dillinger.de/KI/DataScience/processmining/prolothar-queue-mining",
    author="Boris Wiegand",
    author_email="boris.wiegand@stahl-holding-saar.de",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    packages=["prolothar_queue_mining"],
    include_package_data=True,
    ext_modules=cythonize(
        extensions, language_level = "3", annotate=True,
        compiler_directives={'profile': cython_profiling_activated}),
    zip_safe=False,
    install_requires=install_reqs
)
