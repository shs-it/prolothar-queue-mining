# Prolothar Queue Mining

Algorithms for queue mining (discovering discrete event simulations based on waiting queue models) from event logs

Based on the publication
> Boris Wiegand, Dietrich Klakow, and Jilles Vreeken.
> **Why Are We Waiting? Discovering Interpretable Models for Predicting Sojourn and Waiting Times.**
> In: *Proceedings of the SIAM International Conference on Data Mining (SDM), Minneapolis, MN.* 2023, pp. 352–360.

## Prerequisites

Python 3.11+

## Usage

If you want to run the algorithms on your own data, follow the steps below.

### Installing

```bash
pip install prolothar-queue-mining
```

### Creating a dataset

```python
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.inference.queue import CueMin

# our input data are jobs with an ID and their corresponding arrival resp. departure time
observed_arrivals = [
    (Job('A'), 3),
    (Job('B'), 4),
    (Job('C'), 5),
    (Job('D'), 6),
    (Job('E'), 7),
    (Job('F'), 8),
]
observed_departues = [
    (Job('A'), 4),
    (Job('B'), 7),
    (Job('C'), 11),
    (Job('D'), 12),
    (Job('E'), 13),
    (Job('F'), 14),
]

#you can add additional features to a job, example:
Job('4711', {'color': 'blue', size: 12})

cuemin = CueMin(verbose=True)

#if your jobs have features, which can have an influence on the service order or service time:
cuemin = CueMin(verbose=True, categorical_attribute_names = ['color'], numerical_attribute_names = ['size'])

#find and a print a waiting queue model
queue = cuemin.infer_queue(observed_arrivals, observed_departues)
print(queue)

#if you want to use domain knowledge to restrict the number of server, e.g. min 2 and max 4:
queue = cuemin.infer_queue(observed_arrivals, observed_departues, search_strategy_name='linear-2-4')
```

## Development

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Additional Prerequisites
- make (optional)

### Compile Cython code

```bash
make cython
```

### Running the tests

```bash
make test
```

### Deployment

Optional requirement: Create a .pypirc file in the project root directory with your pypi authentication token:
```
[pypi]
username = __token__
password = pypi-AgEIcH...
```

1. Change the version in version.txt
2. Build the package

```bash
make clean_package
make package
```

3. Deploy the version to Pypi:
```bash
 make publish
 ```
or 
```bash
twine upload --skip-existing --verbose --config .pypirc dist/*
```

4. Create and push a tag for this version by

```bash
git tag -a $(cat version.txt) -m "describe this version"
git push --all && git push --tags
```

### Devcontainer

There is a decontainer definition in this project, which helps you to set up your environment.
At Stahl-Holding-Saar, we are behind a corporate proxy and cannot install dependencies from PyPi directly.
I yet have not found a stable solution to set the PIP_INDEX_URL and PIP_TRUSTED_HOST variables dynamically. 
In the current Dockerfile, I hardcoded the values, so you have to adapt them. 
If you know a solution to this problem, please contact me. 

## Versioning

We use [SemVer](http://semver.org/) for versioning.

## Authors

If you have any questions, feel free to ask one of our authors:

* **Boris Wiegand** - boris.wiegand@stahl-holding-saar.de
