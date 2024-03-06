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

```bash
make clean_package || make package && make publish
```

You should also create a tag for the current version

```bash
git tag -a [version] -m "describe what has changed"
git push --tags
```

## Versioning

We use [SemVer](http://semver.org/) for versioning.

## Authors

If you have any questions, feel free to ask one of our authors:

* **Boris Wiegand** - boris.wiegand@stahl-holding-saar.de
