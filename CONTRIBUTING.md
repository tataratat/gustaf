# Contributing
gustaf welcomes and appreciates discussions, issues and pull requests!

## Quick start
Once the repo is forked, one possible starting point would be creating a new python environments, for example, using [conda](https://docs.conda.io/en/latest/miniconda.html) with `python=3.9`
```bash
conda create -n gustafenv python=3.9
conda activate gustafenv
git clone git@github.com:<path-to-your-fork>
cd gustaf
git checkout -b new-feature0
python3 setup.py develop
```

## Style / implementation preferences
- use `if` and `raise` instead of `assert`
- no complex comprehensions: preferably fits in a line, 2 lines max if it is totally necessary
- use first letter abbreviations in element loops:  `for kv in knot_vectors`
- use `i`, `j`, `k`, `l` for pure index: `for i, kv in enumerate(knot_vectors)`


### Formatting and style check
To check the format and style of your code use the following commands:
```bash
pip install pre-commit
precommit run -a
```

## Local docs build
To check if documentations look as intended, you can build it locally.
Remember, `spline` extensions will be empty if you don't have `splinepy` available.
```bash
pip install -r ./docs/requirements.txt
sphinx-apidoc -f -t docs/source/_templates -o docs/source gustaf
sphinx-build -b html docs/source docs/build
```
Now, you can check documentations by opening `docs/build/index.html` with a browser.


### Bash script for format and documentation build checking
Thanks to github, we have a CI that runs tests and checks for PRs.
However, in case you'd like to check this locally, here are sequence of commands as a function:
```bash
# run this at repo's root
function gustaf_commit() {
    pre-commit run -a
    rm -r docs/build
    rm docs/source/*.*.rst docs/source/gustaf.rst docs/source/modules.rst
    sphinx-apidoc -f -t docs/source/_templates -o docs/source gustaf
    sphinx-build -b html docs/source docs/build
    pytest tests
}
```
