# Contributing
gustaf welcomes and appreciates discussions, issues and pull requests!

## Quick start
Once the repo is forked, one possible starting point would be creating a new python environments, for example, using [conda](https://docs.conda.io/en/latest/miniconda.html) with `python=3.9`
```bash
conda create -n gustafenv python=3.9
git clone git@github.com:<path-to-your-fork>
cd gustaf
git checkout -b new-feature0
python3 setup.py develop
```

## Style / implementation preferences
gustaf implementations tries to follow [pep8](pep8.org)'s suggestion closely. A formatter setup will be added soon. For now, followings are some preferences:
- use `if` and `raise` instead of `assert`
- no complex comphrehensions: preferably fits in a line, 2 lines max if it is totally necessary
- use first letter abbreviations in element loops:  `for kv in knot_vectors`
- use `i`, `j`, `k`, `l` for pure index: `for i, kv in enumerate(knot_vectors)`
- vertical alignment only with tabs
- put closing brackets on a separate line, dedented
- if a new feature would open doors to more related functionalities, consider a helper class
- try to avoid looping possibly giant loops

## Pull request suggessions
gustaf is a successor of gustav, which was an internal/experimental library.
Until all the functionalities are fully transferred, it may go through several dramatic changes.
Followings are gentle suggestions for PRs, so that the pre-alpha phase can end as soon as possible:
- small, separable features
- similar implementation style as current implementation (I know, quite ambiguous)
- unit tests  
On the other hand, it is perfect time for suggestions / requests / feedbacks, so let us know!
