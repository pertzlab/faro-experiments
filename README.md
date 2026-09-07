# faro-experiments

Pertzlab experiment notebooks and configs. One folder per experiment;
each is its own uv project pinned to the faro version it runs on.
Experiment data stays on the network share, never in this repo.

## Running an experiment

```
cd <experiment folder>
uv sync
uv run --no-sync jupyter lab   # or open the notebook in VS Code
```

`uv sync` builds the environment from the folder's `pyproject.toml`
(and `uv.lock` when present) with faro at the pinned commit.

## Starting a new experiment

Copy `examples/experiment_template/` from the faro checkout you intend
to pin (so the template matches the API), rename it here following the
`NN_short_name` scheme, and fill in the config cell. Commit the folder
with its `pyproject.toml` pin.

## During active on-scope development

Switch the `[tool.uv.sources]` faro entry to your local worktree
(`{ path = "...", editable = true }`). Before the experiment goes
dormant, set the pin back to the faro commit you actually used and
commit the `uv.lock`.

## Needing a faro change

Library changes never live in an experiment folder. Make a faro branch,
pin this experiment to that branch, and repin to main after the merge.

## Legacy experiments (migrated 2026-09 from the faro repo)

Folders 01-99 below were migrated from `pertzlab/faro:experiments/`.
Their pins are best-effort: the faro main commit that last touched each
folder. They have no `uv.lock`; one is created when an experiment is
first revived. `99_demo_data` is shared demo data used by the demo
notebooks, not an experiment.
