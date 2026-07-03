# Learning note 01 - the DS core

This note explains how the Milestone 1 pipeline fits together and *why* each
piece is shaped the way it is.
The goal is that you can read `sentinel/core/*.py`, understand every decision,
and rewrite pieces yourself.

The whole thing is one line of data flow:

```
FD001 raw files -> RUL target -> rolling-window features -> AutoML -> best model + metrics
       data.py            data.py         features.py         automl.py
```

`pipeline.py` just calls those four things in order.

---

## 1. The problem: RUL as a regression

Each turbofan engine in C-MAPSS is run until it fails, logging 21 sensors +
3 operating settings every cycle.
We want to predict **Remaining Useful Life**: how many cycles are left before
this engine fails.

That's a **regression** problem - the target is a number of cycles, not a
class.
So every row of sensor readings becomes one training example, and its label is
"cycles remaining".

### Why `max_cycle - current_cycle`

For a *training* engine we already know when it died: the last cycle in its log
**is** the failure point.
So for any earlier row, the true remaining life is simply
`(last cycle) - (this cycle)`.
That's the whole RUL label derivation in `data.add_rul`.
An engine that failed at cycle 200, looked at on cycle 50, had 150 cycles left.

### Why cap the RUL (the `DEFAULT_RUL_CAP = 125` knob)

Raw RUL for a fresh engine can be 300+ cycles.
But early in life, nothing has degraded yet - the sensors look identical to a
healthy engine's, so there is no signal that says "you have 312 cycles left"
vs "289 cycles left".
Asking the model to predict those large numbers just teaches it to fit noise.

The standard C-MAPSS fix is a **piecewise-linear RUL**: clip the target at a
cap (125 here).
Below the knee, RUL decreases linearly with each cycle (the degradation phase
we *can* predict); above it, we call the engine "healthy" and label it a flat
125.
The cap is a named constant (`DEFAULT_RUL_CAP`) precisely because it's a
modeling choice you should be able to change and re-measure.

### The test set is different

Test engines are **truncated** before failure - the log just stops mid-life.
So we can't compute their RUL from the data.
NASA ships a separate `RUL_FD001.txt` giving the true RUL at each test engine's
*last observed cycle*.
`data.build_test_eval` therefore keeps only the **last row per test unit** and
attaches that truth value (capped the same way, so train and test targets live
on the same scale).
That's the honest generalization test: one prediction per engine, scored
against the real remaining life.

---

## 2. Features: turning a time series into a table

A single row of sensors is a snapshot.
But failure shows up as a **trend** - a temperature creeping up, a pressure
drifting - not in any one instant.
A plain regressor fed one row at a time can't see trends.

So `features.build_features` computes, **per engine, per sensor**, rolling
statistics over the last `N` cycles (window, default 5):

- **mean** - the recent level (smooths out per-cycle noise)
- **std** - how unstable the sensor has been recently
- **slope** - the least-squares trend: is it rising or falling, how fast
  (`_window_slope`)

Now each row carries a little summary of its own recent history, and the table
is flat and trainable.
Two design points worth internalizing:

- **Windows never cross engines.** We `groupby("unit")` first, so engine 2's
  first cycles never borrow engine 1's tail. Getting this wrong silently leaks
  the future into the past.
- **Drop dead sensors first.** Several FD001 sensors never move across the
  whole fleet (`informative_sensors` finds them by near-zero variance). Constant
  columns are pure noise to a model - we drop them once, decided on the training
  set, and apply the *same* drop list to test so both tables have identical
  columns.

`unit` and `cycle` are carried through the feature table for joining and eval,
but they are **not model inputs** - `unit` is just an identifier (letting the
model "memorize" engine ids would be leakage), and raw `cycle` doesn't mean the
same thing in truncated test logs. `automl` passes them as `ignore_features`.

---

## 3. AutoML: what PyCaret actually does

`automl.train_and_evaluate` is a thin, deterministic wrapper over three PyCaret
calls. Here's what each buys you:

- **`setup(...)`** - defines the experiment: the data, the target, which
  columns to ignore, the CV fold count, the random seed. Under the hood it also
  builds a preprocessing pipeline (imputation, scaling, encoding) that travels
  *with* the model, so the saved artifact preprocesses raw features the same way
  at predict time.
- **`compare_models(...)`** - the AutoML heart. It trains a whole shelf of
  regressors (linear, regularized, trees, random forest, extra trees, gradient
  boosting, LightGBM, KNN...) with cross-validation and returns them ranked by a
  metric (we sort by RMSE). `pull()` grabs that ranking as the **leaderboard**
  DataFrame. This is the "compare many models with little code" story.
- **`finalize_model(best)`** - refits the winning model on *all* the training
  data (CV held out folds; now we use everything) to produce the model we
  actually ship.

We then do the part PyCaret's CV score does **not** cover: predict on the real
held-out FD001 **test set** and compute RMSE / MAE / R2 with plain sklearn. CV
score measures fit on train folds; the test-set score measures generalization
to unseen engines - the number that actually matters.

Finally we `save_model` (the whole preprocessing+model pipeline) and write
`metrics.json` + `leaderboard.csv` to `artifacts/`.

Reproducibility: `pipeline.set_seeds` pins Python/NumPy RNGs and PyCaret gets a
fixed `session_id`, so the leaderboard and metrics come out the same each run.

---

## 4. The seams the agent layer will plug into

M1 is deliberately just functions with clear inputs/outputs, because the
Milestone 2 agent layer (LangGraph) wraps these seams rather than reaching
inside them:

- **`data.load_fd001(...) -> FD001`** - the dataset-loading seam. The V1
  dataset-agnostic goal adds sibling loaders behind the same shape; the
  interviewer sub-agent will eventually choose/parameterize the loader. We did
  **not** build the registry yet (YAGNI) - just left the shape.
- **`features.build_features(df, sensor_cols, window)`** - `window` (and the
  sensor drop threshold) are the knobs the interviewer could set from a user
  conversation.
- **`automl.train_and_evaluate(...) -> TrainResult`** - returns the
  `leaderboard`, fitted `best_model`, and `metrics`. That return value is
  exactly what the orchestrator will react to and the report-writer sub-agent
  will turn into plain-language summaries. The `metrics.json` it writes is the
  "significant event" payload the agent layer reads.

Nothing in this core imports an LLM, and it shouldn't - keeping the DS core
deterministic and independently runnable is what makes the agent layer testable
on top of it.

---

## Try it yourself

Good rewrite exercises, in increasing difficulty:

1. Change `DEFAULT_RUL_CAP` to 100 and 150, rerun, and watch how RMSE/R2 move. Explain
   why.
2. Add a `<sensor>_min` / `<sensor>_max` (range) feature in `build_features`
   and see if the leaderboard improves.
3. Swap the `_window_slope` least-squares fit for a simpler "last minus first
   over window" trend and compare - is the extra rigor worth it?
