# Powerball Weighted Picker Portal

A Streamlit app and companion engine for generating diversified Powerball ticket slates with statistical guidance. It supports position ranges, Bayesian-style priors, exponential recency, deterministic beam search, hybrid strategies, and enforcement of overlap and unique Powerball constraints. Export results to CSV and a detailed Markdown run log.

> Disclaimer: This project is for experimentation and education. It does not predict lottery outcomes. Use responsibly.

---

## Features

* Position range filters for each white-ball position and the Powerball.
* Presets that instantly apply a tuned set of parameters.
* Multiple generation modes:

  * `sample` - stochastic sampling from weighted distributions.
  * `greedy` - choose highest probability at each step.
  * `hybrid` - mix of sampling and greedy selection.
  * `deterministic` - beam search over white-ball sequences with a Top N Powerball chooser.
  * `hybrid_det` - hybrid drafting guided by a beam-derived white-ball pool.
  * `ALL` - seeds from several modes, then expands while enforcing constraints.
* Guidance controls:

  * Beam width and guidance pool size.
  * Guidance strength for white balls and Powerball.
  * Top N Powerball chooser for deterministic mode.
* Diversification controls:

  * Maximum number of shared white balls across tickets.
  * Unique Powerball option across the slate.
  * Reroll attempts and proposals per attempt to satisfy constraints.
* Detailed Markdown run log with parameters, band settings, and picks.
* CSV export of generated tickets.

---

## Repository structure

```
.
├─ streamlit_powerball_portal_v10.py   # Streamlit UI and orchestration
├─ powerball_weighted_picker_v6.py     # Core generation engine with CLI
└─ powerball_drawings.csv              # Example data (date,pos1..pos5,pb)
```

---

## Requirements

* Python 3.10 or newer. Verified with Python 3.13.
* Packages:

  * `streamlit`
  * `pandas`
  * `numpy`
  * `matplotlib`

Install:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit pandas numpy matplotlib
```

---

## Quick start

1. Activate your virtual environment.
2. Start the app:

```bash
streamlit run streamlit_powerball_portal_v10.py
```

3. In the app, upload a CSV with columns:

```
date,pos1,pos2,pos3,pos4,pos5,pb
```

Notes:

* `date` must be parseable by `pandas.to_datetime` (for example `YYYY-MM-DD`).
* `pos1..pos5` are integers in 1..69.
* `pb` is an integer in 1..26.

4. Adjust bands and parameters or select a Preset.
5. Click Generate tickets.
6. Download your tickets as CSV and optionally download or save the Markdown run log.

---

## Default position ranges

The app initializes the sliders with these ranges. You can adjust them at any time.

* Pos1: 4 to 17
* Pos2: 12 to 33
* Pos3: 21 to 49
* Pos4: 33 to 60
* Pos5: 51 to 66
* PB: 4 to 22

These ranges constrain candidates for each position during generation.

---

## How it works

At a high level the app:

1. **Loads historical draws** from your CSV and sorts by date.
2. **Builds posterior counts** for each position and the Powerball:

   * White-ball positions use value counts over a lookback window plus a configurable prior.
   * Powerball uses value counts plus its own prior.
3. **Computes exponential recency multipliers** with a user-defined half-life in draws and per-lane strengths. Recent draws weigh more.
4. **Combines posterior counts with recency** to form per-position probability vectors within your chosen position bands.
5. **Generates candidate tickets** using the selected mode:

   * `deterministic`: beam search proposes top white-ball sequences, then pairs each with a Top N list of Powerballs.
   * `hybrid_det`: builds a high-confidence white-ball pool with the beam, then drafts tickets with a hybrid selector.
   * `hybrid`, `sample`, `greedy`: generate directly from per-position distributions using the requested strategy.
   * `ALL`: seeds from several modes, then expands while enforcing constraints.
6. **Enforces diversification**:

   * Limits the number of shared white balls across tickets.
   * Optional unique Powerball across the slate.
   * Uses reroll attempts with a proposals-per-attempt budget to repair conflicts. Can relax constraints if needed to reach the requested count.
7. **Outputs** a table of tickets and provides CSV and Markdown log downloads.

---

## Streamlit app controls

* **Presets**

  * Balanced, Diversified, Concentrated, Aggressive Deterministic, Gentle Hybrid.
  * Each preset sets:

    * Max shared white balls across tickets.
    * Reroll attempts and proposals per attempt.
    * Beam width and guidance pool.
    * Guidance strengths and PB Top N.
    * Unique Powerball toggle.

* **Controls**

  * Number of sets to generate.
  * Mode.
  * Random seed.

* **Windows and priors**

  * Lookback years for the training window.
  * Prior strength for white balls.
  * Prior strength for Powerball.

* **Recency**

  * Half-life in draws.
  * Recency strength for white balls.
  * Recency strength for Powerball.

* **Diversification**

  * Repeat penalty for whites and a switch to disable it.
  * Max shared white balls across tickets.
  * Reroll attempts.
  * Proposals per attempt.
  * Unique Powerball toggle.

* **Deterministic + Guidance**

  * Beam width.
  * Guidance pool size.
  * Guide strength white.
  * Guide strength PB.
  * PB Top N for deterministic mode.

* **Logging**

  * Save Markdown log to disk.
  * Log folder path.

* **File upload**

  * Upload your historical CSV.
  * The app validates column names and types, and sorts by `date`.

* **Outputs**

  * Generated tickets table.
  * Download tickets as CSV.
  * Download detailed run log as Markdown.
  * Preview the run log in the UI.

---

## Command line engine

`powerball_weighted_picker_v6.py` can run without Streamlit. It expects an input CSV and exposes the core parameters.

Usage:

```bash
python powerball_weighted_picker_v6.py \
  --csv powerball_drawings.csv \
  --sets 5 \
  --mode hybrid \
  --seed 42 \
  --lookback-years 5 \
  --prior-strength-white 200.0 \
  --prior-strength-pb 20.0 \
  --recency-half-life 50 \
  --recency-strength-white 0.6 \
  --recency-strength-pb 0.6 \
  --repeat-penalty 0.25 \
  --beam 64 \
  --overlap-penalty 0.15 \
  --top-pb 2 \
  --guide-pool 40 \
  --guide-strength-white 0.5 \
  --guide-strength-pb 0.5
```

Arguments:

* `--csv` path to input draws. Required. Columns must be `date,pos1,pos2,pos3,pos4,pos5,pb`.
* `--sets` number of tickets to generate. Choices: 1, 3, 5, 10.
* `--mode` one of `sample`, `greedy`, `hybrid`, `deterministic`, `hybrid_det`, `all`.
* `--seed` random seed for non-deterministic modes.
* `--lookback-years` training window size in years.
* `--prior-strength-white` scalar prior strength for white balls.
* `--prior-strength-pb` scalar prior strength for Powerball.
* `--recency-half-life` half-life in draws for exponential recency.
* `--recency-strength-white` recency multiplier for white balls.
* `--recency-strength-pb` recency multiplier for Powerball.
* `--repeat-penalty` penalty for repeating white-ball values while drafting.
* `--no-repeat-penalty` disables the repeat penalty.
* `--beam` beam width for deterministic search.
* `--overlap-penalty` tradeoff used internally when ranking near-duplicate sequences.
* `--top-pb` pick top N Powerball choices in deterministic mode. Choices: 1, 2, 3.
* `--guide-pool` number of white-ball sequences used for guidance in `hybrid_det`.
* `--guide-strength-white` guidance strength for white balls.
* `--guide-strength-pb` guidance strength for Powerball.

The CLI prints the resulting tickets to stdout. Redirect to a file if needed.

---

## Data format

Your CSV must include:

* `date`: parseable by `pandas.to_datetime`.
* `pos1`, `pos2`, `pos3`, `pos4`, `pos5`: integers in 1..69.
* `pb`: integer in 1..26.

The UI sorts by `date` and uses the most recent date as the end of the training window.

---

## Logs

When enabled, the app writes a Markdown report that includes:

* Parameters and their values.
* Position band settings.
* Mode summaries and internal counts where applicable.
* The final ticket slate.

You can preview the log in the UI and download it. If “Also save log to disk” is checked, the app writes a `.md` file to your chosen folder.

---

## Troubleshooting

* **Error**: `st.session_state.overlap_k cannot be modified after the widget with key overlap_k is instantiated.`

  * Cause: In Streamlit, you cannot set a `session_state` value for a widget after that widget has been created in the same run.
  * Fix: Apply presets before creating the widgets and then call `st.rerun()` so the UI instantiates with the new values. Alternatively, queue a pending preset in `session_state` at the very top of the script and consume it before any widgets are built.

* **Date parsing fails**

  * Ensure the `date` column is present and parseable. Format like `YYYY-MM-DD` is safe.

* **Import errors**

  * Reinstall requirements inside your virtual environment:

    ```bash
    pip install --upgrade pip
    pip install streamlit pandas numpy matplotlib
    ```

* **Nothing generates**

  * Start with the `Balanced` preset.
  * Reduce “Max shared white balls across tickets”.
  * Increase “Reroll attempts” and “Proposals per attempt”.
  * Use a larger lookback window so there is more data.

---

## Roadmap

* Optional auto-update of the input CSV from a public data source when recent draws are missing.
* Simple simulation tools to measure overlap and recency exposure across many runs.
* Dockerfile for consistent local runs.

---

## License

Choose a license that fits your needs, for example MIT. Add the license file to the repository.

---

## Acknowledgements

Thanks to the Python and Streamlit communities for excellent tooling that makes rapid iteration possible.
