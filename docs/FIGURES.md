# Figures

`src/figures.py` builds seven results figures from a finished run, in one style and in light and dark
versions. `scripts/make_figures.py` writes them without retraining:

```bash
python scripts/make_figures.py --run_dir runs/<run> --data_dir data/processed --export_dir docs/figures
```

| Figure | Question it answers |
|---|---|
| `cumulative_gains` | How many churners do we reach by contacting the highest-risk users first? |
| `model_comparison` | Which model is best, and are the differences bigger than noise? |
| `base_rate_shift` | Why are the predicted probabilities too low for a new month? |
| `calibration` | Does a predicted 10% mean 10%? |
| `feature_importance` | Which features move the predictions, and where do they come from? |
| `month_end_effect` | How does churn depend on the day of the month a membership ends? |
| `label_choice` | Official labels, or labels rebuilt with the organisers' script? |

Outputs:

- `runs/<run>/plots/<name>.html`: interactive (hover for exact values). The camera icon in the corner
  saves a 2x PNG.
- `<export_dir>/<name>_light.png` and `<name>_dark.png`: for the README, which shows the one that matches
  the reader's GitHub theme.

## Style

- Blue always marks the champion model; grey marks reference values. Colours come from a palette checked
  for colour-blind separation, and every chart with several series has a legend or direct labels.
- Titles state the finding and are computed from the numbers, so a new run rewrites them.
- 2px lines, markers with a ring in the background colour, rounded bar ends, hairline grid.

## PNG export

Static images use Kaleido, which drives a headless Chrome. On a fresh Linux machine:

```bash
pip install -U "plotly>=6.1" "kaleido>=1.0"
plotly_get_chrome -y
# Chrome's system libraries, if the export still fails. Ubuntu 24.04 renamed some of them with a
# t64 suffix, so install one at a time and let unknown names fail on their own:
for p in libnss3 libatk-bridge2.0-0t64 libatk-bridge2.0-0 libcups2t64 libcups2 libxcomposite1 \
         libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 \
         libasound2t64 libasound2; do sudo apt-get install -y -qq "$p" || true; done
```

If export fails, the HTML figures are still written and the warning in `run.log` gives the reason.
