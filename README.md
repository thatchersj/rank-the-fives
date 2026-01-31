# Eton Fives Rankings (GitHub Pages)

This repo hosts a static website (GitHub Pages) that:
- ingests tournament results from `data/TournamentResults.txt`
- recalculates rankings (7-year window + decay, as per the legacy `doRanking.R`)
- publishes:
  - `docs/data/rankings_latest.json` (for the website)
  - `docs/downloads/rankings_latest.csv`
  - `docs/downloads/rankings_latest.xlsx` (with conditional formatting)

## How to publish the site (once)

1. In **Settings → Pages**
2. Set:
   - **Source**: `Deploy from a branch`
   - **Branch**: `main`
   - **Folder**: `/docs`

Your site will be at: `https://<org-or-user>.github.io/<repo>/`

## Updating rankings

### Option A: Submit an Issue (recommended)
1. Open **Issues → New issue → “Submit tournament results”**
2. Paste the new tournament block inside the code fence (must start with a header like `2026 Northern`)
3. Submit

The GitHub Action (workflow `Recalculate rankings`) will:
- append the block to `data/TournamentResults.txt` (refusing duplicates)
- recalculate outputs
- push the updated files back to `main`

### Option B: Edit the results file directly
Edit `data/TournamentResults.txt` and push to `main`. The workflow runs on every push.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas openpyxl

python scripts/calc_rankings.py --results data/TournamentResults.txt --outdir docs/data
cp docs/data/rankings_latest.csv docs/downloads/rankings_latest.csv
python scripts/make_xlsx.py --csv docs/data/rankings_latest.csv --template data/template.xlsx --out docs/downloads/rankings_latest.xlsx
```

Then open `docs/index.html`.
