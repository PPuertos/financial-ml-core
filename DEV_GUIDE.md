# FINML-CORE Developer Guide

This guide documents all processes for developing, validating, and publishing the `finml-core` package. Follow this to keep your workflow consistent.

---

## 1. Validate `pyproject.toml`

Always validate the TOML file before installing or pushing changes.

```bash
# Python >=3.11
python3 -c "import tomllib; f=open('pyproject.toml','rb'); tomllib.load(f); print('TOML is valid ✅')"

# Python <3.11 (requires tomli)
pip install tomli
python3 -c "import tomli; f=open('pyproject.toml','rb'); tomli.load(f); print('TOML is valid ✅')"
```

- Checks for array syntax, missing commas, unclosed strings, etc.
- Run this **every time you update `pyproject.toml`**.

---

## 2. Install package locally (editable)

```bash
# Uninstall previous version if needed
pip uninstall finml-core -y

# Clone the repository (if not already cloned)
git clone https://github.com/PPuertos/financial-ml-core.git
cd financial-ml-core

# Install in editable mode with all extras
python3 -m pip install -e ".[dev,docs,interactive]"
```

- `-e` installs in **editable mode** → changes to code reflect immediately.
- Extras:
  - `dev` → testing & linting tools (`pytest`, `black`, `ruff`)
  - `docs` → documentation tools (`mkdocs`, `mkdocs-material`, `mkdocstrings[python]`)
  - `interactive` → plotting and notebooks (`matplotlib`, `seaborn`, `jupyterlab`)

---

## 3. Documentation workflow (MkDocs)

```bash
# Build documentation locally
mkdocs build

# Serve locally to check changes
mkdocs serve

# Deploy / update GitHub Pages
mkdocs gh-deploy
```

- **Notes on `mkdocs build` vs `serve`:**
  - `mkdocs build` → compiles the docs into the `site/` folder (used for deployment)
  - `mkdocs serve` → runs a local server so you can preview changes in the browser
- **GitHub Pages update:** `mkdocs gh-deploy` deploys the current `site/` folder to GitHub Pages.
- **Commit required?**  
  - If you update `.md` files in `docs/`, you **should commit these changes** so the repository stays updated.
  - `gh-deploy` can deploy without a commit, but it’s best practice to track doc changes in Git.

---

## 4. Git commands

### Common workflow

```bash
# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "Your commit message"

# Pull latest changes
git pull origin main

# Push to GitHub
git push origin main
```

### Tags & releases (optional)

```bash
# Create a version tag
git tag v0.1.2
git push origin v0.1.2
```

- Tags mark release versions in GitHub. Useful for tracking PyPI releases.
- Even if you update `version` in `pyproject.toml`, tagging helps collaborators see release points.

---

## 5. Optional dependencies

You can define extra groups in `pyproject.toml`:

```toml
[project.optional-dependencies]
notebooks = ["jupyter"]
viz = ["plotly", "matplotlib"]
dev-tools = ["pre-commit", "isort"]
```

Install locally as needed:

```bash
pip install -e .[notebooks,viz,dev-tools]
```

---

## 6. DEV_GUIDE.md management

- Keep this guide updated as processes evolve.
- Exclude it from PyPI packaging:

```toml
[tool.setuptools.packages.find]
include = ["finml_core*"]
exclude = ["misc*", "notebooks*", "tests*", "DEV_GUIDE.md"]
```

- This ensures it stays for developers but is **not published to PyPI**.

---

## 7. Recommended workflow summary

### When updating **code**
1. Validate TOML (if changed)
2. Install package locally (`pip install -e .[dev,docs,interactive]`)
3. Run tests (`pytest`)
4. Commit & push changes

### When updating **documentation**
1. Run `mkdocs serve` to check changes
2. Commit `.md` files in `docs/`
3. Deploy with `mkdocs gh-deploy`

### When updating **pyproject.toml**
1. Validate TOML syntax
2. Reinstall package locally
3. Commit & push changes

---

✅ Following this guide ensures reproducibility and a consistent workflow across machines and collaborators.
