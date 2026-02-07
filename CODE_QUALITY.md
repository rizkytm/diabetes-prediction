# Code Quality & Auto-Formatting Guide

This guide explains how to maintain code quality and prevent CI/CD failures due to formatting issues.

---

## 🎯 Problem: CI/CD Formatting Failures

When you push code to GitHub, the CI/CD pipeline runs `black --check` to verify code formatting. If code doesn't meet Black formatting standards, the pipeline fails.

**Error Example:**
```
Run black --check --diff src/ api.py app.py
would reformat /home/runner/work/.../src/__init__.py
Error: Process completed with exit code 1.
```

---

## ✅ Solution: Auto-Format Before Committing

### **Quick Fix (Run This Before Every Commit):**

```bash
# Format imports and code
isort src/ api.py app.py
black src/ api.py app.py

# Verify it passes black check
black --check src/ api.py app.py
```

If no output, your code is properly formatted! ✅

---

## 🛠️ Option 1: Manual Formatting (Recommended Before Commit)

### **Step 1: Install Formatting Tools**

```bash
pip install black isort flake8 pylint
```

Or add to requirements.txt:
```txt
# Code Quality & Formatting
black==24.10.0
isort==5.13.2
flake8==7.1.1
pylint==3.3.2
```

### **Step 2: Format Your Code**

```bash
# Format all Python files
isort src/ api.py app.py
black src/ api.py app.py
```

### **Step 3: Verify Formatting**

```bash
# Check if code is properly formatted
black --check src/ api.py app.py

# Should show no output if everything is good
```

---

## 🤖 Option 2: Use Makefile Command (Easiest!)

```bash
make format
```

This command will:
1. Format imports with `isort`
2. Format code with `black`
3. Show you what was changed

---

## 🔧 Option 3: Git Pre-Commit Hook (Automatic!)

### **Setup Pre-Commit Hooks:**

Pre-commit hooks automatically format your code **before** every commit, so you never forget!

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]
EOF

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files (one-time setup)
pre-commit run --all-files
```

### **How It Works:**

After setup, every time you run `git commit`:
1. Pre-commit automatically formats your code
2. Pre-commit checks for linting errors
3. If issues found, commit is blocked
4. Fix issues, then commit again

**Example:**
```bash
git add .
git commit -m "Add new feature"
# Pre-commit runs automatically:
# [INFO] Formatting files with black...
# [INFO] Import sorting with isort...
# [OK] All checks passed!
# Commit proceeds ✅
```

---

## 📋 Command Reference

### **Formatting Commands:**

```bash
# Format imports
isort src/ api.py app.py

# Format code
black src/ api.py app.py

# Format both (import sorting + code formatting)
isort src/ api.py app.py && black src/ api.py app.py

# Check if formatting is needed (no output = good)
black --check src/ api.py app.py

# Show diff of what would change
black --diff src/ api.py app.py
```

### **Linting Commands:**

```bash
# Check code style
flake8 src/ api.py app.py --max-line-length=100 --extend-ignore=E203,W503

# Check code quality
pylint src/ api.py app.py --disable=C0111,R0913,R0914 --fail-under=8.0

# Run all checks (from Makefile)
make lint
```

### **All-in-One Command:**

```bash
# Format + Check + Lint
make format && make lint
```

---

## 🎨 Black Configuration

### **Create `pyproject.toml` (Optional):**

```toml
[tool.black]
line-length = 100
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  # files
  | notebooks
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

---

## 🔄 CI/CD Pipeline Configuration

The GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) already includes formatting checks:

```yaml
# From .github/workflows/ml-pipeline.yml
- name: Run Black (code formatting check)
  run: black --check --diff src/ api.py app.py

- name: Run isort (import sorting check)
  run: isort --check-only --diff src/ api.py app.py

- name: Run Flake8 (linting)
  run: flake8 src/ api.py app.py --max-line-length=100 --extend-ignore=E203,W503
```

---

## 🛡️ Best Practices

### **1. Format Before Every Commit**

```bash
# Make it a habit!
git add .
make format
git commit -m "Your message"
```

### **2. Use Pre-Commit Hooks**

Setup once, and never worry about formatting again:
```bash
pip install pre-commit
pre-commit install
```

### **3. Run Format in CI/CD**

Already configured! Just make sure to fix formatting if CI fails.

### **4. Format in IDE**

**VS Code:**
- Install extension: "Black Formatter"
- Settings: "Format on Save": true
- Settings: "Default Formatter": "ms-python.black-formatter"

**PyCharm:**
- Settings → Tools → External Tools
- Add Black formatter
- Configure to run on save

---

## 🐛 Troubleshooting

### **Problem: Black formatting is different from my style**

**Solution:**
Black is "opinionated" - it has one style and that's it. You get used to it quickly! Benefits:
- Consistent code across team
- No style debates
- Automatic formatting

### **Problem: Black broke my code**

**Solution:**
Black is very safe, but if you're worried:
```bash
# 1. Check what will change first
black --diff src/ api.py app.py

# 2. Review changes
# 3. If good, apply
black src/ api.py app.py

# 4. Test that code still works
pytest tests/  # or run your app
```

### **Problem: CI still fails after formatting**

**Checklist:**
1. Did you format ALL files?
   ```bash
   isort src/ api.py app.py
   black src/ api.py app.py
   ```

2. Are you using the same version as CI?
   ```bash
   black --version
   # Should show: black, 24.10.0
   ```

3. Are there any uncommitted files?
   ```bash
   git status
   git add .
   ```

4. Try full rebuild:
   ```bash
   make format
   git add .
   git commit -m "Fix formatting"
   git push
   ```

---

## 📊 Summary

| Tool | Purpose | Command |
|------|---------|---------|
| **black** | Code formatting | `black src/ api.py app.py` |
| **isort** | Import sorting | `isort src/ api.py app.py` |
| **flake8** | Linting | `flake8 src/ api.py app.py` |
| **pylint** | Code quality | `pylint src/ api.py app.py` |
| **pre-commit** | Auto-format on commit | `pre-commit install` |

---

## ✅ Quick Checklist

Before pushing to GitHub:

- [ ] Run `make format` or `isort ... && black ...`
- [ ] Run `black --check` to verify
- [ ] Run tests to ensure nothing broke
- [ ] Commit and push

**Or better:** Install pre-commit hooks and let it handle everything automatically! 🎉

---

**Last Updated:** 2026-02-07

**Status:** ✅ Active - Prevents CI failures
