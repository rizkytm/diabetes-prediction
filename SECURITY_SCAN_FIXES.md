# Security Scan Fixes - Summary

## Issues Fixed

### 1. ✅ CodeQL Action v3 Deprecation Warning
**Problem:**
```
Warning: CodeQL Action v3 will be deprecated in December 2026.
Please update all occurrences of the CodeQL Action in your workflow files to v4.
```

**Solution:**
- Updated `github/codeql-action/upload-sarif` from `@v3` to `@v4`
- Added `category: 'trivy'` parameter to distinguish from CodeQL results

### 2. ✅ Missing Security Events Permission
**Problem:**
```
Error: Resource not accessible by integration
This run of the CodeQL Action does not have permission to access the CodeQL Action API endpoints.
Please ensure the workflow has at least the 'security-events: write' permission.
```

**Root Cause:**
GitHub Actions workflows need explicit permissions to upload security scanning results to the Security tab.

**Solution:**
Added permissions block at workflow level:
```yaml
permissions:
  contents: read
  security-events: write  # Required for uploading SARIF files
  actions: read
```

### 3. ✅ Security Scan Fails on Fork PRs
**Problem:**
Security scan fails when running on pull requests from forks because they don't have access to repository secrets and security APIs.

**Solution:**
- Added `continue-on-error: true` to upload step
- Added `if: github.event_name == 'push' || github.event_name == 'pull_request'` condition
- Security scan still runs and generates results, but won't fail if upload fails

## Files Modified

### `.github/workflows/ml-pipeline.yml`

**Changes:**
1. Added `permissions` block at workflow level (lines 10-13)
2. Updated CodeQL action from v3 to v4 (line 187)
3. Added `continue-on-error: true` to SARIF upload (line 188)
4. Added `category: 'trivy'` to distinguish results (line 191)
5. Added explicit condition for security job (line 172)

## Updated Workflow Structure

```yaml
name: ML Pipeline CI/CD

# ✅ NEW: Permissions for security scanning
permissions:
  contents: read
  security-events: write
  actions: read

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint: ...
  test: ...
  build: ...

  security:
    name: Security Vulnerability Scan
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push' || github.event_name == 'pull_request'  # ✅ Explicit condition

    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v4  # ✅ Updated to v4
      continue-on-error: true  # ✅ Won't fail if upload fails
      with:
        sarif_file: 'trivy-results.sarif'
        category: 'trivy'  # ✅ Distinguish from CodeQL results

  deploy: ...
```

## How Security Scan Works Now

### For Push to Main/Develop:
1. ✅ Runs Trivy vulnerability scanner
2. ✅ Generates SARIF file (`trivy-results.sarif`)
3. ✅ Uploads to GitHub Security tab (with proper permissions)
4. ✅ Results visible in repository's Security tab

### For Pull Requests:
1. ✅ Runs Trivy vulnerability scanner
2. ✅ Generates SARIF file
3. ⚠️ Attempts upload (may fail for forks)
4. ✅ Continues even if upload fails (due to `continue-on-error: true`)

## Viewing Security Results

### In GitHub Repository:
1. Go to repository **Security** tab
2. Click **Code scanning alerts**
3. Filter by **trivy** category
4. View vulnerability details and severity

### In GitHub Actions:
1. Go to **Actions** tab
2. Click on workflow run
3. Click on **Security Vulnerability Scan** job
4. View Trivy scan output in logs

## Security Scan Coverage

Trivy scans for:
- ✅ OS package vulnerabilities (in Docker images)
- ✅ Application dependencies (requirements.txt, pip packages)
- ✅ Configuration issues
- ✅ Secrets (basic detection)
- ✅ Infrastructure as Code issues

**Note:** For comprehensive secret scanning, consider adding:
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [Trivy Git Leaks](https://aquasecurity.github.io/trivy/latest/docs/scanning/git-leaks/)

## Troubleshooting

### Security Upload Fails
**Issue:** "Resource not accessible by integration"

**Solutions:**
1. ✅ Already fixed with `permissions` block
2. ✅ Already added `continue-on-error: true`
3. Check if repository has Security tab enabled
   - Settings → Actions → General → Workflow permissions
   - Enable "Read and write permissions"

### No Security Results in UI
**Issue:** Scan runs but no results in Security tab

**Solutions:**
1. Check workflow permissions are set (already done)
2. Verify `security-events: write` permission (already done)
3. Check if using GitHub Team/Enterprise plan (required for Security tab)
4. For free/public repos, Security tab may have limited features

### Trivy Scan Takes Too Long
**Issue:** Security scan job is slow

**Solutions:**
1. Add severity filter:
   ```yaml
   with:
     severity: 'HIGH,CRITICAL'
   ```
2. Skip certain directories:
   ```yaml
   with:
     skip-dirs: 'tests,docs,.github'
   ```

### CodeQL v4 Still Shows Warnings
**Issue:** Still seeing deprecation warnings

**Solution:**
- Ensure all CodeQL actions use v4 (not v3)
- Check if there are other workflow files using v3

## Current Status

✅ **Permissions**: Configured (contents: read, security-events: write)
✅ **CodeQL Action**: Updated to v4
✅ **Trivy Scan**: Running successfully
✅ **SARIF Upload**: Configured with error handling
✅ **Fork PRs**: Won't fail workflow
✅ **Main Pushes**: Full security scanning with upload

## Next Steps (Optional)

### Enhance Security Scanning:
1. **Add Docker Image Scanning**
   ```yaml
   - name: Scan Docker images
     run: |
       trivy image diabetes-streamlit:latest
       trivy image diabetes-api:latest
       trivy image diabetes-mlflow:latest
   ```

2. **Add Dependency Scanning**
   ```yaml
   - name: Run dependency check
     run: |
       pip install safety
       safety check --json > safety-results.json
   ```

3. **Add Secret Scanning**
   ```yaml
   - name: Run Trivy GitLeaks
     run: trivy repo . --scanners git_leaks
   ```

4. **Enable GitHub Advanced Security** (if available)
   - Secret scanning
   - Code scanning with CodeQL
   - Dependency review

## Useful Commands

```bash
# Run Trivy locally
trivy config .
trivy fs .

# Scan Docker image locally
docker build -t diabetes-streamlit -f Dockerfile .
trivy image diabetes-streamlit

# Check for secrets locally
trivy repo . --scanners git_leaks

# View SARIF file
cat trivy-results.sarif | jq '.runs[0].results'
```

## References

- [GitHub CodeQL v4 Documentation](https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/configuring-code-scanning)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/latest/)
- [SARIF Format](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)
- [GitHub Security Permissions](https://docs.github.com/en/actions/security-guides/automatic-token-authentication#permissions-for-the-github_token)
