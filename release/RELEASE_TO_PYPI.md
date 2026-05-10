# Release ghostloop v1.0.0 to PyPI

## One-time setup (Joe to do once)

1. Create a PyPI account at https://pypi.org/account/register/.
2. Create a TestPyPI account at https://test.pypi.org/account/register/ (separate from PyPI).
3. Generate API tokens:
   - PyPI: https://pypi.org/manage/account/token/ → "Add API token", scope "Entire account" first time, then narrow to "Project: ghostloop" after the first upload.
   - TestPyPI: https://test.pypi.org/manage/account/token/ → same flow.
4. Configure `~/.pypirc` (NOT in git):

   ```ini
   [distutils]
     index-servers =
       pypi
       testpypi

   [pypi]
     username = __token__
     password = pypi-AgENdGVzdC5weXBpLm9yZ...    # your real PyPI token

   [testpypi]
     repository = https://test.pypi.org/legacy/
     username = __token__
     password = pypi-AgENdGVzdC5weXBpLm9yZ...    # your real TestPyPI token
   ```

   `chmod 600 ~/.pypirc`.

## Per-release ritual

```bash
# From the repo root:
cd /path/to/ghostloop

# 1. Make sure tests pass.
python3 -m pytest tests/

# 2. Make sure the version in pyproject.toml + ghostloop/__init__.py match
#    the upcoming release.
grep version pyproject.toml | head -1
grep __version__ ghostloop/__init__.py

# 3. Clean previous builds.
rm -rf dist/ build/ *.egg-info/

# 4. Build wheel + sdist.
python3 -m build

# 5. Verify the wheel's metadata + contents.
ls -la dist/
python3 -m twine check dist/*

# 6. Upload to TestPyPI first to verify the listing renders correctly.
python3 -m twine upload --repository testpypi dist/*

# 7. Test-install from TestPyPI in a fresh venv.
python3 -m venv /tmp/test-ghostloop && source /tmp/test-ghostloop/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ghostloop
python -c "import ghostloop; print(ghostloop.__version__)"
deactivate; rm -rf /tmp/test-ghostloop

# 8. If everything looks good, push to real PyPI.
python3 -m twine upload dist/*

# 9. Verify the listing appears at https://pypi.org/project/ghostloop/
#    and the install path works:
pip install --upgrade ghostloop
```

## Post-release

- Tweet / X-post the release link: `https://pypi.org/project/ghostloop/`.
- Update the README install line to recommend `pip install ghostloop` for users who don't need a local clone.
- The GitHub release tag (v1.0.0) is already pushed; a PyPI release matches it 1:1.

## Reverting a bad release

You CANNOT delete a PyPI release once published — only yank it (`twine` doesn't support this; use the web UI). Yanking hides the version from new installs but lets people who already pinned it keep working. Bump the patch version (1.0.1) and re-release with the fix instead.

## Why the dual-package install pattern matters

ghostloop's "core has zero deps" promise breaks if `pip install ghostloop` pulls in mujoco / pybullet / fastapi by default. The `[mujoco]` / `[pybullet]` / `[gym]` / `[ros2]` / `[mcp]` / `[otel]` / `[dashboard]` extras keep the core install lean; users opt into what they need:

```bash
pip install ghostloop                  # core only — works with MockBackend
pip install ghostloop[mcp]             # add MCP server
pip install ghostloop[mujoco,mcp]      # MuJoCo + MCP — common combo
pip install "ghostloop[mujoco,mcp,dashboard,otel]"   # everything except ROS 2
```

ROS 2's `rclpy` is system-installed so the `[ros2]` extra is documentation-only (it doesn't pull anything via pip).
