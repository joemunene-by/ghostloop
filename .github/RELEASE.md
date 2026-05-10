# Releasing ghostloop

Every release is one command.

## Cut a release

```bash
# 1. Bump the version. Edit BOTH files to match — the workflow rejects
#    mismatches.
$EDITOR pyproject.toml         # version = "X.Y.Z"
$EDITOR ghostloop/__init__.py  # __version__ = "X.Y.Z"

# 2. Commit + push.
git add pyproject.toml ghostloop/__init__.py CHANGELOG.md
git commit -m "release: vX.Y.Z — <one-line summary>"
git push

# 3. Tag + push the tag. The publish-pypi.yml workflow fires on tag
#    push. If you skip the tag, nothing publishes.
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z
```

Watch progress at https://github.com/joemunene-by/ghostloop/actions.

## Re-release after a workflow file change

The v1.0.0 git tag predates the workflow files. The first publish
after merging the workflows must either:

- **Push a fresh tag** (e.g. `v1.0.1`) which is the cleanest move:
  ```bash
  $EDITOR pyproject.toml ghostloop/__init__.py   # bump to 1.0.1
  $EDITOR CHANGELOG.md                           # add 1.0.1 entry
  git add -u && git commit -m "release: v1.0.1 — workflow plumbing"
  git tag -a v1.0.1 -m "v1.0.1 — first PyPI release"
  git push origin main v1.0.1
  ```

- **OR delete + re-push v1.0.0** (works but rewrites history; only do
  if no one has pulled v1.0.0 yet):
  ```bash
  git tag -d v1.0.0
  git push origin :refs/tags/v1.0.0
  git tag -a v1.0.0 -m "v1.0.0 — production"
  git push origin v1.0.0
  ```

Recommended: push v1.0.1 with the workflow plumbing as the changelog
note.

## What the publish workflow does

`.github/workflows/publish-pypi.yml`:

1. Verifies the git tag matches `pyproject.toml::version` AND
   `ghostloop/__init__.py::__version__`. Rejects on mismatch.
2. Runs the full test suite. Rejects on failure.
3. Builds sdist + wheel via `python -m build`.
4. Runs `twine check` on both artifacts.
5. Publishes via PyPI Trusted Publishing (OIDC, no API token in repo).
6. Uploads the wheel + sdist as a workflow artifact for offline grab.

If any step fails, nothing publishes. PyPI never sees a half-baked
upload.

## Hugging Face Space

`.github/workflows/publish-hf-space.yml`:

Triggers on any push to `main` that touches `spaces/ghostloop-demo/**`.
Uploads the directory to `huggingface.co/spaces/<HF_USERNAME>/ghostloop-demo`.
Idempotent — reruns are safe; HF builds the Gradio app on the HF side.

Requires `HF_TOKEN` repository secret. One-time setup; covered in
LAUNCH_CHECKLIST.md.

## CI

`.github/workflows/ci.yml` runs on every push to main and every PR:

- pytest on Python 3.10 / 3.11 / 3.12 / 3.13
- sdist + wheel build smoke
- `twine check`

Catches regressions before a release tag triggers `publish-pypi.yml`.

## Versioning policy

ghostloop follows semver:

- **Patch** (1.0.0 → 1.0.1): bug fixes only, no API changes.
- **Minor** (1.0.0 → 1.1.0): new features, additive only. No
  breaking changes to the public API.
- **Major** (1.0.0 → 2.0.0): breaking changes. Document the
  migration path in CHANGELOG.md.

Internal modules (anything not exported from `ghostloop/__init__.py`)
are not part of the public API and may change without a major bump.
The runtime / Backend / PolicyGate / PrimitiveRegistry / Trace / RobotProfile
shapes ARE part of the public API.

## Yanking a bad release

PyPI doesn't let you delete a release; only yank. To yank vX.Y.Z:

1. https://pypi.org/manage/project/ghostloop/release/X.Y.Z/ → "Yank".
2. Bump to X.Y.Z+1 with the fix and re-publish via the normal flow.
3. Note the yank in CHANGELOG.md so users know not to pin to it.
