# ghostloop v1.0 launch checklist

The work is done. The artifacts below are ready; what remains is the
credentialed steps (uploading, submitting, posting). Each item links
to the file with full instructions.

Order matters — earlier items unblock later ones.

## Day 1 — distribution foundations (now automated)

PyPI Trusted Publishing + HuggingFace deploy are wired through GitHub
Actions. Once the one-time setup below is done, every release happens
by pushing a git tag.

- [ ] **One-time PyPI Trusted Publishing setup** (Joe, ~5 min)
  1. Visit https://pypi.org/manage/account/publishing/ → "Add a new pending publisher".
  2. Project name: `ghostloop`. Owner: `joemunene-by`. Repo: `ghostloop`.
     Workflow: `publish-pypi.yml`. Environment name: `pypi`.
  3. Save. PyPI is now ready to accept OIDC pushes from `.github/workflows/publish-pypi.yml`.
  4. (You said you've already wired this — verify the form fields match
     the workflow file exactly. Mismatched workflow filename or env name
     is the most common cause of the OIDC handshake failing.)

- [ ] **One-time HF token paste** (Joe, ~2 min)
  1. Generate a Write-access token at https://huggingface.co/settings/tokens.
  2. In the ghostloop repo on GitHub: Settings → Secrets and variables → Actions
     → New repository secret. Name: `HF_TOKEN`. Value: paste.
  3. (Optional) Add a repo *variable* `HF_USERNAME` if your HF account
     isn't `joemunene-by`. Same screen, "Variables" tab.

- [ ] **Publish v1.0.0 to PyPI** (one command)
  ```bash
  git tag -a v1.0.0 -m "v1.0.0 — production release"
  git push origin v1.0.0
  ```
  The `publish-pypi.yml` workflow runs: version-tag-match check, full
  test suite, sdist + wheel build, `twine check`, OIDC-authenticated
  PyPI upload. Status visible at github.com/joemunene-by/ghostloop/actions.
  (We already pushed the v1.0.0 tag earlier this session — that tag
  predates the workflow file; for the first publish, push a temporary
  v1.0.1 tag with the same artifacts after fixing any version mismatch,
  OR delete + re-tag v1.0.0 after merging this commit.)

- [ ] **Publish the HuggingFace Space** (no command needed)
  The first push to `main` that touches `spaces/ghostloop-demo/**` (or
  any change to the workflow file itself) triggers
  `publish-hf-space.yml`. The workflow creates the Space if it doesn't
  exist and uploads the directory. After this commit lands, the Space
  is live at `huggingface.co/spaces/Ghostgim/ghostloop-demo`
  within ~3 minutes (HF build time on the free tier).

## Day 2 — the demo

- [ ] **Record the demo video** ([marketing/DEMO_VIDEO_SCRIPT.md](marketing/DEMO_VIDEO_SCRIPT.md))
  Five-cut storyboard, two pre-written prompts, shot list, posting
  format guide per platform. Goal: 60 seconds vertical for X /
  LinkedIn, 16:9 horizontal for the README hero + YouTube.

  Setup needed: ghostloop installed locally with `[mcp,mujoco]`,
  Claude Desktop wired to the MCP server, MuJoCo viewer with a
  Franka model loaded. Recording tools: QuickTime / OBS / Cleanshot.

## Day 3 — visibility

- [ ] **Submit MCP marketplace listings** ([marketing/MARKETPLACE_LISTINGS.md](marketing/MARKETPLACE_LISTINGS.md))
  Eight channels in priority order:
  1. PyPI (already done by Day 1)
  2. HuggingFace Space (already done by Day 1)
  3. modelcontextprotocol.io community list — one PR
  4. Smithery.ai — needs `smithery.yaml` at repo root
  5. Awesome-list PRs (5 each):
     - awesome-mcp-servers
     - awesome-robotics-libraries
     - awesome-llm-tooluse
     - awesome-embodied-ai
  6. Tweet the demo video tagging `@AnthropicAI`, `@deepmind`, `@MuJoCoSim`
  7. LinkedIn post — long form, lead with the missing-middle thesis
  8. Show HN post — wait until video has 10k views to maximise lift

- [ ] **Publish the launch blog post** ([blog/2026-05-10_v1_launch.md](blog/2026-05-10_v1_launch.md))
  ~1500 words, ready to paste into a Substack / personal blog /
  Complex Developers site. Includes the eleven-release recap, the
  novel-vs-standard breakdown, the honest "what was hardest" section.

## Week 2 — the deep play

- [ ] **Submit the arXiv preprint** ([paper/SUBMIT_TO_ARXIV.md](paper/SUBMIT_TO_ARXIV.md))
  LaTeX + bibliography written, 6-8 pages, MIT contribution
  positioned correctly vs LangChain / AutoGen / MCP / OpenVLA / π0.
  Need:
  1. arXiv endorsement for cs.RO (one researcher's email + 2-line
     summary). The OWASP CheatSheetSeries credit + the GhostLM /
     ghostloop bodies of work make this easy.
  2. ORCID (60 seconds at orcid.org).
  3. Build the PDF (Overleaf or local pdflatex) and submit at
     arxiv.org/submit.

  Once the arXiv URL exists, update README + Space description +
  blog post + LinkedIn post to reference it.

## Week 3+ — sustained reputation

- [ ] **Run a real OpenVLA reproduction** through `VLABenchmarkSuite`.
  ghostloop's bench harness already ships pre-populated with
  published baselines (`pick_place_widowx`, `manipulation_bridge`,
  `reach_target`). Running an actual OpenVLA-7B checkpoint, recording
  the violation-rate gap under your safety pipeline, publishing the
  numbers as a follow-up arXiv submission — that's a top-tier
  conference paper.

- [ ] **Weekly publishing rhythm** — one post per release / milestone.
  The work is the bottleneck no longer; the narration is.

## What lands where

| Audience | Artifact | URL |
|---|---|---|
| ML researchers | arXiv preprint | `arxiv.org/abs/2605.NNNNN` |
| Python users | PyPI listing | `pypi.org/project/ghostloop` |
| Curious clickers | HuggingFace Space | `huggingface.co/spaces/Ghostgim/ghostloop-demo` |
| MCP power users | Smithery.ai listing | `smithery.ai/server/ghostloop` |
| robotics generalists | Awesome-list entries | five PRs |
| Twitter/X | demo video | platform-native |
| LinkedIn | demo video + long-form post | platform-native |
| HN crowd | Show HN post (week 2) | `news.ycombinator.com` |
| Personal brand | launch blog post | personal site |

## What I'm NOT doing here

- **The actual recording.** I can't operate a camera; you do that.
- **The actual submissions.** I don't have your PyPI / HF / arXiv
  credentials and you wouldn't want me to.
- **The actual posting.** Voice + cadence is yours.
- **The OpenVLA reproduction.** That requires GPU time and a real
  OpenVLA checkpoint — separate project, separate sprint.

Everything *gating* those steps — drafts, scripts, manifests,
configurations, tests, validation — is shipped.
