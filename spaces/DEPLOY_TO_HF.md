# Deploy ghostloop demo to HuggingFace Spaces

## One-time setup

1. Sign in / create account at https://huggingface.co.
2. Generate a write-access token at https://huggingface.co/settings/tokens
   ("New token" → role: Write).
3. Install the CLI: `pip install --user huggingface_hub`.
4. Login locally: `huggingface-cli login` and paste the token.

## First deploy

```bash
# 1. Create the Space (one-time).
huggingface-cli repo create ghostloop-demo --type space \
    --space_sdk gradio

# 2. Clone the empty Space repo.
git clone https://huggingface.co/spaces/joemunene-by/ghostloop-demo \
    /tmp/hf-ghostloop-demo
cd /tmp/hf-ghostloop-demo

# 3. Copy the Space files in.
cp /path/to/ghostloop/spaces/ghostloop-demo/{app.py,requirements.txt,README.md} .

# 4. Push.
git add app.py requirements.txt README.md
git commit -m "Initial ghostloop demo Space"
git push
```

The Space builds automatically (~3 minutes on free CPU tier). When
finished, it's live at:

  https://huggingface.co/spaces/joemunene-by/ghostloop-demo

## Update flow (after PyPI release)

```bash
cd /tmp/hf-ghostloop-demo
git pull
# Edit app.py / requirements.txt as needed
git add -u
git commit -m "Update demo for ghostloop X.Y.Z"
git push
```

## Tips

- **Tier:** the free CPU tier is enough — MockBackend is fast.
  Only upgrade to A10G / T4 if you wire MuJoCo rendering in a future
  iteration.
- **Sleep:** Spaces sleep after 48h idle on the free tier. The first
  visitor pays a 30-second cold-start penalty. For the demo-traffic
  spike around launch, pin "always on" in the Space settings (free
  tier allows it for popular Spaces).
- **Embedding:** the Space iframe-embeds cleanly into the portfolio
  site (`https://huggingface.co/spaces/joemunene-by/ghostloop-demo?embed=true`)
  or any blog post.

## Why this is the highest-conversion artifact

GitHub repo → 30 seconds clone + install + read.
PyPI listing → 1 minute install + open Python REPL.
Space → 5 seconds, click a link, see Claude reasoning about a robot.

Most reputation conversions happen at the lowest-friction surface.
The Space is that surface.
