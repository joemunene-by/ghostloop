# Submit the ghostloop preprint to arXiv

## One-time setup

1. Create an arXiv account: https://arxiv.org/user/register
2. Get endorsement: arXiv requires an endorser for a first submission
   in a category you haven't published in. For cs.RO (Robotics) or
   cs.LG (Machine Learning), email a researcher in the field with a
   2-line summary and the abstract — most are willing to endorse for
   a clear, well-formed paper. The OWASP CheatSheetSeries contribution
   already on your CV makes this easier; mention it.
   Alternative: go through Anthropic / DeepMind / a robotics lab
   contact who can endorse.
3. arXiv account needs an ORCID — sign up at https://orcid.org if you
   don't have one. Free, takes 60 seconds.

## Build the PDF locally first

```bash
cd /path/to/ghostloop/paper

# Install LaTeX if you don't have it (Mac):
#   brew install --cask mactex-no-gui
# (Linux):
#   apt install texlive-full   # heavy, ~5GB
# Or use Overleaf for a no-install path.

# Build:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex     # twice more for cross-refs to settle

# Inspect:
open main.pdf         # macOS
# or xdg-open main.pdf on Linux
```

You should get a 6-8 page PDF. If the bibliography or cross-refs
look off, run the latex sequence again — it's normal to need 3-4
passes the first time.

## Decide on category

Primary: **cs.RO (Robotics)**.
Cross-list: **cs.LG (Machine Learning)** and **cs.SE (Software
Engineering)**.

Workshop targets if you want a peer-reviewed venue too:
- ICRA 2026 RoboTrust workshop (deadline TBA)
- NeurIPS 2026 SafeRL workshop
- CoRL 2026 main track or workshops

The arXiv preprint stands alone; workshop submission is a separate
track and you can do both.

## Submit

1. Sign in at https://arxiv.org/submit
2. Click "Start a new submission"
3. Upload `main.tex` + `refs.bib`. arXiv compiles on its end; if
   your local build worked, theirs will too.
4. Fields to fill:
   - **Title:** ghostloop: A Post-Hoc Analysis Layer for Embodied Agents
   - **Authors:** Joe Munene
   - **Abstract:** copy from `main.tex`
   - **Comments:** "8 pages, MIT-licensed library at github.com/joemunene-by/ghostloop"
   - **MSC class / ACM class:** leave blank
   - **License:** CC BY 4.0 (most permissive; let people remix the figures)
5. Submit. arXiv usually takes 1-3 days to announce; the URL becomes
   `https://arxiv.org/abs/2605.NNNNN` when live.

## After it's live

1. Update `README.md` — change "_[link to be added once preprint is up]_"
   to the real arXiv URL. Same for the HuggingFace Space description
   and the launch blog post.
2. Tweet the link with the demo video — preprints rarely get traction
   without a hook, so the video does most of the work.
3. Email the endorser thanking them; offer reciprocal endorsement
   when they ask.
4. Update your CV / portfolio with the citation:

   ```
   Munene, J. (2026). ghostloop: A Post-Hoc Analysis Layer for
   Embodied Agents. arXiv:2605.NNNNN [cs.RO].
   ```

## Honest read

The paper as drafted is "workshop-grade" — clean writeup of
concrete engineering with novel pieces, but no head-to-head
experimental numbers vs OpenVLA / π0 yet (the limitations section
is honest about this). For a top-tier conference (ICRA / NeurIPS
main track) you'd want the OpenVLA reproduction running through
the bench harness and showing actual transfer-gap statistics.
That's a follow-up paper, not this one.

For arXiv + a workshop, this draft is the right shape: clear
contribution, working code, honest limitations.
