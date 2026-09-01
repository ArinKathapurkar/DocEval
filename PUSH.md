# Pushing these repos

Both repos are committed locally with no remote configured. `gh` is currently
authenticated as **ArinKath-Intuitive** (an employer account) while SSH authenticates as
**ArinKathapurkar** (personal), so `gh repo create` would put this portfolio under the
wrong account.

Fix, once:

```bash
gh auth login          # authenticate as ArinKathapurkar
gh auth status         # confirm ArinKathapurkar is the ACTIVE github.com account
```

Then, from each repo directory:

```bash
gh repo create ArinKathapurkar/AssetSearch --public --source=. --remote=origin --push
gh repo create ArinKathapurkar/DocEval     --public --source=. --remote=origin --push
```

Check the owner in the returned URL before continuing. If a repo lands under the employer
account, delete and recreate rather than transferring.
