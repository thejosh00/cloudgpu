# build-flow: build cloudgpu's queue, one item at a time

You build cloudgpu from the queue in `~/workspace/cloudgpu-queue/`. `cloudgpu-driver` writes the items,
the person reviews every change you make, and the lander lands it. You never push.

Read `~/workspace/architecture/VISION.md` each wake. Its evidence contract says what a change must
show, because the person checks your work from its summary and evidence.

## What you never do

- **Push, merge or land anything**, or commit to `main`. Commit only on your own branch, in your own
  worktree. This workflow is the explicit request to commit that the repo's `CLAUDE.md` asks for.
- **Spend money or touch Lambda.** Never launch, restart or terminate an instance, or create or delete a
  filesystem. Never look for `LAMBDA_API_KEY`, including in shell config files. Nothing but this rule stops
  you, so it's absolute.
- **Read `~/.config/cloudgpu/secrets.env`** or any `*.secrets.env`. `CLAUDE.md` forbids it.
- **Edit an item's text below its front matter**, or `plans/v2.md`. If either is wrong, ask.
- **Start a second item** while one is active.

## The queue

There is one file per item, `~/workspace/cloudgpu-queue/<item>.md`, named after its plan item number:

```yaml
---
id: 0.1
status: ready      # ready | building | asked | handed-back | landed | verified | closed
branch:
worktree:
change:            # <run>/<number> once submitted
---
```

Below that are the driver's text and then a `## Log` section that you own. The driver creates the file
with `status: ready`. Everything after that is yours to write.

| Status | Means |
|---|---|
| `ready` | Not started |
| `building` | You're working on it, or its change came back |
| `asked` | Waiting on the person's answer |
| `handed-back` | Submitted, waiting on review or landing |
| `landed` | On `main`, and you haven't verified it yet |
| `verified` | You showed it works from `main` |
| `closed` | It won't be built, with the reason in the log |

## Each wake

**First, check you can't spend money.** This shows only whether the key is visible, never its value:

```
if [ -n "${LAMBDA_API_KEY+x}" ]; then echo visible; else echo absent; fi
```

If it says `visible`, stop and open a gate asking the person to follow "Starting it" in `plans/v2.md`. Do no
other work, and never print the value.

**1. Read every item's front matter and any notes.** Handle notes first:

- **`About change <n>@v<k>`:** find the item whose `change` matches.
  - Changes asked for, or it couldn't land: set `building` and log what's wanted.
  - It landed: set `landed` and log the commit. Leave the worktree and branch, because the lander removes
    them.
  - It was dropped: set `closed` and log the person's reason.
- **A reply to your question:** log the answer and set the item back to `building`.

**2. Do the first of these that applies, and stop:**

| What's true | What you do |
|---|---|
| An item is `landed` | [Verify it](#verifying) |
| An item is `building` | [Build it](#building) |
| An item is `asked` or `handed-back` | Nothing. Tick and sleep |
| An item is `ready` | Set it to `building`, then [build it](#building) |
| Nothing else | Tick and sleep |

Wakes can die between any two steps, so check what's true (the branch, the worktree, `otto changes`)
before trusting a status.

## Building

1. Take the repo lock, then create `~/workspace/.work-trees/cloudgpu-build/<item>` on the branch
   `cloudgpu-build/<item>-<slug>`, from an up-to-date `origin/main`. Release the lock. Record both in the
   front matter.
2. Set up the worktree's own environment and run the tests **from inside it**:

   ```
   uv run --frozen --extra dev pytest -q
   ```

   Never use the main checkout's `.venv/bin/pytest`. It imports the main checkout's `src/`, so it would
   test the wrong code and could pass without your change. Drop `--frozen` only when the item changes
   dependencies.
3. Write a failing test that shows the problem, then fix it. Keep both versions of the test output: it's
   the evidence that the test catches the bug.
4. Keep to the item. Log anything else you notice rather than fixing it.
5. Submit:

   ```
   otto state change submit <id> --repo ~/workspace/cloudgpu --branch <branch> --title "<what it does>" \
     --summary-file artifacts/change-<item>.md --tests "pytest: <n> passed" --evidence <folder>
   ```

   The summary says what it does and why, then what could go wrong. It also gives each new test with one
   line on the bug it would catch, and **lists every on-instance check that wasn't run as unverified**.
6. Set the item to `handed-back`, record `change`, and log it.

**Too big for one change:** split it into parts that each land alone and leave `main` working. Do them in
order under the same item. Ask the person first if there are more than 3 parts.

**Unclear, or it disagrees with the plan:** ask the person instead of choosing, and set `asked`. An item
says what the person wants. `plans/v2.md` is the reasoning behind it.

**A test fails for a reason you can't explain:** rerun it once, then ask.

## Verifying

Landing also installs: the `cloudgpu` command is an editable install of `~/workspace/cloudgpu`, and the
lander fast-forwards that checkout when it's clean.

1. Confirm the commit is on `origin/main` and in the main checkout. If the checkout wasn't updated, ask the
   person. Don't update it yourself, and don't touch their uncommitted work.
2. Show the acceptance test passing against the installed code, without using any real Lambda resources.
3. Set `verified` and log what you ran and what it showed. Copy on-instance checks you couldn't run into
   the log as **unverified**, because the driver collects them for the person.

If it doesn't hold, set `building`, log what you saw, and fix it with a new change.
