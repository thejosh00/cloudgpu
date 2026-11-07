#!/bin/sh
# The cloudgpu-build run's check (workflows/build-flow.md): a wake only when there's something for it
# to do. No model, no tokens. Exit 0 = nothing new, exit 1 = wake it, anything else = a wake too.
queue="${CLOUDGPU_QUEUE:-$HOME/workspace/cloudgpu-queue}"
dir="${OTTO_HOME:-$HOME/.otto}/runs/$OTTO_RUN_ID"

[ -d "$queue" ] || { echo "no queue at $queue"; exit 2; }

# What happens to its changes reaches it as a one-off note, and a check stands in front of the wake
# that would deliver one.
pending=$(/usr/bin/jq '[(.notes // [])[] | select(.standing != true)] | length' "$dir/run.json") || {
	echo "cannot read run.json"; exit 2
}
if [ "$pending" != "0" ]; then
	echo "a note is waiting"; exit 1
fi

# One word per item: its front-matter status, or "unreadable".
statuses=$(for f in "$queue"/*.md; do
	[ -f "$f" ] || continue
	awk 'NR==1 && $0!="---" {exit} NR>1 && $0=="---" {exit} /^status:/ {s=$2} END {print (s==""?"unreadable":s)}' "$f"
done) || { echo "could not read the queue"; exit 2; }

has() { printf '%s\n' "$statuses" | grep -qx "$1"; }

if has unreadable; then echo "an item has no status"; exit 2; fi
if has landed; then echo "an item landed and wants verifying"; exit 1; fi
if has building; then echo "an item is building"; exit 1; fi
if has asked; then echo "waiting on the person's answer"; exit 0; fi
if has handed-back; then echo "a change waits on review"; exit 0; fi
if has ready; then echo "an item is ready to start"; exit 1; fi
echo "nothing to build"
exit 0
