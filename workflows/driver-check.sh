#!/bin/sh
# The cloudgpu-driver run's check (workflows/drive-flow.md): a wake only when there's something for it
# to do. No model, no tokens. Exit 0 = nothing new, exit 1 = wake it, anything else = a wake too.
#
# artifacts/waiting-on is the wake's cursor, and this only reads it:
#   an item    the plan item it handed off; wake when that item is verified or closed
#   hold       waiting on a person; a note or an answer wakes the run
#   missing    it isn't waiting on anything, so it has work to do
queue="${CLOUDGPU_QUEUE:-$HOME/workspace/cloudgpu-queue}"
dir="${OTTO_HOME:-$HOME/.otto}/runs/$OTTO_RUN_ID"

# What happens to its own changes reaches it as a one-off note, and a check stands in front of the
# wake that would deliver one.
pending=$(/usr/bin/jq '[(.notes // [])[] | select(.standing != true)] | length' "$dir/run.json") || {
	echo "cannot read run.json"; exit 2
}
if [ "$pending" != "0" ]; then
	echo "a note is waiting"; exit 1
fi

if [ ! -f "$dir/artifacts/waiting-on" ]; then
	echo "not waiting on anything"; exit 1
fi
waiting=$(tr -d '[:space:]' < "$dir/artifacts/waiting-on")

case "$waiting" in
hold) echo "waiting on a person"; exit 0 ;;
'' | *[!0-9a-z.-]*) echo "can't read what it is waiting on: $waiting"; exit 2 ;;
esac

item="$queue/$waiting.md"
[ -f "$item" ] || { echo "item $waiting isn't in the queue"; exit 2; }
status=$(awk 'NR==1 && $0!="---" {exit} NR>1 && $0=="---" {exit} /^status:/ {s=$2} END {print s}' "$item")

case "$status" in
verified | closed) echo "item $waiting is $status"; exit 1 ;;
'') echo "item $waiting has no status"; exit 2 ;;
*) echo "item $waiting is $status"; exit 0 ;;
esac
