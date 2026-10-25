#!/usr/bin/env bash
# pick_cpus_test.sh — bin/pick-cpus.sh against faked CPU layouts.
#
# The pinning only helps if it picks one CPU per fast core on *every* machine,
# and a wrong pick does not crash anything, it just runs slower — so these
# layouts are checked here rather than discovered on someone else's laptop.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PICK="$HERE/bin/pick-cpus.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

# fake <name> <online> <perf cores or ""> <cpu:siblings>...
fake() {
  local root="$TMP/$1" spec; mkdir -p "$root/devices/system/cpu"
  echo "$2" > "$root/devices/system/cpu/online"
  [[ -n "$3" ]] && { mkdir -p "$root/devices/cpu_core"; echo "$3" > "$root/devices/cpu_core/cpus"; }
  shift 3
  for spec in "$@"; do
    mkdir -p "$root/devices/system/cpu/cpu${spec%%:*}/topology"
    echo "${spec#*:}" > "$root/devices/system/cpu/cpu${spec%%:*}/topology/thread_siblings_list"
  done
}
# expect <description> <machine> <allowed> <expected output, "" for none>
expect() {
  local got; got="$(PICK_CPUS_SYSFS="$TMP/$2" PICK_CPUS_ALLOWED="$3" "$PICK" 2>&1 | cut -d' ' -f1,2)"
  if [[ "$got" == "$4" ]]; then echo "  ok    $1"; PASS=$((PASS+1))
  else echo "  FAIL  $1: expected '$4', got '$got'"; FAIL=$((FAIL+1)); fi
}

# i9-13900HX: P-cores 0-15 as adjacent hyperthread pairs, E-cores 16-31
specs=(); for c in $(seq 0 15); do specs+=("$c:$((c/2*2))-$((c/2*2+1))"); done
for c in $(seq 16 31); do specs+=("$c:$c"); done
fake hybrid8 0-31 0-15 "${specs[@]}"
expect "8P+16E hybrid: one CPU per P-core"         hybrid8 0-31  "8 5555"
expect "confined to CPUs 0-7: four P-cores"        hybrid8 0-7   "4 55"
expect "confined to E-cores: keeps them"           hybrid8 16-19 "4 f0000"

# 6P+8E hybrid (i7-13700H class): the fixed mask 5555 would land on E-cores
specs=(); for c in $(seq 0 11); do specs+=("$c:$((c/2*2))-$((c/2*2+1))"); done
for c in $(seq 12 19); do specs+=("$c:$c"); done
fake hybrid6 0-19 0-11 "${specs[@]}"
expect "6P+8E hybrid"                              hybrid6 0-19  "6 555"

# AMD-style numbering: CPU n and n+8 share a core
specs=(); for c in $(seq 0 15); do specs+=("$c:$((c%8)),$((c%8+8))"); done
fake amd8 0-15 "" "${specs[@]}"
expect "8 cores, siblings numbered n and n+8"      amd8 0-15     "8 ff"

fake nosmt 0-3 "" 0:0 1:1 2:2 3:3
expect "4 cores without hyperthreading"            nosmt 0-3     "4 f"

# 128 CPUs, siblings n and n+64
specs=(); for c in $(seq 0 127); do specs+=("$c:$((c%64)),$((c%64+64))"); done
fake big 0-127 "" "${specs[@]}"
expect "64 cores / 128 threads"                    big 0-127     "64 ffffffffffffffff"

# CPUs 8-15 offline: no topology for them, and they must not be picked
specs=(); for c in $(seq 0 7); do specs+=("$c:$((c/2*2))-$((c/2*2+1))"); done
fake offline 0-7 "" "${specs[@]}"
expect "offline CPUs are skipped"                  offline 0-15  "4 55"

fake notopo 0-3 ""
expect "no topology: no pinning"                   notopo 0-3    ""

echo "  --    this machine: $("$PICK" || echo 'no pinning')"
echo "pick-cpus: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
