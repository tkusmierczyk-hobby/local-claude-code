#!/usr/bin/env bash
# pick-cpus.sh — the CPUs llama.cpp's compute threads should be pinned to.
#
# Prints "<threads> <hex mask> <cpu list>": one logical CPU per performance
# core, among the CPUs this process may run on. Prints nothing and exits 1 when
# the topology cannot be read, so the caller keeps its unpinned default.
#
# Why: during decode the CPU multiplies the experts with all threads in
# lockstep, so the slowest thread sets the pace. Efficiency cores and second
# hyperthreads cost more than they add: on the reference machine, 8 threads
# pinned to the 8 P-cores decode ~13% faster than 16 unpinned (docs/TUNING.md).
#
#   bin/pick-cpus.sh            # e.g. "8 5555 0,2,4,6,8,10,12,14"
#
# PICK_CPUS_SYSFS and PICK_CPUS_ALLOWED stand in for /sys and for this
# process's allowed-CPU list; test/pick_cpus_test.sh uses them to fake machines.
set -uo pipefail
SYS="${PICK_CPUS_SYSFS:-/sys}"

# "0-3,8,10-11" -> one number per line
expand() {
  tr ',' '\n' <<< "$1" | awk -F- '
    NF == 1 && $1 ~ /^[0-9]+$/                 { print $1 }
    NF == 2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { for (i = $1; i <= $2; i++) print i }'
}
# CPUs present in both lists, as one comma-separated list
intersect() { comm -12 <(expand "$1" | sort) <(expand "$2" | sort) | sort -n | paste -sd, -; }

allowed="${PICK_CPUS_ALLOWED:-$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null)}"
online="$(cat "$SYS/devices/system/cpu/online" 2>/dev/null)"
[[ -n "$allowed" ]] || allowed="$online"
[[ -n "$online" ]] && allowed="$(intersect "$allowed" "$online")"
[[ -n "$allowed" ]] || exit 1

# Intel hybrid CPUs list their performance cores here; elsewhere every core is
# one. A process confined to efficiency cores keeps the cores it was given.
candidates="$allowed"
if [[ -r "$SYS/devices/cpu_core/cpus" ]]; then
  perf="$(intersect "$allowed" "$(cat "$SYS/devices/cpu_core/cpus")")"
  [[ -n "$perf" ]] && candidates="$perf"
fi

# One logical CPU per physical core: skip the siblings of a core already taken.
picked=() seen=" "
for cpu in $(expand "$candidates"); do
  topo="$SYS/devices/system/cpu/cpu$cpu/topology"
  sib="$(cat "$topo/core_cpus_list" 2>/dev/null || cat "$topo/thread_siblings_list" 2>/dev/null)"
  [[ -n "$sib" ]] || exit 1
  [[ "$seen" == *" $sib "* ]] && continue
  seen+="$sib "; picked+=("$cpu")
done
# llama.cpp reads at most 512 CPUs from a mask; beyond that, stay unpinned.
[[ ${#picked[@]} -gt 0 && "${picked[-1]}" -lt 512 ]] || exit 1

# Hex, most significant digit first, as --cpu-mask expects.
mask="$(printf '%s\n' "${picked[@]}" | awk '
  BEGIN { top = 0 }
  { d[int($1 / 4)] += 2 ^ ($1 % 4); if (int($1 / 4) > top) top = int($1 / 4) }
  END { for (i = top; i >= 0; i--) printf "%x", d[i] + 0 }')"
echo "${#picked[@]} $mask $(IFS=,; echo "${picked[*]}")"
