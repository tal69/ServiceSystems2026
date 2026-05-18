#!/usr/bin/env python3
"""
Feasibility and cost checker for the Service-Systems 2026 term project:
"Shuttle-Bus Vehicle and Driver Scheduling".

Usage:
    python check_solution.py  instance.json  solution.json

Exits with code 0 if the solution is feasible (prints OK and cost breakdown),
otherwise prints a list of violations and exits with code 1.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Loading

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Time-dependent travel time

class TravelTable:
    """Step-function travel-time table keyed by departure minute."""

    LEG_ORDER = ["A-B", "B-A", "D-A", "A-D", "D-B", "B-D"]

    def __init__(self, spec: dict):
        self.legs = spec["legs"]
        if self.legs != self.LEG_ORDER:
            raise ValueError(
                f"travel_time.legs must be exactly {self.LEG_ORDER}, got {self.legs}"
            )
        self.buckets = sorted(spec["buckets"], key=lambda b: b["from_min"])
        # Sanity: contiguous, non-empty, non-overlapping
        prev_to = None
        for b in self.buckets:
            if b["from_min"] >= b["to_min"]:
                raise ValueError(f"empty bucket {b}")
            if prev_to is not None and b["from_min"] != prev_to:
                raise ValueError(
                    f"travel-time buckets must be contiguous; gap at {prev_to}->{b['from_min']}"
                )
            if len(b["minutes"]) != len(self.legs):
                raise ValueError(f"bucket {b} has wrong number of leg times")
            prev_to = b["to_min"]

    def lookup(self, frm: str, to: str, start_min: int) -> int:
        key = f"{frm}-{to}"
        if key not in self.legs:
            raise ValueError(f"travel-time leg {key} not tabulated")
        idx = self.legs.index(key)
        for b in self.buckets:
            if b["from_min"] <= start_min < b["to_min"]:
                return int(b["minutes"][idx])
        raise ValueError(f"departure time {start_min} outside travel-time table")


# ---------------------------------------------------------------------------
# Checker

@dataclass
class Violation:
    duty: str | None
    code: str
    message: str

    def __str__(self) -> str:
        tag = f"[{self.duty}] " if self.duty else ""
        return f"  - {tag}{self.code}: {self.message}"


class Checker:
    def __init__(self, instance: dict, solution: dict):
        self.inst = instance
        self.sol = solution
        self.params = instance["parameters"]
        self.trips = {t["trip_id"]: t for t in instance["trips"]}
        self.travel = TravelTable(instance["travel_time"])
        self.violations: list[Violation] = []

    # ---- helpers

    def add(self, duty: str | None, code: str, msg: str) -> None:
        self.violations.append(Violation(duty, code, msg))

    # ---- checks

    def check(self) -> tuple[bool, dict]:
        self._check_top_level()
        seen_trips: dict[int, str] = {}
        vehicle_shifts: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        terminal_timeline: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        total_deadhead_min = 0
        total_driver_cost = 0.0
        vehicles_used: set[str] = set()

        duties = self.sol.get("duties", [])
        if not duties:
            self.add(None, "EMPTY", "solution has no duties")

        for duty in duties:
            dh, drv = self._check_duty(duty, seen_trips, vehicle_shifts,
                                        terminal_timeline, vehicles_used)
            total_deadhead_min += dh
            total_driver_cost  += drv

        # coverage
        required = set(self.trips.keys())
        covered  = set(seen_trips.keys())
        missing  = required - covered
        for t in sorted(missing):
            self.add(None, "C1_COVERAGE",
                     f"trip {t} is not covered by any duty")

        # vehicle continuity
        for vid, shifts in vehicle_shifts.items():
            shifts.sort()
            for i in range(len(shifts) - 1):
                s1, e1, k1 = shifts[i]
                s2, e2, k2 = shifts[i + 1]
                if s2 < e1:
                    self.add(None, "C6_VEHICLE_OVERLAP",
                             f"vehicle {vid}: duties {k1} [{s1},{e1}) "
                             f"and {k2} [{s2},{e2}) overlap")

        # terminal capacity (sweep line)
        cap = int(self.params["terminal_capacity"])
        for term in ("A", "B"):
            events: list[tuple[int, int, str]] = []   # (time, +1/-1, vehicle)
            for s, e, vid in terminal_timeline[term]:
                if s < e:
                    events.append((s, +1, vid))
                    events.append((e, -1, vid))
            events.sort(key=lambda x: (x[0], x[1]))  # ends before starts at same minute
            current: set[str] = set()
            for t, delta, vid in events:
                if delta == +1:
                    current.add(vid)
                else:
                    current.discard(vid)
                if len(current) > cap:
                    self.add(None, "C7_CAPACITY",
                             f"at minute {t} terminal {term} holds "
                             f"{len(current)} vehicles (cap {cap})")
                    break  # one report per terminal is enough

        # cost
        fixed  = self.params["cost_fixed_vehicle"]       * len(vehicles_used)
        variab = self.params["cost_variable_per_min"]    * total_deadhead_min
        total  = fixed + variab + total_driver_cost
        breakdown = {
            "vehicles_used":      len(vehicles_used),
            "fixed_vehicle_cost": fixed,
            "deadhead_minutes":   total_deadhead_min,
            "variable_cost":      variab,
            "driver_cost":        total_driver_cost,
            "total_cost":         total,
        }
        return len(self.violations) == 0, breakdown

    # ---- top-level structure

    def _check_top_level(self) -> None:
        if "instance_id" not in self.sol:
            self.add(None, "SCHEMA", "solution missing 'instance_id'")
        if "duties" not in self.sol or not isinstance(self.sol["duties"], list):
            self.add(None, "SCHEMA", "solution must contain a 'duties' list")

    # ---- per-duty checks

    def _check_duty(
        self,
        duty: dict,
        seen_trips: dict[int, str],
        vehicle_shifts: dict[str, list[tuple[int, int, str]]],
        terminal_timeline: dict[str, list[tuple[int, int, str]]],
        vehicles_used: set[str],
    ) -> tuple[int, float]:
        k = duty.get("duty_id", "?")
        try:
            acts = duty["activities"]
            s0   = duty["shift_start_min"]
            s1   = duty["shift_end_min"]
            b0   = duty["break_start_min"]
            b1   = duty["break_end_min"]
            bloc = duty["break_location"]
            vid  = duty["vehicle_id"]
            did  = duty["driver_id"]
        except KeyError as e:
            self.add(k, "SCHEMA", f"missing field {e.args[0]}")
            return 0, 0.0
        vehicles_used.add(vid)

        # shift length
        length_min = s1 - s0
        L_min = self.params["shift_min_hours"] * 60
        L_max = self.params["shift_max_hours"] * 60
        if not (L_min <= length_min <= L_max):
            self.add(k, "C3_SHIFT_LEN",
                     f"shift length {length_min} min not in [{L_min},{L_max}]")
        if length_min % 15 != 0:
            self.add(k, "C3_SHIFT_LEN_QUARTER",
                     f"shift length {length_min} is not a multiple of 15 minutes")
        if s0 % 15 != 0:
            self.add(k, "C3_SHIFT_START_QUARTER",
                     f"shift start time {s0} is not at an integer quarter-hour")

        # break window
        alpha = self.params["break_min_from_start_hours"] * 60
        beta  = self.params["break_min_from_end_hours"]   * 60
        if b0 < s0 + alpha:
            self.add(k, "C4_BREAK_EARLY",
                     f"break starts at {b0}, must be >= {s0 + alpha}")
        if b1 > s1 - beta:
            self.add(k, "C4_BREAK_LATE",
                     f"break ends at {b1}, must be <= {s1 - beta}")
        if b1 - b0 != self.params["break_length_hours"] * 60:
            self.add(k, "C4_BREAK_LEN",
                     f"break length {b1-b0} min, expected "
                     f"{self.params['break_length_hours']*60}")

        # activity sequence
        if not acts:
            self.add(k, "SCHEMA", "empty activities list")
            return 0, 0.0

        # first / last bookends
        first = acts[0]
        last  = acts[-1]
        if first.get("type") != "deadhead" or first.get("from") != "D":
            self.add(k, "C5_DEPOT_START",
                     "first activity must be a deadhead starting from D")
        if last.get("type") != "deadhead" or last.get("to") != "D":
            self.add(k, "C5_DEPOT_END",
                     "last activity must be a deadhead ending at D")
        if first.get("start_min") != s0:
            self.add(k, "SHIFT_MATCH",
                     f"first activity start {first.get('start_min')} != shift_start {s0}")
        if last.get("end_min") != s1:
            self.add(k, "SHIFT_MATCH",
                     f"last activity end {last.get('end_min')} != shift_end {s1}")

        # iterate activities
        breaks_seen = 0
        prev_end_loc: str | None = None
        prev_end_t:   int | None = None
        deadhead_min = 0

        for i, a in enumerate(acts):
            at = a.get("type")
            s  = a.get("start_min"); e = a.get("end_min")
            if s is None or e is None or e < s:
                self.add(k, "TIME", f"activity {i} ({at}) has bad times {s}->{e}")
                continue

            # contiguity in time
            if prev_end_t is not None and s != prev_end_t:
                self.add(k, "CONTIG_TIME",
                         f"activity {i} starts at {s} but previous ended at {prev_end_t}")

            # per-type rules, compute from/to locations
            from_loc = to_loc = None

            if at == "deadhead":
                frm, to = a.get("from"), a.get("to")
                if (frm, to) not in {("D","A"),("A","D"),("D","B"),("B","D")}:
                    self.add(k, "DEADHEAD_LEG",
                             f"activity {i}: deadhead {frm}->{to} not allowed")
                else:
                    expect = self.travel.lookup(frm, to, s)
                    if e - s != expect:
                        self.add(k, "C2_TRAVEL",
                                 f"activity {i}: deadhead {frm}->{to} duration "
                                 f"{e-s} min, expected {expect}")
                from_loc, to_loc = frm, to
                deadhead_min += (e - s)

            elif at == "service":
                tid = a.get("trip_id")
                if tid not in self.trips:
                    self.add(k, "SERVICE_TRIP",
                             f"activity {i}: unknown trip_id {tid}")
                else:
                    trip = self.trips[tid]
                    if s != trip["departure_min"]:
                        self.add(k, "SERVICE_DEP",
                                 f"activity {i}: service trip {tid} starts at {s}, "
                                 f"scheduled {trip['departure_min']}")
                    expect = self.travel.lookup(trip["origin"], trip["destination"], s)
                    if e - s != expect:
                        self.add(k, "C2_TRAVEL",
                                 f"activity {i}: service trip {tid} duration {e-s}, "
                                 f"expected {expect}")
                    from_loc, to_loc = trip["origin"], trip["destination"]
                    # coverage bookkeeping
                    if tid in seen_trips:
                        self.add(k, "C1_DUP",
                                 f"trip {tid} already covered by duty {seen_trips[tid]}")
                    else:
                        seen_trips[tid] = k

            elif at == "wait":
                at_loc = a.get("at")
                if at_loc not in ("D", "A", "B"):
                    self.add(k, "WAIT_LOC",
                             f"activity {i}: wait at invalid location {at_loc}")
                from_loc = to_loc = at_loc

            elif at == "break":
                breaks_seen += 1
                at_loc = a.get("at")
                if at_loc not in ("D", "A", "B"):
                    self.add(k, "BREAK_LOC",
                             f"activity {i}: break at invalid location {at_loc}")
                if (s, e) != (b0, b1):
                    self.add(k, "BREAK_MATCH",
                             f"activity {i}: break {s}-{e} != duty break {b0}-{b1}")
                if at_loc != bloc:
                    self.add(k, "BREAK_LOC",
                             f"activity {i}: break location {at_loc} != duty "
                             f"break_location {bloc}")
                from_loc = to_loc = at_loc

            else:
                self.add(k, "TYPE", f"activity {i}: unknown type {at!r}")

            # contiguity in space
            if prev_end_loc is not None and from_loc is not None \
               and from_loc != prev_end_loc:
                self.add(k, "CONTIG_LOC",
                         f"activity {i} starts at {from_loc} but "
                         f"previous ended at {prev_end_loc}")

            # record terminal dwell (wait / break / service-not-counted)
            if at in ("wait", "break") and from_loc in ("A", "B") and e > s:
                terminal_timeline[from_loc].append((s, e, vid))

            prev_end_loc = to_loc
            prev_end_t   = e

        if breaks_seen != 1:
            self.add(k, "C4_BREAK_COUNT",
                     f"exactly one break activity required, found {breaks_seen}")

        # vehicle continuity bookkeeping
        vehicle_shifts[vid].append((s0, s1, k))

        # driver cost
        L_h = length_min / 60.0
        driver_cost = 8.0 * self.params["cost_driver_regular_per_h"] \
                     + max(0.0, L_h - 8.0) * self.params["cost_driver_overtime_per_h"]
        return deadhead_min, driver_cost


# ---------------------------------------------------------------------------
# CLI

def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    inst_path, sol_path = argv[1], argv[2]
    inst = load_json(inst_path)
    sol  = load_json(sol_path)

    c = Checker(inst, sol)
    ok, breakdown = c.check()

    if not ok:
        print(f"INFEASIBLE ({len(c.violations)} violation"
              f"{'s' if len(c.violations)!=1 else ''}):")
        for v in c.violations[:50]:
            print(v)
        if len(c.violations) > 50:
            print(f"  ... and {len(c.violations)-50} more.")
        return 1

    print("OK — solution is feasible.")
    print(f"  vehicles used       : {breakdown['vehicles_used']}")
    print(f"  fixed vehicle cost  : {breakdown['fixed_vehicle_cost']:.2f}")
    print(f"  deadhead minutes    : {breakdown['deadhead_minutes']}")
    print(f"  variable cost       : {breakdown['variable_cost']:.2f}")
    print(f"  driver cost         : {breakdown['driver_cost']:.2f}")
    print(f"  TOTAL               : {breakdown['total_cost']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
