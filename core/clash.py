"""
Deterministic scheduling logic.

Nothing in this module is probabilistic or generative. Clash detection
is exact interval arithmetic, and it must stay that way: a language
model asked whether 11:30-13:20 overlaps 12:00-13:50 will usually be
right, and "usually" is not an acceptable standard for a decision that
determines whether a student can graduate on time.

The AI layer calls these functions. It never replaces them.
"""

from __future__ import annotations

import itertools

import pandas as pd


def _venue_label(session: dict, cfg) -> str:
    """Human-readable location, with online sessions named as such."""
    from core.schema import clean_optional
    return clean_optional(session.get("venue")) or cfg.online_label


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """
    True if two half-open intervals intersect.

    Half-open [start, end) is deliberate: a class ending at 09:20 and one
    starting at 09:20 do not clash. Slot-label matching would miss the
    partial overlap case (11:30-13:20 vs 12:00-13:50); interval
    arithmetic does not.
    """
    return a_start < b_end and b_start < a_end


def find_clashes(sessions: pd.DataFrame) -> list[dict]:
    """
    Every pairwise time conflict within a set of sessions.

    Works on any selection: one student's chosen courses, a whole level,
    or the entire timetable.
    """
    out = []
    for day, grp in sessions.groupby("day"):
        rows = grp.to_dict("records")
        for a, b in itertools.combinations(rows, 2):
            if a["course_code"] == b["course_code"] and a["section"] == b["section"]:
                continue
            if overlaps(a["start_min"], a["end_min"],
                        b["start_min"], b["end_min"]):
                out.append({
                    "day": day,
                    "course_a": a["course_code"],
                    "name_a": a["course_name"],
                    "time_a": f"{a['start_time']}-{a['end_time']}",
                    "course_b": b["course_code"],
                    "name_b": b["course_name"],
                    "time_b": f"{b['start_time']}-{b['end_time']}",
                    "overlap_minutes": (
                        min(a["end_min"], b["end_min"])
                        - max(a["start_min"], b["start_min"])
                    ),
                    "same_department": a["department"] == b["department"],
                })
    return out


def venue_double_bookings(sessions: pd.DataFrame) -> list[dict]:
    """
    Two in-person sessions in the same room at overlapping times.

    Included for completeness and for the administrator view. On the
    UGBS Second Semester timetable this correctly returns an empty list:
    the published timetable has no room conflicts. An empty result is a
    finding, not a failure — it tells the School its room allocation is
    sound and that the student-side problem lies elsewhere.
    """
    out = []
    physical = sessions[~sessions["is_online"]]
    for (day, venue), grp in physical.groupby(["day", "venue"]):
        rows = grp.to_dict("records")
        for a, b in itertools.combinations(rows, 2):
            if overlaps(a["start_min"], a["end_min"],
                        b["start_min"], b["end_min"]):
                out.append({
                    "day": day, "venue": venue,
                    "course_a": a["course_code"], "course_b": b["course_code"],
                    "time_a": f"{a['start_time']}-{a['end_time']}",
                    "time_b": f"{b['start_time']}-{b['end_time']}",
                })
    return out


def transition_check(sessions: pd.DataFrame, cfg) -> list[dict]:
    """
    Consecutive sessions the student may not physically reach in time.

    A clash-free schedule can still be unattendable: back-to-back classes
    in distant buildings with a ten-minute changeover. This is the check
    that a timetable PDF can never perform.
    """
    out = []
    for day, grp in sessions.groupby("day"):
        ordered = grp.sort_values("start_min").to_dict("records")
        for a, b in zip(ordered, ordered[1:]):
            gap = b["start_min"] - a["end_min"]
            if gap < 0 or gap > cfg.min_usable_gap:
                continue  # overlapping (a clash) or a genuine free block
            walk = cfg.transit_minutes(a.get("building"), b.get("building"))
            if walk > gap:
                out.append({
                    "day": day,
                    "from_course": a["course_code"],
                    "from_venue": _venue_label(a, cfg),
                    "from_building": a.get("building"),
                    "to_course": b["course_code"],
                    "to_venue": _venue_label(b, cfg),
                    "to_building": b.get("building"),
                    "gap_minutes": gap,
                    "walk_minutes": walk,
                    "shortfall_minutes": walk - gap,
                })
    return out


def mutual_exclusions(sessions: pd.DataFrame, year_group=None) -> list[dict]:
    """
    Course pairs that can never be taken together because every sitting
    of one overlaps every sitting of the other.

    This is the structural finding a student cannot see from a PDF, and
    it is the evidence base for the whole system: on the UGBS Second
    Semester timetable it identifies 23 impossible pairs at Level 300
    and 22 at Level 400.
    """
    df = sessions
    if year_group is not None:
        df = df[df["year_group"] == year_group]

    sittings = {}
    for code, grp in df.groupby("course_code"):
        sittings[code] = grp.to_dict("records")

    out = []
    for a, b in itertools.combinations(sorted(sittings), 2):
        pairs = list(itertools.product(sittings[a], sittings[b]))
        conflicting = [
            (x, y) for x, y in pairs
            if x["day"] == y["day"]
            and overlaps(x["start_min"], x["end_min"],
                         y["start_min"], y["end_min"])
        ]
        if conflicting and len(conflicting) == len(pairs):
            x, y = conflicting[0]
            out.append({
                "course_a": a, "name_a": x["course_name"],
                "dept_a": x["department"],
                "course_b": b, "name_b": y["course_name"],
                "dept_b": y["department"],
                "day": x["day"],
                "time": f"{x['start_time']}-{x['end_time']}",
                "cross_department": x["department"] != y["department"],
            })
    return out


def is_feasible(sessions: pd.DataFrame, cfg) -> tuple[bool, list, list]:
    """Convenience: (no clashes, clash list, infeasible transition list)."""
    clashes = find_clashes(sessions)
    transitions = transition_check(sessions, cfg)
    return (not clashes), clashes, transitions


def conditional_conflicts(sessions: pd.DataFrame, year_group=None) -> list[dict]:
    """
    Course pairs that clash on SOME sittings but not all.

    These are resolvable — but only by choosing the right group/section,
    and only if the student knows which combination works. This is where
    the system adds the most value, because a published timetable gives
    a student no way to work it out.

    Distinguishing these from strict mutual exclusions matters: a strict
    exclusion means "choose one"; a conditional conflict means "both are
    possible, but not every group pairing works".
    """
    df = sessions
    if year_group is not None:
        df = df[df["year_group"] == year_group]

    sittings = {c: g.to_dict("records") for c, g in df.groupby("course_code")}

    out = []
    for a, b in itertools.combinations(sorted(sittings), 2):
        pairs = list(itertools.product(sittings[a], sittings[b]))
        conflicting = [
            (x, y) for x, y in pairs
            if x["day"] == y["day"]
            and overlaps(x["start_min"], x["end_min"],
                         y["start_min"], y["end_min"])
        ]
        if conflicting and len(conflicting) < len(pairs):
            safe = [(x, y) for x, y in pairs if (x, y) not in conflicting]
            x, y = safe[0]
            out.append({
                "course_a": a, "course_b": b,
                "dept_a": x["department"], "dept_b": y["department"],
                "conflicting_pairings": len(conflicting),
                "total_pairings": len(pairs),
                "workable_example": (
                    f"{a} {x['section']} ({x['day']} {x['start_time']}) + "
                    f"{b} {y['section']} ({y['day']} {y['start_time']})"),
                "cross_department": x["department"] != y["department"],
            })
    return out


def integrity_check(sessions: pd.DataFrame, cfg) -> list[dict]:
    """
    Check the timetable itself for scheduling problems.

    Distinct from the loader's validation, which reports how the FILE was
    read. This examines whether the SCHEDULE is sound, and it applies
    whatever the source: a clean PDF of a faulty timetable still produces
    findings here, and a badly parsed PDF of a sound timetable produces
    findings too.

    A check that returns nothing has still done its job. On the UGBS
    Second Semester timetable most of these come back empty, which is
    itself the reportable result: the School's schedule is internally
    sound, so the difficulty students face lies in course selection
    rather than in a faulty timetable. For another institution's data
    the same checks may find real problems, which is why they run on
    every load rather than being hardcoded to an expected answer.

    Each finding carries a severity so the interface can separate
    "this is wrong" from "this is worth a look".
    """
    findings: list[dict] = []

    # --- a room cannot hold two classes at once ----------------------
    for d in venue_double_bookings(sessions):
        findings.append({
            "severity": "error",
            "check": "Room booked twice",
            "detail": (f"{d['venue']} holds {d['course_a']} and "
                       f"{d['course_b']} at the same time on {d['day']} "
                       f"({d['time_a']} / {d['time_b']})"),
        })

    # --- the same group cannot be in two places at once --------------
    physical = sessions[~sessions["is_online"]]
    for (day, section), grp in physical.groupby(["day", "section"]):
        if section == cfg.section_na:
            continue
        rows = grp.to_dict("records")
        for a, b in itertools.combinations(rows, 2):
            if (a["year_group"] == b["year_group"]
                    and a["course_code"] != b["course_code"]
                    and overlaps(a["start_min"], a["end_min"],
                                 b["start_min"], b["end_min"])):
                findings.append({
                    "severity": "info",
                    "check": "Same group label, two classes at once",
                    "detail": (f"{section} at "
                               f"{cfg.year_group_label.lower()} "
                               f"{a['year_group']}: {a['course_code']} and "
                               f"{b['course_code']} overlap on {day}. This "
                               f"is only a clash if the same students are "
                               f"in {section} for both courses — group "
                               f"labels are often set per course."),
                })

    # --- one course listed twice in the same slot --------------------
    dup = (sessions.groupby(["course_code", "section", "day", "start_time"])
           .size().reset_index(name="n"))
    for r in dup[dup["n"] > 1].itertuples():
        findings.append({
            "severity": "warning",
            "check": "Duplicate entry",
            "detail": (f"{r.course_code} ({r.section}) appears {r.n} times "
                       f"on {r.day} at {r.start_time}"),
        })

    # --- sessions that do not sit on the published slot grid ---------
    offgrid = sessions[sessions["slot_label"].isna()]
    for r in offgrid.itertuples():
        findings.append({
            "severity": "warning",
            "check": "Outside the normal timetable grid",
            "detail": (f"{r.course_code} starts at {r.start_time} on "
                       f"{r.day}, which is not one of the published slot "
                       f"times"),
        })

    # --- unusual lesson lengths --------------------------------------
    if len(sessions):
        typical = sessions["duration_minutes"].mode()
        if len(typical):
            usual = int(typical.iloc[0])
            odd = sessions[sessions["duration_minutes"] != usual]
            for r in odd.itertuples():
                findings.append({
                    "severity": "info",
                    "check": "Unusual class length",
                    "detail": (f"{r.course_code} on {r.day} runs "
                               f"{int(r.duration_minutes)} minutes; most "
                               f"classes run {usual}"),
                })

    # --- an in-person class with no room --------------------------
    noroom = sessions[(~sessions["is_online"]) & (sessions["venue"].isna())]
    for r in noroom.itertuples():
        findings.append({
            "severity": "warning",
            "check": "No room given",
            "detail": (f"{r.course_code} on {r.day} at {r.start_time} is "
                       f"in person but has no room listed"),
        })

    # --- one course code sitting under two different cohorts ---------
    spread = sessions.groupby("course_code")["year_group"].nunique()
    for code, n in spread[spread > 1].items():
        groups = sorted(
            sessions[sessions["course_code"] == code]["year_group"].unique())
        findings.append({
            "severity": "info",
            "check": "Course listed under more than one "
                     f"{cfg.year_group_label.lower()}",
            "detail": (f"{code} appears at "
                       f"{', '.join(str(g) for g in groups)} — check this "
                       f"is intended"),
        })

    return findings
