"""
Analytical decision-support layer.

Each function below exists to support a NAMED decision. The project
brief is explicit that the analytical component must not be bolted on,
so the decision each analysis serves is documented in its docstring and
surfaced to the user in the output.
"""

from __future__ import annotations

import pandas as pd


def day_profile(sessions: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Per-day workload profile with a congestion score.

    DECISION SUPPORTED: "Which day can I keep clear for paid work,
    project meetings or study?"

    The score combines contact hours, session count, span, unusable idle
    time, building transitions, infeasible transitions, and early/late
    edges. All weights come from the institution config, so a School
    that cares more about early starts than about walking simply edits
    the YAML.
    """
    w = cfg.congestion_weights
    rows = []

    for day in cfg.teaching_days:
        grp = sessions[sessions["day"] == day].sort_values("start_min")
        if grp.empty:
            rows.append({
                "day": day, "sessions": 0, "contact_hours": 0.0,
                "span_hours": 0.0, "idle_hours": 0.0, "campus_required": False,
                "in_person_sessions": 0, "online_sessions": 0,
                "building_transitions": 0, "tight_transitions": 0,
                "early_start": False, "late_finish": False,
                "congestion_score": 0.0, "low_value_commute": False,
            })
            continue

        recs = grp.to_dict("records")
        contact = grp["duration_minutes"].sum() / 60
        span = (grp["end_min"].max() - grp["start_min"].min()) / 60

        idle = 0
        transitions = 0
        tight = 0
        for a, b in zip(recs, recs[1:]):
            gap = b["start_min"] - a["end_min"]
            if gap >= cfg.min_usable_gap:
                idle += gap
            if a.get("building") and b.get("building") and a["building"] != b["building"]:
                transitions += 1
                if 0 <= gap < cfg.min_usable_gap:
                    if cfg.transit_minutes(a["building"], b["building"]) > gap:
                        tight += 1

        in_person = int((~grp["is_online"]).sum())
        online = int(grp["is_online"].sum())
        early = bool(grp["start_min"].min() <= cfg.early_start_minutes)
        late = bool(grp["end_min"].max() >= cfg.late_finish_minutes)

        score = (
            w["contact_hours"] * contact
            + w["sessions"] * len(grp)
            + w["span_hours"] * span
            + w["idle_hours"] * (idle / 60)
            + w["building_transitions"] * transitions
            + w["tight_transition"] * tight
            + w["early_start"] * early
            + w["late_finish"] * late
        )

        rows.append({
            "day": day,
            "sessions": len(grp),
            "contact_hours": round(contact, 2),
            "span_hours": round(span, 2),
            "idle_hours": round(idle / 60, 2),
            "campus_required": in_person > 0,
            "in_person_sessions": in_person,
            "online_sessions": online,
            "building_transitions": transitions,
            "tight_transitions": tight,
            "early_start": early,
            "late_finish": late,
            "congestion_score": round(score, 1),
            # A trip to campus for a single class: the commute costs more
            # than the session delivers.
            "low_value_commute": bool(
                0 < in_person <= cfg.low_value_commute_threshold),
        })

    return pd.DataFrame(rows)


def free_blocks(sessions: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Usable gaps between classes, and wholly free days.

    DECISION SUPPORTED: "When can my group realistically meet?" and
    "Is this gap long enough to be worth staying on campus for?"

    A gap shorter than the configured minimum is idle time, not free
    time. That distinction is the whole point: three 40-minute gaps are
    a worse day than one two-hour gap, even though the totals match.
    """
    day_start = min(
        int(s["start"].split(":")[0]) * 60 + int(s["start"].split(":")[1])
        for s in cfg.slots)
    day_end = max(
        int(s["end"].split(":")[0]) * 60 + int(s["end"].split(":")[1])
        for s in cfg.slots)

    rows = []
    for day in cfg.teaching_days:
        grp = sessions[sessions["day"] == day].sort_values("start_min")
        if grp.empty:
            rows.append({"day": day, "start": "-", "end": "-",
                         "minutes": day_end - day_start, "type": "free day"})
            continue

        recs = grp.to_dict("records")
        first, last = recs[0], recs[-1]

        if first["start_min"] - day_start >= cfg.min_usable_gap:
            rows.append({
                "day": day,
                "start": f"{day_start // 60:02d}:{day_start % 60:02d}",
                "end": first["start_time"],
                "minutes": first["start_min"] - day_start,
                "type": "morning block"})

        for a, b in zip(recs, recs[1:]):
            gap = b["start_min"] - a["end_min"]
            if gap >= cfg.min_usable_gap:
                rows.append({
                    "day": day, "start": a["end_time"], "end": b["start_time"],
                    "minutes": gap, "type": "midday block"})

        if day_end - last["end_min"] >= cfg.min_usable_gap:
            rows.append({
                "day": day, "start": last["end_time"],
                "end": f"{day_end // 60:02d}:{day_end % 60:02d}",
                "minutes": day_end - last["end_min"], "type": "evening block"})

    return pd.DataFrame(rows)


def common_free_blocks(schedules: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """
    Free time shared by several students.

    DECISION SUPPORTED: "When should our project group meet?"

    Intersects each member's free blocks. This is the analysis that a
    published timetable structurally cannot provide, because it requires
    combining several students' individual selections.
    """
    per_person = {
        name: free_blocks(df, cfg) for name, df in schedules.items()}

    rows = []
    for day in cfg.teaching_days:
        intervals = []
        for name, fb in per_person.items():
            d = fb[(fb["day"] == day) & (fb["start"] != "-")]
            person = [(_m(r["start"]), _m(r["end"])) for _, r in d.iterrows()]
            if fb[(fb["day"] == day) & (fb["start"] == "-")].shape[0]:
                person = [(_m("07:30"), _m("19:20"))]
            intervals.append(person)

        if not intervals or any(len(p) == 0 for p in intervals):
            continue

        common = intervals[0]
        for nxt in intervals[1:]:
            merged = []
            for s1, e1 in common:
                for s2, e2 in nxt:
                    s, e = max(s1, s2), min(e1, e2)
                    if e - s >= cfg.min_usable_gap:
                        merged.append((s, e))
            common = merged
        for s, e in common:
            rows.append({
                "day": day, "start": _hhmm(s), "end": _hhmm(e),
                "minutes": e - s, "members": len(schedules)})

    return pd.DataFrame(rows)


def delivery_mix(sessions: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Online versus in-person load per day.

    DECISION SUPPORTED: "Do I actually need to travel to campus on
    Wednesday, or can I attend from home?"

    The high-value case this catches: a day with several online sessions
    and exactly one in-person class. The student commutes for one class.
    Moving or dropping that single session frees an entire day.
    """
    rows = []
    for day in cfg.teaching_days:
        grp = sessions[sessions["day"] == day]
        in_person = int((~grp["is_online"]).sum())
        online = int(grp["is_online"].sum())
        rows.append({
            "day": day, "in_person": in_person, "online": online,
            "total": len(grp),
            "campus_required": in_person > 0,
            "low_value_commute": bool(
                0 < in_person <= cfg.low_value_commute_threshold),
        })
    return pd.DataFrame(rows)


def venue_utilisation(sessions: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Room load across the week — the administrator view.

    DECISION SUPPORTED (timetable officer, not student): "Which venues
    are over-subscribed, and where is there spare capacity to relieve a
    congested slot?"

    Secondary output. Retained because it is genuinely useful to the
    School, but it is not the student-facing analysis the scenario asks
    for.
    """
    physical = sessions[~sessions["is_online"]]
    total_slots = len(cfg.teaching_days) * len(cfg.slots)

    rows = []
    for venue, grp in physical.groupby("venue"):
        rows.append({
            "venue": venue,
            "building": cfg.building_for(venue),
            "sessions": len(grp),
            "share_of_sessions": round(len(grp) / len(physical), 4),
            "slot_occupancy": round(len(grp) / total_slots, 4),
            "busiest_day": grp["day"].value_counts().idxmax(),
        })
    out = pd.DataFrame(rows).sort_values("sessions", ascending=False)
    return out.reset_index(drop=True)


def slot_pressure(sessions: pd.DataFrame, cfg, year_group=None) -> pd.DataFrame:
    """
    How many courses compete for each day/slot within a year group.

    DECISION SUPPORTED (both audiences): for the student, "which slots
    force me to choose between courses?"; for the School, "which slots
    are over-loaded and should be rebalanced?"

    On the UGBS timetable this surfaces the Tuesday 11:30 Level 300 slot,
    where four courses run simultaneously.
    """
    df = sessions
    if year_group is not None:
        df = df[df["year_group"] == year_group]

    if df.empty:
        return pd.DataFrame(columns=["day", "start_time", "slot",
                                     "concurrent_courses", "departments",
                                     "courses"])

    rows = []
    for (day, start), grp in df.groupby(["day", "start_time"]):
        codes = sorted(grp["course_code"].unique())
        rows.append({
            "day": day, "start_time": start,
            "slot": cfg.slot_for(start),
            "concurrent_courses": len(codes),
            "departments": grp["department"].nunique(),
            "courses": ", ".join(codes),
        })
    out = pd.DataFrame(rows)
    out["day_order"] = out["day"].map(cfg.day_index)
    return (out.sort_values(["concurrent_courses", "day_order"],
                            ascending=[False, True])
            .drop(columns="day_order").reset_index(drop=True))


def _m(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
