"""
Feasible course-combination enumeration and ranking.

This is the system's headline analytical capability.

DECISION SUPPORTED: "Of the elective combinations open to me, which can
I actually register for, and which of those gives me the best week?"

A published timetable answers neither half. A student can read it and
still not know that OMIS 308 and PADM 318 are mutually exclusive, and
certainly cannot rank the surviving options by how liveable the
resulting week would be.
"""

from __future__ import annotations

import itertools

import pandas as pd

from core.clash import find_clashes, transition_check
from analytics.insights import day_profile


def _sessions_for(timetable: pd.DataFrame, codes, sections=None) -> pd.DataFrame:
    """All sessions belonging to a set of course codes."""
    sel = timetable[timetable["course_code"].isin(codes)]
    if sections:
        keep = []
        for _, row in sel.iterrows():
            wanted = sections.get(row["course_code"])
            keep.append(wanted is None or row["section"] == wanted)
        sel = sel[pd.Series(keep, index=sel.index)]
    return sel


def evaluate_schedule(sessions: pd.DataFrame, cfg) -> dict:
    """Score a single candidate schedule. Lower total_score is better."""
    w = cfg.combination_weights
    profile = day_profile(sessions, cfg)
    active = profile[profile["sessions"] > 0]

    clashes = find_clashes(sessions)
    bad_transitions = transition_check(sessions, cfg)

    mean_cong = float(active["congestion_score"].mean()) if len(active) else 0.0
    peak_cong = float(active["congestion_score"].max()) if len(active) else 0.0
    campus_days = int(profile["campus_required"].sum())
    low_value = int(profile["low_value_commute"].sum())
    idle = float(profile["idle_hours"].sum())
    early_days = int(profile["early_start"].sum())

    score = (
        w["mean_congestion"] * mean_cong
        + w["peak_day_congestion"] * peak_cong
        + w["campus_days"] * campus_days
        + w["low_value_commute_days"] * low_value
        + w["total_idle_hours"] * idle
        + w["early_start_days"] * early_days
        + w["infeasible_transitions"] * len(bad_transitions)
    )

    return {
        "feasible": len(clashes) == 0,
        "clashes": clashes,
        "infeasible_transitions": bad_transitions,
        "teaching_days_used": int((profile["sessions"] > 0).sum()),
        "campus_days": campus_days,
        "low_value_commute_days": low_value,
        "mean_congestion": round(mean_cong, 1),
        "peak_congestion": round(peak_cong, 1),
        "total_idle_hours": round(idle, 2),
        "early_start_days": early_days,
        "total_score": round(score, 1),
        "profile": profile,
    }


def rank_combinations(timetable: pd.DataFrame, cfg, *,
                      core_codes: list[str] | None = None,
                      candidate_codes: list[str],
                      choose: int,
                      sections: dict | None = None,
                      max_results: int = 10,
                      max_combinations: int = 20000) -> dict:
    """
    Enumerate every way of choosing `choose` courses from
    `candidate_codes`, discard those that clash (with each other or with
    the fixed core load), score the survivors and return the best.

    Returns a dict containing the ranked table, the counts needed to
    explain the result to a user, and diagnostics for the infeasible
    case — because "no combination works" is itself an answer the
    student needs, and needs a reason for.
    """
    core_codes = core_codes or []
    core_sessions = _sessions_for(timetable, core_codes, sections)

    all_combos = list(itertools.combinations(sorted(set(candidate_codes)), choose))
    if len(all_combos) > max_combinations:
        raise ValueError(
            f"{len(all_combos):,} combinations exceeds the cap of "
            f"{max_combinations:,}. Narrow the candidate list first.")

    ranked, rejected = [], []

    for combo in all_combos:
        sel = _sessions_for(timetable, list(combo), sections)
        full = pd.concat([core_sessions, sel]) if len(core_sessions) else sel
        result = evaluate_schedule(full, cfg)

        record = {
            "combination": " + ".join(combo),
            "courses": list(combo),
            "campus_days": result["campus_days"],
            "low_value_commute_days": result["low_value_commute_days"],
            "mean_congestion": result["mean_congestion"],
            "peak_congestion": result["peak_congestion"],
            "idle_hours": result["total_idle_hours"],
            "early_start_days": result["early_start_days"],
            "tight_transitions": len(result["infeasible_transitions"]),
            "score": result["total_score"],
        }

        if result["feasible"]:
            ranked.append(record)
        else:
            first = result["clashes"][0]
            record["reason"] = (
                f"{first['course_a']} and {first['course_b']} both run "
                f"{first['day']} {first['time_a']}")
            rejected.append(record)

    ranked_df = (pd.DataFrame(ranked).sort_values("score")
                 .head(max_results).reset_index(drop=True)
                 if ranked else pd.DataFrame())

    return {
        "total_combinations": len(all_combos),
        "feasible_count": len(ranked),
        "rejected_count": len(rejected),
        "ranked": ranked_df,
        "rejected_sample": (pd.DataFrame(rejected).head(10)
                            if rejected else pd.DataFrame()),
        "any_feasible": bool(ranked),
    }


def explain_choice(timetable: pd.DataFrame, cfg, codes: list[str],
                   sections: dict | None = None) -> str:
    """
    Plain-language justification for a schedule.

    The AI layer phrases this for the user; the numbers inside it are
    computed here, deterministically. The model never calculates.
    """
    sessions = _sessions_for(timetable, codes, sections)
    result = evaluate_schedule(sessions, cfg)
    profile = result["profile"]
    active = profile[profile["sessions"] > 0]

    lines = [f"Selection: {', '.join(codes)}"]

    if not result["feasible"]:
        lines.append(f"NOT FEASIBLE — {len(result['clashes'])} clash(es):")
        for c in result["clashes"]:
            lines.append(f"  {c['course_a']} vs {c['course_b']} — "
                         f"{c['day']} {c['time_a']} / {c['time_b']} "
                         f"({c['overlap_minutes']} min overlap)")
        return "\n".join(lines)

    lines.append("Feasible — no time conflicts.")
    lines.append(f"Teaching days used: {result['teaching_days_used']} "
                 f"(campus attendance required on {result['campus_days']})")

    if len(active):
        heaviest = active.loc[active["congestion_score"].idxmax()]
        lightest = active.loc[active["congestion_score"].idxmin()]
        lines.append(f"Heaviest day: {heaviest['day']} "
                     f"({heaviest['sessions']} sessions, "
                     f"{heaviest['contact_hours']}h contact, "
                     f"score {heaviest['congestion_score']})")
        lines.append(f"Lightest teaching day: {lightest['day']} "
                     f"(score {lightest['congestion_score']})")

    free_days = profile[profile["sessions"] == 0]["day"].tolist()
    if free_days:
        lines.append(f"Completely free: {', '.join(free_days)}")

    lvc = profile[profile["low_value_commute"]]["day"].tolist()
    if lvc:
        lines.append(f"Low-value commute day(s): {', '.join(lvc)} — "
                     f"a trip to campus for a single in-person class")

    if result["infeasible_transitions"]:
        lines.append("Physically tight transitions:")
        for t in result["infeasible_transitions"]:
            lines.append(f"  {t['day']}: {t['from_course']} ({t['from_venue']}) "
                         f"-> {t['to_course']} ({t['to_venue']}) — "
                         f"{t['gap_minutes']} min gap, "
                         f"{t['walk_minutes']} min walk, "
                         f"short by {t['shortfall_minutes']} min")

    return "\n".join(lines)
