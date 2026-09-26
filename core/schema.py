"""
Canonical schema, configuration loader and validation.

Design rule enforced here: NOTHING institution-specific is written in
code. Every day name, venue, department, slot time and weight is read
from the institution config file. Swapping config swaps institution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import yaml

# Canonical column names used everywhere downstream, regardless of what
# the source file called them.
CANONICAL_COLUMNS = [
    "session_id", "day", "slot_label", "start_time", "end_time",
    "duration_minutes", "course_code", "course_name", "department",
    "year_group", "section", "venue", "building", "delivery_mode",
    "is_online", "effective_from", "effective_to",
]


NULLISH = {"", "nan", "none", "null", "n/a", "na", "-"}


def clean_optional(value):
    """
    Normalise the many ways an absent value arrives to a single None.

    Excel round-trips turn None into NaN, and a defensive astype(str)
    turns NaN into the literal string 'nan'. Both must resolve to None,
    or downstream logic will treat 'nan' as a real venue and compute a
    walking time to it.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if str(value).strip().lower() in NULLISH:
        return None
    return str(value).strip()


def _to_minutes(hhmm: str) -> int:
    """'09:30' -> 570. Times are compared as minutes from midnight."""
    h, m = str(hhmm).strip().split(":")
    return int(h) * 60 + int(m)


def _to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class Config:
    """Wrapper around the institution YAML with convenience accessors."""

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            self.raw = yaml.safe_load(fh)
        self.path = path

    # -- identity -----------------------------------------------------
    @property
    def name(self) -> str:
        return self.raw["institution"]["name"]

    @property
    def short_name(self) -> str:
        return self.raw["institution"]["short_name"]

    # -- calendar -----------------------------------------------------
    @property
    def teaching_days(self) -> list[str]:
        return self.raw["calendar"]["teaching_days"]

    @property
    def slots(self) -> list[dict]:
        return self.raw["calendar"]["slots"]

    def day_index(self, day: str) -> int:
        """Ordering for a day name. Institution-defined, not hardcoded."""
        return self.teaching_days.index(day)

    def slot_for(self, start_time: str) -> str | None:
        for s in self.slots:
            if s["start"] == start_time:
                return s["label"]
        return None

    @property
    def min_usable_gap(self) -> int:
        return self.raw["calendar"]["min_usable_gap_minutes"]

    @property
    def early_start_minutes(self) -> int:
        return _to_minutes(self.raw["calendar"]["early_start_threshold"])

    @property
    def late_finish_minutes(self) -> int:
        return _to_minutes(self.raw["calendar"]["late_finish_threshold"])

    # -- structure ----------------------------------------------------
    @property
    def year_group_label(self) -> str:
        return self.raw["structure"]["year_group_label"]

    @property
    def section_na(self) -> str:
        return self.raw["structure"]["section_not_applicable"]

    # -- departments --------------------------------------------------
    def department_for(self, course_code: str) -> str:
        prefix = str(course_code).strip().split()[0].upper()
        return self.raw["departments"]["prefix_map"].get(prefix, "Unmapped")

    def is_core(self, course_code: str) -> bool:
        prefix = str(course_code).strip().split()[0].upper()
        return prefix in self.raw["departments"]["core_prefixes"]

    # -- venues -------------------------------------------------------
    @property
    def virtual_token(self) -> str:
        return self.raw["venues"]["virtual_token"]

    def building_for(self, venue) -> str | None:
        v = clean_optional(venue)
        if v is None:
            return None
        if v.upper() == self.virtual_token.upper():
            return None
        return self.raw["venues"]["building_map"].get(v, "Unmapped Building")

    def transit_minutes(self, b1: str | None, b2: str | None) -> int:
        """Walking time between two buildings. Online sessions cost 0."""
        b1, b2 = clean_optional(b1), clean_optional(b2)
        if b1 is None or b2 is None:
            return 0  # an online session imposes no travel obligation
        if b1 == b2:
            pass  # still look it up; same-building may be non-zero
        matrix = self.raw["transit"]["matrix"]
        for key in (f"{b1}|{b2}", f"{b2}|{b1}"):
            if key in matrix:
                return matrix[key]
        return self.raw["transit"]["default_minutes"]

    @property
    def changeover_minutes(self) -> int:
        return self.raw["transit"]["changeover_minutes"]

    # -- delivery -----------------------------------------------------
    @property
    def online_label(self) -> str:
        return self.raw["delivery"]["online_label"]

    @property
    def in_person_label(self) -> str:
        return self.raw["delivery"]["in_person_label"]

    @property
    def low_value_commute_threshold(self) -> int:
        return self.raw["delivery"]["low_value_commute_threshold"]

    # -- weights ------------------------------------------------------
    @property
    def congestion_weights(self) -> dict:
        return self.raw["congestion_weights"]

    @property
    def combination_weights(self) -> dict:
        return self.raw["combination_weights"]


@dataclass
class ValidationReport:
    """Result of validating a loaded timetable against the config."""

    rows_in: int = 0
    rows_out: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ambiguities: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"Validation: {self.rows_out}/{self.rows_in} rows accepted"]
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING {w}")
        for a in self.ambiguities:
            lines.append(f"  AMBIGUITY {a['course_code']}: {a['detail']}")
        return "\n".join(lines)


# Column aliases the loader will accept from a source file. Extend this
# when onboarding an institution whose export uses different headers.
COLUMN_ALIASES = {
    "day": ["day", "weekday"],
    "start_time": ["start time", "start", "start_time", "from"],
    "end_time": ["end time", "end", "end_time", "to"],
    "course_code": ["course code", "course_code", "code"],
    "course_name": ["course name", "course_name", "title", "course title"],
    "year_group": ["level", "year", "year_group", "stage"],
    "section": ["group", "section", "class group"],
    "venue": ["venue", "room", "location"],
    "delivery_mode": ["delivery mode", "delivery_mode", "mode"],
    "duration_minutes": ["duration (mins)", "duration", "duration_minutes"],
    "lecturer": ["lecturer", "instructor", "staff"],
    "capacity": ["capacity", "seats"],
    "session_type": ["session type", "type"],
    "effective_from": ["effective from", "effective_from", "valid_from"],
    "effective_to": ["effective to", "effective_to", "valid_to"],
}


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any recognised source header to its canonical name."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = canonical
                break
    return df.rename(columns=rename)


def load_timetable(source: str, cfg: Config, sheet=0):
    """
    Load a timetable from CSV or Excel and normalise it.

    Returns (DataFrame, ValidationReport).
    """
    if source.lower().endswith((".xlsx", ".xlsm", ".xls")):
        # Prefer a sheet named "Timetable" when the caller has not chosen
        # one, so a multi-sheet workbook loads its data sheet, not its
        # documentation sheet.
        if sheet == 0:
            names = pd.ExcelFile(source).sheet_names
            sheet = "Timetable" if "Timetable" in names else 0
        raw = pd.read_excel(source, sheet_name=sheet)
    else:
        raw = pd.read_csv(source)
    return normalise(raw, cfg)


def normalise(raw: pd.DataFrame, cfg: Config):
    """
    Normalise any raw timetable frame to the canonical schema, derive
    the fields the source does not carry, and validate.

    Kept separate from file loading so that data arriving from a PDF
    extraction, an upload, or a future API call passes through exactly
    the same validation as data read from disk. One validation path
    means a PDF cannot bypass a check that a spreadsheet must satisfy.

    Returns (DataFrame, ValidationReport).
    """
    report = ValidationReport(rows_in=len(raw))
    df = _map_columns(raw)

    # --- required fields present? ------------------------------------
    required = cfg.raw["schema"]["required"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"missing required column(s): {missing}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS), report

    # --- drop rows with null required fields -------------------------
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        report.warnings.append(
            f"dropped {before - len(df)} row(s) with null required fields")

    # --- normalise ---------------------------------------------------
    df["day"] = df["day"].astype(str).str.strip().str.title()
    bad_days = sorted(set(df["day"]) - set(cfg.teaching_days))
    if bad_days:
        report.warnings.append(
            f"day value(s) not in institution calendar, dropped: {bad_days}")
        df = df[df["day"].isin(cfg.teaching_days)]

    df["start_time"] = df["start_time"].astype(str).str.strip().str[:5]
    df["end_time"] = df["end_time"].astype(str).str.strip().str[:5]
    df["start_min"] = df["start_time"].map(_to_minutes)
    df["end_min"] = df["end_time"].map(_to_minutes)

    bad_interval = df["end_min"] <= df["start_min"]
    if bad_interval.any():
        report.warnings.append(
            f"dropped {int(bad_interval.sum())} row(s) where end <= start")
        df = df[~bad_interval]

    # Duration is DERIVED, never trusted from source.
    df["duration_minutes"] = df["end_min"] - df["start_min"]

    df["slot_label"] = df["start_time"].map(cfg.slot_for)
    unslotted = df["slot_label"].isna().sum()
    if unslotted:
        report.warnings.append(
            f"{unslotted} session(s) start outside the declared slot grid")

    df["course_code"] = df["course_code"].astype(str).str.strip().str.upper()
    df["course_name"] = df["course_name"].astype(str).str.strip()

    # --- derive department from code prefix via config ---------------
    df["department"] = df["course_code"].map(cfg.department_for)
    unmapped = sorted(df.loc[df["department"] == "Unmapped", "course_code"]
                      .str.split().str[0].unique())
    if unmapped:
        report.warnings.append(
            f"course prefix(es) not in department map: {unmapped}")

    df["is_core"] = df["course_code"].map(cfg.is_core)

    # --- sections: absent means "no section split", not missing ------
    if "section" not in df.columns:
        df["section"] = cfg.section_na
    df["section"] = df["section"].fillna(cfg.section_na).astype(str).str.strip()

    # --- venue / delivery mode: resolve the redundancy ---------------
    if "venue" not in df.columns:
        df["venue"] = None
    df["venue"] = df["venue"].astype(str).str.strip()

    if "delivery_mode" in df.columns:
        df["delivery_mode"] = df["delivery_mode"].astype(str).str.strip()
    else:
        df["delivery_mode"] = cfg.in_person_label

    virtual = cfg.virtual_token.upper()
    df["is_online"] = (
        (df["venue"].str.upper() == virtual)
        | (df["delivery_mode"].str.lower() == cfg.online_label.lower())
    )
    # Authoritative: mode drives everything; a virtual session has no venue.
    df.loc[df["is_online"], "delivery_mode"] = cfg.online_label
    df.loc[~df["is_online"], "delivery_mode"] = cfg.in_person_label
    df.loc[df["is_online"], "venue"] = None

    df["venue"] = df["venue"].map(clean_optional)
    df["building"] = df["venue"].map(cfg.building_for).map(clean_optional)
    unmapped_v = sorted(
        df.loc[df["building"] == "Unmapped Building", "venue"].dropna().unique())
    if unmapped_v:
        report.warnings.append(f"venue(s) not in building map: {unmapped_v}")

    # --- optional validity window ------------------------------------
    for col in ("effective_from", "effective_to"):
        if col not in df.columns:
            df[col] = None

    # --- stable session id -------------------------------------------
    df = df.sort_values(
        ["day", "start_time", "course_code", "section"],
        key=lambda s: s.map(cfg.day_index) if s.name == "day" else s,
    ).reset_index(drop=True)
    df["session_id"] = [f"S{ i + 1 :04d}" for i in range(len(df))]

    # --- ambiguity detection -----------------------------------------
    # A course whose sessions cannot be told apart by section is
    # ambiguous: the system must ask, not guess.
    for code, grp in df.groupby("course_code"):
        slots = grp[["day", "start_time"]].drop_duplicates()
        if len(slots) > 1 and grp["section"].nunique() == 1:
            report.ambiguities.append({
                "course_code": code,
                "detail": (f"{len(slots)} distinct sittings but only one "
                           f"section value ({grp['section'].iloc[0]}) — "
                           f"cannot determine which sitting a student attends"),
            })

    # --- exact duplicates --------------------------------------------
    dup_keys = ["day", "start_time", "course_code", "section"]
    dups = df.duplicated(subset=dup_keys, keep=False)
    if dups.any():
        report.warnings.append(
            f"{int(dups.sum())} row(s) share day/time/course/section")

    report.rows_out = len(df)
    ordered = [c for c in CANONICAL_COLUMNS if c in df.columns]
    extra = [c for c in ("start_min", "end_min", "is_core") if c in df.columns]
    return df[ordered + extra], report


def load_config(path: str) -> Config:
    return Config(path)
