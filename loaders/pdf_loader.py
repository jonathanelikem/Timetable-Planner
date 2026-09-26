"""
PDF timetable ingestion.

Published university timetables come in two broad shapes, and this
module handles both:

  ROW LAYOUT   one row per session, with day and time as columns
  GRID LAYOUT  days across the top, time slots down the side, courses
               in the cells — the shape most schools actually publish

Extraction is best-effort and is deliberately NOT trusted. Every parse
returns a diagnostics object alongside the data, and the interface
shows the user what was extracted and asks them to confirm it before
anything is used. That is a design decision, not a limitation: a
silently mis-parsed timetable would produce confident wrong advice
about whether a student can graduate on time, which is worse than no
advice at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
import pdfplumber

# A course code is letters followed by digits, with optional space:
# "ACCT 304", "OMIS308", "UGBS 102". No institution-specific prefixes
# are hardcoded — the pattern is structural.
COURSE_RE = re.compile(r"\b([A-Z]{2,5})\s?(\d{3}[A-Z]?)\b")
TIME_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})\b")
TIME_RANGE_RE = re.compile(
    r"\b(\d{1,2})[:.](\d{2})\s*(?:-|–|—|to)\s*(\d{1,2})[:.](\d{2})\b")
SECTION_RE = re.compile(r"\b(GP\s?\d{1,2}|GROUP\s?\d{1,2}|[A-Z]{1,2}\d?"
                        r"(?=\s*$))\b", re.I)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
DAY_LOOKUP = {d.lower(): d for d in DAY_NAMES}
DAY_LOOKUP.update({d.lower()[:3]: d for d in DAY_NAMES})


@dataclass
class ExtractionReport:
    """What the parser did, so the user can judge whether to trust it."""

    strategy: str = "none"
    pages: int = 0
    tables_found: int = 0
    rows_extracted: int = 0
    days_seen: list[str] = field(default_factory=list)
    times_seen: list[str] = field(default_factory=list)
    courses_seen: int = 0
    warnings: list[str] = field(default_factory=list)
    unparsed_samples: list[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        """A parse worth showing. The user still confirms it."""
        return (self.rows_extracted >= 5
                and len(self.days_seen) >= 2
                and self.courses_seen >= 3)

    def render(self) -> str:
        lines = [
            f"Strategy: {self.strategy}",
            f"Pages read: {self.pages}",
            f"Sessions extracted: {self.rows_extracted}",
            f"Days found: {', '.join(self.days_seen) or 'none'}",
            f"Distinct courses: {self.courses_seen}",
        ]
        lines += [f"Note: {w}" for w in self.warnings]
        return "\n".join(lines)


def _norm_time(h: str, m: str) -> str:
    return f"{int(h):02d}:{int(m):02d}"


def _find_day(text: str) -> str | None:
    low = str(text).lower()
    for token, day in DAY_LOOKUP.items():
        if re.search(rf"\b{token}", low):
            return day
    return None


def _find_courses(text: str) -> list[str]:
    return [f"{m.group(1).upper()} {m.group(2).upper()}"
            for m in COURSE_RE.finditer(str(text).upper())]


def _find_section(text: str) -> str | None:
    m = re.search(r"\b(GP\s?\d{1,2})\b", str(text).upper())
    if m:
        return m.group(1).replace(" ", "")
    m = re.search(r"\bGROUP\s?(\d{1,2})\b", str(text).upper())
    if m:
        return f"GP{m.group(1)}"
    return None


def _find_venue(text: str, courses: list[str], section: str | None) -> str | None:
    """Whatever remains after stripping codes, sections and times."""
    t = str(text).upper()
    for c in courses:
        t = t.replace(c, " ").replace(c.replace(" ", ""), " ")
    if section:
        t = t.replace(section, " ")
    t = TIME_RANGE_RE.sub(" ", t)
    t = TIME_RE.sub(" ", t)
    t = re.sub(r"\b(ONLINE|VIRTUAL|REMOTE)\b", "ONLINE", t)
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    tokens = [w for w in t.split() if 1 <= len(w) <= 8]
    stop = {"AND", "THE", "OF", "TO", "FOR", "LECTURE", "TUTORIAL",
            "CLASS", "HALL", "ROOM", "VENUE", "NAN", "NONE", "NULL", "NA"}
    tokens = [w for w in tokens if w not in stop]
    return " ".join(tokens[:2]) if tokens else None


def _rows_from_row_table(table: list[list], report: ExtractionReport):
    """
    Layout 1: the table already has one session per row.

    Detected by a header row containing recognisable column names.
    """
    if not table or len(table) < 2:
        return []

    header = [str(c or "").strip().lower() for c in table[0]]
    joined = " ".join(header)
    wanted = ("day" in joined
              and ("time" in joined or "start" in joined)
              and ("course" in joined or "code" in joined))
    if not wanted:
        return []

    def col(*names):
        for i, h in enumerate(header):
            if any(n in h for n in names):
                return i
        return None

    idx = {
        "day": col("day"),
        "start": col("start", "from", "time"),
        "end": col("end", "to"),
        "code": col("course code", "code", "course"),
        "name": col("course name", "title", "name"),
        "level": col("level", "year"),
        "section": col("group", "section", "stream"),
        "venue": col("venue", "room", "location"),
        "mode": col("mode", "delivery"),
    }

    rows = []
    for raw in table[1:]:
        cells = [str(c or "").strip() for c in raw]
        if not any(cells):
            continue

        def get(key):
            i = idx[key]
            return cells[i] if i is not None and i < len(cells) else ""

        day = _find_day(get("day"))
        if not day:
            continue

        start = end = None
        rng = TIME_RANGE_RE.search(get("start") + " " + get("end"))
        if rng:
            start = _norm_time(rng.group(1), rng.group(2))
            end = _norm_time(rng.group(3), rng.group(4))
        else:
            s = TIME_RE.search(get("start"))
            e = TIME_RE.search(get("end"))
            if s:
                start = _norm_time(s.group(1), s.group(2))
            if e:
                end = _norm_time(e.group(1), e.group(2))
        if not start:
            continue

        codes = _find_courses(get("code") or " ".join(cells))
        if not codes:
            continue

        rows.append({
            "day": day,
            "start_time": start,
            "end_time": end,
            "course_code": codes[0],
            "course_name": get("name") or codes[0],
            "year_group": _first_int(get("level")),
            "section": get("section") or _find_section(" ".join(cells)),
            "venue": get("venue") or None,
            "delivery_mode": get("mode") or None,
        })

    if rows:
        report.strategy = "table, one row per session"
    return rows


def _rows_from_grid_table(table: list[list], report: ExtractionReport):
    """
    Layout 2: days across the top, time slots down the side.

    This is how most schools publish. Each cell may hold several
    courses; the parser unpivots the grid back into one row per session.
    """
    if not table or len(table) < 2:
        return []

    header = [str(c or "").strip() for c in table[0]]
    day_cols = {}
    for i, cell in enumerate(header):
        day = _find_day(cell)
        if day:
            day_cols[i] = day
    if len(day_cols) < 2:
        return []

    rows = []
    for raw in table[1:]:
        cells = [str(c or "").strip() for c in raw]
        if not any(cells):
            continue

        # The time for this row comes from whichever leading cell holds one.
        start = end = None
        for cell in cells[:2]:
            rng = TIME_RANGE_RE.search(cell)
            if rng:
                start = _norm_time(rng.group(1), rng.group(2))
                end = _norm_time(rng.group(3), rng.group(4))
                break
            t = TIME_RE.findall(cell)
            if len(t) >= 2:
                start, end = _norm_time(*t[0]), _norm_time(*t[1])
                break
            if len(t) == 1 and start is None:
                start = _norm_time(*t[0])
        if not start:
            continue

        for i, day in day_cols.items():
            if i >= len(cells):
                continue
            content = cells[i]
            if not content:
                continue
            for line in re.split(r"[\n;]+", content):
                codes = _find_courses(line)
                if not codes:
                    continue
                section = _find_section(line)
                rows.append({
                    "day": day,
                    "start_time": start,
                    "end_time": end,
                    "course_code": codes[0],
                    "course_name": codes[0],
                    "year_group": None,
                    "section": section,
                    "venue": _find_venue(line, codes, section),
                    "delivery_mode": None,
                })

    if rows:
        report.strategy = "grid, days across the top"
    return rows


def _rows_from_text(page_text: str, report: ExtractionReport):
    """
    Layout 3: no extractable table structure at all.

    Walks the text line by line, tracking the most recent day heading
    and time, and attaching any course codes found to that context.
    The least reliable path, which is why the interface always shows
    the result for confirmation.
    """
    rows = []
    day = None
    start = end = None

    for line in page_text.splitlines():
        line = line.strip()
        if not line:
            continue

        found_day = _find_day(line)
        if found_day and len(line) < 40:
            day = found_day
            continue

        rng = TIME_RANGE_RE.search(line)
        if rng:
            start = _norm_time(rng.group(1), rng.group(2))
            end = _norm_time(rng.group(3), rng.group(4))

        codes = _find_courses(line)
        if day and start and codes:
            section = _find_section(line)
            rows.append({
                "day": day,
                "start_time": start,
                "end_time": end,
                "course_code": codes[0],
                "course_name": codes[0],
                "year_group": None,
                "section": section,
                "venue": _find_venue(line, codes, section),
                "delivery_mode": None,
            })
        elif codes and not day:
            if len(report.unparsed_samples) < 6:
                report.unparsed_samples.append(line[:90])

    if rows:
        report.strategy = "text lines with day headings"
    return rows


def _first_int(text) -> int | None:
    m = re.search(r"\d{3}", str(text))
    return int(m.group()) if m else None


def extract_from_pdf(file_obj, default_year_group: int | None = None,
                     default_duration: int = 110):
    """
    Read a timetable PDF and return (DataFrame, ExtractionReport).

    Tries the structured strategies first and falls back to text
    parsing. The result is raw and unvalidated: it must still pass
    through core.schema.load_timetable before any analysis runs.
    """
    report = ExtractionReport()
    all_rows: list[dict] = []

    with pdfplumber.open(file_obj) as pdf:
        report.pages = len(pdf.pages)
        for page in pdf.pages:
            tables = page.extract_tables() or []
            report.tables_found += len(tables)

            page_rows: list[dict] = []
            for table in tables:
                page_rows += _rows_from_row_table(table, report)
                if not page_rows:
                    page_rows += _rows_from_grid_table(table, report)

            if not page_rows:
                page_rows = _rows_from_text(page.extract_text() or "", report)

            all_rows += page_rows

    if not all_rows:
        report.warnings.append(
            "No sessions could be read from this file. If the PDF is a "
            "scanned image rather than digital text, it needs OCR first.")
        return pd.DataFrame(), report

    df = pd.DataFrame(all_rows).drop_duplicates()

    # Strip the literal strings that stand in for absent values in many
    # exports, so they do not become fake sections or venues.
    for col in ("section", "venue", "delivery_mode", "course_name"):
        df[col] = df[col].apply(
            lambda v: None if v is None or str(v).strip().lower()
            in {"", "nan", "none", "null", "na", "-"} else str(v).strip())

    # Fill what the layout could not carry.
    if df["end_time"].isna().any():
        report.warnings.append(
            f"{int(df['end_time'].isna().sum())} session(s) had no end time; "
            f"assumed {default_duration} minutes.")
        df["end_time"] = df.apply(
            lambda r: r["end_time"] or _add_minutes(r["start_time"],
                                                    default_duration),
            axis=1)

    if df["year_group"].isna().all():
        if default_year_group:
            df["year_group"] = default_year_group
            report.warnings.append(
                f"No year/level column found; all sessions set to "
                f"{default_year_group}. Correct this before use if the "
                f"file covers several cohorts.")
        else:
            report.warnings.append(
                "No year/level found. Set one before loading.")
    else:
        df["year_group"] = df["year_group"].ffill()

    df["course_name"] = df["course_name"].fillna(df["course_code"])

    report.rows_extracted = len(df)
    report.days_seen = sorted(df["day"].dropna().unique(),
                              key=lambda d: DAY_NAMES.index(d))
    report.times_seen = sorted(df["start_time"].dropna().unique())
    report.courses_seen = int(df["course_code"].nunique())

    if df["course_name"].equals(df["course_code"]):
        report.warnings.append(
            "No course titles found in the file; codes are used as titles.")

    return df.reset_index(drop=True), report


def _add_minutes(hhmm: str, minutes: int) -> str:
    h, m = str(hhmm).split(":")
    total = int(h) * 60 + int(m) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"
