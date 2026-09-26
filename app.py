"""
Timetable decision-support prototype — user interface.

This layer renders and routes. Every scheduling decision, score and
ranking shown here is computed by core/ and analytics/. The interface
never re-implements logic, and the routing layer never computes: it
selects which function to call and phrases the result.

Run:  streamlit run app.py
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

from core.schema import load_config, load_timetable, normalise
from core.clash import (conditional_conflicts, find_clashes, integrity_check,
                        mutual_exclusions, transition_check,
                        venue_double_bookings)
from analytics.insights import (common_free_blocks, day_profile,
                                delivery_mix, free_blocks, slot_pressure,
                                venue_utilisation)
from analytics.combinations import explain_choice, rank_combinations
from loaders.pdf_loader import extract_from_pdf
from ai.intent_router import interpret_intent_with_score, interpreter_in_use

st.set_page_config(page_title="Timetable Planner", layout="wide",
                   initial_sidebar_state="expanded")

INK = "#16202F"
PAPER = "#FCFBF9"
PANEL = "#F2F0EB"
SLATE = "#6E7380"
HAIRLINE = "#E2DFD9"
CONFLICT = "#A81F2D"
VIRTUAL = "#2C6A66"
GOOD = "#2E6B45"

DEPT_HUES = {
    "Accounting": "#41618F",
    "Finance": "#2C6A66",
    "Health Services Management": "#7E5177",
    "Marketing & Entrepreneurship": "#9E6B31",
    "Operations & Management Information Systems": "#356B4F",
    "Organisation & Human Resource Management": "#6B5D96",
    "Public Administration": "#96513F",
    "School-wide Core": "#565E6B",
    "Unmapped": SLATE,
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;450;500;600&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    font-variant-numeric: tabular-nums;
}
.stApp { background: __PAPER__; }
.block-container { padding-top: 3.4rem; max-width: 1500px; }

/* Streamlit floats its own toolbar over the top-right of the page, where
   the Deploy button and developer menu covered the title. Those are hidden
   here, but the toolbar itself is kept, because the button that reopens a
   closed sidebar lives inside it. Hiding the whole toolbar takes that
   button away and leaves the sidebar unreachable once it is closed.

   TO BRING THE DEPLOY BUTTON AND MENU BACK: delete the stAppDeployButton
   and stMainMenu rules below. The menu is also where Streamlit keeps its
   light/dark theme switch, so restoring it lets a viewer change the theme
   — which this interface is not designed for, since its colours assume a
   light background. */
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainMenu"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }

/* Keep the sidebar open/close controls reachable at all times. */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999;
}

h1, h2, h3, h4 { color: __INK__; letter-spacing: -0.015em; }
h2 { font-size: 1.05rem; font-weight: 600; margin: 1.9rem 0 0.2rem 0; }
h3 { font-size: 0.95rem; font-weight: 600; margin: 1.4rem 0 0.2rem 0; }

.mast { border-bottom: 2px solid __INK__; padding-bottom: 0.7rem;
        margin-bottom: 1.3rem; }
.mast .name { font-size: 1.55rem; font-weight: 600; color: __INK__;
              letter-spacing: -0.025em; line-height: 1.1; }
.mast .org { font-size: 0.82rem; color: __SLATE__; margin-top: 0.25rem; }

.lede { color: __SLATE__; font-size: 0.9rem; line-height: 1.6; max-width: 64ch; }
.sub { color: __SLATE__; font-size: 0.84rem; line-height: 1.6; max-width: 74ch; }

.verdict { font-size: 1.18rem; color: __INK__; line-height: 1.45;
           padding: 0.55rem 0 0.85rem 0; max-width: 64ch; font-weight: 450; }
.verdict b { font-weight: 600; }
.verdict.bad { color: __CONFLICT__; }
.verdict.good b { color: __GOOD__; }

.figs { display: flex; gap: 2.4rem; padding: 0.4rem 0 0.9rem 0; flex-wrap: wrap; }
.fig .n { font-size: 1.6rem; font-weight: 600; color: __INK__; line-height: 1.1; }
.fig .n.alert { color: __CONFLICT__; }
.fig .k { font-size: 0.76rem; color: __SLATE__; margin-top: 0.1rem;
          max-width: 18ch; line-height: 1.35; }

table.tt { width: 100%; border-collapse: collapse; table-layout: fixed;
           margin-top: 0.3rem; }
table.tt th { font-size: 0.78rem; font-weight: 500; color: __INK__;
              text-align: left; padding: 0 0 7px 9px;
              border-bottom: 1.5px solid __INK__; }
table.tt th.time { width: 62px; border-bottom: 1.5px solid __INK__; }
table.tt td { border-bottom: 1px solid __HAIRLINE__;
              border-right: 1px solid __HAIRLINE__;
              height: 60px; vertical-align: top; padding: 3px; width: 19.2%;
              overflow: hidden; }
table.tt td.time { border-right: 1.5px solid __HAIRLINE__; border-bottom: none;
                   text-align: right; padding: 5px 11px 0 0; color: __SLATE__;
                   font-size: 0.72rem; width: 62px; line-height: 1.3; }
.blk { border-left: 3px solid; border-radius: 1px; padding: 5px 8px;
       min-height: 50px; box-sizing: border-box; overflow: hidden;
       margin-bottom: 3px; }
.blk:last-child { margin-bottom: 0; }
.blk .code { font-weight: 600; font-size: 0.79rem; color: __INK__;
             letter-spacing: -0.01em; }
.blk .meta { font-size: 0.69rem; color: __SLATE__; margin-top: 1px; line-height: 1.3; }
.blk.clash { border-left-color: __CONFLICT__ !important;
    background: repeating-linear-gradient(135deg,
        #FAE7E9, #FAE7E9 5px, #F3DADC 5px, #F3DADC 10px) !important; }
.blk.clash .code { color: __CONFLICT__; }
.blk.virtual .code::after { content: " · online"; font-weight: 400;
    color: __VIRTUAL__; font-size: 0.68rem; letter-spacing: 0; }

.legend { font-size: 0.73rem; color: __SLATE__; margin-top: 11px; line-height: 1.9; }
.legend span { margin-right: 15px; white-space: nowrap; }
.swatch { display: inline-block; width: 9px; height: 9px; border-radius: 1px;
          margin-right: 5px; vertical-align: middle; }

.note { border-left: 3px solid __SLATE__; background: #F5F4F0;
        padding: 0.6rem 0.85rem; font-size: 0.83rem; color: __INK__;
        line-height: 1.55; margin: 0.6rem 0; max-width: 80ch; }
.note.warn { border-left-color: #9E6B31; background: #FBF5EC; }
.note.bad { border-left-color: __CONFLICT__; background: #FBEFF0; }

section[data-testid="stSidebar"] { background: __PANEL__;
    border-right: 1px solid __HAIRLINE__; }
section[data-testid="stSidebar"] h3 { font-size: 0.8rem; font-weight: 600;
    color: __INK__; margin: 1.2rem 0 0.3rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem;
    border-bottom: 1px solid __HAIRLINE__; }
.stTabs [data-baseweb="tab"] { font-size: 0.88rem; font-weight: 450;
    padding: 0.4rem 0; }
.stDataFrame { font-size: 0.83rem; }
div[data-testid="stExpander"] details { border: 1px solid __HAIRLINE__;
    border-radius: 2px; background: #FAF9F6; }
</style>
"""
for token, value in [("__INK__", INK), ("__PAPER__", PAPER),
                     ("__PANEL__", PANEL), ("__SLATE__", SLATE),
                     ("__HAIRLINE__", HAIRLINE), ("__CONFLICT__", CONFLICT),
                     ("__VIRTUAL__", VIRTUAL), ("__GOOD__", GOOD)]:
    CSS = CSS.replace(token, value)
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_config(path: str):
    return load_config(path)


@st.cache_data(show_spinner=False)
def get_file_timetable(data_path: str, cfg_path: str):
    cfg = load_config(cfg_path)
    return load_timetable(data_path, cfg)


def figs(items):
    """A row of figures. Third element, if true, marks the value as alert."""
    html = '<div class="figs">'
    for item in items:
        value, label = item[0], item[1]
        alert = len(item) > 2 and item[2]
        html += (f'<div class="fig"><div class="n{" alert" if alert else ""}">'
                 f'{value}</div><div class="k">{label}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def note(text: str, kind: str = ""):
    st.markdown(f'<div class="note {kind}">{text}</div>',
                unsafe_allow_html=True)


@contextmanager
def guard(what: str):
    """
    Keep one failing panel from taking the whole app down.

    Streamlit aborts the entire script run on an uncaught exception, so
    a problem in one tab blanks every other tab too. Wrapping each panel
    means a failure stays local and the user is told what to do about it
    in plain language rather than shown a stack trace.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the UI edge
        note(f"Something went wrong while {what}. You can carry on using "
             f"the rest of the planner — try changing your selection above. "
             f"<br><br><span style='font-size:0.78rem;color:#6E7380'>"
             f"Technical detail, for whoever maintains this: "
             f"{type(exc).__name__}: {exc}</span>", "bad")


def listify(items) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def block_html(rows, clashing, cfg) -> str:
    parts = []
    for r in rows:
        hue = DEPT_HUES.get(r["department"], SLATE)
        classes = "blk"
        if r["session_id"] in clashing:
            classes += " clash"
        if r["is_online"]:
            classes += " virtual"
        where = cfg.online_label if r["is_online"] else (r["venue"] or "—")
        section = ("" if r["section"] in (cfg.section_na, None)
                   else f" · {r['section']}")
        time = f'{r["start_time"]}\u2013{r["end_time"]}'
        parts.append(
            f'<div class="{classes}" style="border-left-color:{hue};'
            f'background:{hue}12;">'
            f'<div class="code">{r["course_code"]}</div>'
            f'<div class="meta">{time} \u00b7 {where}{section}</div></div>')
    return "".join(parts)


def render_grid(sessions: pd.DataFrame, cfg, clashing=None):
    """
    The weekly grid — the interface's primary object.

    Conflicts are drawn in place, hatched, in the slot where they occur.
    A student cannot see a clash positionally in a published timetable;
    showing it anywhere other than in the grid would repeat that failure.

    A session whose start time does not exactly match a configured slot
    is placed in the slot that contains it rather than disappearing, and
    anything that cannot be placed at all is listed underneath. A class
    silently missing from the grid is the worst possible failure here,
    because the student would plan around a week that is not their own.
    """
    clashing = clashing or set()
    days = cfg.teaching_days
    display = sessions.copy()

    def display_slot(start_time):
        start_time = str(start_time)
        for slot in cfg.slots:
            if slot["start"] == start_time:
                return slot["start"]
        containing = [slot for slot in cfg.slots
                      if slot["start"] <= start_time < slot["end"]]
        if containing:
            return max(containing, key=lambda slot: slot["start"])["start"]
        return None

    display["_display_slot"] = display["start_time"].apply(display_slot)

    html = '<table class="tt"><tr><th class="time"></th>'
    html += "".join(f"<th>{d}</th>" for d in days) + "</tr>"
    for slot in cfg.slots:
        html += f'<tr><td class="time">{slot["start"]}<br>{slot["end"]}</td>'
        for day in days:
            cell = display[(display["day"] == day)
                           & (display["_display_slot"] == slot["start"])]
            html += "<td>"
            if not cell.empty:
                html += block_html(cell.to_dict("records"), clashing, cfg)
            html += "</td>"
        html += "</tr>"
    html += "</table>"

    legend = '<div class="legend">'
    for d in sorted(display["department"].dropna().unique()):
        hue = DEPT_HUES.get(d, SLATE)
        short = d if len(d) < 36 else d.split("&")[0].strip()
        legend += (f'<span><i class="swatch" style="background:{hue}"></i>'
                   f'{short}</span>')
    if clashing:
        legend += (f'<span><i class="swatch" style="background:{CONFLICT}">'
                   f'</i>clash</span>')
    st.markdown(html + legend + "</div>", unsafe_allow_html=True)

    unplaced = display[display["_display_slot"].isna()]
    if not unplaced.empty:
        note(f"{len(unplaced)} class"
             f"{'es do' if len(unplaced) != 1 else ' does'} not fit the "
             f"normal timetable grid, so "
             f"{'they are' if len(unplaced) != 1 else 'it is'} listed here "
             f"rather than hidden.", "warn")
        st.dataframe(
            unplaced[["day", "course_code", "start_time", "end_time",
                      "venue", "section"]],
            use_container_width=True, hide_index=True)


def course_options(df: pd.DataFrame, level=None, dept=None):
    sel = df
    if level is not None:
        sel = sel[sel["year_group"] == level]
    if dept and dept != "All departments":
        sel = sel[sel["department"] == dept]
    labels = (sel[["course_code", "course_name"]].drop_duplicates()
              .sort_values("course_code"))
    return [f"{r.course_code} — {r.course_name}" for r in labels.itertuples()]


def codes_from(labels):
    return [l.split(" — ")[0] for l in labels]


# ---------------------------------------------------------------------
# Sidebar: configuration and data source
# ---------------------------------------------------------------------
def _config_order(path: Path):
    """Real deployments first, demonstration configs last.

    Plain alphabetical order put example_alt.yaml ahead of ugbs.yaml, so
    the planner opened showing a different institution than the loaded
    timetable belonged to. Anything named "example" is a demonstration of
    portability rather than a deployment, so it sorts to the end.
    """
    return (path.stem.lower().startswith("example"), path.stem.lower())


# Anchored to this file rather than the working directory. Streamlit
# Community Cloud runs from the repository root, so a bare Path("config")
# would resolve against the wrong folder and the planner would start with
# nothing to load.
BASE = Path(__file__).resolve().parent

CONFIGS = {p.stem: str(p)
           for p in sorted((BASE / "config").glob("*.yaml"),
                           key=_config_order)}
DATASETS = {p.name: str(p) for p in sorted((BASE / "data").glob("*.xlsx"))}

if not CONFIGS:
    st.error(
        "No institution configuration found. The planner expects a "
        "'config' folder containing at least one .yaml file, in the same "
        "folder as app.py. If this is a fresh deployment, check that the "
        "config folder was included in the upload.")
    st.stop()

with st.sidebar:
    st.markdown("### Step 1 · Which school")
    cfg_key = st.selectbox(
        "Configuration", list(CONFIGS), index=0, label_visibility="collapsed",
        help="Teaching days, time slots, department names, venue map and "
             "scoring weights all live in a configuration file. Switching "
             "this adapts the whole planner without changing any code.")
    cfg = get_config(CONFIGS[cfg_key])

    st.markdown("### Step 2 · Load the timetable")
    source_kind = st.radio("Source",
                           ["Use prepared data", "Upload a timetable"],
                           label_visibility="collapsed")

st.markdown(
    f'<div class="mast"><div class="name">Timetable planner</div>'
    f'<div class="org">{cfg.name} · '
    f'{cfg.raw["institution"]["academic_period"]} '
    f'{cfg.raw["institution"]["academic_year"]}</div></div>',
    unsafe_allow_html=True)

df = report = None

if source_kind == "Upload a timetable":
    with st.sidebar:
        upload = st.file_uploader("Timetable file",
                                  type=["pdf", "xlsx", "xls", "csv"],
                                  label_visibility="collapsed")
        default_level = st.number_input(
            f"Default {cfg.year_group_label.lower()} if the file does not "
            f"state one", min_value=100, max_value=900, value=400, step=100)

    if upload is None and "confirmed_df" not in st.session_state:
        st.markdown(
            '<div class="lede">Upload the published timetable to begin. '
            'PDF, Excel and CSV are accepted. Nothing is analysed until you '
            'have seen what was read from the file and confirmed it.</div>',
            unsafe_allow_html=True)
        note("The planner can read PDFs where you can select the text. It "
             "cannot read scans or photographs of a timetable — save those "
             "as an Excel file instead.")
        st.stop()

    if upload is not None:
        sig = f"{upload.name}:{upload.size}"
        if st.session_state.get("upload_sig") != sig:
            with st.spinner("Reading the file"):
                name = upload.name.lower()
                if name.endswith(".pdf"):
                    raw, xrep = extract_from_pdf(
                        upload, default_year_group=int(default_level))
                elif name.endswith(".csv"):
                    raw, xrep = pd.read_csv(upload), None
                else:
                    raw, xrep = pd.read_excel(upload), None
            st.session_state.update(raw_df=raw, xrep=xrep, upload_sig=sig)
            st.session_state.pop("confirmed_df", None)

    if "confirmed_df" not in st.session_state:
        raw = st.session_state.get("raw_df")
        xrep = st.session_state.get("xrep")

        st.markdown("## Check what was read")
        st.markdown(
            '<div class="lede">Reading a timetable out of a PDF is not '
            'always perfect, so nothing is used until you have had a look. '
            'Fix anything that came out wrong, then press the button.</div>',
            unsafe_allow_html=True)

        if raw is None or raw.empty:
            note("Nothing could be read from that file. If it is a scan or "
                 "a photograph, the planner cannot read the text. Try a PDF "
                 "you can select text in, or save the timetable as an Excel "
                 "file and upload that instead.", "bad")
            st.stop()

        if xrep:
            figs([(xrep.rows_extracted, "sessions read"),
                  (xrep.courses_seen, "distinct courses"),
                  (len(xrep.days_seen), "teaching days found"),
                  (xrep.pages, "pages read")])
            st.markdown(f'<div class="sub">Timetable layout recognised: '
                        f'<b>{xrep.strategy}</b></div>',
                        unsafe_allow_html=True)
            for w in xrep.warnings:
                note(w, "warn")

        edited = st.data_editor(raw, use_container_width=True, height=340,
                                num_rows="dynamic", key="editor")
        if st.button("This looks right — use it", type="primary"):
            st.session_state["confirmed_df"] = edited
            st.rerun()
        st.stop()

    df, report = normalise(st.session_state["confirmed_df"], cfg)
    with st.sidebar:
        if st.button("Upload a different file"):
            for k in ("confirmed_df", "raw_df", "upload_sig", "xrep"):
                st.session_state.pop(k, None)
            st.rerun()
else:
    with st.sidebar:
        data_key = st.selectbox("Dataset", list(DATASETS), index=0,
                                label_visibility="collapsed")
    df, report = get_file_timetable(DATASETS[data_key], CONFIGS[cfg_key])

if df is None or df.empty:
    note("No usable sessions after validation. Check the source file.", "bad")
    st.stop()

with st.sidebar:
    st.caption(f"{report.rows_out} classes loaded")

    if report.warnings or report.ambiguities:
        n = len(report.warnings) + len(report.ambiguities)
        with st.expander(f"Reading the file — {n} note{'s' if n != 1 else ''}"):
            st.caption("How the timetable was read into the planner.")
            for w in report.warnings:
                st.caption(w)
            for a in report.ambiguities:
                st.caption(f"{a['course_code']}: {a['detail']}")

    issues = []
    with guard("checking the timetable"):
        issues = integrity_check(df, cfg)
    errors = [i for i in issues if i["severity"] == "error"]
    label = ("Timetable checks — all clear" if not issues
             else f"Timetable checks — {len(issues)} "
                  f"finding{'s' if len(issues) != 1 else ''}")
    with st.expander(label):
        if not issues:
            st.caption("Every check passed. No room is booked twice, no "
                       "class is listed twice, and every class sits on the "
                       "published timetable grid.")
        else:
            if errors:
                st.caption(f"{len(errors)} of these look like genuine "
                           f"scheduling errors.")
            for i in issues:
                st.caption(f"{i['check']}: {i['detail']}")

    st.markdown("### Step 3 · About you")
    levels = sorted(df["year_group"].dropna().unique())
    level = st.selectbox(cfg.year_group_label, levels, index=len(levels) - 1)
    depts = ["All departments"] + sorted(
        df[df["year_group"] == level]["department"].unique())
    dept = st.selectbox("Department", depts)

tab_check, tab_plan, tab_insight, tab_ask, tab_group = st.tabs(
    ["1 · Check my courses", "2 · Compare my options",
     "3 · What clashes with what", "4 · Ask a question",
     "5 · Group meeting time"])


# ---------------------------------------------------------------------
with tab_check:
    opts = course_options(df, level, dept)
    st.markdown(
        '<div class="lede">Pick the courses you are planning to register '
        'for. The planner checks whether they clash and shows you what your '
        'week would look like.</div>', unsafe_allow_html=True)
    picked = st.multiselect("Your courses", opts,
                            default=opts[:4] if len(opts) >= 4 else opts)
    codes = codes_from(picked)

    if not codes:
        note("Pick one or more courses above to get started.")
    else:
        sel = df[df["course_code"].isin(codes)]

        multi = {c: sorted(g["section"].unique())
                 for c, g in sel.groupby("course_code")
                 if g["section"].nunique() > 1}
        if multi:
            st.markdown("### Which group are you in?")
            st.markdown(
                '<div class="sub">These courses are taught more than once, '
                'to different groups. Tell the planner which group you are '
                'in so it checks the right class times.</div>',
                unsafe_allow_html=True)
            chosen = {}
            cols = st.columns(min(len(multi), 4))
            for i, (code, sections) in enumerate(multi.items()):
                with cols[i % len(cols)]:
                    chosen[code] = st.selectbox(code, sections,
                                                key=f"sec_{code}")
            sel = sel[sel.apply(
                lambda r: (r["course_code"] not in chosen
                           or r["section"] == chosen[r["course_code"]]),
                axis=1)]

        clashes = find_clashes(sel)
        tight = transition_check(sel, cfg)
        profile = day_profile(sel, cfg)


        clashing_ids = set()
        for c in clashes:
            for _, r in sel.iterrows():
                if (r["day"] == c["day"]
                        and r["course_code"] in (c["course_a"],
                                                 c["course_b"])):
                    clashing_ids.add(r["session_id"])

        free_days = profile[profile["sessions"] == 0]["day"].tolist()
        campus_days = int(profile["campus_required"].sum())
        lvc = profile[profile["low_value_commute"]]["day"].tolist()

        if clashes:
            first = clashes[0]
            extra = (f", and {len(clashes) - 1} other conflict"
                     f"{'s' if len(clashes) > 2 else ''}"
                     if len(clashes) > 1 else "")
            st.markdown(
                f'<div class="verdict bad">This selection will not work. '
                f'<b>{first["course_a"]}</b> and <b>{first["course_b"]}</b> '
                f'both run {first["day"]} {first["time_a"]}{extra}.</div>',
                unsafe_allow_html=True)
        else:
            tail = (f"{listify(free_days)} "
                    f"{'stay' if len(free_days) > 1 else 'stays'} clear."
                    if free_days
                    else f"You are on campus {campus_days} days a week.")
            st.markdown(f'<div class="verdict good">These {len(codes)} '
                        f'courses <b>fit</b>. {tail}</div>',
                        unsafe_allow_html=True)

        figs([(len(clashes), "clashes", bool(clashes)),
              (campus_days, "days you need to be on campus"),
              (f"{profile['contact_hours'].sum():.1f}",
               "hours of class a week"),
              (len(lvc), "days you come in for one class only", bool(lvc))])

        render_grid(sel, cfg, clashing_ids)

        if clashes:
            st.markdown("### Which courses clash")
            st.dataframe(
                pd.DataFrame(clashes)[
                    ["day", "course_a", "time_a", "course_b", "time_b",
                     "overlap_minutes"]
                ].rename(columns={
                    "day": "Day", "course_a": "This course",
                    "time_a": "Runs", "course_b": "Clashes with",
                    "time_b": "Which runs",
                    "overlap_minutes": "Minutes they overlap"}),
                use_container_width=True, hide_index=True)

        if tight:
            st.markdown("### You may not get there in time")
            for t in tight:
                note(f'<b>{t["day"]}</b>: {t["from_course"]} '
                     f'({t["from_venue"]}) to {t["to_course"]} '
                     f'({t["to_venue"]}) — {t["gap_minutes"]} minutes to '
                     f'make a {t["walk_minutes"]}-minute walk.', "warn")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("### Your week, day by day")
            st.dataframe(
                profile[["day", "sessions", "contact_hours",
                         "in_person_sessions", "online_sessions",
                         "idle_hours", "congestion_score"]].rename(columns={
                    "day": "Day", "sessions": "Classes",
                    "contact_hours": "Hours of class",
                    "in_person_sessions": "On campus",
                    "online_sessions": "Online",
                    "idle_hours": "Hours waiting between",
                    "congestion_score": "How heavy (lower is lighter)"}),
                use_container_width=True, hide_index=True)
            if lvc:
                note(f"You travel to campus for a single class on "
                     f"{listify(lvc)}. Moving one of those sessions would "
                     f"free the whole day.")
        with right:
            st.markdown("### Free time you can actually use")
            st.dataframe(
                free_blocks(sel, cfg).rename(columns={
                    "day": "Day", "start": "From", "end": "Until",
                    "minutes": "Minutes", "type": "When"}),
                use_container_width=True, hide_index=True)
            note(f"Only gaps of {cfg.min_usable_gap} minutes or more are "
                 f"shown. Anything shorter is not enough time to be useful.")


# ---------------------------------------------------------------------
with tab_plan:
    st.markdown(
        '<div class="lede">Some courses cannot be taken together because '
        'they run at the same time. Tell the planner which courses are '
        'fixed and which you are still choosing between, and it will work '
        'out every combination that actually fits — then put the best ones '
        'first.</div>', unsafe_allow_html=True)

    opts = course_options(df, level)
    c1, c2 = st.columns(2)
    with c1:
        core_labels = st.multiselect(
            "Courses you have to take", opts,
            help="Your compulsory courses. Leave empty if you are choosing "
                 "everything freely.")
    with c2:
        cand_labels = st.multiselect(
            "Courses you are choosing between",
            [o for o in opts if o not in core_labels],
            help="The optional courses you are deciding between. The "
                 "planner compares every way of choosing from these.")

    core = codes_from(core_labels)
    cands = [c for c in codes_from(cand_labels) if c not in core]

    if not cands:
        note("Add the courses you are deciding between in the box on the "
             "right, and the planner will work out which of them you can "
             "actually combine.")
    else:
        choose = st.number_input(
            "How many of those will you actually take?",
            min_value=1, max_value=len(cands),
            value=min(3, len(cands)), step=1)

        # Where a course is taught more than once, the student must say
        # which sitting they attend. Without this the ranking silently
        # assumes one, and a recommendation built on the wrong sitting is
        # worse than no recommendation.
        sections = {}
        multi = {}
        for code in core + cands:
            available = sorted(
                df[df["course_code"] == code]["section"].dropna().unique())
            if len(available) > 1:
                multi[code] = available

        if multi:
            st.markdown("### Which group are you in?")
            st.markdown(
                '<div class="sub">These courses are taught more than once, '
                'to different groups. Tell the planner which group you '
                'would be in.</div>', unsafe_allow_html=True)
            sec_cols = st.columns(min(len(multi), 4))
            for i, (code, available) in enumerate(multi.items()):
                with sec_cols[i % len(sec_cols)]:
                    sections[code] = st.selectbox(code, available,
                                                  key=f"plan_sec_{code}")

        if st.button("Show me which combinations work", type="primary"):
            with guard("comparing your options"):
                res = rank_combinations(
                    df, cfg, core_codes=core, candidate_codes=cands,
                    choose=int(choose), sections=sections or None)

                if not res["any_feasible"]:
                    st.markdown(
                        f'<div class="verdict bad">None of these '
                        f'{res["total_combinations"]} combinations will '
                        f'work.</div>', unsafe_allow_html=True)
                    note("Every option has a timetable clash. Here is "
                         "what is causing it:", "bad")
                    for _, r in res["rejected_sample"].head(5).iterrows():
                        st.markdown(
                            f'<div class="sub">· {r["reason"]}</div>',
                            unsafe_allow_html=True)
                    note("<b>What to do:</b> remove one of the courses "
                         "named above from your list, or plan to take it "
                         "in a later semester.")
                else:
                    ranked = res["ranked"]
                    best_row = ranked.iloc[0]
                    st.markdown(
                        f'<div class="verdict"><b>'
                        f'{res["feasible_count"]}</b> of '
                        f'{res["total_combinations"]} combinations fit '
                        f'your timetable.</div>', unsafe_allow_html=True)
                    figs([
                        (res["feasible_count"], "combinations that work"),
                        (res["rejected_count"],
                         "ruled out because of clashes", True),
                        (int(best_row["campus_days"]),
                         "days on campus, best option"),
                    ])

                    st.markdown("### Our recommendation")
                    st.markdown(
                        f'<div class="verdict good">Take <b>'
                        f'{best_row["combination"]}</b>.</div>',
                        unsafe_allow_html=True)
                    best = core + best_row["courses"]
                    best_sessions = df[df["course_code"].isin(best)]
                    if sections:
                        best_sessions = best_sessions[
                            best_sessions.apply(
                                lambda r: (r["course_code"] not in sections
                                           or r["section"]
                                           == sections[r["course_code"]]),
                                axis=1)]
                    render_grid(best_sessions, cfg)
                    st.text(explain_choice(df, cfg, best,
                                           sections=sections or None))

                    st.markdown("### All the options that work")
                    display = ranked.rename(columns={
                        "combination": "Courses",
                        "campus_days": "Days on campus",
                        "low_value_commute_days":
                            "Trips for one class only",
                        "mean_congestion": "Average day load",
                        "peak_congestion": "Busiest day load",
                        "idle_hours": "Hours waiting between classes",
                        "score": "Overall (lower is better)"})
                    st.dataframe(
                        display[["Courses", "Days on campus",
                                 "Trips for one class only",
                                 "Hours waiting between classes",
                                 "Overall (lower is better)"]],
                        use_container_width=True, hide_index=True)
                    note("Options are ordered best first. The overall "
                         "figure combines how many days you have to come "
                         "in, how many of those are for a single class, "
                         "how heavy your worst day is, and how long you "
                         "spend waiting between classes.")

                    if not res["rejected_sample"].empty:
                        with st.expander(
                                "Why the other options were ruled out"):
                            for _, r in res["rejected_sample"].iterrows():
                                st.markdown(
                                    f'<div class="sub">· <b>'
                                    f'{r["combination"]}</b> — '
                                    f'{r["reason"]}</div>',
                                    unsafe_allow_html=True)


# ---------------------------------------------------------------------
with tab_insight:
    st.markdown(
        '<div class="lede">Which courses can and cannot go together this '
        'semester. Worth checking before you settle on your choices.</div>',
        unsafe_allow_html=True)

    strict = cond = doubles = []
    with guard("looking for clashes across the timetable"):
        strict = mutual_exclusions(df, level)
        cond = conditional_conflicts(df, level)
        doubles = venue_double_bookings(df)

    figs([(len(strict), "pairs of courses you cannot take together",
           bool(strict)),
          (len(cond), "pairs that work only if you are in the right group"),
          (len(doubles), "rooms booked twice at once", bool(doubles))])

    if not doubles:
        note("The planner checks every timetable it loads for scheduling "
             "errors — rooms booked twice, classes listed twice, classes "
             "outside the normal grid. <b>This timetable passed.</b> That "
             "is worth knowing: it means the difficulty students face is "
             "not a faulty timetable, it is working out which courses they "
             "can combine. The checks still run every time, because a "
             "different semester or a different School may not pass them.")
    else:
        note(f"The planner found {len(doubles)} room booked twice at the "
             f"same time. This is a scheduling error worth reporting to the "
             f"timetable office.", "bad")

    if strict:
        st.markdown("### Courses you cannot take together")
        st.dataframe(
            pd.DataFrame(strict)[
                ["course_a", "dept_a", "course_b", "dept_b", "day", "time"]
            ].rename(columns={
                "course_a": "This course", "dept_a": "Its department",
                "course_b": "Cannot go with", "dept_b": "That department",
                "day": "Day", "time": "Time"}),
            use_container_width=True, hide_index=True)

    if cond:
        st.markdown("### Possible together, but only in certain groups")
        st.dataframe(
            pd.DataFrame(cond)[
                ["course_a", "course_b", "conflicting_pairings",
                 "total_pairings", "workable_example"]
            ].rename(columns={
                "course_a": "This course", "course_b": "And this one",
                "conflicting_pairings": "Group pairings that clash",
                "total_pairings": "Possible pairings",
                "workable_example": "A combination that works"}),
            use_container_width=True, hide_index=True)

    st.markdown("### Times when several courses run at once")
    st.dataframe(slot_pressure(df, cfg, level).head(12),
                 use_container_width=True, hide_index=True)

    st.markdown("### Checks on the timetable itself")
    tt_issues = []
    with guard("checking the timetable"):
        tt_issues = integrity_check(df, cfg)
    if not tt_issues:
        st.markdown(
            '<div class="sub">All checks passed — no room booked twice, no '
            'duplicated class, no class outside the published grid, no '
            'in-person class without a room.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(
            pd.DataFrame(tt_issues).rename(columns={
                "severity": "Type", "check": "Check", "detail": "What we found"}),
            use_container_width=True, hide_index=True)
        note("Findings marked <b>info</b> are worth a look rather than "
             "definitely wrong. Findings marked <b>error</b> are scheduling "
             "problems that should be reported to the timetable office.")

    with st.expander("How busy each room is (for the timetable office)"):
        with guard("summarising room use"):
            st.dataframe(venue_utilisation(df, cfg),
                         use_container_width=True, hide_index=True)
        note("This one is for whoever builds the timetable, not for students "
             "choosing courses.")


# ---------------------------------------------------------------------
def route(question: str, frame: pd.DataFrame, cfg):
    """
    Work out what the student is asking, then answer it by calculation.

    Three layers, each falling back into the next:

      1. A semantic interpreter (ai/intent_router.py) matches the question
         against example phrasings, so wording it has never seen still
         resolves to the right intent.
      2. If that is not confident, keyword rules take over.
      3. If neither settles it, the planner says what it can do instead of
         guessing.

    The interpreter only ever chooses WHICH function runs. Every figure in
    every answer below is computed by core/ or analytics/. That boundary is
    the point: a misrouted question is obvious to the user and easily
    corrected, whereas a miscalculated clash is invisible until
    registration has closed.
    """
    q = question.lower().strip()

    # --- which courses is the question about? ------------------------
    extracted = [re.sub(r"([A-Z]{2,5})(\d{3})", r"\1 \2", c.upper())
                 for c in re.findall(r"\b([a-zA-Z]{2,5}\s?\d{3})\b", question)]
    known = set(frame["course_code"])
    codes = [c for c in extracted if c in known]
    unknown = [c for c in extracted if c not in known]

    # A code that is not in the loaded timetable is reported rather than
    # quietly ignored, because silently dropping it would answer a
    # different question than the one asked.
    if unknown:
        return ("course not found",
                f"I could not find {listify(unknown)} in this timetable. "
                f"Check the code, or switch to the {cfg.year_group_label.lower()} "
                f"that runs it.")

    # --- layer 1: semantic interpretation ----------------------------
    intent, score = interpret_intent_with_score(question)
    used_ai = intent is not None

    # --- layer 2: keyword rules --------------------------------------
    if intent is None:
        if any(w in q for w in ("clash", "conflict", "together", "both",
                                "same time", "overlap")):
            intent = "clash_check"
        elif any(w in q for w in ("free", "gap", "break", "spare",
                                  "available")):
            intent = "free_time"
        elif any(w in q for w in ("lightest", "least busy", "quietest",
                                  "easiest day", "least congested")):
            intent = "workload_lightest"
        elif any(w in q for w in ("busy", "heav", "congest", "worst day")):
            intent = "workload_heaviest"
        elif any(w in q for w in ("campus", "online", "come in", "travel",
                                  "commute", "in person", "in-person")):
            intent = "campus_days"
        elif codes:
            intent = "course_lookup"
        else:
            intent = "unknown"

    label = (f"understood your wording ({score:.0%} match)" if used_ai
             else "matched on keywords")

    sel = frame[frame["course_code"].isin(codes)]

    # --- clash check --------------------------------------------------
    if intent == "clash_check":
        if len(codes) < 2:
            return label, ("Name at least two courses and I will check "
                           "whether they clash.")
        found = find_clashes(sel)
        if not found:
            return label, (f"{listify(codes)} do not clash. You can take "
                           f"them together.")
        return label, " ".join(
            f"{c['course_a']} and {c['course_b']} both run {c['day']} "
            f"{c['time_a']}." for c in found)

    # --- free time ----------------------------------------------------
    if intent == "free_time":
        if not codes:
            return label, "Tell me which courses you are taking."
        fb = free_blocks(sel, cfg)
        if fb.empty:
            return label, "That schedule leaves no usable gaps."
        top = fb.sort_values("minutes", ascending=False).iloc[0]
        if top["type"] == "free day":
            days = fb[fb["type"] == "free day"]["day"].tolist()
            others = fb[fb["type"] != "free day"]
            extra = ""
            if not others.empty:
                n = others.sort_values("minutes", ascending=False).iloc[0]
                extra = (f" Your longest gap on a teaching day is "
                         f"{n['day']} {n['start']}\u2013{n['end']}, "
                         f"{int(n['minutes'])} minutes.")
            return label, (f"{listify(days)} "
                           f"{'are' if len(days) > 1 else 'is'} completely "
                           f"free.{extra}")
        return label, (f"Your longest usable block is {top['day']} "
                       f"{top['start']}\u2013{top['end']}, "
                       f"{int(top['minutes'])} minutes.")

    # --- heaviest / lightest day --------------------------------------
    if intent in ("workload_heaviest", "workload_lightest"):
        if not codes:
            return label, "Tell me which courses you are taking."
        prof = day_profile(sel, cfg)
        active = prof[prof["sessions"] > 0]
        if active.empty:
            return label, "Those courses have no scheduled classes."
        heaviest = intent == "workload_heaviest"
        row = active.loc[active["congestion_score"].idxmax() if heaviest
                         else active["congestion_score"].idxmin()]
        n = int(row["sessions"])
        word = "heaviest" if heaviest else "lightest teaching"
        return label, (f"{row['day']} is your {word} day: {n} "
                       f"class{'es' if n != 1 else ''}, "
                       f"{row['contact_hours']} hours.")

    # --- campus days ---------------------------------------------------
    if intent == "campus_days":
        if not codes:
            return label, "Tell me which courses you are taking."
        mix = delivery_mix(sel, cfg)
        need = mix[mix["campus_required"]]["day"].tolist()
        lvc = mix[mix["low_value_commute"]]["day"].tolist()
        msg = (f"You need to be on campus on {listify(need)}."
               if need else "Nothing requires you on campus.")
        if lvc:
            msg += (f" {listify(lvc)} {'are' if len(lvc) > 1 else 'is'} a "
                    f"single in-person class each \u2014 a whole trip for "
                    f"one class.")
        return label, msg

    # --- course lookup --------------------------------------------------
    if intent == "course_lookup":
        if not codes:
            return label, "Tell me which course you want to look up."
        return label, " ".join(
            f"{r.course_code} runs {r.day} {r.start_time}\u2013{r.end_time}"
            f"{'' if r.is_online else f' in {r.venue}'}"
            f"{'' if r.section == cfg.section_na else f' ({r.section})'}."
            for r in list(sel.itertuples())[:6])

    return (label,
            "I can check whether courses clash, find your free time, tell "
            "you which day is heaviest or lightest, say which days need you "
            "on campus, or look up when a course runs. Include the course "
            "codes.")


with tab_ask:
    st.markdown(
        '<div class="lede">Type a question the way you would say it. '
        'Mention the course codes and the planner will work out the '
        'answer.</div>', unsafe_allow_html=True)

    s_codes = sorted(df[df["year_group"] == level]["course_code"].unique())[:3]
    examples = []
    if len(s_codes) > 1:
        examples += [
            f"Can I take {s_codes[0]} and {s_codes[1]} together?",
            f"When am I free if I take {s_codes[0]} and {s_codes[1]}?",
            f"Which days do I need to be on campus for {s_codes[0]} "
            f"and {s_codes[1]}?",
        ]
    if len(s_codes) > 2:
        joined = ", ".join(s_codes)
        examples += [
            f"Which day is heaviest if I take {joined}?",
            f"Which day is lightest if I take {joined}?",
            f"What is my longest break if I take {joined}?",
        ]

    # Picking an example fills the box rather than replacing it, so the
    # question stays editable — most people want to start from an example
    # and change the course codes.
    if "ask_input" not in st.session_state:
        st.session_state.ask_input = ""

    def load_example():
        if st.session_state.ask_example:
            st.session_state.ask_input = st.session_state.ask_example

    st.selectbox("Try one, or write your own", [""] + examples,
                 key="ask_example", on_change=load_example)
    question = st.text_input(
        "Question", key="ask_input", label_visibility="collapsed",
        placeholder="Can I take these two courses together?")

    if question.strip():
        with guard("answering that question"):
            how, answer = route(question.strip(), df, cfg)
            st.markdown(f'<div class="verdict">{answer}</div>',
                        unsafe_allow_html=True)
            note(f"The planner {how}, then worked the answer out from the "
                 f"timetable. It calculated this \u2014 it did not guess or "
                 f"generate it.")

    with st.expander("How this works"):
        st.markdown(
            f'<div class="sub">Your question is matched against example '
            f'phrasings to work out what you are asking — currently using '
            f'<b>{interpreter_in_use()}</b>. If that is not confident, the '
            f'planner falls back to looking for keywords, and if that also '
            f'fails it tells you what it can do instead of guessing.'
            f'<br><br>Whichever route is taken, the answer itself is always '
            f'calculated from the timetable. The language layer only '
            f'chooses which calculation to run. That separation is '
            f'deliberate: a misunderstood question is obvious to you and '
            f'easy to correct, whereas a miscalculated clash would not be '
            f'noticed until registration had closed.</div>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------
with tab_group:
    st.markdown(
        '<div class="lede">Find a time when everyone in your group is '
        'free. Add each person and the courses they take, and the planner '
        'works out when all of you are free at once. A published timetable '
        'cannot answer this, because it needs several people\'s course '
        'choices at the same time.</div>', unsafe_allow_html=True)

    member_count = st.number_input("How many people are in the group?",
                                   min_value=2, max_value=6, value=3, step=1)

    group_opts = course_options(df, level)
    schedules = {}
    missing_members = []

    for i in range(int(member_count)):
        left, right = st.columns([1, 3])
        with left:
            name = st.text_input("Name", value=f"Person {i + 1}",
                                 key=f"group_name_{i}").strip()
        with right:
            picks = st.multiselect("Their courses", group_opts,
                                   key=f"group_courses_{i}")

        member_codes = codes_from(picks)
        display_name = name or f"Person {i + 1}"

        if not member_codes:
            missing_members.append(display_name)
            continue

        member_sessions = df[(df["year_group"] == level)
                             & (df["course_code"].isin(member_codes))]

        # Each person may be in a different group for the same course, so
        # ask per person rather than assuming the group shares sittings.
        member_multi = {
            code: sorted(member_sessions[
                member_sessions["course_code"] == code]["section"]
                .dropna().unique())
            for code in member_codes
            if member_sessions[member_sessions["course_code"] == code
                               ]["section"].nunique() > 1}

        if member_multi:
            st.caption(f"Which group is {display_name} in for these?")
            sec_cols = st.columns(min(len(member_multi), 4))
            chosen_sections = {}
            for j, (code, available) in enumerate(member_multi.items()):
                with sec_cols[j % len(sec_cols)]:
                    chosen_sections[code] = st.selectbox(
                        code, available, key=f"group_sec_{i}_{code}")
            member_sessions = member_sessions[member_sessions.apply(
                lambda r: (r["course_code"] not in chosen_sections
                           or r["section"] == chosen_sections[r["course_code"]]),
                axis=1)]

        key = display_name
        if key in schedules:
            key = f"{display_name} ({i + 1})"
        schedules[key] = member_sessions
        st.divider()

    minimum = st.select_slider(
        "How long does the meeting need to be?",
        options=[60, 90, 120, 150, 180], value=120,
        format_func=lambda m: (f"{m // 60} hour" if m == 60
                               else f"{m // 60} hours" if m % 60 == 0
                               else f"{m // 60} hr {m % 60} min"))

    if st.button("Find a time that works for everyone", type="primary"):
        if missing_members:
            note(f"Add at least one course for {listify(missing_members)} "
                 f"before searching.", "bad")
        else:
            with guard("looking for a shared free time"):
                common = common_free_blocks(schedules, cfg)

                if common.empty:
                    st.markdown(
                        '<div class="verdict bad">There is no time when '
                        'everyone is free.</div>', unsafe_allow_html=True)
                    note("Try a shorter meeting, or split into two smaller "
                         "groups.")
                else:
                    suitable = (common[common["minutes"] >= minimum]
                                .sort_values(["minutes", "day"],
                                             ascending=[False, True])
                                .reset_index(drop=True))

                    if suitable.empty:
                        longest = common.sort_values(
                            "minutes", ascending=False).iloc[0]
                        st.markdown(
                            f'<div class="verdict bad">Everyone is free at '
                            f'the same time, but never for '
                            f'{minimum} minutes.</div>',
                            unsafe_allow_html=True)
                        note(f"The longest you all share is "
                             f"<b>{longest['day']} {longest['start']}"
                             f"\u2013{longest['end']}</b>, "
                             f"{int(longest['minutes'])} minutes. Shorten "
                             f"the meeting to fit it.")
                    else:
                        best = suitable.iloc[0]
                        st.markdown(
                            f'<div class="verdict good">Meet '
                            f'<b>{best["day"]} {best["start"]}'
                            f'\u2013{best["end"]}</b>. That gives all '
                            f'{int(member_count)} of you '
                            f'{int(best["minutes"])} minutes.</div>',
                            unsafe_allow_html=True)
                        figs([(len(suitable), "times that work"),
                              (int(best["minutes"]), "minutes in the longest"),
                              (int(member_count), "people")])
                        st.markdown("### Every time that works")
                        st.dataframe(
                            suitable[["day", "start", "end", "minutes"]],
                            use_container_width=True, hide_index=True)
