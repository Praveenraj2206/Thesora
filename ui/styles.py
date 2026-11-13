"""Visual design for the Thesora UI: light ("paper") and dark ("ink") palettes plus all CSS.

Streamlit has one theme per app, so both palettes are applied with CSS custom properties that
override Streamlit's own colours. ``build_css(mode)`` returns the <style> block for one mode.
"""
from __future__ import annotations

LIGHT = "light"
DARK = "dark"

PALETTES: dict[str, dict[str, str]] = {
    LIGHT: {  # warm paper, ink text, deep indigo accent
        "bg": "#F7F5F0",
        "surface": "#FFFFFF",
        "surface-2": "#FBFAF7",
        "sidebar": "#F0ECE3",
        "text": "#1E2230",
        "muted": "#6A6F7C",
        "faint": "#9A9EA8",
        "border": "#E4DED2",
        "accent": "#3446A6",
        "accent-strong": "#27368A",
        "accent-soft": "#ECEEF9",
        "accent-ink": "#FFFFFF",
        "good": "#2E7D5B",
        "good-soft": "#E7F3EC",
        "warn": "#A8661A",
        "warn-soft": "#FBF1E2",
        "bad": "#B2443A",
        "bad-soft": "#FBEAE7",
        "shadow": "0 1px 2px rgba(30,34,48,.04), 0 4px 16px rgba(30,34,48,.06)",
        "code": "#F2EFE8",
    },
    DARK: {  # deep ink, soft paper text, periwinkle accent
        "bg": "#121419",
        "surface": "#1A1D24",
        "surface-2": "#1F232B",
        "sidebar": "#16181E",
        "text": "#E7E5DF",
        "muted": "#A0A5B0",
        "faint": "#727884",
        "border": "#2B303A",
        "accent": "#8E9CF5",
        "accent-strong": "#AAB5FF",
        "accent-soft": "#242A45",
        "accent-ink": "#121419",
        "good": "#5BBE8E",
        "good-soft": "#1C2E26",
        "warn": "#E3AA55",
        "warn-soft": "#33291A",
        "bad": "#E58276",
        "bad-soft": "#3A2220",
        "shadow": "0 1px 2px rgba(0,0,0,.3), 0 6px 20px rgba(0,0,0,.25)",
        "code": "#23272F",
    },
}

FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&"
    "family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=JetBrains+Mono:wght@400;500&display=swap"
)

_BASE_CSS = """
@import url('__FONTS__');

:root, .stApp { color-scheme: __SCHEME__; }
.stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
/* Hide Streamlit's menu, deploy button and status widget, but NOT the toolbar itself: the button
   that re-opens a collapsed sidebar (stExpandSidebarButton) lives inside it. */
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stToolbarActions"],
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"] { display: none; }
[data-testid="stToolbar"] { background: transparent; }
[data-testid="stExpandSidebarButton"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; box-shadow: var(--shadow);
}
[data-testid="stExpandSidebarButton"]:hover { border-color: var(--accent); }
[data-testid="stExpandSidebarButton"] svg { color: var(--accent) !important; }
.block-container { max-width: 980px; padding-top: 2.2rem; padding-bottom: 7rem; }

/* typography */
.stApp p, .stApp li, .stApp label, .stApp span, .stApp td, .stApp th,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { color: var(--text); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  color: var(--text); letter-spacing: -0.01em;
}
[data-testid="stMarkdownContainer"] h2 { font-size: 1.15rem; margin-top: 1.1rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
.stApp a { color: var(--accent); }
.stApp code { background: var(--code); color: var(--text); border-radius: 4px; }
hr { border-color: var(--border) !important; }

/* sidebar */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: var(--sidebar); }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
[data-testid="stSidebarContent"] { padding-top: 0; }
[data-testid="stSidebarHeader"] { height: 2.2rem; padding-top: .6rem; }
[data-testid="stBottomBlockContainer"] { max-width: 980px; }
[data-testid="stSidebarCollapseButton"] svg { color: var(--muted); }

/* buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  border-radius: 10px; font-weight: 500; box-shadow: none; transition: all .15s ease;
}
.stButton > button:hover, [data-testid="stPopover"] button:hover {
  border-color: var(--accent); color: var(--accent); background: var(--accent-soft);
}
.stButton > button:disabled { opacity: .45; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--accent) !important; color: var(--accent-ink) !important; border-color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: var(--accent-strong) !important; border-color: var(--accent-strong) !important;
}
.stButton > button[kind="primary"] p { color: var(--accent-ink) !important; }
.stButton > button p { color: inherit; }

/* segmented control (mode switch) */
[data-testid="stButtonGroup"] button {
  background: var(--surface); color: var(--muted); border-color: var(--border);
}
[data-testid="stButtonGroup"] button p { color: inherit; }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--accent-soft); color: var(--accent); border-color: var(--accent);
}

/* inputs */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface); border: 1.5px dashed var(--border); border-radius: 12px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted); }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] > div {
  background: var(--surface-2) !important; border-radius: 8px; color: var(--text);
}
[data-testid="stFileUploaderFile"] svg { color: var(--muted); }
[data-testid="stFileChip"] { background: var(--surface-2) !important; border: 1px solid var(--border); }
[data-testid="stFileChip"] * { color: var(--muted) !important; }
[data-testid="stFileChipName"], [data-testid="stFileChipName"] * { color: var(--text) !important; }
[data-testid="stFileChip"] svg { color: var(--muted) !important; }
[data-testid="stTextAreaRootElement"] {
  background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
}
[data-testid="stTextAreaRootElement"]:focus-within { border-color: var(--accent) !important; }
[data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFile"] small { color: var(--text); }
[data-testid="stFileUploaderFile"] svg, [data-testid="stFileUploaderDeleteBtn"] svg { color: var(--muted); }
[data-testid="stFileUploader"] section button { border-radius: 8px; }
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="base-input"] {
  background: var(--surface) !important; border-color: var(--border) !important; border-radius: 10px !important;
}
[data-testid="stTextArea"] textarea, [data-testid="stTextArea"] > div > div,
[data-baseweb="textarea"] textarea, [data-baseweb="input"] input {
  background: var(--surface) !important; color: var(--text) !important; caret-color: var(--accent);
}
textarea::placeholder, input::placeholder { color: var(--faint) !important; }
[data-testid="stChatInput"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow);
}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb="textarea"] {
  background: var(--surface) !important; border: none !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--accent) !important; border-radius: 10px; }
[data-testid="stChatInputSubmitButton"] svg { color: var(--accent-ink) !important; fill: var(--accent-ink) !important; }

/* toggles */
[data-testid="stToggle"] label p, [data-testid="stCheckbox"] label p { color: var(--muted); font-size: .9rem; }

/* alerts */
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stAlertContainer"] { background: var(--surface-2); border: 1px solid var(--border); color: var(--text); }
[data-testid="stAlertContainer"] p { color: var(--text); }

/* bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"][class*="border"],
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"] {
  border-color: var(--border) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div .th-card-mark),
div:has(> .stElementContainer .th-card-mark) {
  background: var(--surface); border: 1px solid var(--border) !important; border-radius: 16px !important;
  box-shadow: var(--shadow);
}

/* tables (synthesis) */
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; font-size: .92rem; margin: .4rem 0 1rem; }
[data-testid="stMarkdownContainer"] th {
  background: var(--accent-soft); color: var(--text); text-align: left; font-weight: 600;
  border: 1px solid var(--border); padding: .55rem .7rem;
}
[data-testid="stMarkdownContainer"] td { border: 1px solid var(--border); padding: .55rem .7rem; vertical-align: top; }
[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: var(--surface-2); }

/* -------- Thesora components -------- */
.th-card-mark { display: none; }
.stElementContainer:has(.th-card-mark), [data-testid="stElementContainer"]:has(.th-card-mark) { display: none; }
.th-brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .1rem; }
.th-logo {
  width: 38px; height: 38px; border-radius: 10px; background: var(--accent); color: var(--accent-ink);
  display: grid; place-items: center; font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.35rem;
}
.th-wordmark { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.55rem; font-weight: 600; color: var(--text); line-height: 1.1; }
.th-tagline { color: var(--muted); font-size: .82rem; }
.th-hero h1 { font-size: 2rem; margin: 0 0 .25rem; padding: 0; }
.th-hero p { color: var(--muted); margin: 0 0 .9rem; font-size: .98rem; }
.th-eyebrow {
  font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .7rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--faint); margin: 1.1rem 0 .45rem;
}
.th-pill {
  display: inline-flex; align-items: center; gap: .4rem; padding: .28rem .65rem; border-radius: 999px;
  font-size: .78rem; font-weight: 500; border: 1px solid var(--border); background: var(--surface); color: var(--muted);
}
.th-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--faint); display: inline-block; }
.th-pill.good { color: var(--good); background: var(--good-soft); border-color: transparent; }
.th-pill.good .th-dot { background: var(--good); }
.th-pill.warn { color: var(--warn); background: var(--warn-soft); border-color: transparent; }
.th-pill.warn .th-dot { background: var(--warn); }
.th-pill.bad { color: var(--bad); background: var(--bad-soft); border-color: transparent; }
.th-pill.bad .th-dot { background: var(--bad); }

.th-doc {
  display: flex; gap: .6rem; align-items: flex-start; padding: .6rem .7rem; margin-bottom: .45rem;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
}
.th-doc-icon {
  flex: none; width: 26px; height: 32px; border-radius: 4px; background: var(--accent-soft); color: var(--accent);
  font-size: .55rem; font-weight: 600; display: grid; place-items: center; font-family: 'JetBrains Mono', monospace;
}
.th-doc-name { font-size: .86rem; font-weight: 500; color: var(--text); word-break: break-all; }
.th-doc-meta { font-size: .75rem; color: var(--muted); }
.th-empty { color: var(--faint); font-size: .85rem; padding: .3rem 0; }
.th-stat { color: var(--muted); font-size: .8rem; margin-top: .3rem; }

.th-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-top: 1.2rem; }
.th-step { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; box-shadow: var(--shadow); }
.th-step-n { font-family: 'JetBrains Mono', monospace; color: var(--accent); font-size: .75rem; }
.th-step-t { font-weight: 600; margin: .3rem 0 .2rem; color: var(--text); }
.th-step-d { color: var(--muted); font-size: .85rem; line-height: 1.45; }
.th-legend { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
@media (max-width: 760px) { .th-steps { grid-template-columns: 1fr; } }

.th-q { display: flex; justify-content: flex-end; margin: 1.4rem 0 .6rem; }
.th-q-bubble {
  max-width: 80%; background: var(--accent-soft); color: var(--text); padding: .7rem 1rem;
  border-radius: 16px 16px 4px 16px; font-size: .97rem; line-height: 1.5;
}
.th-q-mode {
  display: block; font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--accent); margin-bottom: .2rem;
}
.th-answer-head { display: flex; align-items: center; justify-content: space-between; gap: .6rem; flex-wrap: wrap; margin-bottom: .2rem; }
.th-answer-label { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.02rem; color: var(--text); }
.th-meta { color: var(--faint); font-size: .76rem; font-family: 'JetBrains Mono', monospace; }

.cite {
  display: inline-block; min-width: 1.25rem; padding: 0 .32rem; margin: 0 .08rem; border-radius: 6px;
  background: var(--accent-soft); color: var(--accent); font-size: .74rem; font-weight: 600; line-height: 1.35rem;
  text-align: center; vertical-align: .08rem; font-family: 'JetBrains Mono', monospace;
}
.cite.unverified { background: var(--warn-soft); color: var(--warn); }

.th-conf { margin: .9rem 0 .2rem; }
.th-conf-top { display: flex; justify-content: space-between; font-size: .8rem; color: var(--muted); margin-bottom: .3rem; }
.th-conf-top b { color: var(--text); font-weight: 600; }
.th-conf-track { height: 7px; border-radius: 999px; background: var(--border); overflow: hidden; }
.th-conf-fill { height: 100%; border-radius: 999px; }
.th-conf-fill.good { background: var(--good); } .th-conf-fill.warn { background: var(--warn); } .th-conf-fill.bad { background: var(--bad); }

.th-notice {
  display: flex; gap: .75rem; align-items: flex-start; padding: .85rem 1rem; border-radius: 12px;
  border: 1px solid var(--border); background: var(--surface-2); margin: .3rem 0;
}
.th-notice-icon { flex: none; width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: .85rem; }
.th-notice.warn .th-notice-icon { background: var(--warn-soft); color: var(--warn); }
.th-notice.bad .th-notice-icon { background: var(--bad-soft); color: var(--bad); }
.th-notice.info .th-notice-icon { background: var(--accent-soft); color: var(--accent); }
.th-notice.good .th-notice-icon { background: var(--good-soft); color: var(--good); }
.th-notice-title { font-weight: 600; color: var(--text); font-size: .92rem; }
.th-notice-body { color: var(--muted); font-size: .87rem; line-height: 1.45; }

.th-sources-title {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--faint); margin: 1.1rem 0 .5rem;
}
details.th-src { border: 1px solid var(--border); border-radius: 12px; background: var(--surface-2); margin-bottom: .5rem; overflow: hidden; }
details.th-src[open] { background: var(--surface); }
details.th-src summary { list-style: none; cursor: pointer; padding: .65rem .85rem; display: flex; gap: .7rem; align-items: center; }
details.th-src summary::-webkit-details-marker { display: none; }
.th-src-n {
  flex: none; width: 1.6rem; height: 1.6rem; border-radius: 7px; display: grid; place-items: center;
  background: var(--accent-soft); color: var(--accent); font-weight: 600; font-size: .8rem; font-family: 'JetBrains Mono', monospace;
}
.th-src-main { flex: 1; min-width: 0; }
.th-src-doc { font-size: .88rem; font-weight: 500; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.th-src-loc { font-size: .76rem; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.th-src-rel { flex: none; width: 74px; }
.th-src-rel-track { height: 4px; border-radius: 999px; background: var(--border); overflow: hidden; }
.th-src-rel-fill { height: 100%; background: var(--accent); border-radius: 999px; }
.th-src-rel-label { font-size: .68rem; color: var(--faint); text-align: right; margin-top: .2rem; font-family: 'JetBrains Mono', monospace; }
.th-badge { flex: none; font-size: .72rem; font-weight: 600; padding: .18rem .5rem; border-radius: 999px; }
.th-badge.good { background: var(--good-soft); color: var(--good); }
.th-badge.warn { background: var(--warn-soft); color: var(--warn); }
.th-badge.muted { background: var(--border); color: var(--muted); }
.th-src-text {
  padding: .2rem 1rem .9rem 3.15rem; color: var(--text); font-family: 'Source Serif 4', Georgia, serif;
  font-size: .95rem; line-height: 1.6; border-top: 1px dashed var(--border); padding-top: .75rem;
}
.th-chev { color: var(--faint); font-size: .8rem; transition: transform .15s; }
details.th-src[open] .th-chev { transform: rotate(90deg); }

.th-refine {
  border: 1px solid var(--accent-soft); background: linear-gradient(180deg, var(--accent-soft), transparent 140%);
  border-radius: 14px; padding: .9rem 1rem; margin: .3rem 0 .6rem;
}
.th-refine-label { font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }
.th-refine-text { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.08rem; color: var(--text); margin: .35rem 0 .3rem; line-height: 1.45; }
.th-refine-why { color: var(--muted); font-size: .84rem; }
.th-footer { color: var(--faint); font-size: .74rem; margin-top: 1.4rem; line-height: 1.5; }
"""


def build_css(mode: str) -> str:
    """Return the full <style> block for ``mode`` (``light`` or ``dark``)."""
    palette = PALETTES.get(mode, PALETTES[LIGHT])
    variables = "\n".join(f"  --{name}: {value};" for name, value in palette.items())
    body = _BASE_CSS.replace("__FONTS__", FONTS_URL).replace("__SCHEME__", "dark" if mode == DARK else "light")
    return f"<style>\n:root {{\n{variables}\n}}\n{body}\n</style>"
