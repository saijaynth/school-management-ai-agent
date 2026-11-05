"""Deterministic question normalisation applied before the parser model.

Small seq2seq models are good at *copying* but poor at calendar arithmetic, so every date
expression ("Nov 2", "2nd November 2026", "02/11/2026", "today") is rewritten to ISO
``YYYY-MM-DD`` first. The same function is used when building the training set, so the model only
ever sees (and learns from) the normalised form.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
_D_MON = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_ALT})\b\.?(?:,?\s*(\d{{4}}))?", re.I)
_MON_D = re.compile(rf"\b({_ALT})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s*(\d{{4}}))?", re.I)
_REL = {"today": 0, "yesterday": -1, "tomorrow": 1}
_WEEKDAYS = {"mon": "Monday", "tue": "Tuesday", "tues": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
             "thur": "Thursday", "thurs": "Thursday", "fri": "Friday", "sat": "Saturday"}


def _year_for(month: int, day: int, start: Optional[dt.date], end: Optional[dt.date]) -> int:
    if start and end:
        for y in (start.year, end.year):
            try:
                if start <= dt.date(y, month, day) <= end:
                    return y
            except ValueError:
                continue
        return start.year
    return dt.date.today().year


def normalize_question(text: str, ay_start: Optional[dt.date] = None, ay_end: Optional[dt.date] = None,
                       today: Optional[dt.date] = None) -> str:
    today = today or dt.date.today()
    out = text

    def iso(y: int, m: int, d: int, raw: str) -> str:
        try:
            return dt.date(y, m, d).isoformat()
        except ValueError:
            return raw

    out = _ISO.sub(lambda m: iso(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0)), out)
    out = _DMY.sub(lambda m: iso(int(m.group(3)), int(m.group(2)), int(m.group(1)), m.group(0)), out)

    def d_mon(m: re.Match) -> str:
        mon, day = MONTHS[m.group(2).lower()], int(m.group(1))
        year = int(m.group(3)) if m.group(3) else _year_for(mon, day, ay_start, ay_end)
        return iso(year, mon, day, m.group(0))

    def mon_d(m: re.Match) -> str:
        mon, day = MONTHS[m.group(1).lower()], int(m.group(2))
        year = int(m.group(3)) if m.group(3) else _year_for(mon, day, ay_start, ay_end)
        return iso(year, mon, day, m.group(0))

    out = _D_MON.sub(d_mon, out)
    out = _MON_D.sub(mon_d, out)
    for word, delta in _REL.items():
        out = re.sub(rf"\b{word}\b", (today + dt.timedelta(days=delta)).isoformat(), out, flags=re.I)
    for abbr, full in _WEEKDAYS.items():
        out = re.sub(rf"\b{abbr}\b", full, out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip()
