"""Word version of the hand-check sheet, and reading the answers back into the CSV.

    python scripts/handcheck_word.py make   # data/handcheck_100.csv -> data/handcheck_100.docx
    python scripts/handcheck_word.py read   # answers typed in the .docx -> 1s in data/handcheck_100.csv

In the Word file, type the category numbers after each "Answer N:" (e.g. "Answer 3: 2, 7").
Leave it empty if the review has no real complaint. Anything after "--" is saved as a note.
The key is repeated in the page header, so it is always on screen.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.batch_labels import COMPLAINTS
from src.handcheck import read_sheet

CSV = REPO / "data" / "handcheck_100_done.csv"   # blank sheet: handcheck_100.csv
DOCX = REPO / "data" / "handcheck_100.docx"
CATS = list(COMPLAINTS)                              # 1 = cleanliness ... 14 = other
NUM = {i + 1: c for i, c in enumerate(CATS)}
MUTED = RGBColor(0x5A, 0x64, 0x78)
RULES = [
    "Mark what the guest actually complains about, by meaning (\"Dodgy lift\" = maintenance). Nothing added, nothing missed.",
    "No complaint at all (\"all good\", \"nothing\") = leave the answer empty.",
    "Mild wishes count (\"breakfast could have more choice\" = breakfast).",
    "Price of one thing (\"breakfast overpriced\") = that thing + value.",
    "Wrong bed type = checkin only if they say they booked/asked for something else; otherwise bed_comfort.",
    "temperature = the guest's own room only. Hot lobby / broken air con in the bar = facilities.",
    "other = a real complaint that fits none of the categories.",
]


def make():
    sheet = read_sheet(CSV)
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(12)

    hdr = doc.sections[0].header.paragraphs[0]
    r = hdr.add_run("  ".join(f"{i}={c}" for i, c in NUM.items()))
    r.font.size, r.font.color.rgb = Pt(9), MUTED

    doc.add_heading("Hand check: 100 hotel complaints", level=1)
    doc.add_paragraph("For each review, type the numbers of every category it complains about after \"Answer N:\", "
                      "separated by commas (e.g. 2, 7). Leave it empty if there is no real complaint. "
                      "Add a note after -- if you are unsure (e.g. 11 -- could also be value).")
    t = doc.add_table(rows=0, cols=2)
    t.style = "Light List Accent 1"
    from src.batch_labels import CATEGORIES
    for i, c in NUM.items():
        row = t.add_row().cells
        row[0].text = f"{i}  {c}"
        row[1].text = CATEGORIES[c]
    doc.add_paragraph()
    for rule in RULES:
        doc.add_paragraph(rule, style="List Bullet")
    doc.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)

    for n, row in enumerate(sheet.itertuples(), start=1):
        p = doc.add_paragraph()
        h = p.add_run(f"Review {n} of {len(sheet)}")
        h.bold, h.font.size = True, Pt(13)
        m = p.add_run(f"    id {row.review_id}, score {row.score}")
        m.font.size, m.font.color.rgb = Pt(10), MUTED
        q = doc.add_paragraph()
        qr = q.add_run(row.complaint_text)
        qr.font.size = Pt(14)
        a = doc.add_paragraph()
        ar = a.add_run(f"Answer {n}: ")
        ar.bold = True
        doc.add_paragraph()
    doc.save(DOCX)
    print(f"Saved {DOCX}")


def read():
    sheet = pd.read_csv(CSV, encoding="utf-8-sig", keep_default_na=False)
    answers, problems = {}, []
    for p in Document(DOCX).paragraphs:
        mt = re.match(r"\s*Answer\s+(\d+)\s*:(.*)", p.text)
        if not mt:
            continue
        n, rest = int(mt.group(1)), mt.group(2)
        body, _, note = rest.partition("--")
        nums = [int(x) for x in re.findall(r"\d+", body)]
        bad = [x for x in nums if x not in NUM]
        if bad:
            problems.append(f"Answer {n}: {bad} is not a category number (1-14)")
        answers[n] = ({NUM[x] for x in nums if x in NUM}, note.strip())
    missing = [n for n in range(1, len(sheet) + 1) if n not in answers]
    if missing:
        problems.append(f"No 'Answer N:' line found for reviews {missing}")
    empty = [n for n, (cats, _) in answers.items() if not cats]
    for n, (cats, note) in answers.items():
        i = n - 1
        for c in CATS:
            sheet.at[i, c] = "1" if c in cats else ""
        sheet.at[i, "notes"] = note
    sheet.to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"Read {len(answers)} answers into {CSV.name}; {len(empty)} marked as no complaint: {empty}")
    for pr in problems:
        print("CHECK:", pr)


if __name__ == "__main__":
    {"make": make, "read": read}[sys.argv[1]]()
