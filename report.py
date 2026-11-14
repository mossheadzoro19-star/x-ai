"""
report.py - builds a one-page PDF summary of a single analysis (UI-free, NFR9).

Contains: original X-ray, Grad-CAM overlay, prediction, confidence, usefulness scores,
explanation text and the disclaimer. No filenames, no patient identifiers, nothing written to disk.
"""
import io
import re
from datetime import datetime
from typing import Dict, Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

DISCLAIMER = ("Educational research prototype - not a diagnostic device. Heatmaps show model sensitivity, "
              "not verified pathology. Confidence is a statistical estimate, not a disease probability. "
              "The clinician retains final decision authority.")


def _plain(text: str) -> str:
    """Strip markdown/emoji so the built-in PDF fonts can render it."""
    text = re.sub(r"[#*`>_]", "", text or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    return text.encode("latin-1", "ignore").decode("latin-1")


def build_pdf(res: Dict, explanation: str, case_id: str, latency_s: Optional[float] = None) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    m = 40
    forest, lime, grey = colors.HexColor("#2C3A2E"), colors.HexColor("#CEE39B"), colors.HexColor("#6B6F69")

    # header band
    c.setFillColor(forest)
    c.rect(0, H - 80, W, 80, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 20)
    c.drawString(m, H - 46, "Chest X-ray XAI - analysis summary")
    c.setFont("Helvetica", 9)
    c.setFillColor(lime)
    c.drawString(m, H - 64, f"Case ref {case_id}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Team 19, KLE Tech")

    # images
    y_img = H - 100 - 230
    for i, (arr, cap) in enumerate([(res["original_image"], "Original X-ray"), (res["overlay"], "Grad-CAM overlay")]):
        x = m + i * 255
        c.drawImage(ImageReader(Image.fromarray(arr)), x, y_img, width=230, height=230)
        c.setFillColor(grey)
        c.setFont("Helvetica", 8)
        c.drawString(x, y_img - 12, cap)

    # scores
    u = res["usefulness"]
    y = y_img - 40
    c.setFillColor(forest)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(m, y, f"Prediction: {res['fact_class'].capitalize()}  -  calibrated confidence {res['confidence']*100:.1f}%")
    y -= 18
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    c.drawString(m, y, f"Closest alternative: {res['foil_class'].capitalize()} ({res['foil_confidence']*100:.1f}%)")
    y -= 15
    c.drawString(m, y, f"Explanation Usefulness Score: {u['usefulness_score']:+.2f} raw  |  {u['usefulness_normalized']:.2f} normalised (0-1)")
    y -= 15
    if res["confidence"] < 0.70:
        c.setFillColor(colors.HexColor("#B4532A"))
        c.drawString(m, y, "LOW CONFIDENCE (< 70%) - treat this result as uncertain.")
        c.setFillColor(colors.black)
        y -= 15
    if latency_s is not None:
        c.setFillColor(grey)
        c.setFont("Helvetica", 8)
        c.drawString(m, y, f"Analysis time {latency_s:.1f} s")
        y -= 12

    # explanation
    y -= 8
    c.setFillColor(forest)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(m, y, "Explanation")
    y -= 14
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 9)
    for line in simpleSplit(_plain(explanation), "Helvetica", 9, W - 2 * m)[:28]:
        c.drawString(m, y, line)
        y -= 12

    # disclaimer
    c.setFillColor(grey)
    c.setFont("Helvetica-Oblique", 8)
    yy = 52
    for line in simpleSplit(DISCLAIMER, "Helvetica-Oblique", 8, W - 2 * m):
        c.drawString(m, yy, line)
        yy -= 10
    c.showPage()
    c.save()
    return buf.getvalue()
