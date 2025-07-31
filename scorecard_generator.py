# scorecard_generator.py

import fpdf
from config import SCORECARD_PDF

def generate_scorecard(punch_log):
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Boxing Match Scorecard", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.cell(50, 10, "Fighter", 1)
    pdf.cell(50, 10, "Time (MM:SS)", 1)
    pdf.cell(50, 10, "Punch Count", 1)
    pdf.ln()

    for fighter, time, score in punch_log:
        pdf.cell(50, 10, fighter, 1)
        pdf.cell(50, 10, time, 1)
        pdf.cell(50, 10, str(score), 1)
        pdf.ln()

    pdf.output(SCORECARD_PDF)
    print(f"✅ Scorecard saved to {SCORECARD_PDF}")
