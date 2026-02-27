#!/usr/bin/env python3
"""
Generate per-resident credential pages (PDF) with a shared Play Store QR code.

- QR code is the SAME for everyone (Play Store link)
- Username + initial password are UNIQUE per resident (parsed from your creds PDF)

INPUT:
  --creds-pdf  path to a PDF that contains rows like:
    1 FC F163 FC163 TG-FCF163
    (index ... username password)

OUTPUT:
  output/pages/<index>_<username>.pdf
  output/GuestPass_<Estate>_AllResidents.pdf

INSTALL:
  pip install reportlab qrcode[pil] pillow PyPDF2
"""

import argparse
import io
import os
import re
from dataclasses import dataclass
from typing import List

import qrcode
from PIL import Image
from PyPDF2 import PdfMerger, PdfReader

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader


PLAYSTORE_URL_DEFAULT = "https://play.google.com/store/apps/details?id=com.visitormanagementvms"


@dataclass
class CredRow:
    idx: int
    house_owner: str
    username: str
    initial_password: str


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def parse_creds_from_text(text: str) -> List[CredRow]:
    """
    Parse lines like:
      1 FC F163 FC163 TG-FCF163

    Rule:
      - line starts with integer index
      - last 2 tokens are username and password
      - everything between is house/owner name
    """
    rows: List[CredRow] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # skip headers
        if line.lower().startswith("#") or "house / owner" in line.lower():
            continue

        m = re.match(r"^(\d+)\s+(.*)$", line)
        if not m:
            continue

        idx = int(m.group(1))
        rest = m.group(2).strip()
        parts = rest.split()
        if len(parts) < 3:
            continue

        username = parts[-2]
        initial_password = parts[-1]
        house_owner = " ".join(parts[:-2]).strip()

        # sanity checks
        if len(username) > 60 or len(initial_password) > 120:
            continue

        rows.append(CredRow(idx=idx, house_owner=house_owner, username=username, initial_password=initial_password))

    rows.sort(key=lambda r: r.idx)
    return rows


def make_qr_image(url: str, box_size: int = 10, border: int = 2) -> Image.Image:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    if not isinstance(img, Image.Image):
        img = img.convert("RGB")
    return img


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:120]


def draw_page(
    pdf_path: str,
    estate: str,
    appname: str,
    playstore_url: str,
    qr_img: Image.Image,
    row: CredRow,
):
    # --- GuestPass theme colors ---
    DEEP_GREEN = colors.HexColor("#03A36F")
    MID_GREEN = colors.HexColor("#65C395")
    SOFT_GREEN = colors.HexColor("#5AC28E")
    LAVENDER = colors.HexColor("#DDC6B8")
    GOLD = colors.HexColor("#F6BC4F")
    MUSTARD = colors.HexColor("#FEB604")
    WHITE = colors.HexColor("#FFFFFF")
    DARK_TEXT = colors.HexColor("#111827")
    MUTED_TEXT = colors.HexColor("#374151")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    margin = 18 * mm

    # Background
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # Header bar
    header_h = 18 * mm
    c.setFillColor(DEEP_GREEN)
    c.rect(0, H - header_h, W, header_h, stroke=0, fill=1)

    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, H - 12 * mm, f"{appname} — Resident Access")

    # Subheader info
    c.setFillColor(DARK_TEXT)
    c.setFont("Helvetica", 11)
    c.drawString(margin, H - header_h - 8 * mm, f"Estate: {estate}")
    c.drawString(margin, H - header_h - 14 * mm, f"House / Owner: {row.house_owner}")

    # QR placement (top-right)
    qr_size = 55 * mm
    qr_x = W - margin - qr_size
    qr_y = H - header_h - 10 * mm - qr_size

    qr_buf = io.BytesIO()
    qr_img.save(qr_buf, format="PNG")
    qr_buf.seek(0)
    qr_reader = ImageReader(qr_buf)
    c.drawImage(qr_reader, qr_x, qr_y, width=qr_size, height=qr_size, mask="auto")

    # QR caption
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED_TEXT)
    c.drawRightString(W - margin, qr_y - 4 * mm, "Scan to download GuestPass")
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawRightString(W - margin, qr_y - 8 * mm, "Google Play Store")

    # Account details box (moved down)
    clearance = 18 * mm
    box_top = (qr_y - 12 * mm) - clearance
    min_top = 70 * mm
    if box_top < min_top:
        box_top = min_top

    box_left = margin
    box_width = W - (2 * margin)
    box_height = 62 * mm

    c.setStrokeColor(DEEP_GREEN)
    c.setLineWidth(1.2)
    c.setFillColor(WHITE)
    c.roundRect(box_left, box_top - box_height, box_width, box_height, 10, stroke=1, fill=1)

    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(DEEP_GREEN)
    c.drawString(box_left + 10 * mm, box_top - 12 * mm, "Account Details")

    c.setStrokeColor(MUSTARD)
    c.setLineWidth(2.2)
    c.line(box_left + 10 * mm, box_top - 16 * mm, box_left + 70 * mm, box_top - 16 * mm)

    c.setFont("Helvetica", 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(box_left + 10 * mm, box_top - 28 * mm, "Username:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(box_left + 40 * mm, box_top - 28 * mm, row.username)

    c.setFont("Helvetica", 12)
    c.drawString(box_left + 10 * mm, box_top - 42 * mm, "Initial password:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(box_left + 55 * mm, box_top - 42 * mm, row.initial_password)

    c.setFont("Helvetica", 10.2)
    c.setFillColor(MUTED_TEXT)
    c.drawString(
        box_left + 10 * mm,
        box_top - 56 * mm,
        "Sign in using the details above, then change your password."
    )

    # Footer
    c.setFont("Helvetica", 9.5)
    c.setFillColor(MUTED_TEXT)
    c.drawString(margin, 16 * mm, f"Play Store link: {playstore_url}")

    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(DEEP_GREEN)
    c.drawString(margin, 10 * mm, "Or search GuestPass by Signal Africa Microsystems Ltd on Play Store.")

    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creds-pdf", required=True, help="Path to credentials PDF")
    ap.add_argument("--outdir", default="./output", help="Output directory")
    ap.add_argument("--estate", default="Forest Court", help="Estate / site name printed on page")
    ap.add_argument("--appname", default="GuestPass", help="App name printed on page")
    ap.add_argument("--playstore-url", default=PLAYSTORE_URL_DEFAULT, help="Play Store URL to encode in QR")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pages_dir = os.path.join(args.outdir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    text = extract_text_from_pdf(args.creds_pdf)
    rows = parse_creds_from_text(text)
    if not rows:
        raise SystemExit("No credentials rows parsed. Confirm the PDF contains selectable text (not scanned images).")

    qr_img = make_qr_image(args.playstore_url)

    generated_paths: List[str] = []
    for r in rows:
        fname = f"{r.idx:03d}_{sanitize_filename(r.username)}.pdf"
        out_path = os.path.join(pages_dir, fname)
        draw_page(out_path, args.estate, args.appname, args.playstore_url, qr_img, r)
        generated_paths.append(out_path)

    combined_path = os.path.join(args.outdir, f"{args.appname}_{sanitize_filename(args.estate)}_AllResidents.pdf")
    merger = PdfMerger()
    for p in generated_paths:
        merger.append(p)
    merger.write(combined_path)
    merger.close()

    print(f"Generated {len(generated_paths)} resident pages in: {pages_dir}")
    print(f"Combined PDF: {combined_path}")


if __name__ == "__main__":
    main()
