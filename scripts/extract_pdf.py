#!/usr/bin/env python3
"""Extract text from PDF files using pdftotext."""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def extract_pdf_text(pdf_path: str, output_path: Optional[str] = None) -> str:
    """Extract text from a PDF file using pdftotext.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the extracted text

    Returns:
        Extracted text content
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Run pdftotext command
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")

    text = result.stdout

    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        print(f"Saved extracted text to: {output_path}")

    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <pdf_path> [output_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    text = extract_pdf_text(pdf_path, output_path)

    if not output_path:
        print(text)
