"""Fix font formatting and improve visual consistency of the MVP PPT.

Changes:
1. Unify all fonts to "Microsoft YaHei" (雅黑)
2. Set consistent sizes for different text levels:
   - Page number (小标题序号): 14pt
   - Main title: 30pt bold
   - Subtitle/hint: 12pt
   - Large KPI numbers: 32pt bold
   - KPI labels: 12pt bold
   - Body text: 14pt
   - Evidence footer: 10pt
   - Draft tag: removed
3. Apply consistent color scheme
4. Remove "Draft" text boxes
5. Fix paragraph spacing
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# --- Constants ---
FONT_MAIN = "Microsoft YaHei"  # 雅黑
FONT_MONO = "Consolas"

# Color palette
COLOR_DARK = RGBColor(0x1A, 0x1A, 0x1A)       # 主标题/正文
COLOR_RED = RGBColor(0xC7, 0x00, 0x0B)         # 强调红
COLOR_BLUE = RGBColor(0x00, 0x52, 0x9B)        # 强调蓝
COLOR_GRAY = RGBColor(0x66, 0x66, 0x66)        # 辅助灰
COLOR_LIGHT_GRAY = RGBColor(0x99, 0x99, 0x99)  # 证据口径灰
COLOR_KPI = RGBColor(0xC7, 0x00, 0x0B)         # KPI 数字红
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# Font sizes (in Pt)
SIZE_PAGE_NUM = Pt(14)
SIZE_TITLE = Pt(30)
SIZE_SUBTITLE = Pt(12)
SIZE_KPI_NUM = Pt(32)
SIZE_KPI_LABEL = Pt(12)
SIZE_BODY = Pt(14)
SIZE_EVIDENCE = Pt(10)
SIZE_DRAFT = Pt(10)

ROOT = Path(__file__).resolve().parents[2]
PPT_PATH = ROOT / "deliverables" / "ppt" / "fall_detection_mvp_draft.pptx"
OUTPUT_PATH = PPT_PATH.parent / "fall_detection_mvp_v2.pptx"


def _is_page_number_shape(shape) -> bool:
    """Check if shape contains slide page number like '01', '02', etc."""
    if not shape.has_text_frame:
        return False
    text = shape.text_frame.text.strip()
    return len(text) <= 3 and text.isdigit() and 1 <= int(text) <= 20


def _is_draft_shape(shape) -> bool:
    """Check if shape is a Draft tag."""
    if not shape.has_text_frame:
        return False
    text = shape.text_frame.text.strip()
    return text == "Draft"


def _is_evidence_shape(shape) -> bool:
    """Check if shape contains evidence reference text."""
    if not shape.has_text_frame:
        return False
    text = shape.text_frame.text.strip()
    return text.startswith("证据口径：") or text.startswith("证据：")


def _is_kpi_number_shape(shape) -> bool:
    """Check if shape is a KPI number (large bold digits with unit)."""
    if not shape.has_text_frame:
        return False
    text = shape.text_frame.text.strip()
    # KPI numbers: "87.5%", "80%", "100%", "8", "19.0201ms", "52.6", "6", "2"
    kpi_patterns = [
        r'^\d+\.?\d*%$',           # "87.5%", "80%"
        r'^\d+\.?\d*ms$',          # "19.0201ms"
        r'^\d+\.?\d+$',            # "52.6", "8", "6", "2"
        r'^\d+\.?\d*%?$',          # general number
        r'^384x384$',              # resolution
        r'^27\.68%$',              # percentage
        r'^QDQ$',                  # QDQ
        r'^Conv$',                 # Conv
    ]
    if len(text) <= 6:
        # Short text that looks like a number
        try:
            float(text.replace('%', '').replace('ms', ''))
            return shape.width > Emu(2000000)  # Large shapes only
        except ValueError:
            pass
    return False


def _is_kpi_label_shape(shape) -> bool:
    """Check if shape is a KPI label (describes what the number means)."""
    if not shape.has_text_frame:
        return False
    kpi_labels = [
        "视频级 Accuracy", "Precision", "Recall", "标注视频数",
        "TP", "FP", "TN", "FN",
        "CPU ORT 平均推理", "FPS from mean", "合规通过项", "设计目标项",
        "固定输入", "INT8 格式", "INT8/FP32 体积", "量化算子",
        "多人状态隔离键", "功能验证口径", "无需可穿戴", "隐私与弱网可用",
        "事件级告警", "截图/视频/JSON",
    ]
    text = shape.text_frame.text.strip()
    return text in kpi_labels


def _is_title_shape(shape) -> bool:
    """Check if shape is the main slide title."""
    if not shape.has_text_frame:
        return False
    # Titles typically have size around 317500 EMU (about 25pt) and contain Chinese
    text = shape.text_frame.text.strip()
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    is_large = shape.width > Emu(6000000)  # Wide text box
    return has_chinese and is_large and len(text) > 10


def _is_subtitle_shape(shape) -> bool:
    """Check if shape is the subtitle/hint text."""
    if not shape.has_text_frame:
        return False
    text = shape.text_frame.text.strip()
    # Subtitles are shorter hints like "评委先看应用价值，再看模型指标"
    return len(text) > 5 and len(text) < 40 and not text.startswith("证据")


def _is_body_shape(shape) -> bool:
    """Check if shape contains body bullet points."""
    if not shape.has_text_frame:
        return False
    text = shape.text_frame.text.strip()
    return text.startswith("•") or text.startswith("-")


def _is_placeholder_hint(shape) -> bool:
    """Check if shape is an image placeholder hint like '封面图待增强'."""
    if not shape.has_text_frame:
        return False
    hints = ["待补", "待增强", "建议补图", "建议替换"]
    text = shape.text_frame.text.strip()
    return any(h in text for h in hints)


def _apply_font(run, font_name: str, font_size, bold: bool | None = None, color: RGBColor | None = None):
    """Apply font settings to a run."""
    if font_name:
        run.font.name = font_name
    if font_size:
        run.font.size = font_size
    if bold is not None:
        run.font.bold = bold
    if color:
        run.font.color.rgb = color


def fix_fonts_and_formatting(prs: Presentation) -> Presentation:
    """Main function: fix fonts and formatting across all slides."""

    for slide_idx, slide in enumerate(prs.slides):
        shapes_to_remove = []

        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue

            # --- Remove Draft tags ---
            if _is_draft_shape(shape):
                shapes_to_remove.append(shape)
                continue

            # --- Fix page numbers ---
            if _is_page_number_shape(shape):
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, FONT_MAIN, SIZE_PAGE_NUM, bold=True, color=COLOR_RED)

            # --- Fix evidence footers ---
            elif _is_evidence_shape(shape):
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, FONT_MAIN, SIZE_EVIDENCE, bold=False, color=COLOR_LIGHT_GRAY)

            # --- Fix body text (bullet points) ---
            elif _is_body_shape(shape):
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, FONT_MAIN, SIZE_BODY, bold=None, color=COLOR_DARK)

            # --- Fix placeholder hints ---
            elif _is_placeholder_hint(shape):
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, FONT_MAIN, Pt(16), bold=True, color=COLOR_GRAY)

            # --- Fix KPI numbers ---
            elif _is_kpi_number_shape(shape):
                text = shape.text_frame.text.strip()
                is_mono = bool(re.match(r'^\d+x\d+$', text))  # 384x384
                font = FONT_MONO if is_mono else FONT_MAIN
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, font, SIZE_KPI_NUM, bold=True, color=COLOR_KPI)

            # --- Fix KPI labels ---
            elif _is_kpi_label_shape(shape):
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, FONT_MAIN, SIZE_KPI_LABEL, bold=True, color=COLOR_GRAY)

            # --- Fix main titles ---
            elif _is_title_shape(shape):
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, FONT_MAIN, SIZE_TITLE, bold=True, color=COLOR_DARK)

            # --- Fix subtitles ---
            elif _is_subtitle_shape(shape):
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        _apply_font(run, FONT_MAIN, SIZE_SUBTITLE, bold=False, color=COLOR_GRAY)

            # --- Catch-all: fix any remaining inherit fonts ---
            else:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        if run.font.name is None or run.font.name == "inherit":
                            run.font.name = FONT_MAIN
                        # Keep existing size if set, otherwise default to body size
                        if run.font.size is None:
                            run.font.size = SIZE_BODY
                        # Fix color if not set
                        try:
                            _ = run.font.color.rgb
                        except (AttributeError, Exception):
                            run.font.color.rgb = COLOR_DARK

        # Remove Draft shapes
        for shape in shapes_to_remove:
            sp = shape._element
            sp.getparent().remove(sp)

    return prs


def fix_paragraph_spacing(prs: Presentation) -> Presentation:
    """Fix paragraph line spacing for better readability."""
    from pptx.oxml.ns import qn

    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                # Set line spacing to 1.2x
                pPr = para._pPr
                if pPr is None:
                    pPr = para._p.get_or_add_pPr()

                # Remove existing line spacing
                for existing in pPr.findall(qn('a:lnSpc')):
                    pPr.remove(existing)

                # Add 1.2x line spacing
                lnSpc = pPr.makeelement(qn('a:lnSpc'), {})
                spcPct = lnSpc.makeelement(qn('a:spcPct'), {'val': '120000'})
                lnSpc.append(spcPct)
                pPr.append(lnSpc)

    return prs


def fix_vertical_alignment(prs: Presentation) -> Presentation:
    """Fix text frame vertical alignment to middle for better centering."""
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            # Only fix KPI number shapes and centered shapes
            if _is_kpi_number_shape(shape):
                shape.text_frame.word_wrap = True


def main():
    print(f"Reading: {PPT_PATH}")
    prs = Presentation(str(PPT_PATH))

    # Apply fixes
    prs = fix_fonts_and_formatting(prs)
    prs = fix_paragraph_spacing(prs)
    fix_vertical_alignment(prs)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    print(f"Saved: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
