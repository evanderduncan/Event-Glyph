"""
Event Glyph — Symbol Extractor
Extracts individual symbol crops from the Dreyfuss Symbol Sourcebook.

Cell boundaries are detected from the actual printed grid lines, then inset
inward so the thick border is never included in the extracted symbol.

Folder structure created automatically:
    sourcebook/
    └── dataset/
        ├── crops/          ← raw extracted symbol PNGs (128x128)
        ├── tagged/         ← crops sorted into chapter subfolders after tagging
        │   ├── astronomy/
        │   ├── meteorology/
        │   └── ... (one folder per Dreyfuss chapter)
        ├── previews/       ← grid preview images for verification
        ├── metadata.json   ← full record of every symbol + tags
        └── log.txt         ← processing log

Usage:
    # Test grid detection on one page
    python grid_extractor.py --preview --input sourcebook/pages/page_054.png

    # Batch extract all pages
    python grid_extractor.py --batch --input sourcebook/pages/

    # Tag extracted symbols
    python grid_extractor.py --tag

    # Re-crop existing crops to remove border bleed
    python grid_extractor.py --recrop
"""

import cv2
import numpy as np
import json
import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

COLS         = 6
ROWS         = 7
SYMBOL_RATIO = 0.78    # top 78% of cell = symbol, bottom 22% = label text
OUTPUT_SIZE  = 128     # saved crop size in pixels
PADDING      = 14      # whitespace inside canvas
BORDER_INSET = 12      # pixels to inset from each detected grid line
                       # increase if border still bleeds, decrease if symbol is clipped
LEFT_MARGIN  = 70      # fallback only — used if line detection fails
RIGHT_MARGIN = 80
MIN_INK      = 50000   # minimum ink sum to treat cell as non-empty

OUTPUT_DIR   = Path("sourcebook/dataset")

# Dreyfuss chapter taxonomy — two-letter keys
TAXONOMY = {
    "ac": "accommodations_travel",
    "ag": "agriculture",
    "ar": "architecture",
    "as": "astronomy",
    "bi": "biology",
    "bu": "business",
    "ch": "chemistry",
    "co": "communications",
    "en": "engineering",
    "fo": "folklore",
    "ge": "geography",
    "gl": "geology",
    "hg": "handling_of_goods",
    "he": "home_economics",
    "ma": "manufacturing",
    "mt": "mathematics",
    "me": "medicine",
    "mo": "meteorology",
    "mu": "music",
    "ph": "photography",
    "py": "physics",
    "re": "recreation",
    "ri": "religion",
    "sa": "safety",
    "tr": "traffic",
    "vc": "vehicle_controls",
    "d":  "delete",
    "u":  "unsure",
}

TAXONOMY_HINT = """
  [ac] accommodations & travel    [ag] agriculture
  [ar] architecture               [as] astronomy
  [bi] biology                    [bu] business
  [ch] chemistry                  [co] communications
  [en] engineering                [fo] folklore
  [ge] geography                  [gl] geology
  [hg] handling of goods          [he] home economics
  [ma] manufacturing              [mt] mathematics
  [me] medicine                   [mo] meteorology
  [mu] music                      [ph] photography
  [py] physics                    [re] recreation
  [ri] religion                   [sa] safety
  [tr] traffic                    [vc] vehicle controls
  [d]  delete   [u] unsure   [SPACE] skip   [z] undo   [q] quit & save
"""


# ─────────────────────────────────────────────
# Folder setup
# ─────────────────────────────────────────────

def setup_folders():
    folders = [
        OUTPUT_DIR / "crops",
        OUTPUT_DIR / "previews",
        OUTPUT_DIR / "tagged",
    ]
    for tag in TAXONOMY.values():
        if tag not in ("delete", "unsure"):
            folders.append(OUTPUT_DIR / "tagged" / tag)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._write(f"\n{'='*55}")
        self._write(f"Event Glyph — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self._write(f"{'='*55}")

    def _write(self, msg):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def page_start(self, page_id, page_num, total):
        self._write(f"\n[{page_num}/{total}] {page_id}")

    def page_done(self, page_id, extracted, empty):
        self._write(f"  ✓  {extracted} symbols saved  |  {empty} empty cells ignored")

    def page_skipped(self, page_id, reason):
        self._write(f"  —  skipped: {reason}")

    def page_error(self, page_id, error):
        self._write(f"  ✗  error: {error}")

    def summary(self, processed, total_symbols, skipped, errors):
        self._write(f"\n{'='*55}")
        self._write(f"Batch complete")
        self._write(f"  Pages processed : {processed}")
        self._write(f"  Pages skipped   : {skipped}")
        self._write(f"  Errors          : {errors}")
        self._write(f"  Total symbols   : {total_symbols}")
        self._write(f"  Crops           : {OUTPUT_DIR / 'crops'}")
        self._write(f"  Metadata        : {OUTPUT_DIR / 'metadata.json'}")
        self._write(f"{'='*55}\n")


# ─────────────────────────────────────────────
# Grid line detection
# ─────────────────────────────────────────────

def cluster_positions(positions, gap=20):
    """Merge nearby line detections into single positions."""
    if not positions:
        return []
    positions = sorted(set(positions))
    clusters  = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] < gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]


def detect_grid_lines(gray):
    """
    Use Hough line detection to find the actual printed grid lines.
    Returns (h_lines, v_lines) — sorted lists of y and x positions.
    Returns (None, None) if detection fails.
    """
    h, w  = gray.shape
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=150,
        minLineLength=int(min(w, h) * 0.3),
        maxLineGap=15
    )

    if lines is None:
        return None, None

    h_raw = []
    v_raw = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 5 or angle > 175:
            h_raw.append((y1 + y2) // 2)
        elif 85 < angle < 95:
            v_raw.append((x1 + x2) // 2)

    h_lines = cluster_positions(h_raw)
    v_lines = cluster_positions(v_raw)

    return h_lines, v_lines


def find_horizontal_grid_lines(gray, line_width_threshold=0.55, min_gap=50):
    """
    Fallback: scan pixel rows for thick dark horizontal lines.
    Used when Hough detection doesn't find enough vertical lines.
    """
    h, w   = gray.shape
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    inv    = 255 - binary
    lines  = []
    last_y = -min_gap

    for y in range(h):
        row_sum = inv[y, :].sum()
        if row_sum > (line_width_threshold * w * 255):
            if y - last_y > min_gap:
                lines.append(y)
                last_y = y

    return lines


# ─────────────────────────────────────────────
# Cell boundary calculation
# ─────────────────────────────────────────────

def get_cell_boundaries(img):
    """
    Build cell list by detecting the actual printed grid lines and
    insetting BORDER_INSET pixels from each detected line.

    Strategy:
    1. Try full Hough detection (horizontal + vertical lines)
    2. If vertical detection fails, use horizontal lines + fixed column division
    3. If all detection fails, fall back to fixed margins

    The inset ensures the thick printed border is never included
    in the extracted symbol region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    inset = BORDER_INSET

    h_lines, v_lines = detect_grid_lines(gray)

    # ── Strategy 1: full detection ───────────────────────────────
    if (h_lines and v_lines and
            len(h_lines) >= 2 and len(v_lines) >= 2):

        print(f"  Grid lines detected: {len(h_lines)}h × {len(v_lines)}v")
        cells = []
        for row_idx in range(len(h_lines) - 1):
            for col_idx in range(len(v_lines) - 1):
                x1 = v_lines[col_idx]     + inset
                y1 = h_lines[row_idx]     + inset
                x2 = v_lines[col_idx + 1] - inset
                y2 = h_lines[row_idx + 1] - inset
                # Skip cells that are too small (merged or partial)
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue
                cells.append((x1, y1, x2, y2, row_idx, col_idx))

        if len(cells) >= 6:
            return cells

    # ── Strategy 2: horizontal detection + fixed columns ─────────
    print("  Vertical line detection weak — using horizontal lines + fixed columns")

    if not h_lines or len(h_lines) < 2:
        h_lines = find_horizontal_grid_lines(gray)

    if len(h_lines) >= 2:
        usable_w = w - LEFT_MARGIN - RIGHT_MARGIN
        cell_w   = usable_w // COLS
        v_lines  = [LEFT_MARGIN + col * cell_w for col in range(COLS + 1)]

        cells = []
        for row_idx in range(len(h_lines) - 1):
            for col_idx in range(COLS):
                x1 = v_lines[col_idx]     + inset
                y1 = h_lines[row_idx]     + inset
                x2 = v_lines[col_idx + 1] - inset
                y2 = h_lines[row_idx + 1] - inset
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue
                cells.append((x1, y1, x2, y2, row_idx, col_idx))

        if len(cells) >= 6:
            return cells

    # ── Strategy 3: fixed margin fallback ────────────────────────
    print("  Using fixed margin fallback")
    top    = 210
    bottom = h - 410
    step   = (bottom - top) // ROWS
    h_fb   = [top + i * step for i in range(ROWS + 1)]

    usable_w = w - LEFT_MARGIN - RIGHT_MARGIN
    cell_w   = usable_w // COLS
    v_fb     = [LEFT_MARGIN + col * cell_w for col in range(COLS + 1)]

    cells = []
    for row_idx in range(ROWS):
        for col_idx in range(COLS):
            x1 = v_fb[col_idx]     + inset
            y1 = h_fb[row_idx]     + inset
            x2 = v_fb[col_idx + 1] - inset
            y2 = h_fb[row_idx + 1] - inset
            cells.append((x1, y1, x2, y2, row_idx, col_idx))

    return cells


# ─────────────────────────────────────────────
# Image processing
# ─────────────────────────────────────────────

def load_and_threshold(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )
    return img, gray, thresh


def is_empty(cell_thresh):
    return (255 - cell_thresh).sum() < MIN_INK


def extract_symbol(cell_gray, split_y):
    """
    Extract symbol from upper portion of an already-inset cell.
    The cell has already been inset from the border, so no additional
    shaving is needed — just crop to split_y and centre on canvas.
    """
    region = cell_gray[:split_y, :]

    if region.size == 0:
        return np.ones((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8) * 255

    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv    = 255 - binary
    coords = cv2.findNonZero(inv)

    if coords is None:
        return np.ones((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8) * 255

    x, y, bw, bh = cv2.boundingRect(coords)
    cropped  = binary[y:y+bh, x:x+bw]
    inner    = OUTPUT_SIZE - 2 * PADDING
    scale    = min(inner / max(bw, 1), inner / max(bh, 1))
    new_w    = max(1, int(bw * scale))
    new_h    = max(1, int(bh * scale))
    resized  = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas   = np.ones((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8) * 255
    ox = (OUTPUT_SIZE - new_w) // 2
    oy = (OUTPUT_SIZE - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = resized
    return canvas


def ocr_label(cell_gray, split_y):
    """
    OCR the label region (below split_y).
    PSM 6 handles multi-line labels correctly.
    """
    if not OCR_AVAILABLE:
        return ""

    region = cell_gray[split_y:, :]
    if region.size == 0:
        return ""

    enlarged = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text  = pytesseract.image_to_string(binary, config="--psm 6 --oem 3")
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        return " ".join(lines).lower()
    except Exception:
        return ""


# ─────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────

def save_preview(img, cells, page_id):
    """
    Draw the detected + inset cell boundaries on the image.
    Green = extraction zone (already inset from border)
    Orange = symbol / label split line
    """
    preview = img.copy()
    for (x1, y1, x2, y2, row, col) in cells:
        split_y = y1 + int((y2 - y1) * SYMBOL_RATIO)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 90), 2)
        cv2.line(preview, (x1, split_y), (x2, split_y), (0, 140, 255), 2)
    out = OUTPUT_DIR / "previews" / f"{page_id}_preview.png"
    cv2.imwrite(str(out), preview)
    return out


# ─────────────────────────────────────────────
# Single page extraction
# ─────────────────────────────────────────────

def extract_page(image_path, page_id=None, preview=False, logger=None):
    image_path = Path(image_path)
    page_id    = page_id or image_path.stem
    crops_dir  = OUTPUT_DIR / "crops"

    img, gray, thresh = load_and_threshold(image_path)
    cells = get_cell_boundaries(img)

    if preview:
        out = save_preview(img, cells, page_id)
        msg = f"  Preview saved: {out.resolve()}"
        print(msg) if not logger else logger._write(msg)

    metadata  = []
    extracted = 0
    empty     = 0

    for (x1, y1, x2, y2, row, col) in cells:
        cell_gray   = gray[y1:y2, x1:x2]
        cell_thresh = thresh[y1:y2, x1:x2]

        if cell_gray.size == 0:
            continue
        if is_empty(cell_thresh):
            empty += 1
            continue

        cell_h     = y2 - y1
        split_y    = int(cell_h * SYMBOL_RATIO)
        symbol_img = extract_symbol(cell_gray, split_y)
        label_text = ocr_label(cell_gray, split_y)

        filename = f"{page_id}_r{row:02d}_c{col:02d}.png"
        cv2.imwrite(str(crops_dir / filename), symbol_img)

        metadata.append({
            "file":        filename,
            "page":        page_id,
            "row":         row,
            "col":         col,
            "symbol_name": label_text,
            "tags":        [],
            "notes":       ""
        })
        extracted += 1

    return metadata, extracted, empty


# ─────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────

def batch_extract(input_dir):
    setup_folders()
    input_dir = Path(input_dir)
    log       = Logger(OUTPUT_DIR / "log.txt")
    exts      = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    pages     = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)

    if not pages:
        log._write(f"No images found in {input_dir}")
        return

    log._write(f"Found {len(pages)} pages")
    log._write(f"OCR         : {'enabled — PSM 6 multi-line' if OCR_AVAILABLE else 'disabled'}")
    log._write(f"Border inset: {BORDER_INSET}px per edge")

    meta_path    = OUTPUT_DIR / "metadata.json"
    all_meta     = []
    already_done = set()

    if meta_path.exists():
        with open(meta_path) as f:
            all_meta = json.load(f)
        already_done = {r["page"] for r in all_meta}
        log._write(f"Resuming — {len(already_done)} pages already done")

    total_symbols = len(all_meta)
    skipped_pages = 0
    error_pages   = 0
    processed     = 0

    for i, page_path in enumerate(pages, 1):
        page_id = page_path.stem
        log.page_start(page_id, i, len(pages))

        if page_id in already_done:
            log.page_skipped(page_id, "already extracted")
            skipped_pages += 1
            continue

        try:
            meta, extracted, empty = extract_page(
                page_path, page_id=page_id, logger=log
            )

            if extracted == 0:
                log.page_skipped(
                    page_id,
                    f"no symbols found ({empty} empty cells — likely intro or index page)"
                )
                skipped_pages += 1
                continue

            all_meta.extend(meta)
            total_symbols += extracted
            processed     += 1
            log.page_done(page_id, extracted, empty)

            with open(meta_path, "w") as f:
                json.dump(all_meta, f, indent=2)

        except Exception as e:
            log.page_error(page_id, str(e))
            error_pages += 1
            continue

    log.summary(processed, total_symbols, skipped_pages, error_pages)


# ─────────────────────────────────────────────
# Re-crop existing crops
# ─────────────────────────────────────────────

def recrop_existing():
    """
    Re-apply clean extraction to already-saved crops.
    Useful if the batch ran before the border inset logic was in place.
    Simply re-centres each existing crop on a fresh canvas with PADDING.
    """
    crops_dir = OUTPUT_DIR / "crops"
    pngs      = list(crops_dir.glob("*.png"))

    if not pngs:
        print("No crops found.")
        return

    print(f"Re-cropping {len(pngs)} existing crops...")
    done = 0

    for png in pngs:
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv    = 255 - binary
        coords = cv2.findNonZero(inv)

        if coords is None:
            continue

        x, y, bw, bh = cv2.boundingRect(coords)
        cropped  = binary[y:y+bh, x:x+bw]
        inner    = OUTPUT_SIZE - 2 * PADDING
        scale    = min(inner / max(bw, 1), inner / max(bh, 1))
        new_w    = max(1, int(bw * scale))
        new_h    = max(1, int(bh * scale))
        resized  = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas   = np.ones((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8) * 255
        ox = (OUTPUT_SIZE - new_w) // 2
        oy = (OUTPUT_SIZE - new_h) // 2
        canvas[oy:oy+new_h, ox:ox+new_w] = resized

        cv2.imwrite(str(png), canvas)
        done += 1

    print(f"Done — {done} crops updated")


# ─────────────────────────────────────────────
# Tagging UI
# ─────────────────────────────────────────────

def run_tagging_ui():
    setup_folders()
    crops_dir = OUTPUT_DIR / "crops"
    meta_path = OUTPUT_DIR / "metadata.json"

    if not meta_path.exists():
        print("No metadata.json found. Run --batch first.")
        return

    with open(meta_path) as f:
        records = json.load(f)

    tagged   = sum(1 for r in records if r.get("tags") and "delete" not in r["tags"])
    untagged = sum(1 for r in records if not r.get("tags"))

    print(f"\n{'='*55}")
    print(f"Event Glyph — Tagging UI")
    print(f"{'='*55}")
    print(f"Total    : {len(records)}")
    print(f"Tagged   : {tagged}")
    print(f"Remaining: {untagged}")
    print(TAXONOMY_HINT)

    history    = []
    key_buffer = ""

    for i, record in enumerate(records):
        if record.get("tags") and "unsure" not in record["tags"]:
            continue

        img_path = crops_dir / record["file"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Redraw display with current key buffer
        while True:
            display = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)

            # Progress bar
            progress = int((i / len(records)) * 400)
            cv2.rectangle(display, (0, 0), (progress, 5), (0, 180, 90), -1)

            # Info
            name = (record.get("symbol_name") or record["file"])[:40]
            cv2.putText(display, f"{i+1}/{len(records)}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
            cv2.putText(display, name,
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)

            # Key buffer hint
            if key_buffer:
                cv2.putText(display, f"key: {key_buffer}_",
                            (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 80, 0), 1)

            if record.get("tags"):
                cv2.putText(display, " + ".join(record["tags"]),
                            (8, 386), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 0), 1)

            cv2.imshow("Event Glyph — Tagger  [q=quit]", display)
            key  = cv2.waitKey(0) & 0xFF
            char = chr(key) if key < 128 else ""

            if key == ord("q"):
                cv2.destroyAllWindows()
                # Save and exit immediately
                with open(meta_path, "w") as f:
                    json.dump(records, f, indent=2)
                print(f"\nSaved and quit — {tagged} tagged so far")
                return

            elif key == ord("z") and history:
                prev = history.pop()
                records[prev]["tags"] = []
                key_buffer = ""
                print(f"  Undid: {records[prev].get('symbol_name') or records[prev]['file']}")
                break  # redisplay same symbol

            elif key == ord(" "):
                key_buffer = ""
                break  # skip to next

            elif char == "d":
                record["tags"] = ["delete"]
                history.append(i)
                key_buffer = ""
                print(f"  {name} → delete")
                break

            elif char == "u":
                record["tags"] = ["unsure"]
                history.append(i)
                key_buffer = ""
                print(f"  {name} → unsure")
                break

            elif char.isalpha():
                key_buffer += char
                if len(key_buffer) == 1:
                    # Wait for second character — redraw with buffer hint
                    continue
                elif len(key_buffer) == 2:
                    if key_buffer in TAXONOMY:
                        tag = TAXONOMY[key_buffer]
                        record["tags"] = [tag]
                        history.append(i)
                        print(f"  {name} → {tag}")
                        key_buffer = ""
                        break
                    else:
                        print(f"  Unknown key [{key_buffer}] — try again")
                        key_buffer = ""
                        continue

    cv2.destroyAllWindows()

    # Save metadata
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)

    # Sort tagged crops into chapter subfolders
    print("\nSorting crops into chapter folders...")
    copied = 0
    for record in records:
        if not record.get("tags"):
            continue
        tag = record["tags"][0]
        if tag in ("delete", "unsure"):
            continue
        src = crops_dir / record["file"]
        dst = OUTPUT_DIR / "tagged" / tag / record["file"]
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    tagged_total = sum(1 for r in records if r.get("tags") and "delete" not in r["tags"])
    print(f"\nDone")
    print(f"  {tagged_total} symbols tagged")
    print(f"  {copied} crops sorted into tagged/ chapter folders")
    print(f"\nLoad in PyTorch with:")
    print(f"  datasets.ImageFolder(root='{OUTPUT_DIR}/tagged', transform=transform)")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Glyph — Symbol Extractor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preview", action="store_true",
                       help="Test grid detection on a single page")
    group.add_argument("--batch",   action="store_true",
                       help="Extract symbols from all pages in a folder")
    group.add_argument("--tag",     action="store_true",
                       help="Run the tagging UI")
    group.add_argument("--recrop",  action="store_true",
                       help="Re-apply clean extraction to existing crops")

    parser.add_argument("--input", help="Image file (--preview) or folder (--batch)")
    args = parser.parse_args()

    if args.preview:
        if not args.input:
            print("--preview requires --input <image path>")
            sys.exit(1)
        setup_folders()
        extract_page(args.input, preview=True)
        print(f"Check: {(OUTPUT_DIR / 'previews').resolve()}")

    elif args.batch:
        if not args.input:
            print("--batch requires --input <folder path>")
            sys.exit(1)
        batch_extract(args.input)

    elif args.tag:
        run_tagging_ui()

    elif args.recrop:
        recrop_existing()