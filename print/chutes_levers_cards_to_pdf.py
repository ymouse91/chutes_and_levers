#!/usr/bin/env python3
"""
Generate printable Chutes & Levers puzzle cards from challenges.json.

Output layout:
- A4 portrait pages
- 9 cards per page (3 x 3)
- Card size 63.5 mm x 89 mm
- Fronts on one page, matching backs on the next page
- Back positions are mirrored horizontally: (row, col) -> (row, 2-col)
- Landscape puzzles are rotated 90 degrees clockwise to print in portrait
- START / GOAL labels rotate with the puzzle so they read correctly when the
  physical card is turned to landscape for play.
- The pawn is not drawn.

Usage:
    python chutes_levers_cards_to_pdf.py \
        --input challenges.json \
        --output c_and_l_cards.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = A4
CARD_W = 63.5 * mm
CARD_H = 89.0 * mm
SHEET_COLS = 3
SHEET_ROWS = 3
GRID_STROKE = 0.6
WALL_STROKE = 2.0
CHUTE_STROKE = 2.6
LEVER_STROKE = 1.9
CARD_BORDER = 0.8
WRAP_PREVIEW_FRAC = 0.25


@dataclass
class Slot:
    x: float
    y: float
    w: float
    h: float
    row: int
    col: int


@dataclass
class PreparedSide:
    side: Dict
    cols: int
    rows: int
    label_rotation_deg: int


@dataclass
class PreparedPuzzle:
    name: str
    front: PreparedSide
    back: PreparedSide


CORNER_OFFSETS = {
    "TL": (0.25, 0.75),
    "TR": (0.75, 0.75),
    "BL": (0.25, 0.25),
    "BR": (0.75, 0.25),
}

ROTATE_CORNER_CW = {
    "TL": "TR",
    "TR": "BR",
    "BR": "BL",
    "BL": "TL",
}

ROTATE_CORNER_CCW = {
    "TL": "BL",
    "BL": "BR",
    "BR": "TR",
    "TR": "TL",
}

CHUTE_COLORS = [
    colors.HexColor("#2ecc71"),
    colors.HexColor("#9b59b6"),
    colors.HexColor("#3498db"),
    colors.HexColor("#e67e22"),
    colors.HexColor("#e84393"),
    colors.HexColor("#1abc9c"),
    colors.HexColor("#f1c40f"),
    colors.HexColor("#e74c3c"),
    colors.HexColor("#00b894"),
    colors.HexColor("#6c5ce7"),
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deduce_side_size(side: Dict) -> Tuple[int, int]:
    max_col = -1
    max_row = -1

    for wall in side.get("walls", []):
        max_col = max(max_col, wall.get("c", -1))
        max_row = max(max_row, wall.get("r", -1))

    for item in side.get("chutes", []):
        max_col = max(max_col, item.get("col", -1))
        max_row = max(max_row, item.get("row", -1))

    for item in side.get("levers", []):
        max_col = max(max_col, item.get("col", -1))
        max_row = max(max_row, item.get("row", -1))

    for key in ("start", "goal"):
        item = side.get(key)
        if item:
            max_col = max(max_col, item.get("col", -1))
            max_row = max(max_row, item.get("row", -1))

    if max_col < 0 or max_row < 0:
        return (0, 0)
    return (max_col + 1, max_row + 1)


def fallback_side_size(primary: Dict, secondary: Dict, initial_orientation: int) -> Tuple[int, int]:
    cols, rows = deduce_side_size(primary)
    if cols and rows:
        return cols, rows

    other_cols, other_rows = deduce_side_size(secondary)
    if other_cols and other_rows:
        return other_cols, other_rows

    # Final fallback from orientation convention used in this project.
    return (7, 5) if initial_orientation == 1 else (5, 7)


def wall_segments(side: Dict) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    segments = []
    for wall in side.get("walls", []):
        r = wall["r"]
        c = wall["c"]
        if wall.get("B"):
            segments.append(((c, r + 1), (c + 1, r + 1)))
        if wall.get("R"):
            segments.append(((c + 1, r), (c + 1, r + 1)))
    return segments


def rotate_point_cw(x: int, y: int, old_rows: int) -> Tuple[int, int]:
    return old_rows - y, x


def rotate_cell_cw(col: int, row: int, old_rows: int) -> Tuple[int, int]:
    return old_rows - 1 - row, col


def rotate_wall_cw(wall_name: str) -> str:
    mapping = {"T": "R", "R": "B", "B": "L", "L": "T"}
    return mapping[wall_name]


def rotate_wall_ccw(wall_name: str) -> str:
    mapping = {"T": "L", "L": "B", "B": "R", "R": "T"}
    return mapping[wall_name]


def rotate_corner_cw(corner: str) -> str:
    return ROTATE_CORNER_CW[corner]


def rotate_corner_ccw(corner: str) -> str:
    return ROTATE_CORNER_CCW[corner]


def segments_to_walls(segments: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Dict]:
    wall_map: Dict[Tuple[int, int], Dict[str, bool]] = {}

    for (x1, y1), (x2, y2) in segments:
        if y1 == y2:  # horizontal -> bottom wall of cell above
            y = y1
            c1, c2 = sorted((x1, x2))
            for c in range(c1, c2):
                cell = (y - 1, c)
                wall_map.setdefault(cell, {})["B"] = True
        elif x1 == x2:  # vertical -> right wall of cell left of it
            x = x1
            r1, r2 = sorted((y1, y2))
            for r in range(r1, r2):
                cell = (r, x - 1)
                wall_map.setdefault(cell, {})["R"] = True
        else:
            raise ValueError("Wall segment is not axis-aligned")

    result = []
    for (r, c), flags in sorted(wall_map.items()):
        entry = {"r": r, "c": c}
        entry.update(flags)
        result.append(entry)
    return result


def rotate_side_cw(side: Dict, cols: int, rows: int) -> Tuple[Dict, int, int]:
    new_side = {
        "walls": [],
        "start": None,
        "goal": None,
        "chutes": [],
        "levers": [],
    }

    rotated_segments = []
    for p1, p2 in wall_segments(side):
        rp1 = rotate_point_cw(*p1, old_rows=rows)
        rp2 = rotate_point_cw(*p2, old_rows=rows)
        rotated_segments.append((rp1, rp2))
    new_side["walls"] = segments_to_walls(rotated_segments)

    for key in ("start", "goal"):
        item = side.get(key)
        if item:
            c, r = rotate_cell_cw(item["col"], item["row"], old_rows=rows)
            new_side[key] = {"col": c, "row": r}

    for item in side.get("chutes", []):
        c, r = rotate_cell_cw(item["col"], item["row"], old_rows=rows)
        new_side["chutes"].append(
            {
                "col": c,
                "row": r,
                "wall": rotate_wall_cw(item["wall"]),
            }
        )

    for item in side.get("levers", []):
        c, r = rotate_cell_cw(item["col"], item["row"], old_rows=rows)
        new_side["levers"].append(
            {
                "col": c,
                "row": r,
                "corner": rotate_corner_cw(item["corner"]),
            }
        )

    return new_side, rows, cols


def rotate_point_ccw(x: int, y: int, old_cols: int) -> Tuple[int, int]:
    return y, old_cols - x


def rotate_cell_ccw(col: int, row: int, old_cols: int) -> Tuple[int, int]:
    return row, old_cols - 1 - col


def rotate_side_ccw(side: Dict, cols: int, rows: int) -> Tuple[Dict, int, int]:
    new_side = {
        "walls": [],
        "start": None,
        "goal": None,
        "chutes": [],
        "levers": [],
    }

    rotated_segments = []
    for p1, p2 in wall_segments(side):
        rp1 = rotate_point_ccw(*p1, old_cols=cols)
        rp2 = rotate_point_ccw(*p2, old_cols=cols)
        rotated_segments.append((rp1, rp2))
    new_side["walls"] = segments_to_walls(rotated_segments)

    for key in ("start", "goal"):
        item = side.get(key)
        if item:
            c, r = rotate_cell_ccw(item["col"], item["row"], old_cols=cols)
            new_side[key] = {"col": c, "row": r}

    for item in side.get("chutes", []):
        c, r = rotate_cell_ccw(item["col"], item["row"], old_cols=cols)
        new_side["chutes"].append(
            {
                "col": c,
                "row": r,
                "wall": rotate_wall_ccw(item["wall"]),
            }
        )

    for item in side.get("levers", []):
        c, r = rotate_cell_ccw(item["col"], item["row"], old_cols=cols)
        new_side["levers"].append(
            {
                "col": c,
                "row": r,
                "corner": rotate_corner_ccw(item["corner"]),
            }
        )

    return new_side, rows, cols


def prepare_puzzle(puzzle: Dict) -> PreparedPuzzle:
    initial_orientation = int(puzzle.get("initialOrientation", 0))
    front_raw = deepcopy(puzzle["card"]["front"])
    back_raw = deepcopy(puzzle["card"]["back"])

    front_cols, front_rows = fallback_side_size(front_raw, back_raw, initial_orientation)
    back_cols, back_rows = fallback_side_size(back_raw, front_raw, initial_orientation)

    rotate_to_portrait = front_cols > front_rows or initial_orientation == 1
    front_label_rotation_deg = -90 if rotate_to_portrait else 0
    back_label_rotation_deg = 90 if rotate_to_portrait else 0

    if rotate_to_portrait:
        front_raw, front_cols, front_rows = rotate_side_cw(front_raw, front_cols, front_rows)
        back_raw, back_cols, back_rows = rotate_side_ccw(back_raw, back_cols, back_rows)

    return PreparedPuzzle(
        name=puzzle.get("name", puzzle.get("id", "Unnamed")),
        front=PreparedSide(front_raw, front_cols, front_rows, front_label_rotation_deg),
        back=PreparedSide(back_raw, back_cols, back_rows, back_label_rotation_deg),
    )


def make_slots() -> List[Slot]:
    total_w = SHEET_COLS * CARD_W
    total_h = SHEET_ROWS * CARD_H
    margin_x = (PAGE_W - total_w) / 2.0
    margin_y = (PAGE_H - total_h) / 2.0

    slots = []
    for row in range(SHEET_ROWS):
        for col in range(SHEET_COLS):
            x = margin_x + col * CARD_W
            y = PAGE_H - margin_y - (row + 1) * CARD_H
            slots.append(Slot(x=x, y=y, w=CARD_W, h=CARD_H, row=row, col=col))
    return slots


def slot_for_back(front_slot: Slot, slots: List[Slot]) -> Slot:
    target_row = front_slot.row
    target_col = SHEET_COLS - 1 - front_slot.col
    for slot in slots:
        if slot.row == target_row and slot.col == target_col:
            return slot
    raise KeyError("Back slot not found")


def draw_rotated_text(
    c: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    font_name: str,
    font_size: float,
    angle_deg: float,
    fill_color=colors.black,
    align: str = "center",
) -> None:
    c.saveState()
    c.translate(x, y)
    c.rotate(angle_deg)
    c.setFillColor(fill_color)
    c.setFont(font_name, font_size)
    if align == "center":
        c.drawCentredString(0, -font_size * 0.35, text)
    elif align == "left":
        c.drawString(0, -font_size * 0.35, text)
    else:
        raise ValueError(f"Unsupported alignment: {align}")
    c.restoreState()


def draw_arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, size: float) -> None:
    angle = math.atan2(y2 - y1, x2 - x1)
    a1 = angle + math.radians(160)
    a2 = angle - math.radians(160)
    c.line(x2, y2, x2 + size * math.cos(a1), y2 + size * math.sin(a1))
    c.line(x2, y2, x2 + size * math.cos(a2), y2 + size * math.sin(a2))


def draw_board_contents(
    c: canvas.Canvas,
    side: PreparedSide,
    board_x: float,
    board_y: float,
    cell: float,
    cols: int,
    rows: int,
    shift_cols: int = 0,
    shift_rows: int = 0,
) -> None:
    board_w = cell * cols
    board_h = cell * rows

    # grid
    c.saveState()
    c.setLineWidth(GRID_STROKE)
    c.setStrokeColor(colors.HexColor("#a0a0a0"))
    for i in range(cols + 1):
        x = board_x + (i + shift_cols) * cell
        c.line(x, board_y + shift_rows * cell, x, board_y + board_h + shift_rows * cell)
    for j in range(rows + 1):
        y = board_y + (j + shift_rows) * cell
        c.line(board_x + shift_cols * cell, y, board_x + board_w + shift_cols * cell, y)
    c.restoreState()

    # walls
    c.saveState()
    c.setLineWidth(WALL_STROKE)
    c.setStrokeColor(colors.black)
    for wall in side.side.get("walls", []):
        r = wall["r"]
        col = wall["c"]
        x = board_x + (col + shift_cols) * cell
        y_top = board_y + board_h - (r + shift_rows) * cell
        y_bottom = y_top - cell
        if wall.get("B"):
            c.line(x, y_bottom, x + cell, y_bottom)
        if wall.get("R"):
            c.line(x + cell, y_bottom, x + cell, y_top)
    c.restoreState()

    # chutes
    chutes = side.side.get("chutes", [])
    for idx, chute in enumerate(chutes):
        c.saveState()
        color = CHUTE_COLORS[idx % len(CHUTE_COLORS)]
        c.setStrokeColor(color)
        c.setLineWidth(CHUTE_STROKE)
        col = chute["col"]
        row = chute["row"]
        wall_name = chute["wall"]
        x = board_x + (col + shift_cols) * cell
        y_top = board_y + board_h - (row + shift_rows) * cell
        y_bottom = y_top - cell
        mx = x + cell / 2.0
        my = y_bottom + cell / 2.0
        inset = cell * 0.08
        arrow = cell * 0.18
        stem_offset = cell * 0.04

        if wall_name == "T":
            sx, sy = mx, my + stem_offset
            ex, ey = mx, y_top - inset
        elif wall_name == "B":
            sx, sy = mx, my - stem_offset
            ex, ey = mx, y_bottom + inset
        elif wall_name == "L":
            sx, sy = mx - stem_offset, my
            ex, ey = x + inset, my
        elif wall_name == "R":
            sx, sy = mx + stem_offset, my
            ex, ey = x + cell - inset, my
        else:
            c.restoreState()
            continue

        draw_arrow(c, sx, sy, ex, ey, arrow)
        c.restoreState()

    # levers
    for lever in side.side.get("levers", []):
        c.saveState()
        c.setStrokeColor(colors.HexColor("#7d3cff"))
        c.setFillColor(colors.HexColor("#7d3cff"))
        c.setLineWidth(LEVER_STROKE)

        col = lever["col"]
        row = lever["row"]
        corner = lever["corner"]
        x = board_x + (col + shift_cols) * cell
        y_top = board_y + board_h - (row + shift_rows) * cell
        y_bottom = y_top - cell

        corner_x = x + (cell * 0.14 if corner in ("TL", "BL") else cell * 0.86)
        corner_y = y_bottom + (cell * 0.86 if corner in ("TL", "TR") else cell * 0.14)
        dx = cell * (0.14 if corner in ("TL", "BL") else -0.14)
        dy = cell * (-0.14 if corner in ("TL", "TR") else 0.14)
        end_x = corner_x + dx
        end_y = corner_y + dy
        c.line(corner_x, corner_y, end_x, end_y)
        c.circle(corner_x, corner_y, cell * 0.038, stroke=1, fill=1)
        c.restoreState()

    # start / goal labels
    for key, fill in (("start", colors.HexColor("#0f8f3a")), ("goal", colors.HexColor("#c62828"))):
        item = side.side.get(key)
        if not item:
            continue
        col = item["col"]
        row = item["row"]
        cx = board_x + (col + shift_cols + 0.5) * cell
        cy = board_y + board_h - (row + shift_rows + 0.5) * cell
        text = key.upper()
        font_size = min(10, max(6.5, cell * 0.22))
        draw_rotated_text(
            c,
            cx,
            cy,
            text,
            "Helvetica-Bold",
            font_size,
            side.label_rotation_deg,
            fill_color=fill,
            align="center",
        )


def draw_grid_side(c: canvas.Canvas, side: PreparedSide, card_x: float, card_y: float, card_w: float, card_h: float, puzzle_name: str = "") -> None:
    # Card frame
    c.saveState()
    c.setStrokeColor(colors.black)
    c.setLineWidth(CARD_BORDER)
    c.rect(card_x, card_y, card_w, card_h, stroke=1, fill=0)
    c.restoreState()

    cols = side.cols or 5
    rows = side.rows or 7

    # Maximize the playable board area while keeping the puzzle name on the card.
    # The torus preview strips are drawn outside the main board, so the margins
    # need just enough room for those strips and the lower caption.
    pad_x = 1.2 * mm
    pad_y = 1.8 * mm
    name_h = 6.5 * mm

    inner_x = card_x + pad_x
    inner_w = card_w - 2 * pad_x
    inner_y = card_y + pad_y
    inner_h = card_h - 2 * pad_y - name_h

    cell = min(inner_w / cols, inner_h / rows)
    board_w = cell * cols
    board_h = cell * rows
    board_x = inner_x + (inner_w - board_w) / 2.0
    board_y = inner_y + (inner_h - board_h) / 2.0 + 2 * mm

    # puzzle name at top
    name_font_size = 7.5
    name_y = card_y + card_h - pad_y - name_font_size * 0.75 + 1 * mm
    c.saveState()
    c.setFont("Helvetica-Bold", name_font_size)
    c.setFillColor(colors.HexColor("#222222"))
    c.drawCentredString(card_x + card_w / 2.0, name_y, puzzle_name)
    c.restoreState()

    preview = cell * WRAP_PREVIEW_FRAC

    # Lines that sit exactly on the boundary between the main board and
    # torus-preview strips must not be clipped in half. Expand each clip
    # rectangle slightly, but never outside the current card, so thick walls
    # keep the same visual width and nothing can bleed into neighbouring cards.
    clip_bleed = max(WALL_STROKE, CHUTE_STROKE, LEVER_STROKE, GRID_STROKE) * 2.0

    def clip_to_card_rect(rx: float, ry: float, rw: float, rh: float, bleed: float = 0.0):
        ix = max(rx - bleed, card_x)
        iy = max(ry - bleed, card_y)
        ix2 = min(rx + rw + bleed, card_x + card_w)
        iy2 = min(ry + rh + bleed, card_y + card_h)
        if ix2 <= ix or iy2 <= iy:
            return None
        return ix, iy, ix2 - ix, iy2 - iy

    # main board
    main_clip = clip_to_card_rect(board_x, board_y, board_w, board_h, clip_bleed)
    if main_clip:
        c.saveState()
        p = c.beginPath()
        p.rect(*main_clip)
        c.clipPath(p, stroke=0, fill=0)
        draw_board_contents(c, side, board_x, board_y, cell, cols, rows, 0, 0)
        c.restoreState()

    # torus previews: opposite edges shown outside the board
    preview_clips = [
        (board_x - preview, board_y, preview, board_h, -cols, 0),
        (board_x + board_w, board_y, preview, board_h, cols, 0),
        (board_x, board_y + board_h, board_w, preview, 0, -rows),
        (board_x, board_y - preview, board_w, preview, 0, rows),
        (board_x - preview, board_y + board_h, preview, preview, -cols, -rows),
        (board_x + board_w, board_y + board_h, preview, preview, cols, -rows),
        (board_x - preview, board_y - preview, preview, preview, -cols, rows),
        (board_x + board_w, board_y - preview, preview, preview, cols, rows),
    ]

    for clip_x, clip_y, clip_w, clip_h, shift_cols, shift_rows in preview_clips:
        expanded_clip = clip_to_card_rect(clip_x, clip_y, clip_w, clip_h, clip_bleed)
        if not expanded_clip:
            continue
        c.saveState()
        p = c.beginPath()
        p.rect(*expanded_clip)
        c.clipPath(p, stroke=0, fill=0)
        draw_board_contents(c, side, board_x, board_y, cell, cols, rows, shift_cols, shift_rows)
        c.restoreState()




def draw_sheet(
    c: canvas.Canvas,
    puzzles: List[PreparedPuzzle],
    slots: List[Slot],
    draw_fronts: bool,
    start_index: int,
) -> None:
    for offset, puzzle in enumerate(puzzles):
        front_slot = slots[offset]
        slot = front_slot if draw_fronts else slot_for_back(front_slot, slots)
        side = puzzle.front if draw_fronts else puzzle.back
        draw_grid_side(c, side, slot.x, slot.y, slot.w, slot.h, puzzle_name=puzzle.name)

    # page footer
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#555555"))
    sheet_num = (start_index // 9) + 1
    side_text = "Fronts" if draw_fronts else "Backs"
    c.drawCentredString(PAGE_W / 2.0, 6 * mm, f"Sheet {sheet_num} - {side_text}")
    c.restoreState()

    c.showPage()


def build_pdf(puzzles: List[PreparedPuzzle], output_path: Path) -> None:
    c = canvas.Canvas(str(output_path), pagesize=A4)
    slots = make_slots()

    for start in range(0, len(puzzles), 9):
        batch = puzzles[start : start + 9]
        draw_sheet(c, batch, slots, draw_fronts=True, start_index=start)
        draw_sheet(c, batch, slots, draw_fronts=False, start_index=start)

    c.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chutes & Levers puzzle cards PDF")
    parser.add_argument("--input", default="challenges.json", help="Path to challenges.json")
    parser.add_argument("--output", default="c_and_l_cards.pdf", help="Output PDF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    data = load_json(input_path)
    raw_puzzles = data.get("puzzles", [])
    prepared = [prepare_puzzle(p) for p in raw_puzzles]
    build_pdf(prepared, output_path)

    print(f"Loaded {len(prepared)} puzzles from {input_path}")
    print(f"Wrote PDF to {output_path}")


if __name__ == "__main__":
    main()
