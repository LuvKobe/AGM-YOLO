from __future__ import annotations

from pathlib import Path

from tools.prepare_vsb import parse_annotation


def test_vsb_decimal_comma_ltrb(tmp_path: Path):
    f = tmp_path / "1_anno.txt"
    f.write_text("Knot_OK 0,10 0,20 0,30 0,60\nQuartzity 0,1 0,1 0,2 0,2\n", encoding="utf-8")
    labels, excluded, unknown = parse_annotation(f)
    assert len(labels) == 1
    c, xc, yc, w, h = labels[0]
    assert c == 0
    assert abs(xc - 0.20) < 1e-9
    assert abs(yc - 0.40) < 1e-9
    assert abs(w - 0.20) < 1e-9
    assert abs(h - 0.40) < 1e-9
    assert excluded["Quartzity"] == 1
    assert not unknown
