"""Zoomable page viewer for the citation panel (N9): a sharp page image with the cited passage outlined.

The page is rendered once at high resolution (``PANEL_PAGE_ZOOM`` x 72 dpi) and shown in a small HTML
component, so it stays crisp when zoomed. Highlights are drawn on top as boxes in page coordinates.

Controls: trackpad pinch or Ctrl + mouse wheel zooms where the pointer is, scrolling or dragging moves
around, double-click zooms in, and the toolbar has -, +, Fit and "Go to highlight". Everything runs in
the browser with no external scripts (works offline).
"""
from __future__ import annotations

import base64
import html
import json
from typing import Any, Sequence


def viewer_html(png: bytes, page_width: float, page_height: float, boxes: Sequence[dict[str, Any]],
                height: int, label: str = "") -> str:
    """HTML for one page. ``boxes`` are highlight rectangles in PDF points (x, y, width, height; top-left origin)."""
    image = base64.b64encode(png).decode("ascii")
    rects = [{"x": b["x"] / page_width, "y": b["y"] / page_height,
              "w": b["width"] / page_width, "h": b["height"] / page_height} for b in boxes]
    return _TEMPLATE.replace("__IMG__", image).replace("__RECTS__", json.dumps(rects)) \
        .replace("__RATIO__", f"{page_height / page_width:.6f}").replace("__HEIGHT__", str(height)) \
        .replace("__LABEL__", html.escape(label)).replace("__HAS__", "true" if rects else "false")


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg:#f6f4ef; --ink:#2b2a27; --line:#d9d4c7; --btn:#ffffff; --mark:rgba(255,205,40,.45); --edge:#e08a00; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1d1d1b; --ink:#ece9e1; --line:#3a3934; --btn:#2a2926; } }
  html, body { margin:0; padding:0; background:transparent; font:13px system-ui, -apple-system, Segoe UI, sans-serif; color:var(--ink); }
  .bar { display:flex; gap:6px; align-items:center; margin:0 0 6px; flex-wrap:wrap; }
  .bar button { border:1px solid var(--line); background:var(--btn); color:var(--ink); border-radius:8px;
                padding:3px 10px; cursor:pointer; font:inherit; line-height:1.6; }
  .bar button:hover { border-color:var(--edge); }
  .bar .pct { min-width:46px; text-align:center; opacity:.8; }
  .bar .hint { flex-basis:100%; opacity:.6; font-size:11.5px; }
  .stage { position:relative; height:__HEIGHT__px; overflow:hidden; border:1px solid var(--line); border-radius:10px;
           background:var(--bg); cursor:grab; touch-action:none; user-select:none; }
  .stage.dragging { cursor:grabbing; }
  .page { position:absolute; left:0; top:0; transform-origin:0 0; box-shadow:0 2px 14px rgba(0,0,0,.18); background:#fff; }
  .page img { display:block; width:100%; height:100%; pointer-events:none; }
  .mark { position:absolute; background:var(--mark); mix-blend-mode:multiply; border-bottom:2px solid var(--edge);
          border-radius:2px; pointer-events:none; }
  .mark.pulse { animation:glow 1.2s ease-in-out 2; }
  @keyframes glow { 0%, 100% { opacity:1; } 50% { opacity:.35; } }
</style></head>
<body>
<div class="bar">
  <button id="out" title="Zoom out">−</button><span class="pct" id="pct">100%</span>
  <button id="in" title="Zoom in">+</button>
  <button id="fit" title="Fit the page width">Fit</button>
  <button id="go" title="Zoom to the highlighted passage">Go to highlight</button>
  <span class="hint">Pinch or Ctrl + scroll to zoom · drag to move</span>
</div>
<div class="stage" id="stage" aria-label="__LABEL__">
  <div class="page" id="page"><img alt="__LABEL__" src="data:image/png;base64,__IMG__"></div>
</div>
<script>
(function () {
  const rects = __RECTS__, ratio = __RATIO__, hasMarks = __HAS__;
  const stage = document.getElementById('stage'), page = document.getElementById('page'), pct = document.getElementById('pct');
  const MIN = 0.5, MAX = 6;           // relative to "fit width"
  let base = 1, zoom = 1, tx = 0, ty = 0;

  rects.forEach(r => {
    const m = document.createElement('div');
    m.className = 'mark pulse';
    Object.assign(m.style, { left: (r.x * 100) + '%', top: (r.y * 100) + '%', width: (r.w * 100) + '%', height: (r.h * 100) + '%' });
    page.appendChild(m);
  });
  if (!hasMarks) document.getElementById('go').style.display = 'none';

  function layout() {
    base = stage.clientWidth - 16;            // page width at zoom 1 (fit width, small margin)
    page.style.width = base + 'px';
    page.style.height = (base * ratio) + 'px';
  }
  function clamp() {
    const w = base * zoom, h = base * ratio * zoom, sw = stage.clientWidth, sh = stage.clientHeight;
    tx = w + 16 <= sw ? (sw - w) / 2 : Math.min(8, Math.max(sw - w - 8, tx));
    ty = h + 16 <= sh ? Math.max(8, ty) : Math.min(8, Math.max(sh - h - 8, ty));
    if (h + 16 <= sh) ty = 8;
  }
  function apply() {
    clamp();
    page.style.transform = `translate(${tx}px, ${ty}px) scale(${zoom})`;
    pct.textContent = Math.round(zoom * 100) + '%';
  }
  function zoomAt(next, cx, cy) {
    next = Math.min(MAX, Math.max(MIN, next));
    const px = (cx - tx) / zoom, py = (cy - ty) / zoom;   // page point under the pointer stays put
    zoom = next; tx = cx - px * zoom; ty = cy - py * zoom; apply();
  }
  function centre() { return [stage.clientWidth / 2, stage.clientHeight / 2]; }
  function goToMarks() {
    if (!rects.length) return;
    const x0 = Math.min(...rects.map(r => r.x)), y0 = Math.min(...rects.map(r => r.y));
    const x1 = Math.max(...rects.map(r => r.x + r.w)), y1 = Math.max(...rects.map(r => r.y + r.h));
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const fitW = (sw - 32) / ((x1 - x0) * base), fitH = (sh - 32) / ((y1 - y0) * base * ratio);
    zoom = Math.min(MAX, Math.max(1, Math.min(fitW, fitH, 2.2)));
    tx = sw / 2 - ((x0 + x1) / 2) * base * zoom;
    ty = sh / 2 - ((y0 + y1) / 2) * base * ratio * zoom;
    apply();
  }

  stage.addEventListener('wheel', e => {
    e.preventDefault();
    const box = stage.getBoundingClientRect();
    if (e.ctrlKey || e.metaKey) {                          // trackpad pinch arrives as ctrl + wheel
      zoomAt(zoom * Math.exp(-e.deltaY * 0.01), e.clientX - box.left, e.clientY - box.top);
    } else {
      tx -= e.deltaX; ty -= e.deltaY; apply();
    }
  }, { passive: false });

  const pointers = new Map(); let pinch = null, last = null;
  stage.addEventListener('pointerdown', e => {
    stage.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]);
    stage.classList.add('dragging'); last = [e.clientX, e.clientY];
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), z: zoom };
    }
  });
  stage.addEventListener('pointermove', e => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    const box = stage.getBoundingClientRect();
    if (pointers.size === 2 && pinch) {                   // touch pinch
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      zoomAt(pinch.z * d / pinch.d, (a[0] + b[0]) / 2 - box.left, (a[1] + b[1]) / 2 - box.top);
    } else if (last) {
      tx += e.clientX - last[0]; ty += e.clientY - last[1]; last = [e.clientX, e.clientY]; apply();
    }
  });
  function end(e) { pointers.delete(e.pointerId); if (pointers.size < 2) pinch = null; if (!pointers.size) { stage.classList.remove('dragging'); last = null; } }
  stage.addEventListener('pointerup', end); stage.addEventListener('pointercancel', end);
  stage.addEventListener('dblclick', e => {
    const box = stage.getBoundingClientRect();
    zoomAt(zoom < 1.8 ? zoom * 2 : 1, e.clientX - box.left, e.clientY - box.top);
  });

  document.getElementById('in').onclick = () => zoomAt(zoom * 1.25, ...centre());
  document.getElementById('out').onclick = () => zoomAt(zoom / 1.25, ...centre());
  document.getElementById('fit').onclick = () => { zoom = 1; tx = 8; ty = 8; apply(); };
  document.getElementById('go').onclick = goToMarks;

  new ResizeObserver(() => { const z = zoom; layout(); zoom = z; apply(); }).observe(stage);
  layout(); zoom = 1; tx = 8; ty = 8; apply();
  if (hasMarks) setTimeout(goToMarks, 150);
})();
</script>
</body></html>
"""
