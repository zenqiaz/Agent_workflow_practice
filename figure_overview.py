#!/usr/bin/env python3
"""
QC Agent paper — Figure 1: System Overview
Layout: top banner (architecture) | bottom-left (planning pipeline) | bottom-right (execution + tasks)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = '#F8F9FA'
C_LOCAL = '#D6EAF8'   # blue   — local client
C_REM   = '#D5F5E3'   # green  — remote server
C_LLM   = '#E8DAEF'   # purple — LLM / planner
C_SKILL = '#FEF5E4'   # yellow — skills / chemistry
C_TOOL  = '#EBF5FB'   # pale blue — tools / graph
C_RES   = '#EAFAF1'   # pale green — results / ORCA
C_PLAN  = '#FDEBD0'   # orange — plan output
EDGE    = '#2C3E50'
DARK    = '#1C2833'
GRAY    = '#7F8C8D'
AFWD    = '#1A5276'   # forward arrow  (local → remote)
ABCK    = '#1D6A39'   # return  arrow  (remote → local)
AINT    = '#626567'   # internal arrow

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, ec=EDGE, lw=0.9, r=0.015, zorder=2, **kw):
    """Rounded rectangle. r = corner radius in axis units.
    Compensates position so the outer extent is exactly (x, y)–(x+w, y+h)."""
    rr = min(r, w / 4, h / 4)
    p = FancyBboxPatch(
        (x + rr, y + rr), max(w - 2 * rr, 1e-3), max(h - 2 * rr, 1e-3),
        boxstyle=f'round,pad={rr}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder, **kw)
    ax.add_patch(p)


def arr(ax, x0, y0, x1, y1, c=AINT, lw=1.3, ms=12):
    """Annotate-based arrow in data coordinates."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                mutation_scale=ms))


def txt(ax, x, y, s, fs=7.5, c=DARK, ha='center', va='center',
        bold=False, italic=False, zorder=3, **kw):
    fw = 'bold' if bold else 'normal'
    fi = 'italic' if italic else 'normal'
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c,
            fontweight=fw, fontstyle=fi, zorder=zorder, **kw)


# ── Figure & axes ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
gs = GridSpec(2, 2, figure=fig,
              height_ratios=[0.40, 0.60],
              width_ratios=[0.42, 0.58],
              hspace=0.09, wspace=0.06,
              left=0.02, right=0.98, top=0.97, bottom=0.02)

ax_top = fig.add_subplot(gs[0, :])
ax_bl  = fig.add_subplot(gs[1, 0])
ax_br  = fig.add_subplot(gs[1, 1])

for ax in (ax_top, ax_bl, ax_br):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(BG)


# ═════════════════════════════════════════════════════════════════════════════
# Panel a — System Architecture
# ═════════════════════════════════════════════════════════════════════════════
ax = ax_top
txt(ax, 0.005, 0.97, 'a', fs=13, bold=True, va='top', ha='left')

# ── Local client box ──────────────────────────────────────────────────────────
rbox(ax, 0.02, 0.04, 0.43, 0.88, C_LOCAL, lw=1.5, r=0.04)
txt(ax, 0.235, 0.915, 'Local Client  (Windows)', fs=9, bold=True)

lc = [
    ('REPL Interface',               C_TOOL),
    ('Compound Identification',      C_SKILL),
    ('Skill Dispatch',               C_SKILL),
    ('Planner LLM  (gpt-4.1-mini)',  C_LLM),
    ('LangGraph Orchestrator',       C_PLAN),
]
bw, bh = 0.355, 0.107
lc_ys = [0.795, 0.643, 0.491, 0.339, 0.187]   # box centres

for (label, c), cy in zip(lc, lc_ys):
    rbox(ax, 0.043, cy - bh / 2, bw, bh, c, lw=0.7, r=0.02)
    txt(ax, 0.22, cy, label, fs=7.5)

for i in range(len(lc_ys) - 1):
    arr(ax, 0.22, lc_ys[i] - bh / 2,
            0.22, lc_ys[i + 1] + bh / 2, ms=10)

# ── Remote server box ─────────────────────────────────────────────────────────
rbox(ax, 0.55, 0.04, 0.43, 0.88, C_REM, lw=1.5, r=0.04)
txt(ax, 0.765, 0.915, 'Remote Server  (k8s pod, 36 CPUs)', fs=9, bold=True)

rc = [
    ('FastMCP Server',                C_TOOL),
    ('ORCA 6.1.1  +  OPI',            C_RES),
    ('SOLVATOR  (microsolvation)',    C_RES),
    ('MPI  parallel workers',         C_SKILL),
]
rc_ys = [0.755, 0.560, 0.365, 0.170]

for (label, c), cy in zip(rc, rc_ys):
    rbox(ax, 0.565, cy - bh / 2, 0.400, bh, c, lw=0.7, r=0.02)
    txt(ax, 0.765, cy, label, fs=7.5)

for i in range(len(rc_ys) - 1):
    arr(ax, 0.765, rc_ys[i] - bh / 2,
            0.765, rc_ys[i + 1] + bh / 2, ms=10)

# ── Bidirectional connection arrows ───────────────────────────────────────────
arr(ax, 0.450, 0.63, 0.550, 0.63, c=AFWD, lw=2.2, ms=14)
txt(ax, 0.500, 0.675, 'MCP tools  /  SSH (kubectl exec)',
    fs=7, c=AFWD, italic=True)

arr(ax, 0.550, 0.42, 0.450, 0.42, c=ABCK, lw=2.2, ms=14)
txt(ax, 0.500, 0.375, 'JSON artifacts  /  structured results',
    fs=7, c=ABCK, italic=True)


# ═════════════════════════════════════════════════════════════════════════════
# Panel b — Planning Pipeline
# ═════════════════════════════════════════════════════════════════════════════
ax = ax_bl
txt(ax, 0.005, 0.99, 'b', fs=13, bold=True, va='top', ha='left')
txt(ax, 0.500, 0.975, 'Planning Pipeline', fs=9.5, bold=True, va='top')

pipeline = [
    ('User Natural Language Query',                                             C_TOOL),
    ('Compound Identification\n(LLM extract → PubChem / OPSIN → user confirm)', C_SKILL),
    ('Skill Dispatch\n(method · pKa · thermo · solvation · NBO · TDDFT · scan · TS)',
                                                                                C_SKILL),
    ('Planner LLM  (gpt-4.1-mini)\n+ injected skill context',                  C_LLM),
    ('JSON Workflow Plan',                                                       C_PLAN),
]
pipe_ys = [0.895, 0.725, 0.515, 0.295, 0.095]
pipe_h  = 0.115
pipe_bw = 0.86

for (label, c), cy in zip(pipeline, pipe_ys):
    rbox(ax, 0.07, cy - pipe_h / 2, pipe_bw, pipe_h, c, lw=0.9, r=0.025)
    txt(ax, 0.50, cy, label, fs=7.5, multialignment='center')

for i in range(len(pipe_ys) - 1):
    arr(ax, 0.50, pipe_ys[i] - pipe_h / 2,
            0.50, pipe_ys[i + 1] + pipe_h / 2, ms=11)

# Small right-pointing arrow at the JSON Plan level → crosses into panel c
ax.annotate('', xy=(1.03, 0.095), xytext=(0.93, 0.095),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color=AFWD, lw=1.8,
                            mutation_scale=13),
            clip_on=False)


# ═════════════════════════════════════════════════════════════════════════════
# Panel c — Execution & Task Types
# ═════════════════════════════════════════════════════════════════════════════
ax = ax_br
txt(ax, 0.005, 0.99, 'c', fs=13, bold=True, va='top', ha='left')
txt(ax, 0.500, 0.975, 'Execution & Supported Task Types',
    fs=9.5, bold=True, va='top')

# ── Execution flow strip ──────────────────────────────────────────────────────
eflow = [
    ('JSON\nPlan',          0.018, C_PLAN),
    ('LangGraph\nGraph',    0.208, C_TOOL),
    ('MCP\nTool nodes',     0.398, C_TOOL),
    ('ORCA\n(k8s)',         0.588, C_REM),
    ('Artifacts\n& Plots',  0.778, C_LLM),
]
ew, eh, ey = 0.152, 0.110, 0.845

for label, x, c in eflow:
    rbox(ax, x, ey - eh / 2, ew, eh, c, lw=0.9, r=0.025)
    txt(ax, x + ew / 2, ey, label, fs=7.5, multialignment='center')

for i in range(len(eflow) - 1):
    arr(ax, eflow[i][1] + ew, ey, eflow[i + 1][1], ey, ms=11)

txt(ax, 0.50, 0.715, 'node types:   tool  |  llm  |  calc_expr',
    fs=6.5, c=GRAY, italic=True)

# ── Divider & section header ──────────────────────────────────────────────────
ax.axhline(0.660, color='#CCD1D1', lw=0.8, xmin=0.01, xmax=0.99)
txt(ax, 0.50, 0.635, 'Supported Calculation Types', fs=8.5, bold=True)

# ── Task chips (2 × 4 grid) ───────────────────────────────────────────────────
tasks = [
    ('Geometry\nOptimization',    C_LOCAL),
    ('pKa\nCalculation',          C_LLM),
    ('IR Spectrum\n(freq job)',   C_REM),
    ('UV-Vis\n(TD-DFT)',          C_PLAN),
    ('PES Scan',                  '#FAE5D3'),
    ('TS Search\n(OptTS)',        '#F9EBEA'),
    ('NBO Analysis',              C_SKILL),
    ('Microsolvation\n(SOLVATOR)', C_TOOL),
]

cols = 4
cw, ch = 0.214, 0.115
gx, gy = 0.028, 0.032
x0, y0 = 0.035, 0.587

for i, (label, c) in enumerate(tasks):
    col = i % cols
    row = i // cols
    cx = x0 + col * (cw + gx)
    cy = y0 - row * (ch + gy) - ch
    rbox(ax, cx, cy, cw, ch, c, lw=0.8, r=0.02)
    txt(ax, cx + cw / 2, cy + ch / 2, label, fs=7, multialignment='center')


# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
out = 'D:/brick/D/20260217/working/figure_overview'
fig.savefig(out + '.png', dpi=200, bbox_inches='tight', facecolor=BG)
fig.savefig(out + '.svg', bbox_inches='tight', facecolor=BG)
print(f'Saved:\n  {out}.png\n  {out}.svg')
