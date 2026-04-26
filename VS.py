import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Wedge
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.ticker as ticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── PALETTE (Apple Fitness exact colours) ──────────────────────────────────────
MOVE_COLOR   = '#FA3535'   # Vivid red
EXER_COLOR   = '#A3FF57'   # Bright lime-green
STAND_COLOR  = '#46D0FB'   # Cyan blue
STEPS_COLOR  = '#FFD426'   # Amber yellow
DIST_COLOR   = '#5AC8FA'   # Sky teal
FLOORS_COLOR = '#CF7AFF'   # Soft violet

BG_DARK   = '#0B0C0F'
BG_CARD   = '#13151A'
BG_CARD2  = '#1A1C22'
WHITE     = '#F2F4F8'
MUTED     = '#6B7280'
DIM       = '#2E3038'

# ── DATA ───────────────────────────────────────────────────────────────────────
np.random.seed(42)
N = 30

days_short = [f"{d:02d}" for d in range(1, N+1)]

steps    = np.random.randint(5000, 14000, N).astype(float)
distance = np.round(steps * 0.00070, 2)
floors   = np.random.randint(4, 23, N).astype(float)
move     = np.random.randint(200, 620, N).astype(float)
exercise = np.random.randint(15, 65, N).astype(float)
stand    = np.random.randint(5, 15, N).astype(float)

MOVE_GOAL  = 450
EXER_GOAL  = 30
STAND_GOAL = 12
STEP_GOAL  = 10000
DIST_GOAL  = 6.5
FLOOR_GOAL = 10

# Today's ring values (avg of last 7)
today_move  = int(move[-7:].mean())
today_exer  = int(exercise[-7:].mean())
today_stand = int(stand[-7:].mean())

# ── FIGURE SETUP ───────────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 22
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_DARK, dpi=130)

# Overall grid: header row + content rows
outer = GridSpec(
    5, 1,
    figure=fig,
    hspace=0.38,
    top=0.96, bottom=0.04,
    left=0.04, right=0.97,
    height_ratios=[0.6, 2.6, 1.5, 1.5, 1.5]
)

# ────────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ────────────────────────────────────────────────────────────────────────────────

def set_card_bg(ax, color=BG_CARD, radius=0.03, alpha=1.0):
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_visible(False)

def clear_ax(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

def draw_ring(ax, cx, cy, radius, width, value, goal, color, track_color='#1E2028'):
    """Draw a single Apple-style activity ring with glowing tip."""
    progress = min(value / goal, 1.35)        # allow >100%
    start_angle = 90                           # 12 o'clock
    sweep = 360 * progress

    # Track
    ring_bg = Wedge((cx, cy), radius, 0, 360,
                    width=width, facecolor=track_color, zorder=2)
    ax.add_patch(ring_bg)

    # Gradient simulation: draw many thin arcs to fake a gradient fill
    n_segs = 120
    for i in range(n_segs):
        seg_start = start_angle - (sweep * i / n_segs)
        seg_end   = start_angle - (sweep * (i+1) / n_segs)
        if seg_start == seg_end:
            continue
        t = i / n_segs
        # slightly darken toward the start
        alpha = 0.55 + 0.45 * t
        seg = Wedge((cx, cy), radius, min(seg_end, seg_start),
                    max(seg_end, seg_start), width=width,
                    facecolor=color, alpha=alpha, zorder=3)
        ax.add_patch(seg)

    # Tip glow circle
    if sweep > 5:
        tip_angle_rad = np.radians(start_angle - sweep)
        mid_r = radius - width / 2
        tx = cx + mid_r * np.cos(tip_angle_rad)
        ty = cy + mid_r * np.sin(tip_angle_rad)

        glow = plt.Circle((tx, ty), width * 0.65, color=color, alpha=0.30, zorder=4)
        ax.add_patch(glow)
        tip = plt.Circle((tx, ty), width * 0.45, color=color, alpha=1.0, zorder=5)
        ax.add_patch(tip)

    # Start cap
    start_rad = np.radians(start_angle)
    mid_r = radius - width / 2
    sx = cx + mid_r * np.cos(start_rad)
    sy = cy + mid_r * np.sin(start_rad)
    cap = plt.Circle((sx, sy), width * 0.45, color=color, zorder=5)
    ax.add_patch(cap)


def draw_bar_chart(ax, values, color, goal=None, title='', subtitle='',
                   unit='', goal_label=''):
    """Full-featured bar chart with styled axes."""
    ax.set_facecolor(BG_CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)

    x = np.arange(len(values))
    max_val = max(values) * 1.12

    # Bars
    bar_colors = []
    for v in values:
        if goal and v >= goal:
            bar_colors.append(color)
        else:
            bar_colors.append(color + '66')

    bars = ax.bar(x, values, color=bar_colors, width=0.68,
                  linewidth=0, zorder=3)

    # Round top edges via line trick
    for bar, v in zip(bars, values):
        bx = bar.get_x() + bar.get_width() / 2
        ax.plot([bx], [v], 'o', color=bar.get_facecolor(),
                markersize=bar.get_width() * 18, zorder=2, alpha=0)

    # X-axis ticks: every 5 days
    tick_pos = list(range(0, len(values), 5))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([days_short[i] for i in tick_pos],
                       fontsize=7, color=MUTED, fontfamily='DejaVu Sans')
    ax.tick_params(axis='x', length=0, pad=4)

    # Y-axis minimal
    ax.set_yticks([])
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(0, max_val)

    # Title block
    ax.text(0, 1.14, title, transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', color=WHITE,
            fontfamily='DejaVu Sans', va='top')
    ax.text(0, 1.055, subtitle, transform=ax.transAxes,
            fontsize=6.8, color=MUTED,
            fontfamily='DejaVu Sans', va='top')

    # Current avg badge
    avg = int(np.mean(values))
    unit_clean = unit.strip()
    avg_text = f'Avg {avg:,}'
    if unit_clean:
        avg_text = f'{avg_text} {unit_clean}'

    ax.text(
        0.98, 1.14, avg_text,
        transform=ax.transAxes,
        fontsize=7.2, color=color, ha='right', va='top',
        fontfamily='DejaVu Sans', fontweight='bold',
        clip_on=False,
        bbox=dict(boxstyle='round,pad=0.34', facecolor=BG_CARD2, edgecolor='none', alpha=0.92),
    )

    ax.tick_params(axis='both', which='both', bottom=False, left=False)
    ax.grid(axis='y', color=DIM, linewidth=0.4, alpha=0.5, zorder=0)


def draw_progress_bar(ax, x, y, width, height, pct, color, bg=DIM):
    """Draw a rounded horizontal progress bar."""
    # Background
    bg_bar = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0", facecolor=bg, zorder=2)
    ax.add_patch(bg_bar)
    # Fill
    fill_w = min(pct, 1.0) * width
    if fill_w > 0.001:
        fill = mpatches.FancyBboxPatch(
            (x, y), fill_w, height,
            boxstyle="round,pad=0", facecolor=color, zorder=3)
        ax.add_patch(fill)


# ════════════════════════════════════════════════════════════════════════════════
# ROW 0 — HEADER
# ════════════════════════════════════════════════════════════════════════════════
ax_hdr = fig.add_subplot(outer[0])
ax_hdr.set_facecolor(BG_DARK)
ax_hdr.axis('off')

ax_hdr.text(0.0, 0.72, 'Activity Dashboard',
            transform=ax_hdr.transAxes,
            fontsize=28, fontweight='bold', color=WHITE,
            fontfamily='DejaVu Sans', va='center')
ax_hdr.text(0.0, 0.22, 'January 2026  ·  30 Days  ·  Personal Health & Fitness Metrics',
            transform=ax_hdr.transAxes,
            fontsize=10, color=MUTED, fontfamily='DejaVu Sans', va='center')

# Coloured accent dots
for i, (c, lbl) in enumerate([(MOVE_COLOR,'Move'),(EXER_COLOR,'Exercise'),
                               (STAND_COLOR,'Stand'),(STEPS_COLOR,'Steps'),
                               (DIST_COLOR,'Distance'),(FLOORS_COLOR,'Floors')]):
    xpos = 0.56 + i * 0.075
    ax_hdr.plot(xpos, 0.72, 'o', color=c, markersize=9,
                transform=ax_hdr.transAxes, zorder=5)
    ax_hdr.text(xpos, 0.22, lbl, transform=ax_hdr.transAxes,
                fontsize=6.5, color=c, ha='center', fontfamily='DejaVu Sans')

# Thin separator line
ax_hdr.plot([0, 1], [-0.3, -0.3], color=DIM, linewidth=0.8,
            transform=ax_hdr.transAxes, clip_on=False)


# ════════════════════════════════════════════════════════════════════════════════
# ROW 1 — RINGS PANEL (left) + KPI CARDS (right)
# ════════════════════════════════════════════════════════════════════════════════
inner1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                 wspace=0.06, width_ratios=[1, 1.55])

# ── LEFT: Concentric Rings ───────────────────────────────────────────────────
ax_rings = fig.add_subplot(inner1[0])
ax_rings.set_facecolor(BG_CARD)
ax_rings.set_aspect('equal')
ax_rings.axis('off')
ax_rings.set_xlim(-1.1, 1.1)
ax_rings.set_ylim(-1.45, 1.25)

for sp in ax_rings.spines.values():
    sp.set_visible(False)

# Draw three concentric rings (outer=Move, mid=Exercise, inner=Stand)
ring_specs = [
    (0, 0, 0.95, 0.16, today_move,  MOVE_GOAL,  MOVE_COLOR,  'Move',     f'{today_move}', 'kcal', MOVE_GOAL),
    (0, 0, 0.70, 0.16, today_exer,  EXER_GOAL,  EXER_COLOR,  'Exercise', f'{today_exer}', 'min',  EXER_GOAL),
    (0, 0, 0.45, 0.16, today_stand, STAND_GOAL, STAND_COLOR, 'Stand',    f'{today_stand}','hrs',  STAND_GOAL),
]

for cx, cy, r, w, val, goal, col, name, vstr, unit, g in ring_specs:
    draw_ring(ax_rings, cx, cy, r, w, val, goal, col)

# Central value display
ax_rings.text(0, 0.13, f'{today_move}', fontsize=20, fontweight='bold',
              color=MOVE_COLOR, ha='center', va='center', fontfamily='DejaVu Sans')
ax_rings.text(0, -0.08, 'kcal', fontsize=8.4, color=WHITE,
              ha='center', va='center', fontfamily='DejaVu Sans', fontweight='bold')
ax_rings.text(0, -0.20, 'today', fontsize=7, color=MUTED,
              ha='center', va='center', fontfamily='DejaVu Sans')

# Legend below rings
legend_items = [
    (MOVE_COLOR,  f'Move    {today_move} / {MOVE_GOAL} kcal'),
    (EXER_COLOR,  f'Exercise  {today_exer} / {EXER_GOAL} min'),
    (STAND_COLOR, f'Stand   {today_stand} / {STAND_GOAL} hrs'),
]
for i, (col, txt) in enumerate(legend_items):
    y = -1.15 - i * 0.12
    ax_rings.plot(-0.55, y, 'o', color=col, markersize=7, zorder=5)
    ax_rings.text(-0.40, y, txt, fontsize=8.5, color=WHITE, va='center',
                  fontfamily='DejaVu Sans')


# ── RIGHT: KPI card grid (2x3) ───────────────────────────────────────────────
inner1r = GridSpecFromSubplotSpec(3, 2, subplot_spec=inner1[1],
                                  wspace=0.08, hspace=0.12)

kpi_data = [
    ('Total Steps',       f'{int(steps.sum()):,}',     f'Avg {int(steps.mean()):,} / day',  STEPS_COLOR),
    ('Total Distance',    f'{distance.sum():.1f} km',  f'Avg {distance.mean():.1f} km / day', DIST_COLOR),
    ('Floors Climbed',    f'{int(floors.sum())}',      f'Avg {floors.mean():.1f} / day',    FLOORS_COLOR),
    ('Active Days',       '30 / 30',                   '100 % consistency',                 EXER_COLOR),
    ('Peak Steps Day',    f'{int(steps.max()):,}',     f'Jan {days_short[steps.argmax()]}', MOVE_COLOR),
    ('Best Distance',     f'{distance.max():.2f} km',  f'Jan {days_short[distance.argmax()]}', STAND_COLOR),
]

for idx, (label, val, sub, col) in enumerate(kpi_data):
    r, c = divmod(idx, 2)
    axk = fig.add_subplot(inner1r[r, c])
    axk.set_facecolor(BG_CARD2)
    for sp in axk.spines.values(): sp.set_visible(False)
    axk.axis('off')
    axk.set_xlim(0,1); axk.set_ylim(0,1)

    # Accent left bar
    axk.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.08), 0.022, 0.84, boxstyle="round,pad=0",
        facecolor=col, zorder=3))

    axk.text(0.10, 0.80, label, fontsize=7.5, color=MUTED,
             va='top', fontfamily='DejaVu Sans')
    axk.text(0.10, 0.50, val, fontsize=15.5, fontweight='bold', color=col,
             va='center', fontfamily='DejaVu Sans')
    axk.text(0.10, 0.14, sub, fontsize=6.8, color=MUTED,
             va='bottom', fontfamily='DejaVu Sans')

    # Subtle glow background
    axk.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0",
        facecolor=col, alpha=0.035, zorder=1))


# ════════════════════════════════════════════════════════════════════════════════
# ROW 2 — STEPS + DISTANCE bar charts
# ════════════════════════════════════════════════════════════════════════════════
inner2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.10)

ax_steps = fig.add_subplot(inner2[0])
draw_bar_chart(ax_steps, steps, STEPS_COLOR,
               goal=STEP_GOAL,
               title='Daily Step Count',
               subtitle='January 2026  ·  Goal: 10,000 steps',
               unit=' steps', goal_label='10K')

ax_dist = fig.add_subplot(inner2[1])
draw_bar_chart(ax_dist, distance, DIST_COLOR,
               goal=DIST_GOAL,
               title='Distance Covered (km)',
               subtitle='January 2026  ·  Goal: 6.5 km / day',
               unit=' km', goal_label='6.5 km')


# ════════════════════════════════════════════════════════════════════════════════
# ROW 3 — FLOORS + MOVE bar charts
# ════════════════════════════════════════════════════════════════════════════════
inner3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], wspace=0.10)

ax_floors = fig.add_subplot(inner3[0])
draw_bar_chart(ax_floors, floors, FLOORS_COLOR,
               goal=FLOOR_GOAL,
               title='Floors / Stairs Climbed',
               subtitle='January 2026  ·  Goal: 10 flights',
               unit=' ft', goal_label='10 ft')

ax_move = fig.add_subplot(inner3[1])
draw_bar_chart(ax_move, move, MOVE_COLOR,
               goal=MOVE_GOAL,
               title='Move Ring — Calories Burned',
               subtitle='January 2026  ·  Goal: 450 kcal',
               unit=' kcal', goal_label='450')


# ════════════════════════════════════════════════════════════════════════════════
# ROW 4 — EXERCISE + STAND bar charts
# ════════════════════════════════════════════════════════════════════════════════
inner4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4], wspace=0.10)

ax_exer = fig.add_subplot(inner4[0])
draw_bar_chart(ax_exer, exercise, EXER_COLOR,
               goal=EXER_GOAL,
               title='Exercise Ring — Active Minutes',
               subtitle='January 2026  ·  Goal: 30 min',
               unit=' min', goal_label='30 min')

ax_stand = fig.add_subplot(inner4[1])
draw_bar_chart(ax_stand, stand, STAND_COLOR,
               goal=STAND_GOAL,
               title='Stand Ring — Active Hours',
               subtitle='January 2026  ·  Goal: 12 hrs',
               unit=' hrs', goal_label='12 hrs')


# ════════════════════════════════════════════════════════════════════════════════
# CARD BORDERS — draw subtle rounded outlines on key axes
# ════════════════════════════════════════════════════════════════════════════════
for ax in [ax_steps, ax_dist, ax_floors, ax_move, ax_exer, ax_stand]:
    ax.set_facecolor(BG_CARD)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ════════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════════
output_dir = Path(__file__).resolve().parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)
out = output_dir / 'apple_fitness_dashboard.png'
plt.savefig(out, dpi=160, bbox_inches='tight',
            facecolor=BG_DARK, edgecolor='none')
print(f'Saved → {out}')
plt.close()



