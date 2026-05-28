r"""
Builds the Saturday talk as an editable PowerPoint (.pptx).
Clean / flat / highly-visual: rendered LaTeX equations, all four figures, both simulation videos
embedded as playable H.264 clips (with poster frames), a future-work slide, and an opening
"what we tried & abandoned" arc.  Numbers are pulled verbatim from out_gate/*.json (see JOURNEY.md).

  /home/user/miniconda3/envs/lerobot/bin/python gate/build_deck.py
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = "/home/user/Desktop/Saptarshi/impedance_flow_vla"
FIG = f"{ROOT}/figs"
MED = f"{FIG}/pptx_media"
os.makedirs(MED, exist_ok=True)

# ---------- palette ----------
ACCENT   = RGBColor(0x1F, 0x77, 0xB4)   # impedance blue (matches the figures)
ACCENT_D = RGBColor(0x12, 0x4A, 0x73)
INK      = RGBColor(0x1E, 0x26, 0x30)   # near-black slate
MUTE     = RGBColor(0x5B, 0x64, 0x70)
HAIR     = RGBColor(0xD7, 0xDE, 0xE6)
PANEL    = RGBColor(0xF2, 0xF6, 0xFA)
PANEL2   = RGBColor(0xEA, 0xF1, 0xF8)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
RED      = RGBColor(0xC0, 0x39, 0x2B)
GREEN    = RGBColor(0x21, 0x7A, 0x46)
GOLD     = RGBColor(0xC8, 0x8A, 0x00)
NAVY_BG  = RGBColor(0x12, 0x1A, 0x24)
FONT     = "Calibri"
FONT_H   = "Calibri"

SW, SH = 13.333, 7.5
LM = 0.72                       # left margin
CW = SW - 2 * LM                # content width

prs = Presentation()
prs.slide_width  = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]

# ---------- low-level helpers ----------

def slide():
    return prs.slides.add_slide(BLANK)

def box(s, x, y, w, h, fill=None, line=None, line_w=1.0, rounded=False, radius=0.06):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                             Inches(x), Inches(y), Inches(w), Inches(h))
    shp.shadow.inherit = False
    if rounded:
        try: shp.adjustments[0] = radius
        except Exception: pass
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(line_w)
    return shp

def tbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf

def para(tf, runs, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
         before=0, after=6, lh=1.06, name=FONT, first=False, bullet=False, italic=False):
    """runs: str OR list of (text, dict-overrides)."""
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(before); p.space_after = Pt(after); p.line_spacing = lh
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, ov in runs:
        r = p.add_run(); r.text = text
        f = r.font
        f.size = Pt(ov.get("size", size)); f.bold = ov.get("bold", bold)
        f.italic = ov.get("italic", italic); f.name = ov.get("name", name)
        f.color.rgb = ov.get("color", color)
    return p

def header(s, kicker, title, page):
    box(s, LM, 0.52, 0.14, 0.66, fill=ACCENT)                       # accent tab
    tf = tbox(s, LM + 0.28, 0.50, CW - 0.28, 0.3)
    para(tf, kicker.upper(), size=12.5, bold=True, color=ACCENT, after=0, first=True)
    tf2 = tbox(s, LM + 0.28, 0.74, CW - 0.28, 0.62)
    para(tf2, title, size=27, bold=True, color=INK, after=0, first=True, lh=1.0)
    box(s, LM, 1.46, CW, 0.018, fill=HAIR)                          # hairline rule
    footer(s, page)

def footer(s, page):
    tf = tbox(s, LM, SH - 0.42, CW * 0.7, 0.3)
    para(tf, "Impedance-Shaped Flow Matching for VLA Policies", size=9.5, color=MUTE, first=True)
    tf2 = tbox(s, SW - LM - 1.2, SH - 0.42, 1.2, 0.3)
    para(tf2, str(page), size=10.5, bold=True, color=MUTE, align=PP_ALIGN.RIGHT, first=True)

def fit(iw, ih, bw, bh):
    s = min(bw / iw, bh / ih)
    return iw * s, ih * s

def place_image(s, path, bx, by, bw, bh, shadow_panel=False):
    iw, ih = Image.open(path).size
    w, h = fit(iw, ih, bw, bh)
    x, y = bx + (bw - w) / 2, by + (bh - h) / 2
    if shadow_panel:
        box(s, x - 0.06, y - 0.06, w + 0.12, h + 0.12, fill=WHITE, line=HAIR, line_w=1.0)
    return s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))

def place_movie(s, path, poster, bx, by, bw, bh):
    iw, ih = Image.open(poster).size
    w, h = fit(iw, ih, bw, bh)
    x, y = bx + (bw - w) / 2, by + (bh - h) / 2
    box(s, x - 0.05, y - 0.05, w + 0.10, h + 0.10, fill=INK)        # frame
    mv = s.shapes.add_movie(path, Inches(x), Inches(y), Inches(w), Inches(h),
                            poster_frame_image=poster, mime_type="video/mp4")
    # play button badge
    bb = box(s, x + w/2 - 0.35, y + h/2 - 0.35, 0.7, 0.7, fill=None)
    tf = tbox(s, x + w/2 - 0.35, y + h/2 - 0.46, 0.7, 0.7, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, "▶", size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)
    cap = tbox(s, x, y + h + 0.06, w, 0.3)
    para(cap, "▶  click to play in Slide Show", size=10,
         color=MUTE, align=PP_ALIGN.CENTER, first=True)
    return mv

def render_eq(tex, name, fontsize=26, color="#1E2630"):
    path = f"{MED}/{name}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=fontsize, color=color)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path

def chip(s, x, y, w, h, label, fill, txt=WHITE, size=11.5):
    box(s, x, y, w, h, fill=fill, rounded=True, radius=0.5)
    tf = tbox(s, x, y, w, h, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, label, size=size, bold=True, color=txt, align=PP_ALIGN.CENTER, first=True)

# ---------- equations (rendered crisp) ----------
EQ_MAIN = render_eq(
    r"$v_\theta(a_\tau,\tau,o)\;=\;M^{-1}\left(\,-\,D\,\dot{a}_\tau\;-\;K\,(a_\tau-a_{\mathrm{goal}})\;+\;f_\theta(a_\tau,\tau,o)\,\right)$",
    "eq_main", fontsize=24)
EQ_MAIN_W = render_eq(
    r"$v_\theta(a_\tau,\tau,o)=M^{-1}\left(-D\,\dot{a}_\tau-K(a_\tau-a_{\mathrm{goal}})+f_\theta\right)$",
    "eq_main_w", fontsize=22, color="#FFFFFF")
EQ_VAN = render_eq(r"$v_\theta(a_\tau,\tau,o)=\mathrm{MLP}_\theta(\cdot)$  (unstructured)", "eq_van", fontsize=18, color="#5B6470")

# =======================================================================================
# SLIDE 1 — TITLE
# =======================================================================================
s = slide()
box(s, 0, 0, SW, SH, fill=NAVY_BG)
box(s, 0, 0, 0.22, SH, fill=ACCENT)
tf = tbox(s, LM, 1.15, CW, 0.4)
para(tf, "ROBOT LEARNING  ·  FLOW-MATCHING POLICIES  ·  CONTACT-RICH MANIPULATION",
     size=13, bold=True, color=RGBColor(0x7F, 0xB2, 0xD9), first=True)
tf = tbox(s, LM, 1.62, CW, 1.9)
para(tf, "Impedance-Shaped Flow Matching", size=46, bold=True, color=WHITE, first=True, lh=1.0, after=2)
para(tf, "for VLA Policies", size=46, bold=True, color=WHITE, after=10, lh=1.0)
para(tf, "Making the action expert's flow-ODE drift a closed-loop impedance law — compliance becomes structural, not a downstream add-on.",
     size=16.5, color=RGBColor(0xC4, 0xCF, 0xDA))
# equation hero on a light card
box(s, LM, 4.05, CW, 1.35, fill=RGBColor(0x1B, 0x26, 0x33), line=RGBColor(0x2C, 0x3B, 0x4C), line_w=1.0, rounded=True, radius=0.06)
place_image(s, EQ_MAIN_W, LM + 0.3, 4.18, CW - 0.6, 1.1)
tf = tbox(s, LM, 6.55, CW, 0.5)
para(tf, [("IS Lab · Changwon National University", {"bold": True, "color": WHITE, "size": 14}),
          ("        Group presentation — Saturday, 2026-05-30", {"color": RGBColor(0x9F,0xAD,0xBC), "size": 14})],
     first=True)

# =======================================================================================
# SLIDE 2 — the road here (1): ideas we brainstormed
# =======================================================================================
s = slide()
header(s, "How we got here  ·  1 / 2", "We started somewhere else: Adjoint Attribution", 2)
tf = tbox(s, LM, 1.65, CW, 0.7)
para(tf, [("Prior project: ", {"bold": True}),
          ("“Why attention explanations for flow-matching robot policies are provably wrong.”  Use the ", {}),
          ("adjoint sensitivity", {"bold": True, "color": ACCENT}),
          (" of the action-generating ODE as the principled attribution. Four bids for novelty:", {})],
     size=15, color=INK, first=True)
cards = [
    ("1", "Attention is “provably wrong”", "Attention maps disagree with true ODE sensitivity."),
    ("2", "Second-order / interaction adjoint", "Go beyond first-order gradients — curvature & token interactions."),
    ("3", "Adjoint-KL fine-tuning", "Use the adjoint signal to fine-tune the policy (performance lever)."),
    ("4", "Test-time goal-steering", "Steer generation along adjoint directions at inference."),
]
cw = (CW - 3 * 0.3) / 4
for i, (n, t, d) in enumerate(cards):
    x = LM + i * (cw + 0.3)
    box(s, x, 2.55, cw, 2.7, fill=PANEL, line=HAIR, line_w=1.0, rounded=True, radius=0.05)
    box(s, x, 2.55, cw, 0.62, fill=ACCENT, rounded=True, radius=0.05)
    box(s, x, 2.9, cw, 0.27, fill=ACCENT)   # square off the bottom of the header band
    tf = tbox(s, x, 2.6, cw, 0.55, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, "HYPOTHESIS " + n, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)
    tf = tbox(s, x + 0.18, 3.35, cw - 0.36, 1.85)
    para(tf, t, size=15, bold=True, color=INK, first=True, after=8)
    para(tf, d, size=12.5, color=MUTE, lh=1.12)
tf = tbox(s, LM, 5.5, CW, 0.8)
para(tf, [("The goal we set: ", {"bold": True}),
          ("genuine novelty ", {"bold": True, "color": ACCENT}),
          ("+ an ", {}), ("improving number", {"bold": True, "color": ACCENT}),
          (". Every one of these four was tested against that bar.", {})],
     size=15, color=INK, first=True)

# =======================================================================================
# SLIDE 3 — the road here (2): why we abandoned it
# =======================================================================================
s = slide()
header(s, "How we got here  ·  2 / 2", "… and why we abandoned it", 3)
rows = [
    ("Attention “provably wrong”", "Confound.", "Attention vs adjoint actually agree on rank (Spearman +0.52, top-1 100%) once token granularity is matched."),
    ("2nd-order interaction adjoint", "Scale artifact.", "The 90–122× state-token curvature was an embedding-norm effect (state ≈7 vs image ≈3700, ~518×)."),
    ("Adjoint-KL fine-tuning", "Null.", "+1.7 vs a proper control — inside the noise at n = 12."),
    ("Test-time goal-steering", "Net-negative.", "Helps free-space reach (37→75%) but collapses every contact task (push, window → 0%)."),
]
y = 1.66
rh = 0.86
for t, verdict, d in rows:
    box(s, LM, y, CW, rh - 0.12, fill=PANEL, line=HAIR, line_w=1.0, rounded=True, radius=0.06)
    box(s, LM, y, 0.1, rh - 0.12, fill=RED, rounded=False)
    tf = tbox(s, LM + 0.28, y, 0.55, rh - 0.12, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, "✗", size=22, bold=True, color=RED, align=PP_ALIGN.CENTER, first=True)
    tf = tbox(s, LM + 0.95, y, 3.5, rh - 0.12, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, t, size=14.5, bold=True, color=INK, first=True)
    tf = tbox(s, LM + 4.5, y, CW - 4.6, rh - 0.12, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, [(verdict + "  ", {"bold": True, "color": RED}), (d, {"color": INK})], size=13, first=True, lh=1.05)
    y += rh
# survivor + lesson
box(s, LM, y + 0.02, CW, 0.78, fill=PANEL2, line=ACCENT, line_w=1.25, rounded=True, radius=0.06)
box(s, LM, y + 0.02, 0.1, 0.78, fill=GREEN)
tf = tbox(s, LM + 0.28, y + 0.02, 0.55, 0.78, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "✓", size=22, bold=True, color=GREEN, align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, LM + 0.95, y + 0.02, CW - 1.1, 0.78, anchor=MSO_ANCHOR.MIDDLE)
para(tf, [("One result survived (analysis-only): ", {"bold": True, "color": GREEN}),
          ("an interventional ground truth showed attention's modality ranking is the ", {}),
          ("reverse", {"bold": True}), (" of causal truth, and the adjoint's matches. Not the method/perf paper we wanted → ", {}),
          ("abandoned.", {"bold": True, "color": RED})], size=12.5, first=True, lh=1.05)
tf = tbox(s, LM, y + 0.92, CW, 0.5)
para(tf, [("Lesson kept: ", {"bold": True, "color": ACCENT}),
          ("model-internal attributions evaporate under scale/granularity normalization. Validate the core object in isolation, clear a kill-gate before scaling, report numbers honestly.", {})],
     size=13.5, color=INK, first=True)

# =======================================================================================
# SLIDE 4 — the pivot
# =======================================================================================
s = slide()
header(s, "The pivot", "From explaining compliance to building it", 4)
tf = tbox(s, LM, 1.85, CW, 1.9)
para(tf, [("If steering a trained flow policy ", {}), ("collapses", {"bold": True, "color": RED}),
          (" contact tasks, the problem isn't the explanation — the policy has ", {}),
          ("no notion of physical compliance", {"bold": True, "color": ACCENT}),
          (" in how it generates motion.", {})], size=18, color=INK, first=True, lh=1.15, after=10)
para(tf, [("So stop ", {}), ("attributing", {"italic": True}),
          (" behaviour after the fact. Build the right inductive bias ", {}),
          ("into the generator itself.", {"bold": True})], size=18, color=INK, lh=1.15)
# two contrasting panels
box(s, LM, 4.0, CW/2 - 0.2, 2.3, fill=PANEL, line=HAIR, line_w=1.0, rounded=True, radius=0.04)
tf = tbox(s, LM + 0.3, 4.25, CW/2 - 0.8, 1.85)
para(tf, "BEFORE", size=13, bold=True, color=RED, first=True, after=8)
para(tf, "Unstructured neural drift", size=17, bold=True, color=INK, after=6)
para(tf, "Compliance bolted on downstream (external controller / F-T sensor), or merely explained post-hoc.", size=13.5, color=MUTE, lh=1.15)
box(s, LM + CW/2 + 0.2, 4.0, CW/2 - 0.2, 2.3, fill=PANEL2, line=ACCENT, line_w=1.25, rounded=True, radius=0.04)
tf = tbox(s, LM + CW/2 + 0.5, 4.25, CW/2 - 0.8, 1.85)
para(tf, "OURS", size=13, bold=True, color=ACCENT, first=True, after=8)
para(tf, "Drift IS an impedance law", size=17, bold=True, color=INK, after=6)
para(tf, "A closed-loop mass–spring–damper attractor, end-to-end differentiable, baked into the flow ODE. No downstream controller.", size=13.5, color=INK, lh=1.15)
# arrow
tf = tbox(s, LM + CW/2 - 0.25, 4.85, 0.5, 0.6, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "→", size=34, bold=True, color=ACCENT, align=PP_ALIGN.CENTER, first=True)

# =======================================================================================
# SLIDE 5 — thesis / the one new object
# =======================================================================================
s = slide()
header(s, "The contribution", "The one new object: an impedance-shaped flow drift", 5)
box(s, LM, 1.75, CW, 1.5, fill=PANEL, line=ACCENT, line_w=1.25, rounded=True, radius=0.05)
place_image(s, EQ_MAIN, LM + 0.35, 1.95, CW - 0.7, 1.1)
tf = tbox(s, LM, 3.4, CW, 0.4)
para(tf, "The flow-matching action expert's ODE velocity field is itself a 2nd-order closed-loop impedance law.",
     size=14.5, italic=True, color=MUTE, align=PP_ALIGN.CENTER, first=True)
items = [
    ("M, K, D", "task-conditioned SPD matrices (mass / stiffness / damping), Cholesky-parameterized, predicted from features."),
    ("ȧ — velocity state", "the flow is a 2nd-order augmented ODE, integrated semi-implicitly (symplectic Euler). The naive finite-difference form is unstable — the toy gate proved it first."),
    ("a_goal", "an attractor (chunk target or a predicted attractor token) the drift is pulled toward."),
    ("End-to-end", "fully differentiable. No downstream controller. No force/torque sensor at inference."),
]
y = 4.0
for tag, d in items:
    box(s, LM, y + 0.07, 0.16, 0.46, fill=ACCENT)
    tf = tbox(s, LM + 0.35, y, 2.7, 0.7, anchor=MSO_ANCHOR.TOP)
    para(tf, tag, size=14.5, bold=True, color=ACCENT, first=True)
    tf = tbox(s, LM + 3.2, y, CW - 3.2, 0.7, anchor=MSO_ANCHOR.TOP)
    para(tf, d, size=13.5, color=INK, first=True, lh=1.08)
    y += 0.72

# =======================================================================================
# SLIDE 6 — why it's novel (positioning)
# =======================================================================================
s = slide()
header(s, "Positioning", "Why it's novel — and what it is not", 6)
comp = [
    ("CompliantVLA-adaptor", "Predicts K, D from a VLM but feeds an EXTERNAL impedance controller wrapped around a frozen VLA."),
    ("Flow with the Force Field", "Standard unstructured flow drift + a DOWNSTREAM passive impedance controller for compliance."),
    ("ACP / Diffusion-Impedance / DMP", "Stiffness lives on the OUTPUT head or a DMP decoder — not in the generative drift."),
]
y = 1.7
for name, d in comp:
    box(s, LM, y, CW, 0.92, fill=PANEL, line=HAIR, line_w=1.0, rounded=True, radius=0.05)
    tf = tbox(s, LM + 0.3, y, 3.7, 0.92, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, name, size=15, bold=True, color=INK, first=True)
    tf = tbox(s, LM + 4.2, y, CW - 4.5, 0.92, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, d, size=13.5, color=MUTE, first=True, lh=1.1)
    y += 1.04
box(s, LM, y + 0.02, CW, 1.15, fill=ACCENT, rounded=True, radius=0.05)
tf = tbox(s, LM + 0.4, y + 0.02, CW - 0.8, 1.15, anchor=MSO_ANCHOR.MIDDLE)
para(tf, [("Ours:  ", {"bold": True, "size": 18, "color": WHITE}),
          ("the velocity field IS the impedance law", {"bold": True, "size": 18, "color": WHITE})],
     first=True, after=4)
para(tf, "A closed-loop K/D/M attractor conditioned on features, embedded inside the action expert — end-to-end, no external controller, no F/T sensor. (The prior-art check confirms this intersection is unclaimed as of May 2026.)",
     size=12.5, color=RGBColor(0xE2, 0xEE, 0xF8), lh=1.1)

# =======================================================================================
# SLIDE 7 — MECHANISM VIDEO
# =======================================================================================
s = slide()
header(s, "Explainable simulation  ·  mechanism", "How one action is generated through the flow ODE", 7)
place_movie(s, f"{MED}/mechanism_anim.mp4", f"{MED}/poster_mechanism.png", LM, 1.66, CW, 3.95)
tf = tbox(s, LM, 5.8, CW, 1.3)
para(tf, [("Vanilla flow", {"bold": True, "color": MUTE}),
          (" travels a straight, unstructured path and arrives at full speed. ", {}),
          ("Impedance flow (ours)", {"bold": True, "color": ACCENT}),
          (" is a damped spring toward the predicted attractor — it decelerates to a ", {}),
          ("soft landing.", {"bold": True})], size=15, color=INK, first=True, lh=1.12, after=6)
para(tf, [("ζ = 1 (ours, critically damped) is smooth; ζ = 0.3 overshoots — which is exactly why we enforce ", {"color": MUTE}),
          ("ζ ≥ 1", {"bold": True, "color": ACCENT}),
          (" via D = 2ζ√K.", {"color": MUTE})], size=14, lh=1.1)

# =======================================================================================
# SLIDE 8 — TOY GATE
# =======================================================================================
s = slide()
header(s, "De-risking  ·  validate the core in isolation", "Toy gate — cleared before touching the model", 8)
place_image(s, f"{FIG}/toy_impedance.png", LM, 1.66, CW * 0.52, 3.5, shadow_panel=True)
place_image(s, f"{FIG}/toy_stability.png", LM + CW * 0.54, 1.66, CW * 0.46, 3.5, shadow_panel=True)
y = 5.45
checks = [
    ("SPD heads verified", "Cholesky M = LLᵀ + εI stays positive-definite."),
    ("Damping-ratio control", "ζ = 0.3 → 36% overshoot,  ζ = 1 → clean,  ζ = 2 → monotonic."),
    ("Stability bound", "at N Euler steps usable stiffness K ≲ (steps)² — the soft/compliant regime, which is what contact wants."),
]
for i, (t, d) in enumerate(checks):
    x = LM + i * (CW / 3)
    tf = tbox(s, x, y, CW / 3 - 0.25, 1.4)
    para(tf, [("✓  ", {"bold": True, "color": GREEN, "size": 15}), (t, {"bold": True, "color": INK, "size": 14})], first=True, after=3)
    para(tf, d, size=12, color=MUTE, lh=1.12)

# =======================================================================================
# SLIDE 9 — how we tested (fair comparison)
# =======================================================================================
s = slide()
header(s, "Stage-1 setup", "A fair test: only the drift parameterization differs", 9)
left = [
    ("Same everything", "Compact state-conditioned flow policy. Vanilla vs impedance arm trained with an identical objective; only the ODE drift differs."),
    ("Param-matched", "Both arms at ~400k parameters — the win is the structure, not capacity."),
    ("Honest protocol", "3 seeds × 30 closed-loop eval episodes; no tuning toward a target."),
]
y = 1.8
for t, d in left:
    box(s, LM, y + 0.06, 0.16, 0.5, fill=ACCENT)
    tf = tbox(s, LM + 0.35, y, CW * 0.54, 0.95)
    para(tf, t, size=16, bold=True, color=INK, first=True, after=3)
    para(tf, d, size=13.5, color=MUTE, lh=1.12)
    y += 1.05
# metrics panel on the right
bx = LM + CW * 0.6
box(s, bx, 1.8, CW * 0.4, 3.3, fill=PANEL, line=HAIR, line_w=1.0, rounded=True, radius=0.05)
tf = tbox(s, bx + 0.3, 2.05, CW * 0.4 - 0.6, 3.0)
para(tf, "WHAT WE MEASURE", size=12.5, bold=True, color=ACCENT, first=True, after=10)
for t, d in [("Success rate", "did the task complete"),
             ("Action jerk", "2nd-difference of actions — smoothness"),
             ("Peak contact force", "95th-pct of contact force — compliance"),
             ("Sample efficiency", "metric vs # demonstrations")]:
    para(tf, [("▪  ", {"color": ACCENT, "bold": True}), (t, {"bold": True, "color": INK})], size=14, after=0)
    para(tf, "      " + d, size=11.5, color=MUTE, after=8, lh=1.05)
tf = tbox(s, LM, 5.45, CW, 1.0)
para(tf, [("Benchmark: ", {"bold": True}),
          ("MetaWorld full-state demos — ", {}),
          ("peg-insert-side", {"bold": True, "color": ACCENT}),
          (" (tight-tolerance contact, the hard task) and ", {}),
          ("door-open", {"bold": True}),
          (" (easy-from-state control). ManiSkill2/3 contact suite is the next, primary benchmark.", {})],
     size=14, color=INK, first=True, lh=1.12)

# =======================================================================================
# SLIDE 10 — RESULT: peg-insert (primary)
# =======================================================================================
s = slide()
header(s, "Result  ·  primary task", "peg-insert-side: a textbook sample-efficiency win", 10)
place_image(s, f"{FIG}/fig1_peg_sample_efficiency.png", LM, 1.62, CW, 3.55, shadow_panel=True)
y = 5.4
stats = [("+22 pp", "success @ 10 demos\n(30% → 52%)", ACCENT),
         ("−42%", "lower action jerk\nat low data", ACCENT),
         ("−18%", "peak contact force\n@ matched success (n=25)", ACCENT)]
cw3 = (CW - 2 * 0.3) / 3
for i, (big, d, c) in enumerate(stats):
    x = LM + i * (cw3 + 0.3)
    box(s, x, y, cw3, 1.45, fill=PANEL2, line=HAIR, line_w=1.0, rounded=True, radius=0.06)
    tf = tbox(s, x + 0.2, y + 0.12, cw3 - 0.4, 0.7)
    para(tf, big, size=30, bold=True, color=c, first=True)
    tf = tbox(s, x + 0.2, y + 0.78, cw3 - 0.4, 0.6)
    for j, ln in enumerate(d.split("\n")):
        para(tf, ln, size=12.5, color=INK, first=(j == 0), after=0, lh=1.0)

# =======================================================================================
# SLIDE 11 — ROLLOUT VIDEO
# =======================================================================================
s = slide()
header(s, "Explainable simulation  ·  rollout", "Impedance inserts where vanilla misses — 10 demos", 11)
place_movie(s, f"{MED}/peg_rollout_compare.mp4", f"{MED}/poster_rollout.png", LM, 1.66, CW, 4.0)
tf = tbox(s, LM, 5.95, CW, 1.1)
para(tf, [("MuJoCo peg-insert, identical seeds, both arms trained on just 10 demos (3 episodes shown). ", {}),
          ("Impedance succeeds where vanilla fails.", {"bold": True, "color": ACCENT})],
     size=15, color=INK, first=True, lh=1.12, after=5)
para(tf, "Note: the live force gauge reads higher for impedance here precisely because it makes the insertion contact that vanilla misses — the −18% compliance number is the matched-success comparison on the previous slide.",
     size=11.5, italic=True, color=MUTE, lh=1.08)

# =======================================================================================
# SLIDE 12 — RESULT: door-open (secondary)
# =======================================================================================
s = slide()
header(s, "Result  ·  secondary task (honest scope)", "door-open saturates — so the prior is redundant there", 12)
place_image(s, f"{FIG}/fig2_door_secondary.png", LM, 1.66, CW * 0.62, 3.7, shadow_panel=True)
bx = LM + CW * 0.66
tf = tbox(s, bx, 1.9, CW * 0.34, 3.6)
para(tf, "WHY THIS MATTERS", size=12.5, bold=True, color=ACCENT, first=True, after=10)
para(tf, [("Door-open is easy from state", {"bold": True, "color": INK}),
          (" — vanilla already hits 82% at 10 demos. No sample-efficiency headroom → ", {}),
          ("the two arms tie on success.", {"bold": True})], size=14, color=MUTE, lh=1.18, after=10)
para(tf, [("The impedance prior helps exactly where ", {}),
          ("physics is the bottleneck", {"bold": True, "color": ACCENT}),
          (" (tight-tolerance insertion) and is harmlessly redundant where it isn't.", {})],
     size=14, color=MUTE, lh=1.18, after=10)
para(tf, "That is the honest, expected signature of a good structural prior — not a universal +X%.",
     size=13.5, italic=True, color=INK, lh=1.15)

# =======================================================================================
# SLIDE 13 — RESULTS AT A GLANCE (table)
# =======================================================================================
s = slide()
header(s, "Evidence", "Stage-1 results at a glance", 13)
data = [
    ["Setting", "Success  (van → imp)", "Action jerk  (van → imp)", "Peak force  (van → imp)"],
    ["peg-insert · 10 demos", "30% → 52%   (+22pp)", "0.99 → 0.57   (−42%)", "509 → 674"],
    ["peg-insert · 25 demos", "74% → 73%   (tie)", "0.41 → 0.24   (−41%)", "536 → 441   (−18%)"],
    ["peg-insert · 100 demos", "88% → 88%   (saturated)", "0.19 → 0.26", "443 → 403"],
    ["door-open · 100 demos", "93% → 93%   (tie)", "0.18 → 0.14   (−22%)", "2202 → 2199"],
]
nrows, ncols = len(data), len(data[0])
tbl_w, tbl_h = CW, 3.7
gt = s.shapes.add_table(nrows, ncols, Inches(LM), Inches(1.8), Inches(tbl_w), Inches(tbl_h)).table
gt.first_row = False; gt.horz_banding = False
widths = [3.3, 3.0, 3.05, 2.58]
for c, wv in enumerate(widths):
    gt.columns[c].width = Inches(wv)
gt.rows[0].height = Inches(0.72)
for r in range(1, nrows):
    gt.rows[r].height = Inches((tbl_h - 0.72) / (nrows - 1))
for r in range(nrows):
    for c in range(ncols):
        cell = gt.cell(r, c)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.14); cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)
        if r == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = ACCENT
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r % 2 else PANEL
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        run = p.add_run(); run.text = data[r][c]
        f = run.font; f.name = FONT; f.size = Pt(13.5 if r == 0 else 13)
        f.bold = (r == 0 or c == 0)
        f.color.rgb = WHITE if r == 0 else INK
        # green-tint the headline win cell
        if r == 1 and c == 1:
            cell.fill.fore_color.rgb = RGBColor(0xDD, 0xF0, 0xE2); f.bold = True; f.color.rgb = GREEN
tf = tbox(s, LM, 5.7, CW, 0.9)
para(tf, [("matched ~400k params · identical training objective · 3 seeds · 30 eval episodes.   ", {"italic": True, "color": MUTE, "size": 12.5})],
     first=True)
para(tf, [("Read: ", {"bold": True, "color": ACCENT}),
          ("a big low-data gain that closes as data saturates (sample efficiency), smoother actions at low data, lower force at matched success — the structural prior helps where physics matters.", {"color": INK})],
     size=13.5, lh=1.12)

# =======================================================================================
# SLIDE 14 — FUTURE WORK
# =======================================================================================
s = slide()
header(s, "Future work", "From Stage-1 proof to a contact-rich VLA paper", 14)
fw = [
    ("Stage-2 · conditioning ablation", "VLM-predicted M/K/D vs fixed; stiffness-output-only baseline — isolate what the structure buys."),
    ("Stage-3 · scale to SmolVLA", "Port the impedance drift into the SmolVLA action expert with the full 2nd-order conditional flow matching."),
    ("Primary benchmark · ManiSkill2/3", "PegInsertionSide, PlugCharger, TurnFaucet, AssemblyNut — reviewer-accepted contact suite. MetaWorld as cross-sim. (Not LIBERO — kinematic.)"),
    ("Efficiency angle · ODE steps", "5 vs 10 integration steps — same accuracy, fewer steps = lower latency."),
    ("Credibility lever · real arm", "Peg / USB insertion with a wrist F/T sensor."),
]
y = 1.66
for i, (t, d) in enumerate(fw):
    box(s, LM, y, CW, 0.8, fill=PANEL if i % 2 == 0 else WHITE, line=HAIR, line_w=1.0, rounded=True, radius=0.05)
    chip(s, LM + 0.22, y + 0.23, 0.34, 0.34, str(i + 1), ACCENT, size=14)
    tf = tbox(s, LM + 0.78, y, 4.1, 0.8, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, t, size=14.5, bold=True, color=INK, first=True, lh=1.0)
    tf = tbox(s, LM + 5.0, y, CW - 5.2, 0.8, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, d, size=12.8, color=MUTE, first=True, lh=1.08)
    y += 0.86
box(s, LM, y + 0.04, CW, 0.5, fill=PANEL2, line=ACCENT, line_w=1.0, rounded=True, radius=0.08)
tf = tbox(s, LM + 0.3, y + 0.04, CW - 0.6, 0.5, anchor=MSO_ANCHOR.MIDDLE)
para(tf, [("Kill-gate:  ", {"bold": True, "color": ACCENT}),
          ("GO only if the ManiSkill contact subset shows ≥ +5pp over vanilla SmolVLA at matched params. Fail fast; report honestly.", {"color": INK})],
     size=13.5, first=True)

# =======================================================================================
# SLIDE 15 — TAKEAWAYS
# =======================================================================================
s = slide()
box(s, 0, 0, SW, SH, fill=NAVY_BG)
box(s, 0, 0, 0.22, SH, fill=ACCENT)
tf = tbox(s, LM, 0.7, CW, 0.5)
para(tf, "TAKEAWAYS", size=14, bold=True, color=RGBColor(0x7F, 0xB2, 0xD9), first=True)
pts = [
    ("A genuinely new object", "the flow-ODE drift made a closed-loop impedance law — not an external controller, not a post-hoc explanation. Prior-art-clear."),
    ("It works where it should", "+22pp sample efficiency, −42% jerk, and −18% contact force at matched success on tight-tolerance insertion; harmlessly redundant on easy tasks."),
    ("Earned by discipline", "toy gate → param-matched Stage-1 → SmolVLA next. The lesson from the abandoned project, applied: validate, gate, report honestly."),
]
y = 1.7
for i, (t, d) in enumerate(pts):
    chip(s, LM, y + 0.05, 0.5, 0.5, str(i + 1), ACCENT, size=18)
    tf = tbox(s, LM + 0.8, y, CW - 0.9, 1.4)
    para(tf, t, size=22, bold=True, color=WHITE, first=True, after=4, lh=1.0)
    para(tf, d, size=15, color=RGBColor(0xC4, 0xCF, 0xDA), lh=1.14)
    y += 1.55
box(s, LM, 6.65, CW, 0.02, fill=RGBColor(0x2C, 0x3B, 0x4C))
tf = tbox(s, LM, 6.8, CW, 0.4)
para(tf, [("Impedance-Shaped Flow Matching for VLA Policies", {"bold": True, "color": WHITE, "size": 13}),
          ("    ·    IS Lab, CWNU    ·    2026-05-30", {"color": RGBColor(0x9F,0xAD,0xBC), "size": 13})],
     first=True)

out = f"{ROOT}/Impedance_Flow_VLA_Saturday.pptx"
prs.save(out)
print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")
