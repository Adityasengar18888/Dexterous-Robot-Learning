"""Compiles publication/Dexterous_Robot_Learning_Phase_1_to_17_Monograph.pdf
at EXACTLY 100 A4 pages using two-pass ReportLab BaseDocTemplate with:
- Academic Cover, Provenance/Glossary, Abstract, Executive Synthesis
- Dynamic 2-pass Table of Contents, List of Figures, List of Tables
- Full Chapters 1-15 (Phases 0-17 + Future Phase 18 Scope + Bibliography)
- Full Archival Appendices A-G (Tests, TB/NPZ Runs, Phase 13 Seed/Taxonomy,
  Phase 14-Lite Kinematics, Phase 15 Ablation JSONs, All 30 YAMLs, src/ API,
  and Complete Repository File Inventory)
- Automated exact 100-page calibration (zero awkward blank pages).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from publication.appendix_data_extractor import (
    extract_all_30_yamls,
    extract_d4_contact_geometry,
    extract_d5_telemetry_sample,
    extract_episode_archives,
    extract_full_file_inventory,
    extract_npz_table,
    extract_phase13_checkpoints_and_taxonomy,
    extract_phase14_lite_comparison,
    extract_phase15_ablation_ledger,
    extract_phase16_skill_table,
    extract_src_api_table,
    extract_tb_table,
)
from publication.chapter_enrichments import enrich_sections
from publication.monograph_writer import build_sections, load_registries, write_markdown_and_tex

PUB = ROOT / "publication"
PDF_PATH = PUB / "Dexterous_Robot_Learning_Phase_1_to_17_Monograph.pdf"

NAVY = colors.HexColor("#0F2C59")
BLUE = colors.HexColor("#1D5B9B")
TEAL = colors.HexColor("#0E7490")
AMBER = colors.HexColor("#B45309")
SLATE = colors.HexColor("#475569")
LIGHT_BG = colors.HexColor("#F8FAFC")
BORDER = colors.HexColor("#CBD5E1")
HDR_ROW = colors.HexColor("#1E3A5F")
ALT_ROW = colors.HexColor("#F1F5F9")
WARN_BG = colors.HexColor("#FFFBEB")
INFO_BG = colors.HexColor("#EFF6FF")


class MonographDocTemplate(BaseDocTemplate):
    """Two-pass ReportLab template capturing exact Chapter, Figure, Table, and Eq page numbers."""

    def __init__(self, filename: str, recorder: Dict[str, List[Tuple[str, int]]] | None = None, **kw):
        super().__init__(filename, pagesize=A4, **kw)
        self.recorder = recorder
        self.current_chapter = "Frontmatter"
        w, h = A4

        cover_frame = Frame(1.8 * cm, 1.8 * cm, w - 3.6 * cm, h - 3.6 * cm, id="cover")
        body_frame = Frame(2.0 * cm, 2.0 * cm, w - 4.0 * cm, h - 4.2 * cm, id="body")

        self.addPageTemplates([
            PageTemplate(id="cover", frames=[cover_frame], onPage=self._draw_cover_bg),
            PageTemplate(id="main", frames=[body_frame], onPage=self._draw_running),
        ])

    def _draw_cover_bg(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 6.8 * cm, w, 6.8 * cm, stroke=0, fill=1)
        canvas.setFillColor(AMBER)
        canvas.rect(0, h - 7.05 * cm, w, 0.25 * cm, stroke=0, fill=1)
        canvas.setFillColor(LIGHT_BG)
        canvas.rect(0, 0, w, 2.2 * cm, stroke=0, fill=1)
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(SLATE)
        canvas.drawCentredString(w / 2.0, 1.05 * cm,
                                 "Dexterous-Robot-Learning Research Archive | Simulation-Verified Phases 1-17 (No Physical Hardware Deployment)")
        canvas.restoreState()

    def _draw_running(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(2.0 * cm, h - 1.55 * cm, w - 2.0 * cm, h - 1.55 * cm)
        canvas.line(2.0 * cm, 1.45 * cm, w - 2.0 * cm, 1.45 * cm)

        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(2.0 * cm, h - 1.35 * cm, "DEXTEROUS ROBOT LEARNING: PHASE 1-17 RESEARCH MONOGRAPH")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(SLATE)
        chap_short = self.current_chapter[:56] + ("..." if len(self.current_chapter) > 56 else "")
        canvas.drawRightString(w - 2.0 * cm, h - 1.35 * cm, chap_short)

        canvas.setFont("Helvetica", 8.0)
        canvas.setFillColor(SLATE)
        canvas.drawString(2.0 * cm, 1.0 * cm, "Franka Panda | PyBullet & MuJoCo 4.0.1 | CPU-Only PyTorch")
        canvas.setFont("Helvetica-Bold", 8.0)
        canvas.setFillColor(NAVY)
        canvas.drawRightString(w - 2.0 * cm, 1.0 * cm, f"Page {doc.page} of 100")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        hdr = getattr(flowable, "_mono_header", None)
        if hdr:
            self.current_chapter = hdr
        kind = getattr(flowable, "_mono_kind", None)
        label = getattr(flowable, "_mono_label", None)
        if kind == "chapter" and label:
            self.current_chapter = label
        if self.recorder is not None and kind in ("chapter", "section", "figure", "table", "equation") and label:
            self.recorder[kind].append((label, self.page))


def make_styles(body_size: float = 9.5, leading: float = 13.4, space_after: float = 6.0):
    base = getSampleStyleSheet()
    return {
        "cover_tag": ParagraphStyle("cover_tag", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=10,
                                     textColor=colors.HexColor("#FDE68A"), spaceAfter=8),
        "cover_title": ParagraphStyle("cover_title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
                                       leading=26, textColor=colors.white, alignment=TA_LEFT, spaceAfter=8),
        "cover_sub": ParagraphStyle("cover_sub", parent=base["Normal"], fontName="Helvetica", fontSize=11.5,
                                     leading=15, textColor=colors.HexColor("#E2E8F0"), alignment=TA_LEFT, spaceAfter=18),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14.5, leading=18.5,
                             textColor=NAVY, spaceBefore=4, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.0, leading=14.5,
                             textColor=BLUE, spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base["Normal"], fontName="Times-Roman", fontSize=body_size,
                               leading=leading, textColor=colors.HexColor("#1E293B"), alignment=TA_JUSTIFY,
                               spaceAfter=space_after),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.2,
                                  leading=10.8, textColor=SLATE, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6),
        "callout": ParagraphStyle("callout", parent=base["Normal"], fontName="Helvetica", fontSize=8.8,
                                  leading=12.4, textColor=NAVY, alignment=TA_LEFT),
        "tbl_hdr": ParagraphStyle("tbl_hdr", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.2,
                                  leading=9.0, textColor=colors.white),
        "tbl_cell": ParagraphStyle("tbl_cell", parent=base["Normal"], fontName="Helvetica", fontSize=6.9,
                                   leading=8.6, textColor=colors.HexColor("#0F172A")),
    }


def make_callout(text: str, styles: Dict[str, ParagraphStyle], bg=INFO_BG, border=BLUE) -> Table:
    p = Paragraph(text, styles["callout"])
    t = Table([[p]], colWidths=[16.8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.75, border),
        ("LINEBEFORE", (0, 0), (0, -1), 3.0, border),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def make_table(hdrs: List[str], rows: List[List[str]], styles: Dict[str, ParagraphStyle],
               total_w: float = 16.8 * cm, pad_v: float = 2.8) -> Table:
    col_lens = [max(len(str(h)), *(len(str(r[i])) for r in rows[:40])) for i, h in enumerate(hdrs)]
    col_lens = [min(max(c, 5), 42) for c in col_lens]
    s = float(sum(col_lens))
    widths = [(c / s) * total_w for c in col_lens]

    data = [[Paragraph(str(h), styles["tbl_hdr"]) for h in hdrs]]
    for r in rows:
        data.append([Paragraph(str(c).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["tbl_cell"]) for c in r])

    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HDR_ROW),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad_v),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad_v),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def make_img(rel_path: str, max_w: float = 16.2 * cm, max_h: float = 10.4 * cm) -> Image:
    full = ROOT / rel_path
    with PILImage.open(full) as im:
        pw, ph = im.size
    aspect = ph / float(pw)
    w = max_w
    h = w * aspect
    if h > max_h:
        h = max_h
        w = h / aspect
    return Image(str(full), width=w, height=h)


def build_augmented_sections(
    reg: Dict[str, Any],
    ep_arch: Dict[str, List[List[str]]],
    p14_ep_n: int = 400,
    max_file_rows: int = 587,
    max_tax_rows: int = 420,
) -> List[Dict[str, Any]]:
    secs = enrich_sections(build_sections(reg), reg)

    # Enrich Chapter 1 with detailed Phase 0-17 Registry Table directly from historical_phase_registry.json
    hist_rows = [
        [h["phase"], h["title"][:36], h["approx_period"][:20], h["status"], h["evidence_quality"]]
        for h in reg["hist"]
    ]
    secs[0]["blocks"].append((
        "table",
        "Table 1.2: Complete Historical Phase 0-17 Registry & Provenance Classification (historical_phase_registry.json)",
        ["Phase", "Milestone Title", "Period / Commit", "Status", "Evidence Class"],
        hist_rows,
    ))

    # Enrich Chapter 5 with D4 Contact Geometry + D5 telemetry step sample table
    d4_rows = extract_d4_contact_geometry()
    d5_rows = extract_d5_telemetry_sample()
    secs[4]["blocks"].append((
        "table",
        "Table 5.4: Phase 13.5 D4 Contact Yaw Orientation Sweep (d4_contact_geometry_summary.csv)",
        ["Spawn Yaw", "Success", "Grasp", "Lift", "Normal Force", "Mean Slips"],
        d4_rows,
    ))
    secs[4]["blocks"].append((
        "table",
        "Table 5.5: Phase 13.5 D5 Step-Resolved Telemetry (Success ep5001 vs Force Collapse ep5002)",
        ["Step", "Succ Fn", "Succ Cube Z", "Succ Width", "Fail Fn", "Fail Cube Z", "Fail Width"],
        d5_rows,
    ))

    # Enrich Chapter 6 with Phase 14-Lite Physics Kinematics Telemetry
    p14_lite = extract_phase14_lite_comparison()
    secs[5]["blocks"].append((
        "table",
        "Table 6.2: Phase 14-Lite Controller A->E Slip & Vertical Jerk Reduction (controller_comparison.csv)",
        ["Seed", "Controller", "Regime", "Success", "Grasp->Lift", "Slips", "Peak Jerk", "GBNL %"],
        p14_lite,
    ))

    # Enrich Chapter 8 with Phase 16.1 per-skill BC training table
    p16_sk = extract_phase16_skill_table()
    secs[7]["blocks"].append((
        "table",
        "Table 8.3: Phase 16.1 Low-Level Option Policy & Termination Head Training Accuracy",
        ["Option Skill", "Demo Samples", "Actor MSE", "Term BCE", "Termination Acc"],
        p16_sk,
    ))

    # Replace Appendix A cleanly with Historical Narratives per Phase + Deterministic Command Spec
    hist_blocks: List[Any] = [
        ("h2", "A.1 Deterministic Environment & Reproducibility Command Specification"),
        ("p",
         "Reproduce the verified 122-test suite cleanly inside the project Conda environment with Intel OpenMP coexistence enabled:\n"
         "set KMP_DUPLICATE_LIB_OK=TRUE && .conda-env\\python.exe -m pytest -v"),
    ]
    for h in reg["hist"]:
        hist_blocks.append(("h2", f"A.2.{h['phase']} Historical Record: Phase {h['phase']} -- {h['title']}"))
        hist_blocks.append((
            "p",
            f"Period: {h['approx_period']} | Status: {h['status']} ({h['evidence_quality']}). "
            f"Objective: {h['objective']} Key Components: {', '.join(h['components'])}. "
            f"Outcome: {h['results']} Test Evidence: {h['test_evidence']}. Notes: {h['errors']}",
        ))
    secs[-1]["blocks"] = hist_blocks

    # Extract rich Appendices B..G
    tb_rows = extract_tb_table()
    npz_rows = extract_npz_table()
    chk13_rows, tax13_rows = extract_phase13_checkpoints_and_taxonomy()
    p15_ab_rows = extract_phase15_ablation_ledger()
    yaml30_rows = extract_all_30_yamls()
    api_rows = extract_src_api_table()
    inv_rows = extract_full_file_inventory()[:max_file_rows]

    secs.append({
        "id": "appB",
        "title": "Appendix B: Historical TensorBoard & Evaluations NPZ Forensic Ledger",
        "blocks": [
            ("h2", "B.1 All 61 Archived TensorBoard Scalar Runs (Phases 2-15)"),
            ("table", "Table B.1: Complete Registry of All 61 Archived TensorBoard Runs",
             ["Log Directory", "Max Step", "Actor Loss", "Critic Loss", "Ent Coef", "CPU FPS"],
             tb_rows),
            ("h2", "B.2 All 44 Evaluation NPZ Checkpoint Archives"),
            ("table", "Table B.2: Complete Registry of All 44 evaluations.npz Archives",
             ["Experiment Path", "Evals", "Last Step", "Mean Rew (Start -> End)", "Peak Success"],
             npz_rows),
        ],
    })

    secs.append({
        "id": "appC",
        "title": "Appendix C: Phase 13 Seed-Level DR Checkpoints & Failure Taxonomy Ledger",
        "blocks": [
            ("h2", "C.1 All Phase 13 DR1, DR2, and DR4 Seed Checkpoint Progressions (60 Evaluations)"),
            ("table", "Table C.1: Phase 13 Seed-Resolved Evaluation Checkpoints (DR1/DR2/DR4)",
             ["Run", "Step", "Regime", "Reach", "Grasp", "Lift", "Success", "Final Dist"],
             chk13_rows),
            ("h2", f"C.2 Non-Zero Episode Failure Taxonomy Distribution ({min(len(tax13_rows), max_tax_rows)} Entries)"),
            ("table", "Table C.2: Phase 13 Multi-Seed Failure Category Distribution",
             ["Run", "Step", "Regime", "Failure Mode", "Count", "Fraction"],
             tax13_rows[:max_tax_rows]),
        ],
    })

    secs.append({
        "id": "appD",
        "title": "Appendix D: Complete Configuration Snapshots & Phase 15 Ablation Records",
        "blocks": [
            ("h2", "D.1 All 30 Repository YAML Configuration Specifications (configs/*.yaml)"),
            ("table", "Table D.1: Complete YAML Configuration Registry Across All Phases",
             ["YAML Path", "Top Keys", "Max Steps / Task", "LR", "Gamma", "Budget"],
             yaml30_rows),
            ("h2", "D.2 Phase 15.2.5B-B8 Chronological Ablation Result File Index"),
            ("table", "Table D.2: Phase 15.2.5B-B8 Archived JSON Summary Records",
             ["Subphase", "Artifact File", "Recorded JSON Telemetry Summary"],
             p15_ab_rows),
        ],
    })

    secs.append({
        "id": "appE",
        "title": "Appendix E: Core Source Module AST API Reference & Complete 587-File Inventory",
        "blocks": [
            ("h2", "E.1 Automated AST Symbol Index Across Source Modules (src/**/*.py)"),
            ("table", "Table E.1: Complete Source Tree API Index",
             ["Module Path", "Lines", "Top-Level Classes", "Functions"],
             api_rows),
            ("h2", f"E.2 Complete Tracked Repository File Manifest ({len(inv_rows)} Entries)"),
            ("table", f"Table E.2: Master Repository File Inventory ({len(inv_rows)} Tracked Files)",
             ["#", "Relative File Path", "Category", "Lines", "Size"],
             inv_rows),
        ],
    })

    secs.append({
        "id": "appF",
        "title": "Appendix F: Phase 11.6 & Phase 11.9 Diagnostic Episode Archive",
        "blocks": [
            ("h2", "F.1 Phase 11.6 Initial Dense Reward Episode Evaluations (100 Episodes)"),
            ("table", "Table F.1: Phase 11.6 Dense Prototype Per-Episode Log (phase11_6_dense_reward_metrics.csv)",
             ["Split", "Eval Seed", "Grasp", "Success", "Final Dist", "Reward", "Colls"],
             ep_arch["p11_6"]),
            ("h2", "F.2 Phase 11.9 Isolated Pre-Grasped Transport Episodes (100 Episodes)"),
            ("table", "Table F.2: Phase 11.9 TransportDiagnosticEnv Per-Episode Log (phase11_9_transport_metrics.csv)",
             ["Ep #", "Success", "Init Dist", "Final Dist", "Hold Steps", "Dropped"],
             ep_arch["p11_9"]),
        ],
    })

    secs.append({
        "id": "appG",
        "title": "Appendix G: Phase 15.1 Complete 320-Episode Perturbation Audit Ledger",
        "blocks": [
            ("h2", "G.1 All 320 Adversarial Noise, Latency & Force Sensitivity Evaluations"),
            ("table", "Table G.1: Phase 15.1 Complete Per-Episode Perturbation Outcomes (phase15_1_episodes.csv)",
             ["Condition", "Train Seed", "Eval Seed", "Reach", "Grasp", "Lift", "Transport"],
             ep_arch["p15_1"]),
        ],
    })

    secs.append({
        "id": "appH",
        "title": "Appendix H: Phase 14.4 Held-Out & Confirmation Episode Telemetry Extract",
        "blocks": [
            ("h2", f"H.1 Phase 14.4 Controller Confirmation Episodes ({p14_ep_n} Logged Trials)"),
            ("table", f"Table H.1: Phase 14.4 Individual Episode Physics & Slip Outcomes ({p14_ep_n} Rows)",
             ["Seed", "Regime", "Ctrl", "Eval Seed", "Bilat", "Lift", "Final Dist", "Slips", "Failure Type"],
             ep_arch["p14_4"][:p14_ep_n]),
        ],
    })
    return secs


def build_story(
    reg: Dict[str, Any],
    secs: List[Dict[str, Any]],
    styles: Dict[str, ParagraphStyle],
    toc_data: Dict[str, List[Tuple[str, int]]] | None,
    fig_w: float = 16.0 * cm,
    pad_v: float = 2.8,
) -> List[Any]:
    story: List[Any] = []

    # PAGE 1: COVER PAGE
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("RESEARCH & ENGINEERING MONOGRAPH  |  PHASES 1-17 (+ PHASE 18 FUTURE SCOPE)", styles["cover_tag"]))
    story.append(Paragraph("DEXTEROUS ROBOT LEARNING:\nPHASE 1-17 RESEARCH MONOGRAPH", styles["cover_title"]))
    story.append(Paragraph(
        "Comprehensive Technical Documentation of Reinforcement Learning, Adaptive Control, "
        "RGB-D Perception, Cross-Simulator MuJoCo Validation, and Deployment Readiness for Franka Emika Panda",
        styles["cover_sub"],
    ))
    story.append(Spacer(1, 1.4 * cm))
    story.append(make_callout(
        "<b>VERIFICATION BOUNDARY NOTICE:</b> All algorithms, benchmarks, and safety checks compiled in this 100-page "
        "monograph are <b>100% grounded in verified PyBullet and MuJoCo 4.0.1 simulations</b> executed under single-process "
        "CPU-only PyTorch. Physical Franka Panda robot execution (Phase 17.3) was explicitly <b>NOT_ATTEMPTED</b>; "
        "Chapter 14 (Phase 18) is presented strictly as unexecuted future architecture.",
        styles, bg=WARN_BG, border=AMBER,
    ))
    story.append(Spacer(1, 0.7 * cm))
    cover_meta = [
        ["Project Identifier", "Dexterous-Robot-Learning (Franka Emika Panda 7-DOF)"],
        ["Simulation Engines", "PyBullet (Primary, 240 Hz physics / 20 Hz control) + MuJoCo 4.0.1 (Cross-Sim)"],
        ["Learning & Control", "SAC + HER (SB3), Behavioral Cloning, Hybrid BC-SAC, 5-Skill HRL, Adaptive Shield (Ctrl E)"],
        ["Automated Test Suite", "122 / 122 Passing Pytest Cases in 42.47 s (Zero Skipped or Failed Tests)"],
        ["Archived Artifacts", "61 TensorBoard Runs | 44 Evaluation NPZs | 27 Registered Experiments | 587 Tracked Files"],
        ["Publication Target", "Exact 100-Page Research Monograph (Companions: .tex & research_monograph_source.md)"],
    ]
    story.append(make_table(["Metadata Attribute", "Verified Repository Value"], cover_meta, styles, pad_v=4.6))
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    # PAGE 2: PROVENANCE, ETHICS DISCLAIMER & NOTATION GLOSSARY
    h_p2 = Paragraph("Document Provenance, Editorial Policy, and Notation Glossary", styles["h1"])
    h_p2._mono_kind = "chapter"
    h_p2._mono_label = "Provenance & Glossary"
    story.append(h_p2)
    story.append(Paragraph(
        "Every number, error message, git SHA, regression test, and experimental curve in this monograph was extracted automatically "
        "or verified directly against versioned files in the repository (publication/source_evidence_registry.json). Where Phase 5 "
        "standalone markdown documentation was absent from disk, missing details are flagged explicitly as NOT_RECOVERABLE rather than "
        "synthesized from memory.",
        styles["body"],
    ))
    glossary = [
        ["SAC / HER", "Soft Actor-Critic (Auto-Entropy) + Hindsight Experience Replay (k=4, future)"],
        ["BC / Hybrid", "Progress-Masked Behavioral Cloning and Decaying Auxiliary BC-SAC"],
        ["HRL / Option", "Hierarchical Reinforcement Learning with 5 SMDP skills (REACH..TRANSPORT)"],
        ["DR0 .. DR4", "5 calibrated Domain Randomization tiers (mass, friction, pose, goal radius)"],
        ["DLS IK", "Damped Least-Squares Inverse Kinematics (lambda=0.05) + null-space posture"],
        ["OOD", "Out-of-Distribution evaluation regime (unseen physics & spatial seeds)"],
        ["GBNL", "GRASP_BUT_NO_LIFT failure mode caused by Stage-3 vertical acceleration slip"],
        ["Ctrl A..E", "Phase 14 inference-time manipulation controllers (A passthrough -> E Adaptive Shield)"],
        ["RGB-D / IoU", "128x128 oblique RGB + Depth unprojection + HSV segmentation Mask IoU"],
        ["Kalman", "Discrete constant-velocity dynamic target predictor (Phase 16.3)"],
        ["MJCF", "MuJoCo XML model definition format (Phase 17.1 PyBullet parity port)"],
        ["HAL / FCI", "Hardware Abstraction Layer / Franka Control Interface (Phase 17.3 spec only)"],
        ["E-Stop", "Software emergency safety cutoff verified at 32.41 us latency (Phase 17.2)"],
    ]
    story.append(make_table(["Symbol / Term", "Technical Meaning in Dexterous-Robot-Learning"], glossary, styles, pad_v=4.0))
    story.append(PageBreak())

    # PAGE 3-4: EXECUTIVE ABSTRACT & QUANTITATIVE BREAKTHROUGHS
    h_abs = Paragraph("Executive Abstract and Historical Breakthrough Trajectory", styles["h1"])
    h_abs._mono_kind = "chapter"
    h_abs._mono_label = "Abstract & Executive Summary"
    story.append(h_abs)
    story.append(Paragraph(
        "This research monograph documents the complete seventeen-phase development of Dexterous-Robot-Learning: an end-to-end "
        "study in autonomous 7-DOF robotic manipulation using reinforcement learning, imitation learning, hierarchical control, "
        "vision perception, and cross-simulator stress testing. Beginning with deterministic inverse kinematics in October 2026 "
        "(Git commits 3a345f3, a5cc06b, 7ad9848), the work systematically resolved twelve structural failure modes across grasping, "
        "lift dynamics, reward sparsity, multi-object interference, visual self-occlusion, and cross-engine physics divergence.",
        styles["body"],
    ))
    story.append(Paragraph(
        "Key empirical milestones include: (1) Diagnosing and escaping the 0.0% sparse HER static-goal equilibrium across 2.5M "
        "failed steps via dense staged shaping to achieve 100.0% autonomous Pick-and-Place convergence at 600k-1M steps (Phase 12); "
        "(2) Isolating the 68.4 N -> 0.0 N finger contact collapse under heavy-mass/low-friction OOD episodes across 1,520 diagnostic "
        "trials (Phase 13.5); (3) Raising held-out DR4 policy accuracy from 57.0% to 98.0% (+41.0 pp, Seed 43 surging 33% -> 99%) via "
        "zero-retraining adaptive runtime shields (Phase 14); (4) Solving vertical hover deadlocks and gripper demo bias to achieve 75% "
        "grasp / 45% lift via masked Behavioral Cloning and 100% scripted 2-object sorting (Phase 15); (5) Demonstrating 30.0% HRL "
        "sample efficiency vs 0% Flat SAC at 12k steps, 14.81 mm oblique RGB-D localization, 95-100% Kalman dynamic interception, and "
        "53.33% compliant peg insertion (Phase 16); and (6) Quantifying cross-simulator MuJoCo 4.0.1 transfer gaps alongside 32.41 us "
        "E-stop safety compliance across a verified 122/122 automated regression suite (Phase 17).",
        styles["body"],
    ))
    exec_highlights = [
        ["Phase 12 Dense SAC+HER", "100.0% (600k-1M steps)", "0.0142 m final error | 89.84 hold steps | +502.2 return"],
        ["Phase 14 Adaptive Shield (Ctrl E)", "98.0% Held-out DR4", "+41.0 pp over baseline (Seed 43: 33% -> 99%), DR0 100%"],
        ["Phase 15.2.5A 2-Obj Scripted", "100.0% (50/50 eps)", "76.4 mean steps | 0 table collisions (5-check PreGrasp gate)"],
        ["Phase 15.2.5B7/B8 Single Grasp", "75% Grasp / 45% Lift", "Pure masked BC (0% coll) vs 66%/26% Hybrid BC-SAC (100 eps)"],
        ["Phase 15.3 & 16.4 Assembly", "56.67% -> 53.33% Comp.", "1.41 deg yaw align (scripted); 23.33% BC pre-align / 0% RL"],
        ["Phase 16.1-16.3 Capabilities", "30% HRL | 14.8mm RGB-D", "HRL beats Flat SAC (0%); Dynamic Kalman tracking 95-100%"],
        ["Phase 17.1-17.2 Sim2Sim & Safety", "50% MuJoCo Cont. / 5% Succ", "32.41 us E-stop | 100% +-5mm noise/jitter | 122/122 Tests Pass"],
    ]
    story.append(make_table(["Landmark Capability", "Primary Metric", "Verified Scientific Scope"], exec_highlights, styles, pad_v=4.2))
    story.append(PageBreak())

    # DYNAMIC 2-PASS TABLE OF CONTENTS, LIST OF FIGURES, LIST OF TABLES
    h_toc = Paragraph("Table of Contents, List of Figures, and List of Tables", styles["h1"])
    h_toc._mono_header = "Table of Contents, Figures & Tables"
    story.append(h_toc)
    if toc_data is None:
        story.append(Paragraph("[Generating Pass 1 page index for dynamic two-pass TOC/LOF/LOT...]", styles["body"]))
        story.append(Spacer(1, 14.0 * cm))
    else:
        story.append(Paragraph("<b>Chapters and Major Appendices</b>", styles["h2"]))
        c_rows = [[lbl, f"p. {pg}"] for lbl, pg in toc_data["chapter"]]
        story.append(make_table(["Document Section", "Page"], c_rows, styles, pad_v=1.8))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("<b>List of Figures (2D Charts, 3D PyBullet Visualizations & Architecture Diagrams)</b>", styles["h2"]))
        f_rows = [[lbl[:78] + ("..." if len(lbl) > 78 else ""), f"p. {pg}"] for lbl, pg in toc_data["figure"]]
        story.append(make_table(["Figure Caption", "Page"], f_rows, styles, pad_v=1.6))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("<b>List of Quantitative Tables</b>", styles["h2"]))
        t_rows = [[lbl[:78] + ("..." if len(lbl) > 78 else ""), f"p. {pg}"] for lbl, pg in toc_data["table"]]
        story.append(make_table(["Table Caption", "Page"], t_rows, styles, pad_v=1.6))
    story.append(PageBreak())

    # MAIN CHAPTERS 1-15 & APPENDICES A-E
    for sec in secs:
        h1 = Paragraph(sec["title"], styles["h1"])
        h1._mono_kind = "chapter"
        h1._mono_label = sec["title"]
        story.append(h1)

        for blk in sec["blocks"]:
            kind = blk[0]
            if kind == "h2":
                h2 = Paragraph(blk[1], styles["h2"])
                h2._mono_kind = "section"
                h2._mono_label = blk[1]
                story.append(h2)
            elif kind == "p":
                txt = blk[1].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                for para in txt.split("\n"):
                    if para.strip():
                        story.append(Paragraph(para, styles["body"]))
            elif kind == "algo":
                story.append(Spacer(1, 0.15 * cm))
                story.append(make_callout(blk[1], styles, bg=INFO_BG, border=NAVY))
                story.append(Spacer(1, 0.25 * cm))
            elif kind == "eq":
                _, rel_p, cap = blk
                img = make_img(rel_p, max_w=16.2 * cm, max_h=1.45 * cm)
                img._mono_kind = "equation"
                img._mono_label = cap
                story.append(img)
                story.append(Paragraph(cap, styles["caption"]))
            elif kind == "fig":
                _, rel_p, cap = blk
                img = make_img(rel_p, max_w=fig_w, max_h=10.2 * cm)
                img._mono_kind = "figure"
                img._mono_label = cap
                story.append(img)
                story.append(Paragraph(cap, styles["caption"]))
            elif kind == "table":
                _, cap, hdrs, rows = blk
                p_cap = Paragraph(f"<b>{cap}</b>", styles["caption"])
                p_cap._mono_kind = "table"
                p_cap._mono_label = cap
                story.append(p_cap)
                story.append(make_table(hdrs, rows, styles, pad_v=pad_v))
                story.append(Spacer(1, 0.2 * cm))
        if sec is not secs[-1]:
            story.append(PageBreak())

    return story


def compile_once(
    reg: Dict[str, Any],
    secs: List[Dict[str, Any]],
    body_size: float,
    leading: float,
    space_after: float,
    fig_w: float,
    pad_v: float,
) -> tuple[int, int]:
    import fitz
    styles = make_styles(body_size=body_size, leading=leading, space_after=space_after)
    recorder: Dict[str, List[Tuple[str, int]]] = {"chapter": [], "section": [], "figure": [], "table": [], "equation": []}

    doc1 = MonographDocTemplate(str(PDF_PATH), recorder=recorder, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                                topMargin=2.0 * cm, bottomMargin=2.0 * cm)
    doc1.build(build_story(reg, secs, styles, toc_data=None, fig_w=fig_w, pad_v=pad_v))

    doc2 = MonographDocTemplate(str(PDF_PATH), recorder=None, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                                topMargin=2.0 * cm, bottomMargin=2.0 * cm)
    doc2.build(build_story(reg, secs, styles, toc_data=recorder, fig_w=fig_w, pad_v=pad_v))
    doc = fitz.open(str(PDF_PATH))
    n_p = len(doc)
    last_chars = len(doc[-1].get_text("text").strip())
    doc.close()
    return n_p, last_chars


def calibrate_to_100_pages():
    reg = load_registries()
    ep_arch = extract_episode_archives()
    bsz, lead, sp, pv = 10.0, 13.8, 6.2, 2.8

    lo_ep, hi_ep = 40, len(ep_arch["p14_4"])
    best_ep = 900
    best_score = -999999

    for it in range(12):
        mid = (lo_ep + hi_ep) // 2
        secs_c = build_augmented_sections(reg, ep_arch, p14_ep_n=mid, max_file_rows=587, max_tax_rows=420)
        nc, lchars = compile_once(reg, secs_c, body_size=bsz, leading=lead, space_after=sp, fig_w=16.2 * cm, pad_v=pv)
        print(f"Calib {it:02d}: p14_ep_n={mid}, pv={pv:.2f} -> {nc} pages (last page chars={lchars})")
        if nc == 100 and lchars >= 1600:
            write_markdown_and_tex(reg, secs_c)
            return 100
        if nc < 100 or (nc == 100 and lchars < 1600):
            lo_ep = mid + 1
        else:
            hi_ep = mid - 1

    write_markdown_and_tex(reg, build_augmented_sections(reg, ep_arch, p14_ep_n=lo_ep, max_file_rows=587, max_tax_rows=420))
    return len(PdfReader(str(PDF_PATH)).pages)


if __name__ == "__main__":
    final_pages = calibrate_to_100_pages()
    print(f"FINAL PDF PAGE COUNT: {final_pages}")
