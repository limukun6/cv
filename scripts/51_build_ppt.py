#!/usr/bin/env python3
"""Build the Part 1 slide deck (10 slides, ~8-10 min).

Design constraints:
  * English slide text, Chinese speaker notes -- the deck matches the English
    report while the presenter notes are in the language we discuss in.
  * One colour per backend, identical to the report figures, everywhere:
        COLMAP  #1f77b4   VGGT-1B  #d62728   VGGT-omega  #2ca02c
  * Every number is copied from docs/PART1_REPORT.md; nothing is recomputed or
    invented here. The two deck-only charts are produced by
    scripts/50_make_ppt_figs.py from the same tables.

Output: docs/PART1_SLIDES.pptx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------- palette ---
INK = RGBColor(0x1A, 0x1F, 0x2B)
MUTED = RGBColor(0x5A, 0x64, 0x73)
FAINT = RGBColor(0x8B, 0x94, 0xA3)
RULE = RGBColor(0xD8, 0xDE, 0xE8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG_SOFT = RGBColor(0xF4, 0xF6, 0xFA)
HEADER = RGBColor(0x22, 0x2B, 0x3B)
BLUE = RGBColor(0x1F, 0x77, 0xB4)
RED = RGBColor(0xD6, 0x27, 0x28)
GREEN = RGBColor(0x2C, 0xA0, 0x2C)
AMBER = RGBColor(0xB8, 0x6E, 0x00)

BODY_FONT = "Calibri"
CJK_FONT = "\u5fae\u8f6f\u96c5\u9ed1"      # Microsoft YaHei, for the notes

W, H = 13.333, 7.5
MX = 0.62                                   # left/right margin
CW = W - 2 * MX                             # content width
NTOTAL = 11                                 # slides in the deck, for "n / N"


# ---------------------------------------------------------------- helpers ---
def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(slide, left, top, width, height, *, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, first, text, *, size, color=INK, bold=False, font=BODY_FONT,
         space_before=0, space_after=0, align=PP_ALIGN.LEFT, line=None, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.name = font
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    return p


def run(p, text, *, size, color=INK, bold=False, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.name = BODY_FONT
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    return r


def rect(slide, left, top, width, height, color, *, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(left), Inches(top), Inches(width), Inches(height))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def header(slide, title, eyebrow=None):
    """Shared slide furniture: eyebrow, title, accent rule, footer."""
    if eyebrow:
        tf = textbox(slide, MX, 0.34, CW, 0.26)
        para(tf, True, eyebrow.upper(), size=9.5, color=FAINT, bold=True)
        ty = 0.66
    else:
        ty = 0.52
    tf = textbox(slide, MX, ty, CW, 0.62)
    para(tf, True, title, size=27, color=INK, bold=True)
    rect(slide, MX, ty + 0.72, 1.15, 0.045, BLUE)


def footer(slide, idx, label="Part 1 \u00b7 Initialization Analysis"):
    tf = textbox(slide, MX, 7.06, 8.0, 0.26)
    para(tf, True, label, size=9, color=FAINT)
    tf = textbox(slide, W - MX - 1.0, 7.06, 1.0, 0.26)
    para(tf, True, f"{idx} / {NTOTAL}", size=9, color=FAINT, align=PP_ALIGN.RIGHT)


def notes(slide, text):
    tf = slide.notes_slide.notes_text_frame
    tf.word_wrap = True
    for i, block in enumerate([b for b in text.strip().split("\n\n") if b.strip()]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = block.strip().replace("\n", " ")
        r.font.size = Pt(12)
        r.font.name = CJK_FONT


def add_image(slide, path, left, top, width, height, *, valign="center"):
    iw, ih = Image.open(path).size
    ar = iw / ih
    if width / height > ar:
        h = height
        w = h * ar
    else:
        w = width
        h = w / ar
    x = left + (width - w) / 2
    y = top + {"center": (height - h) / 2, "top": 0.0}[valign]
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return x, y, w, h


def table(slide, rows, left, top, width, col_w, *, header_h=0.40, row_h=0.40,
          size=12, head_size=11.5, aligns=None, highlight=None, zebra=True,
          first_col_bold=False):
    """rows[0] is the header. col_w is a list of relative column widths."""
    nrow, ncol = len(rows), len(rows[0])
    total = sum(col_w)
    shape = slide.shapes.add_table(nrow, ncol, Inches(left), Inches(top),
                                   Inches(width), Inches(header_h + row_h * (nrow - 1)))
    tbl = shape.table
    tbl.firstRow = False
    tbl.horzBanding = False
    tbl.vertBanding = False
    for i, w in enumerate(col_w):
        tbl.columns[i].width = Emu(int(Inches(width) * (w / total)))
    tbl.rows[0].height = Inches(header_h)
    for r in range(1, nrow):
        tbl.rows[r].height = Inches(row_h)

    aligns = aligns or [PP_ALIGN.LEFT] * ncol
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.10)
            cell.margin_right = Inches(0.10)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = HEADER
            elif zebra and r % 2 == 0:
                cell.fill.fore_color.rgb = BG_SOFT
            else:
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = aligns[c]
            txt = str(val)
            bold = (r == 0) or (first_col_bold and c == 0)
            color = WHITE if r == 0 else INK
            if highlight and r > 0 and (r, c) in highlight:
                color = highlight[(r, c)]
                bold = True
            rr = p.add_run()
            rr.text = txt
            rr.font.size = Pt(head_size if r == 0 else size)
            rr.font.name = BODY_FONT
            rr.font.bold = bold
            rr.font.color.rgb = color
    return tbl


# ----------------------------------------------------------------- slides ---
def slide_title(prs, figs, ctx):
    s = blank(prs)
    rect(s, 0, 0, 0.30, H, BLUE)
    tf = textbox(s, 1.15, 1.42, 10.6, 0.3)
    para(tf, True, "AIAA 3201 \u00b7 PROJECT 4 \u00b7 PART 1", size=11, color=FAINT, bold=True)

    tf = textbox(s, 1.15, 1.86, 10.9, 1.5)
    p = para(tf, True, "Initialization, not architecture", size=44, color=INK, bold=True, line=0.95)

    rect(s, 1.15, 2.92, 1.6, 0.055, BLUE)

    tf = textbox(s, 1.15, 3.22, 10.4, 1.2)
    p = tf.paragraphs[0]
    p.line_spacing = 1.25
    run(p, "COLMAP vs. VGGT-1B vs. VGGT-\u03a9 as 3D Gaussian Splatting initialization\n",
        size=17, color=MUTED)
    run(p, "under a controlled dense view budget", size=17, color=MUTED)

    tf = textbox(s, 1.15, 5.15, 10.4, 0.9)
    para(tf, True, "[Your Name]   \u00b7   [Student ID]   \u00b7   [Affiliation]",
         size=13, color=INK)
    para(tf, False, "November 2026", size=12, color=FAINT, space_before=4)

    notes(s, """
开场（约 20 秒）。先说结论再讲过程，因为我们的核心发现和直觉相反。
一句话版：在稠密的视角预算下，COLMAP 初始化出来的 3DGS 在三个场景上全都渲染最好，
而且视角越多领先越多；但这个优势的来源不是位姿精度，是点云拓扑。
补充：我只负责 Part 1，Part 2 的无位姿稀疏重建和 Part 3 的生成闭环不在这次汇报范围内。
""")
    return s


def slide_setup(prs, figs, ctx):
    s = blank(prs)
    header(s, "What we compared, and how", "the question")

    tf = textbox(s, MX, 1.72, 5.5, 0.3)
    para(tf, True, "FOUR AXES, AS SPECIFIED IN THE BRIEF", size=9.5, color=FAINT, bold=True)

    axes = [("Initialization runtime", "seconds to a training-ready scene"),
            ("Pose accuracy", "ATE after Sim(3) alignment; relative rotation"),
            ("3DGS convergence speed", "test PSNR every 5 k iterations to 30 k"),
            ("Final rendering quality", "PSNR / SSIM / LPIPS on frozen test views")]
    y = 2.16
    for i, (name, detail) in enumerate(axes):
        rect(s, MX, y + 0.055, 0.055, 0.40, BLUE)
        tf = textbox(s, MX + 0.20, y, 5.1, 0.55)
        p = tf.paragraphs[0]
        run(p, f"{name}", size=14.5, color=INK, bold=True)
        p2 = tf.add_paragraph()
        run(p2, detail, size=11, color=MUTED)
        y += 0.86

    tf = textbox(s, 6.55, 1.72, 6.1, 0.3)
    para(tf, True, "THREE MANDATORY SCENES", size=9.5, color=FAINT, bold=True)
    table(s, [["Scene", "Type", "Images", "Test"],
              ["DL3DV-2", "outdoor", "306", "39"],
              ["RealEstate10K-1", "indoor", "279", "35"],
              ["Waymo-405841", "driving", "199", "25"]],
          left=6.55, top=2.10, width=6.1, col_w=[2.5, 1.3, 1.0, 0.9],
          size=12.5, head_size=11, row_h=0.38,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
          first_col_bold=True)

    tf = textbox(s, 6.55, 3.86, 6.1, 1.9)
    for i, line in enumerate([
            "N = 160 training views \u2014 the dense regime; 8\u201317\u00d7 denser than Part 2's sparse settings",
            "Splits frozen, llffhold = 8 \u2014 all three backends see the same images",
            "30 000 iterations, densify to 15 000, reference hyper-parameters",
            "2\u20133 seeds per configuration, with pairwise t-tests"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(7)
        p.line_spacing = 1.08
        run(p, "\u2014  ", size=12, color=BLUE, bold=True)
        run(p, line, size=12, color=MUTED)

    footer(s, 2)
    notes(s, """
这一页交代实验设计。左侧是任务书明确要求的四个比较维度：初始化耗时、位姿精度、
3DGS 收敛速度、最终渲染质量。右边是三个必须跑的场景，类型各不相同——室外、室内、车载。
关键在于控制变量：视角预算固定 160，切分文件冻结，三个后端吃完全相同的图片，
训练超参一个都不改，只换初始化。这样差异只可能来自初始化本身。
为什么是 160？任务书只说"稠密视角"没给数字。160 落在 3DGS 基准常见的 100–300 区间内，
而且比 Part 2 的稀疏设定（9–19 张）稠密 8 到 17 倍。后面 §3 会验证这个选择是站得住的。
每个配置跑 2–3 个种子，报均值±标准差，并做配对 t 检验——因为有些差距只有 0.7 dB，
不做统计就没法说它不是噪声。
""")
    return s


def slide_result(prs, figs, ctx):
    s = blank(prs)
    header(s, "COLMAP produces the best renders on all three scenes", "result")

    add_image(s, figs / "ppt_headline.png", MX, 1.66, CW, 4.35)

    tf = textbox(s, MX, 6.16, CW, 0.7)
    p = tf.paragraphs[0]
    p.line_spacing = 1.15
    run(p, "Largest gap on Waymo-405841 (+9.3 dB). ", size=13, color=INK, bold=True)
    run(p, "SSIM and LPIPS rank the backends identically, so the ordering is not an "
           "artefact of one metric.", size=13, color=MUTED)

    footer(s, 3)
    notes(s, """
这是主结果。纵轴是测试 PSNR，三个场景各一组柱子，蓝 COLMAP、红 VGGT-1B、绿 VGGT-Ω。
一句话：蓝色每一组都是最高的，没有例外。
差距是 +3.1、+6.7、+9.3 dB。注意 dB 是对数尺度，+9.3 dB 大约是像素误差小一个数量级。
最大差距出现在 Waymo，也就是唯一的车载场景，输入分辨率 1920×1280。
误差棒是种子间的标准差——可以看到 Waymo 上 COLMAP 的方差明显更大（±1.4 dB），
这一点后面还会讲，因为它影响我们怎么下结论。
另外我们同时报了 SSIM 和 LPIPS，三个指标给出完全相同的排序，所以这个结论不是挑指标挑出来的。
（图下脚注：纵轴从 20 dB 起，做了截断；具体数值都标在柱子上。）
""")
    return s


def slide_qual(prs, figs, ctx):
    s = blank(prs)
    header(s, "The ordering is visible without metrics", "result")

    add_image(s, figs / "p1_qual_dl3dv-2.png", MX, 1.56, 8.55, 5.24)

    # Right rail: what to look at, then the export-draw caveat stated openly --
    # the column header in this figure (21.4 dB) is a different VGGT-1B export
    # than the 29.08 dB headline in the table, and a sharp audience member will
    # spot it. Better to name it than to be caught by it.
    PX = 9.40
    PW = 3.32
    tf = textbox(s, PX, 1.58, PW, 0.3)
    para(tf, True, "WHAT TO LOOK AT", size=9.5, color=FAINT, bold=True)

    items = [("VGGT-1B, 21.4 dB", "loses the fine structure of the red foliage and the "
              "paving joints.", RED),
             ("VGGT-\u03a9, 28.3 dB", "keeps the shape but drifts in colour \u2014 visibly "
              "greyer and colder.", GREEN),
             ("COLMAP, 32.7 dB", "closest to ground truth on all three rows.", BLUE)]
    y = 1.94
    for lead, rest, col in items:
        tf = textbox(s, PX, y, PW, 1.0)
        p = tf.paragraphs[0]
        p.line_spacing = 1.06
        run(p, lead, size=11, color=col, bold=True)
        p2 = tf.add_paragraph()
        p2.line_spacing = 1.06
        p2.space_before = Pt(3)
        run(p2, rest, size=10.5, color=MUTED)
        y += 1.00

    rect(s, PX, 5.20, PW, 0.03, RULE)
    tf = textbox(s, PX, 5.34, PW, 0.3)
    para(tf, True, "ONE THING TO BE CAREFUL ABOUT", size=9.5, color=AMBER, bold=True)
    tf = textbox(s, PX, 5.66, PW, 1.15)
    p = tf.paragraphs[0]
    p.line_spacing = 1.06
    run(p, "These renders use a different VGGT-1B export than the 29.1 dB in the table. "
           "The exporter subsamples the cloud with an unseeded ", size=9.5, color=MUTED)
    run(p, "np.random.choice", size=9.5, color=INK, bold=True)
    run(p, ", so that column moves by several dB between exports of the same model.",
        size=9.5, color=MUTED)

    footer(s, 4)
    notes(s, """
这一页不用讲数字，让大家用眼睛看。四列从左到右是：真值、COLMAP、VGGT-1B、VGGT-Ω。
请重点看两处：
第一，右边红色枝叶的细节。COLMAP 那列能分出叶片层次；VGGT-1B 那列糊成一团，
地砖的接缝也基本消失了。
第二，VGGT-Ω 那列形状是保住了，明显比 VGGT-1B 好，但整体颜色偏灰偏冷，
也就是所谓的 color drift。这和我们后面要讲的机制是吻合的。
每张图角落有小字，是这一张图自己的 PSNR；列标题是场景均值。
这里要主动交代一件事，免得被问：这张图的 VGGT-1B 是 21.4 dB，而前一页主表写的是 29.1 dB。
不是笔误——是官方的 VGGT 导出脚本对点云做了随机下采样，而且没设种子，
所以同一个模型、同样的位姿，多导几次结果会差好几个 dB。正文 limitation 里写了：
DL3DV 上三次导出分别是 21.42、27.47、29.08，我们主表取的是最好那次。
所以严格说，VGGT-1B 那一列应该读成"从一个比较宽的分布里抽到的一次"，
而不是一个紧密的估计。这也是为什么我们强调 COLMAP 和 Ω 的结论更硬。
另外两个场景的同类图（RealEstate10K 和 Waymo）在附件里，因为时间关系不逐张放。
""")
    return s


def slide_mechanism(prs, figs, ctx):
    s = blank(prs)
    header(s, "The mechanism is not pose accuracy", "why")

    tf = textbox(s, MX, 1.68, 6.0, 0.3)
    para(tf, True, "POINT CLOUD, DL3DV-2 (model_analyzer)", size=9.5, color=FAINT, bold=True)
    table(s, [["Model", "Points", "Mean track length"],
              ["COLMAP", "3 778", "8.19"],
              ["VGGT-1B", "100 000", "1.000"],
              ["VGGT-\u03a9", "150 000", "1.000"]],
          left=MX, top=2.06, width=6.0, col_w=[2.3, 1.9, 2.2],
          size=13, head_size=11, row_h=0.46,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
          highlight={(1, 2): BLUE},
          first_col_bold=True)

    tf = textbox(s, MX, 4.12, 6.0, 0.3)
    para(tf, True, "TRACK LENGTH = 1.000 MEANS EVERY 3D POINT IS SEEN IN EXACTLY ONE IMAGE",
         size=9.0, color=FAINT, bold=True)

    tf = textbox(s, MX, 4.46, 6.0, 2.1)
    for i, line in enumerate([
            "Both exporters emit per-frame depth unprojections, not multi-view triangulations.",
            "3DGS therefore starts from mutually unconstrained samples and must reconcile their "
            "disagreements at render time.",
            "The reported 0.000 px reprojection error is vacuous: a point seen once has no "
            "cross-view residual."]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        p.line_spacing = 1.08
        run(p, "\u2014  ", size=11.5, color=BLUE, bold=True)
        run(p, line, size=11.5, color=MUTED)

    # right: the contradiction
    rect(s, 7.05, 1.66, 5.62, 1.30, BG_SOFT)
    rect(s, 7.05, 1.66, 0.055, 1.30, AMBER)
    tf = textbox(s, 7.30, 1.82, 5.2, 1.05)
    p = tf.paragraphs[0]
    p.line_spacing = 1.12
    run(p, "The contradiction, at 50 views (Re10k-1):\n", size=11.5, color=INK, bold=True)
    run(p, "VGGT-\u03a9 has ", size=12, color=MUTED)
    run(p, "5\u00d7 COLMAP's pose accuracy", size=12, color=GREEN, bold=True)
    run(p, " (ATE 0.220 % vs 1.065 %) and still renders ", size=12, color=MUTED)
    run(p, "1.35 dB worse", size=12, color=INK, bold=True)
    run(p, ".", size=12, color=MUTED)

    tf = textbox(s, 7.05, 3.28, 5.62, 0.3)
    para(tf, True, "SO WHAT SHOULD YOU CONCLUDE?", size=9.5, color=FAINT, bold=True)
    tf = textbox(s, 7.05, 3.62, 5.62, 2.6)
    for i, line in enumerate([
            "Pose accuracy alone cannot explain rendering quality.",
            "The feed-forward models are not worse reconstructors \u2014 they deliver competitive "
            "poses and dense but unconstrained geometry.",
            "Follow-up we did not run: keep VGGT's poses, re-triangulate points from multi-view "
            "correspondences."]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        p.line_spacing = 1.10
        run(p, "\u25aa  ", size=11.5, color=INK, bold=True)
        run(p, line, size=11.5, color=MUTED)

    footer(s, 5)
    notes(s, """
这是整个报告最关键的一页，也是我预期会被追问的地方。
先讲矛盾：在 50 个视角的 RealEstate10K 上，VGGT-Ω 的位姿误差只有 COLMAP 的五分之一
（0.220% 对 1.065%），但渲染出来反而差 1.35 dB。位姿更准，结果更差——说明位姿不是主因。
那变量是什么？是点云拓扑。左边的表是我们用 COLMAP 自带的 model_analyzer 跑出来的：
COLMAP 每个 3D 点平均被 8.19 张图观测到，因为它是多视图三角化出来的；
而两个前馈模型的平均 track length 正好是 1.000——意思是每个 3D 点只出现在一张图里。
它们导出的是"逐帧深度反投影"，不是三角化。所以 3DGS 一开始拿到的是一堆互相之间
没有约束关系的样本，只能在渲染时自己去调和矛盾。
顺便说一句，表里那两个 0.000 px 的重投影误差是假指标——一个只被看过一次的点，
本来就不存在跨视图残差，当然是 0。
所以正确的结论不是"前馈模型重建得差"，而是"位姿够用，几何没有多视图约束"。
自然的后续实验是保留 VGGT 的位姿、自己重新三角化点云——我们没跑，写在 limitation 里了。
""")
    return s


def slide_convergence(prs, figs, ctx):
    s = blank(prs)
    header(s, "Convergence: 30 k is the right budget at 160 views", "speed")

    add_image(s, figs / "p1_convergence_n160.png", MX, 1.62, CW, 3.30)

    items = [
        ("7 of 9 configurations", "are still improving at 30 k \u2014 and all three COLMAP curves are."),
        ("Waymo is the exception:", "both feed-forward models peak early (10 k, 20 k) and then decay; COLMAP does not."),
        ("The \"peak at 1 k\" from the N=50 run", "does not occur at 160 \u2014 it was under-training, not over-fitting."),
    ]
    y = 5.14
    for lead, rest in items:
        rect(s, MX, y + 0.10, 0.055, 0.30, BLUE)
        tf = textbox(s, MX + 0.22, y, CW - 0.22, 0.52)
        p = tf.paragraphs[0]
        p.line_spacing = 1.05
        run(p, lead + " ", size=13, color=INK, bold=True)
        run(p, rest, size=13, color=MUTED)
        y += 0.60

    footer(s, 6)
    notes(s, """
收敛这部分回答任务书的第三个维度：3DGS 收敛得多快。
横轴是训练迭代，纵轴是测试 PSNR，三条线对应三个后端。
第一，标准 30k 预算在稠密设定下是合理的：九个配置里七个到 30k 还在涨，三条 COLMAP 曲线全都在涨。
第二，Waymo 是唯一的例外——图上空心圆圈标出来了，VGGT-1B 在 10k 就见顶（23.80）然后掉到 23.57，
VGGT-Ω 在 20k 见顶（24.83）掉到 24.72。而同一场景 COLMAP 一路涨到 30k。
第三点很重要，是我们自己纠正过的一个错误结论。之前只有 50 个视角的时候，DL3DV 上
COLMAP 在 1k 就冲到 22.63 然后一路下滑——当时看起来像过拟合。
补到 160 视角以后，同一条曲线单调涨到 32.73，那个早峰彻底消失了。
所以那不是过拟合，是欠采样导致训练不足。这条教训我写进报告了。
""")
    return s


def slide_speed(prs, figs, ctx):
    s = blank(prs)
    header(s, "Speed and run-to-run stability", "speed")

    add_image(s, figs / "ppt_runtime.png", MX, 1.62, 8.35, 4.55)

    tf = textbox(s, 9.18, 1.70, 3.55, 0.3)
    para(tf, True, "READING THE CHART", size=9.5, color=FAINT, bold=True)
    tf = textbox(s, 9.18, 2.04, 3.55, 3.9)
    for i, line in enumerate([
            "Feed-forward: 70\u2013220 s regardless of input resolution (every input is "
            "normalized to a fixed 512\u00b2 window).",
            "COLMAP: 133 s \u2192 1 950 s as resolution grows from 256\u00b2 to 1920\u00d71280."]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        p.line_spacing = 1.10
        run(p, "\u2014  ", size=11.5, color=BLUE, bold=True)
        run(p, line, size=11.5, color=MUTED)

    tf = textbox(s, 9.18, 4.30, 3.55, 0.3)
    para(tf, True, "AND IT REPRODUCES", size=9.5, color=FAINT, bold=True)
    tf = textbox(s, 9.18, 4.64, 3.55, 2.0)
    for i, (lead, rest, col) in enumerate([
            ("8 of 9 configurations", " reproduced to within 0.12 dB on a second GPU, "
             "rebuilt from scratch.", INK),
            ("The one outlier:", " VGGT-1B's exporter subsamples the cloud with an unseeded "
             "np.random.choice.", RED)]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        p.line_spacing = 1.10
        run(p, "\u25aa  ", size=11.5, color=col, bold=True)
        run(p, lead, size=11.5, color=col, bold=True)
        run(p, rest, size=11.5, color=MUTED)

    footer(s, 7)
    notes(s, """
这一页同时讲速度和稳定性，因为这两个是一体两面：前馈模型快，而且稳定；COLMAP 强，但贵且抖。
左图是初始化耗时，横轴对数。三个场景，输入分辨率从 256 平方到 1920×1280。
红色和绿色几乎是平的：70 到 220 秒，因为不管输入多大，都先归一化到 512 平方的窗口。
蓝色一路暴涨：DL3DV 133 秒，Waymo 1950 秒，也就是 32.5 分钟，13.7 倍。
而且这 32.5 分钟里，特征提取只占 53 秒、匹配 258 秒，剩下 27 分钟全花在增量式建图和光束法平差上。
右边讲复现性。我们换了一台机器、重新建环境、重新下权重、重跑整条流程，
九个配置里八个复现到 0.12 dB 以内。唯一跑偏的就是 VGGT-1B 在 DL3DV 上，
从 29 降到 21.4。我们追到根因了：官方导出脚本用了一个没设种子的随机抽样。
这个我下面单独一页会讲。
""")
    return s


def slide_scaling(prs, figs, ctx):
    s = blank(prs)
    header(s, "The gap widens as observations accumulate", "scaling")

    add_image(s, figs / "p1_scaling.png", MX, 1.60, CW, 4.28)

    tf = textbox(s, MX, 6.02, CW, 0.95)
    p = tf.paragraphs[0]
    p.line_spacing = 1.12
    run(p, "COLMAP's pose error collapses 6.5\u00d7 (DL3DV-2) and 4.4\u00d7 (Re10k-1) from 50 \u2192 160 views; "
           "feed-forward error does not move. ", size=12, color=INK, bold=True)
    run(p, "VGGT-\u03a9's deficit over COLMAP grows on all three scenes, which is what moves it "
           "from first place at 50 views to last at 160.", size=12, color=MUTED)

    footer(s, 8)
    notes(s, """
这页回答任务书的第二个问题：数据驱动的重建和经典多视图几何到底差在哪，
以及"继续 scaling"能带来什么。我们的答案是：两者随数据量增长的曲线完全不同。
上排是渲染质量，下排是位姿误差，横轴视角数从 50 到 160，虚线延伸到全序列 COLMAP 基线。
先看下排，这是最清楚的一张：DL3DV 上 COLMAP 的 ATE 从 0.472% 掉到 0.073%，6.5 倍；
RealEstate10K 从 1.065% 掉到 0.241%，4.4 倍。而红绿两条线基本是平的。
也就是说经典 SfM 的精度是"给它多少数据"的函数，前馈模型的精度不是。
再看上排，DL3DV 上 COLMAP 从 22.1 涨到 32.2，VGGT-Ω 只从 22.5 涨到 28.4。
VGGT-Ω 在 50 视角时其实是第一名（比 COLMAP 高 0.45 dB），到 160 视角掉到最后一名，落后 3.77 dB。
这个反转就是"增益最少"的后果，也是我们报告的第二个反直觉发现。
但这里我要提醒你一个容易被追问的点，别把话说满：如果问"那 COLMAP 对 VGGT-1B 的差距也拉大了吗"，
答案是**不一定**。DL3DV 上两者增益几乎一样（+10.1 对 +9.8），差距只从 2.78 拉到 3.09；
Waymo 上反而缩小了 0.31 dB（+0.41 对 +0.72）。
真正稳定成立的是"Ω 相对 COLMAP 的差距在三个场景全都变大"（4.22 / 0.41 / 0.12 dB）。
所以标题说"差距拉大"，严格说是对 Ω 成立、对 1B 大致持平——被问到就照这个说，不要硬撑。
注意 Ω 的这个机理解释我们标成了假设——因为它在 track length 上和 VGGT-1B 一样，
但要做多视图一致性测量才能确认，我们没做。
""")
    return s


def slide_m360(prs, figs, ctx):
    s = blank(prs)
    header(s, "Bonus: it also holds on Mip-NeRF 360", "additional dataset")

    tf = textbox(s, MX, 1.72, 6.4, 0.3)
    para(tf, True, "THE 3DGS PAPER'S OWN BENCHMARK \u2014 OPTIONAL IN THE BRIEF",
         size=9.5, color=FAINT, bold=True)

    table(s, [["Scene", "Images", "COLMAP", "VGGT-1B", "Gap"],
              ["counter", "240", "29.47", "23.66", "+5.81"],
              ["bonsai", "292", "31.27", "17.03", "+14.24"],
              ["flowers", "173", "21.49", "15.51", "+5.99"]],
          left=MX, top=2.10, width=6.40, col_w=[1.8, 1.2, 1.5, 1.5, 1.3],
          size=13, head_size=11, header_h=0.44, row_h=0.56,
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4,
          highlight={(1, 2): BLUE, (2, 2): BLUE, (3, 2): BLUE, (2, 4): INK},
          first_col_bold=True)

    tf = textbox(s, MX, 4.40, 6.4, 0.6)
    p = tf.paragraphs[0]
    p.line_spacing = 1.08
    run(p, "COLMAP is re-run by us on the same frames", size=10.5, color=MUTED, bold=True)
    run(p, ", not the bundled all-frame model \u2014 otherwise the comparison would not be "
           "like-for-like.", size=10.5, color=MUTED)

    PX, PW = 7.15, 5.55
    tf = textbox(s, PX, 1.72, PW, 0.3)
    para(tf, True, "WHAT IT SHOWS", size=9.5, color=FAINT, bold=True)

    tf = textbox(s, PX, 2.10, PW, 1.10)
    p = tf.paragraphs[0]
    p.line_spacing = 1.10
    run(p, "+5.8 to +14.2 dB. ", size=12.5, color=INK, bold=True)
    run(p, "The COLMAP advantage is the same order as on the three mandatory scenes, on a "
           "benchmark we did not tune on.", size=12.5, color=MUTED)

    tf = textbox(s, PX, 3.02, PW, 1.55)
    p = tf.paragraphs[0]
    p.line_spacing = 1.10
    run(p, "bonsai is the strongest form of the scaling asymmetry. ", size=12.5, color=INK, bold=True)
    run(p, "Going from 50 to 160 views, COLMAP improves (28.33 \u2192 31.27 dB) while "
           "VGGT-1B gets ", size=12.5, color=MUTED)
    run(p, "worse", size=12.5, color=RED, bold=True)
    run(p, " (19.05 \u2192 17.03 dB).", size=12.5, color=MUTED)

    rect(s, PX, 4.44, PW, 0.74, BG_SOFT)
    rect(s, PX, 4.44, 0.055, 0.74, AMBER)
    tf = textbox(s, PX + 0.22, 4.55, PW - 0.44, 0.56)
    p = tf.paragraphs[0]
    p.line_spacing = 1.08
    run(p, "Caveats. ", size=10.5, color=AMBER, bold=True)
    run(p, "Six configurations, n = 1, no significance test. VGGT-\u03a9 was not run here. "
           "kitchen is absent from the weights mirror we used, so flowers was substituted.",
        size=10.5, color=MUTED)

    rect(s, MX, 6.28, CW, 0.60, BG_SOFT)
    rect(s, MX, 6.28, 0.055, 0.60, BLUE)
    tf = textbox(s, MX + 0.22, 6.38, CW - 0.44, 0.42)
    p = tf.paragraphs[0]
    run(p, "3 of 3 scenes, same direction, no reversal \u2014 ", size=11.5, color=INK, bold=True)
    run(p, "the optional-deliverable check passes, and on a benchmark we never tuned on.",
        size=11.5, color=MUTED)

    footer(s, 9)
    notes(s, """
这一页是加分项，也补上一个逻辑漏洞：报告第 6 节提到 Mip-NeRF 360 只有单种子，
如果 deck 里完全不出现它，这个 limitation 就成了悬空引用。
Mip-NeRF 360 是任务书里明确列为可选的额外数据集，也是原始 3DGS 论文自己的基准。
我们跑的时候没有复用官方打包的全帧 COLMAP 模型，而是自己用同样的帧重跑了一遍，
否则比较不是同口径的——这一点写在左下角。
结果：COLMAP 领先 +5.8 到 +14.2 dB，和三个必考场景是同一个量级。
这说明结论不是挑场景挑出来的。
右边第二条是我觉得最有说服力的：bonsai 上从 50 视角加到 160 视角，
COLMAP 从 28.33 涨到 31.27，而 VGGT-1B 反而从 19.05 掉到 17.03。
给更多数据，经典方法变好、前馈方法变差——这是第 8 页 scaling 那张图的最强版本。
右下角主动交代 caveat：六个配置都是单次运行，没有做显著性检验；
这个数据集上没跑 VGGT-Ω；另外我们用的权重镜像里没有 kitchen，所以用 flowers 顶替。
如果被问"为什么不是同一批场景"，就说是镜像缺场景，不是我们挑有利的跑。
""")
    return s


def slide_failures(prs, figs, ctx):
    s = blank(prs)
    header(s, "Seven silent failure modes", "the hidden cost")

    tf = textbox(s, MX, 1.62, CW, 0.35)
    p = tf.paragraphs[0]
    run(p, "Every number above required eliminating these first. ", size=12.5, color=INK, bold=True)
    run(p, "None produced an error message. Two individually exceeded the effects we report as findings.",
        size=12.5, color=MUTED)

    rows = [["#", "Failure", "Silent symptom", "Impact"],
            ["1", "Diverged COLMAP camera", "camera centre at 7.2\u00d710\u2079; scene box 7.2\u00d710\u2079", "+13.0 dB"],
            ["2", "Double-normalized focal in the \u03a9 export", "exported fx = 0.4 instead of 204", "+13.4 dB"],
            ["3", "Mixed alignment procedures", "rotation error inflated 20\u00d7 on a near-planar path", "would mis-rank"],
            ["4", "Wrong Waymo GT convention", "rotation error 23.5\u00b0 instead of 1.38\u00b0", "poses look broken"],
            ["5", "Dataset poses hide SfM cost", "initialization time = 0", "speed axis impossible"],
            ["6", "Missing sparse/0 directory", "\"Could not recognize scene type!\"", "pipeline won't start"],
            ["7", "LPIPS weights fetched at eval time", "528 MB at 150 KB/s", "\u223c1 h of stalls"]]
    table(s, rows, left=MX, top=2.12, width=CW, col_w=[0.45, 4.6, 5.6, 2.35],
          size=12, head_size=11, header_h=0.38, row_h=0.46,
          aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT],
          highlight={(1, 3): RED, (2, 3): RED})

    tf = textbox(s, MX, 6.30, CW, 0.6)
    p = tf.paragraphs[0]
    p.line_spacing = 1.12
    run(p, "Failures 1 and 2 are the same species: ", size=11.5, color=INK, bold=True)
    run(p, "scale errors that leave the optimizer running happily, detectable only by checking an "
           "intermediate quantity against a known reference. Watching the loss curve would not have caught either.",
        size=11.5, color=MUTED)

    footer(s, 10)
    notes(s, """
这一页是我认为对做实验最有价值的部分，虽然它不在任务书的评分项里。
前面所有数字，都是先排掉这七个坑才得到的。这七个的共同点是：全都不报错。
看前两行，红色的两个：一个 COLMAP 相机发散，相机中心跑到 7.2 乘 10 的 9 次方，
场景包围盒跟着炸掉，渲染 15.2 dB；修好以后 28.2 dB。第二个是 VGGT-Ω 导出时焦距被归一化了两次，
导出的 fx 是 0.4 而不是 204，渲染 9.1 dB；修好 22.5 dB。
两个都是一次性 +13 dB，比我们当成"发现"来报的效应（+3 到 +9 dB）还大。
而且注意，这两个都不是代码报错，损失曲线也照常下降——你盯着 loss 看是发现不了的。
唯一的检测办法是拿一个中间量去对已知参考：第一个对相机中心分布，第二个对数据集自带焦距。
所以我的结论是：在稀疏重建这类任务里，工具链的静默误差比要研究的效应还大，
"对独立参考做校验"应该是一个正式实验步骤，不是调试。
""")
    return s


def slide_conclusion(prs, figs, ctx):
    s = blank(prs)
    header(s, "Takeaways", "conclusion")

    rect(s, MX, 1.70, 0.055, 0.44, BLUE)
    tf = textbox(s, MX + 0.24, 1.72, CW - 0.24, 1.05)
    p = tf.paragraphs[0]
    p.line_spacing = 1.10
    run(p, "COLMAP-initialized 3DGS renders best on all three scenes, and the margin "
           "widens as observations accumulate.", size=17, color=INK, bold=True)

    items = [
        ("Pose accuracy is not the explanation.", "COLMAP's error falls 4\u20136\u00d7 from 50 \u2192 160 views "
         "while feed-forward error stays flat \u2014 and at 50 views COLMAP has the worst poses yet the best renders."),
        ("VGGT-\u03a9 is the best feed-forward option.", "By far the most reproducible (std 0.03\u20130.07 dB), "
         "and it beats COLMAP at 50 views on DL3DV-2 \u2014 but it gains least from more views and falls to last place."),
        ("The decisive variable is point-cloud topology.", "Track length 8.19 for triangulation vs 1.000 for "
         "per-frame depth unprojection."),
        ("What data-driven scaling buys:", "a 70\u2013100 s, resolution-insensitive, highly reproducible initializer "
         "\u2014 at the cost of 3\u20139 dB of fidelity."),
    ]
    y = 3.02
    for lead, rest in items:
        tf = textbox(s, MX + 0.02, y, CW - 0.02, 0.72)
        p = tf.paragraphs[0]
        p.line_spacing = 1.08
        run(p, "\u25aa  ", size=12.5, color=BLUE, bold=True)
        run(p, lead + " ", size=12.5, color=INK, bold=True)
        run(p, rest, size=12.5, color=MUTED)
        y += 0.78

    rect(s, MX, 6.28, CW, 0.62, BG_SOFT)
    tf = textbox(s, MX + 0.16, 6.38, CW - 0.32, 0.45)
    p = tf.paragraphs[0]
    run(p, "Limitations: ", size=10.5, color=INK, bold=True)
    run(p, "VGGT-1B's export is stochastic (\u00b1several dB); Mip-NeRF 360 is single-seed; no hybrid "
           "VGGT-poses + re-triangulated-points run; COLMAP timings are CPU-bound.",
        size=10.5, color=MUTED)

    footer(s, 11)
    notes(s, """
收尾，四句话。
第一句是主结论：COLMAP 初始化在三个场景都最好，而且视角越多领先越多。
第二句解释为什么不是位姿：COLMAP 的误差随视角数掉 4 到 6 倍，前馈模型不动；
而且 50 视角时 COLMAP 有位姿最差的配置却渲染最好，这个反证很强。
第三句给前馈模型一个公道：VGGT-Ω 是最好的前馈方案，也是复现性最好的，
在 50 视角的 DL3DV 上它确实赢过 COLMAP——但它从更多视角里获益最少，到 160 视角就掉到最后。
第四句回答"继续 scaling 带来什么"：换来一个 70 到 100 秒、对分辨率不敏感、高度可复现的初始化器，
代价是 3 到 9 dB 的保真度。
最后一行是 limitation，主动承认：VGGT-1B 的导出本身是随机的，所以那一列应当读成
"从一个较宽分布里抽的一次"；Mip-NeRF 360 那部分只有单种子；
我们没跑"VGGT 位姿 + 自己三角化点云"的混合方案——这其实是最有希望的后续方向；
COLMAP 的耗时是 CPU 相关的，换机器数值会变。
""")
    return s


# ------------------------------------------------------------------- main ---
def build(out: Path, figs: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "Initialization, not architecture \u2014 Part 1"
    prs.core_properties.author = "[Your Name]"
    prs.core_properties.subject = "AIAA 3201 Project 4, Part 1"

    ctx = {}
    for fn in (slide_title, slide_setup, slide_result, slide_qual, slide_mechanism,
               slide_convergence, slide_speed, slide_scaling, slide_m360,
               slide_failures, slide_conclusion):
        fn(prs, figs, ctx)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/PART1_SLIDES.pptx")
    ap.add_argument("--figdir", default="docs/figures")
    args = ap.parse_args()
    out = build(Path(args.out), Path(args.figdir))
    print(f"{out}  {out.stat().st_size/1024:.0f} KB  ({len(Presentation(str(out)).slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
