"""Render the MiniGPT diagram: python assets/minigpt_flow.py.

Requires matplotlib and a CJK font (e.g. Noto Sans CJK SC or Microsoft YaHei).
The architecture follows MiniGPT and TransformerBlock in the stage 01 notebook.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def render():
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei")
    font = next((name for name in candidates if name in available), None)
    if font is None:
        raise RuntimeError("Install a CJK font such as Noto Sans CJK SC, then rerun.")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

    fig, ax = plt.subplots(figsize=(9.6, 16.8), dpi=180)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 100), ylim=(175, 0))
    ax.set_axis_off()
    ink, muted, line = "#172B43", "#516276", "#65758A"
    blue, green = "#EDF4FC", "#EBF6F0"

    def text(x, y, label, size=12, color=ink, weight="normal", ha="center"):
        return ax.text(x, y, label, fontsize=size, color=color,
                       weight=weight, ha=ha, va="center", zorder=4)

    def box(x, y, w, h, label, detail=None, fill=blue, size=12):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=1",
            facecolor=fill, edgecolor="#C7D4E2", linewidth=1, zorder=2))
        text(x + w/2, y + h*(0.37 if detail else 0.5), label, size, weight="bold")
        if detail:
            text(x + w/2, y + h*0.75, detail, 10.5, muted)

    def arrow(points, color=line, dashed=False):
        xs, ys = zip(*points)
        if len(points) > 2:
            ax.plot(xs[:-1], ys[:-1], color=color, linewidth=1.3,
                    linestyle="--" if dashed else "-", zorder=3)
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
            linewidth=1.3, color=color, linestyle="--" if dashed else "-", zorder=3))

    def add(x, y):
        ax.add_patch(Circle((x, y), 1.65, facecolor="white",
                            edgecolor=line, linewidth=1.2, zorder=4))
        text(x, y - 0.1, "+", 16)

    text(7, 5, "Mini-GPT 数据流", 21, weight="bold", ha="left")
    text(7, 10, "Decoder-only · Causal Self-Attention · Pre-LN", 11, muted, ha="left")

    box(8, 16, 38, 9, "Token IDs", "[B, T]")
    box(54, 16, 38, 9, "Position IDs", "[T]")
    arrow([(27, 25), (27, 29)])
    arrow([(73, 25), (73, 29)])
    box(8, 29, 38, 11, "Token Embedding", "[B, T, C]")
    box(54, 29, 38, 11, "Position Embedding", "[T, C] → 广播到 [B, T, C]")
    arrow([(27, 40), (27, 45), (48.35, 45)])
    arrow([(73, 40), (73, 45), (51.65, 45)])
    add(50, 45)
    arrow([(50, 46.65), (50, 61)])
    text(64, 49, "X [B, T, C]", 11, muted)

    ax.add_patch(FancyBboxPatch(
        (14, 53), 72, 69, boxstyle="round,pad=0.1,rounding_size=1.5",
        facecolor="#F8FAFD", edgecolor="#AABDD0", linewidth=1.2, zorder=1))
    text(18, 56.5, "Transformer Block × N", 12, weight="bold", ha="left")
    box(30, 61, 40, 7, "LayerNorm")
    arrow([(50, 68), (50, 72)])
    box(26, 72, 48, 14, "Multi-Head Self-Attention", "Q/K 匹配 → Mask → Softmax → A × V", size=11)
    arrow([(50, 86), (50, 89.35)])
    arrow([(50, 59.5), (20, 59.5), (20, 91), (48.35, 91)])
    text(21.5, 78, "残差", 10, muted, ha="left")
    add(50, 91)
    arrow([(50, 92.65), (50, 96)])
    box(30, 96, 40, 7, "LayerNorm")
    arrow([(50, 103), (50, 107)])
    box(30, 107, 40, 7, "FFN / MLP")
    arrow([(50, 114), (50, 117.35)])
    arrow([(50, 94), (80, 94), (80, 119), (51.65, 119)])
    text(78.5, 108, "残差", 10, muted, ha="right")
    add(50, 119)

    arrow([(50, 120.65), (50, 126)])
    box(30, 126, 40, 7, "Final LayerNorm", fill=green)
    arrow([(50, 133), (50, 137)])
    box(30, 137, 40, 7, "LM Head", fill=green)
    arrow([(50, 144), (50, 148)])
    text(50, 150.5, "Logits [B, T, V]", 13, weight="bold")
    arrow([(50, 153), (50, 155), (26, 155), (26, 158)])
    arrow([(50, 153), (50, 155), (74, 155), (74, 158)])
    box(6, 158, 40, 11, "训练", "所有位置计算下一 Token Loss", fill=green)
    box(54, 158, 40, 11, "生成", "最后位置选 Token → 追加到前缀", fill=green)
    arrow([(94, 163.5), (97, 163.5), (97, 13), (27, 13), (27, 16)], dashed=True)
    text(73, 172, "重复输入更新后的前缀", 10, muted)

    output = Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=180, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    render()
