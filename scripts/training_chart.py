"""Training metrics chart generation — ASCII terminal output and PNG export.

Parses TensorBoard events files written by kohya-ss during training, then renders:
  1. A compact ASCII chart printed to the terminal after training completes
  2. A PNG chart saved alongside the output model

kohya-ss only writes step-level logs (loss, LR) when a tracker is active.
train.py enables TensorBoard logging to a temp folder so metrics are always captured.
"""

from pathlib import Path


# ── TensorBoard events parser ────────────────────────────────────────────


def _find_events_file(tb_dir: Path) -> Path | None:
    """Find the latest events.out.tfevents.* file under tb_dir (recursive).

    kohya-ss writes to: tb_dir/<timestamp>/network_train/events.out.tfevents.<ts>.<host>.<pid>
    """
    if not tb_dir.exists():
        return None
    matches = sorted(tb_dir.rglob("events.out.tfevents.*"))
    return matches[-1] if matches else None


def parse_tensorboard_events(
    tb_dir: Path,
) -> tuple[list[int], list[float], list[float]]:
    """Parse TensorBoard events from a kohya-ss training run.

    Reads the events file and extracts per-step loss and learning rate values.

    Returns:
        (steps, avg_loss, lr_values) — parallel lists indexed by step.
        Empty lists if no events file found or parsing fails.
    """
    events_path = _find_events_file(tb_dir)
    if events_path is None:
        return [], [], []

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return [], [], []

    try:
        ea = EventAccumulator(str(events_path))
        ea.Reload()
    except Exception:
        return [], [], []

    tags = ea.Tags()["scalars"]
    if not tags:
        return [], [], []

    # Find the loss/average tag
    loss_tag = None
    for candidate in ("loss/average", "loss/train_average"):
        if candidate in tags:
            loss_tag = candidate
            break
    if loss_tag is None:
        for tag in tags:
            if "loss" in tag.lower() and "average" in tag.lower():
                loss_tag = tag
                break

    # Find the primary LR tag (lr/unet, lr/dit, lr/model, or first lr/* that isn't lr/d*)
    lr_tag = None
    for candidate in ("lr/unet", "lr/dit", "lr/model"):
        if candidate in tags:
            lr_tag = candidate
            break
    if lr_tag is None:
        for tag in tags:
            if tag.startswith("lr/") and not tag.startswith("lr/d"):
                lr_tag = tag
                break

    steps: list[int] = []
    avg_loss: list[float] = []
    lr_values: list[float] = []

    if loss_tag is not None:
        loss_events = ea.Scalars(loss_tag)
        for ev in loss_events:
            steps.append(ev.step)
            avg_loss.append(ev.value)

    if lr_tag is not None and steps:
        lr_events = ea.Scalars(lr_tag)
        lr_by_step = {ev.step: ev.value for ev in lr_events}
        for step in steps:
            lr_values.append(lr_by_step.get(step, 0.0))

    return steps, avg_loss, lr_values


# ── ASCII chart ──────────────────────────────────────────────────────────


def _smooth(values: list[float], window: int = 5) -> list[float]:
    """Simple moving-average smooth."""
    if len(values) < window:
        return list(values)
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def generate_ascii_chart(
    steps: list[int],
    values: list[float],
    title: str,
    *,
    width: int = 60,
    height: int = 12,
    smooth_window: int = 15,
) -> str:
    """Generate a compact ASCII line chart for terminal display.

    Args:
        steps: X-axis step numbers.
        values: Y-axis data values.
        title: Chart title shown above the plot.
        width: Chart width in characters.
        height: Chart height in lines.
        smooth_window: Moving-average window size for smoothing.

    Returns:
        A multi-line string containing the ASCII chart.
    """
    if not values:
        return f"{title}\n  (no data)\n"

    smoothed = _smooth(values, smooth_window)
    n = len(smoothed)

    vmin = min(smoothed)
    vmax = max(smoothed)
    vrange = vmax - vmin if vmax > vmin else 1.0

    smin = steps[0] if steps else 0
    smax = steps[-1] if steps else 1

    # Build grid: height rows x width cols
    grid: list[list[str]] = [[" " for _ in range(width)] for _ in range(height)]

    # Plot points
    for i in range(n):
        col = int((steps[i] - smin) / (smax - smin if smax > smin else 1) * (width - 1))
        normalized = (smoothed[i] - vmin) / vrange
        row = height - 1 - int(normalized * (height - 1))
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        grid[row][col] = "*"

    # Format value labels
    lines: list[str] = [f"  {title}"]
    lines.append("  " + "-" * (width + 10))

    for r in range(height):
        normalized_val = 1.0 - r / (height - 1) if height > 1 else 0.5
        val = vmin + normalized_val * vrange
        label = f"{val:>10.4f} |"
        line_chars = grid[r]
        lines.append(f"{label}{''.join(line_chars)}")

    # X-axis
    x_label = f"{'steps':>10} |"
    lines.append(x_label + "-" * width)

    # Simpler bottom label
    bottom = f"{'':>10} " + f"{smin}" + " " * (width - len(str(smin)) - len(str(smax)) - 2) + f"{smax}"
    lines.append(bottom)

    # Stats
    lines.append(f"{'':>10} min={min(values):.4f}  max={max(values):.4f}  last={values[-1]:.4f}  steps={n}")
    lines.append("")

    return "\n".join(lines)


# ── PNG chart ────────────────────────────────────────────────────────────


def generate_png_chart(
    steps: list[int],
    loss_values: list[float],
    lr_values: list[float] | None,
    output_path: Path,
    *,
    title: str = "Training Metrics",
) -> None:
    """Generate a PNG chart with loss and LR curves.

    Uses matplotlib. Writes a two-panel chart:
      - Top panel: loss (current + smoothed average)
      - Bottom panel: learning rate schedule

    Args:
        steps: X-axis step numbers.
        loss_values: Per-step average loss values.
        lr_values: Per-step LR values (may be None if not captured).
        output_path: File path to write the PNG to.
        title: Chart title.
    """
    # Lazy import — matplotlib may not be installed in minimal envs
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    smooth_window = max(1, len(steps) // 40) if len(steps) > 30 else 1
    smoothed_loss = _smooth(loss_values, smooth_window) if smooth_window > 1 else loss_values

    fig, axes = plt.subplots(
        2 if lr_values else 1,
        1,
        figsize=(10, 6 if lr_values else 3.5),
        dpi=120,
        gridspec_kw={"height_ratios": [2, 1]} if lr_values else None,
    )
    if not lr_values:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Loss panel ──
    ax_loss = axes[0]
    ax_loss.plot(steps, loss_values, color="#e74c3c", alpha=0.3, linewidth=0.6, label="raw loss")
    ax_loss.plot(steps, smoothed_loss, color="#e74c3c", linewidth=1.2, label="smoothed avg loss")
    ax_loss.set_ylabel("Loss", color="#e74c3c")
    ax_loss.tick_params(axis="y", labelcolor="#e74c3c")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlabel("Step")

    # ── LR panel ──
    if lr_values:
        ax_lr = axes[1]
        ax_lr.plot(steps, lr_values, color="#3498db", linewidth=1.2, label="learning rate")
        ax_lr.set_ylabel("LR", color="#3498db")
        ax_lr.tick_params(axis="y", labelcolor="#3498db")
        ax_lr.set_yscale("log")
        ax_lr.legend(loc="upper right", fontsize=8)
        ax_lr.grid(True, alpha=0.3)
        ax_lr.set_xlabel("Step")
        ax_lr.yaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout()

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Terminal display helper ──────────────────────────────────────────────


def print_training_summary(
    steps: list[int],
    loss_values: list[float],
    lr_values: list[float] | None,
) -> None:
    """Print ASCII charts to the terminal after training completes."""
    print("")
    print("  +----------------------------------------------------------+")
    print("  |              Training Metrics Summary                    |")
    print("  +----------------------------------------------------------+")
    print("")

    loss_chart = generate_ascii_chart(
        steps,
        loss_values,
        "Average Loss",
        smooth_window=max(1, len(steps) // 25),
    )
    print(loss_chart)

    if lr_values:
        lr_chart = generate_ascii_chart(
            steps,
            lr_values,
            "Learning Rate",
            smooth_window=max(1, len(steps) // 25),
        )
        print(lr_chart)
