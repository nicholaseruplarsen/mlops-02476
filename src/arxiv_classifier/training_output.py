import re
import torch
import random
import platform
import subprocess
import numpy as np
from datetime import datetime
from tqdm import tqdm as _tqdm_orig

try:
    from nvitop import Device as NvDevice

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def hex_to_ansi(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


class Colors:
    GREEN = "\033[38;5;118m"
    CYAN = "\033[38;5;51m"
    GRAY = "\033[90m"
    YELLOW = "\033[38;2;255;255;230m"
    ORANGE = "\033[38;5;214m"
    TEAL = hex_to_ansi("#00CED1")
    RED_ORANGE = hex_to_ansi("#FF6347")
    VIOLET = hex_to_ansi("#9932CC")
    GOLDENROD = hex_to_ansi("#DAA520")
    RESET = "\033[0m"


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device_name(device):
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        short = name.replace("NVIDIA ", "").replace("GeForce ", "").split("(")[0].strip()
        return short
    elif device.type == "mps":
        return _get_apple_chip_name("MPS")
    elif device.type == "cpu":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return _get_apple_chip_name("CPU")
        return "CPU"
    return "CPU"


def _get_apple_chip_name(suffix):
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return f"{result.stdout.strip()} {suffix}"
    except Exception:
        return f"Apple Silicon {suffix}"


def _is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        pass
    return False


class ProgressBar:
    def __init__(self, total, desc, device, baseline=None, update_freq=50, color=None, is_eval=False, batch_size=1):
        self.total = total
        self.batch_size = batch_size
        # self.total_batches = (total + batch_size - 1) // batch_size
        self.device = device
        self.baseline = baseline
        self.update_freq = update_freq
        self.loss_sum = torch.tensor(0.0, device=device)
        self.count = 0
        self.batch_count = 0
        self.pbar = None

        if not _is_notebook():
            device_name = get_device_name(device)
            if color:
                self.color = color
                self.dev_display = (
                    f"{Colors.ORANGE}{device_name}{Colors.RESET}"
                    if is_eval
                    else f"{Colors.GREEN}{device_name}{Colors.RESET}"
                )
            elif device.type == "cuda":
                self.color = "#76B900"
                self.dev_display = f"{Colors.GREEN}{device_name}{Colors.RESET}"
            elif device.type == "mps" or "Apple" in device_name:
                self.color = "#00C2FF"
                self.dev_display = f"{Colors.CYAN}{device_name}{Colors.RESET}"
            else:
                self.color = "#888888"
                self.dev_display = f"{Colors.GRAY}{device_name}{Colors.RESET}"

            if is_eval:
                self.color = "#FFA500"

            timestamp = datetime.now().strftime("%H:%M:%S")
            self.pbar = _tqdm_orig(
                total=self.total,
                # total=self.total_batches,
                desc=f"{timestamp} - {desc:>5}",
                bar_format="{desc} │ {postfix} │{bar:43}│ [{elapsed} {rate_fmt}] {n:,}/{total:,} │ " + self.dev_display,
                colour=self.color,
                leave=False,
                dynamic_ncols=False,
                ncols=200,
                postfix={"": self._format_loss()},
                unit=" samples",
            )

    def _format_loss(self):
        if self.count == 0:
            return f"{Colors.YELLOW}{' ' * 19}{Colors.RESET}".ljust(19 + len(Colors.YELLOW) + len(Colors.RESET))
        loss = (self.loss_sum / self.count).item()
        if self.baseline and self.baseline != 0:
            imp = (self.baseline - loss) / self.baseline * 100
            s = f"{loss:.6f} ({imp:+.1f}%)"
        else:
            s = f"{loss:.6f}"
        return f"{Colors.YELLOW} {s}{Colors.RESET}".ljust(19 + len(Colors.YELLOW) + len(Colors.RESET))

    def _get_gpu_stats(self, device_idx=0):
        if not NVML_AVAILABLE:
            return None
        try:
            dev = NvDevice(device_idx)
            util = dev.gpu_utilization()
            mem = dev.memory_info()
            temp = dev.temperature()
            power = dev.power_usage()
            power_limit = dev.power_limit()

            C = Colors
            return (
                f"{C.RESET}("
                f"{C.TEAL}{util:2d}%{C.RESET}, "
                f"{C.RED_ORANGE}{temp}°C{C.RESET}, "
                f"{C.GOLDENROD}{mem.used / 1024**3:.1f}/{mem.total / 1024**3:.1f}GB{C.RESET}, "
                f"{C.VIOLET}{power / 1000:.0f}/{power_limit / 1000:.0f}W{C.RESET})"
            )
        except Exception:
            return None

    def update(self, batch_loss, batch_size):
        self.loss_sum += batch_loss.detach() * batch_size
        self.count += batch_size
        self.batch_count += 1

        if self.pbar:
            self.pbar.update(batch_size)
            if self.batch_count % self.update_freq == 0:
                loss_str = self._format_loss()
                if self.device.type == "cuda":
                    gpu = self._get_gpu_stats(self.device.index or 0)
                    if gpu:
                        self.pbar.bar_format = f"{{desc}} │{{postfix}}│{{bar:43}}│ [{{elapsed}} {{rate_fmt}}] {{n:,}}/{{total:,}} │ {self.dev_display} {gpu}"
                self.pbar.set_postfix_str(loss_str)

    def get_loss(self):
        if self.count == 0:
            return 0.0
        return (self.loss_sum / self.count).item()

    def close(self):
        if self.pbar:
            self.pbar.close()


def log_training_config(train_size: int, val_size: int, device, num_workers: int, batch_size: int) -> None:
    """Log training configuration at start."""
    device_name = get_device_name(device)
    print(f"Training: {train_size:,} samples, Validation: {val_size:,} samples")
    print(f"Device: {Colors.GREEN}{device_name}{Colors.RESET}, Workers: {num_workers}, Batch size: {batch_size}")


def log_epoch_summary(
    epoch: int, epochs: int, train_loss: float, val_loss: float, val_acc: float, improved: bool
) -> None:
    """Log epoch summary after train/val."""
    marker = f" {Colors.GREEN}*{Colors.RESET}" if improved else ""
    print(
        f"Epoch {epoch + 1}/{epochs} │ "
        f"Train: {Colors.YELLOW}{train_loss:.4f}{Colors.RESET} │ "
        f"Val: {Colors.YELLOW}{val_loss:.4f}{Colors.RESET} │ "
        f"Acc: {Colors.CYAN}{val_acc:.2%}{Colors.RESET}{marker}"
    )


def log_training_complete(output_dir) -> None:
    """Log training completion message."""
    print(f"{Colors.GREEN}Training complete.{Colors.RESET} Models saved to {output_dir}")


class EpochLogger:
    def __init__(self, logger, baselines, show_accuracy=True):
        self.logger = logger
        self.baselines = baselines
        self.show_accuracy = show_accuracy
        self.col_widths = [5, 19, 19, 19, 8, 8, 8, 8] if show_accuracy else [5, 19, 19, 19, 8, 8]
        self._print_header()

    def _c(self, x):
        return f"{Colors.YELLOW}{x}{Colors.RESET}"

    def _print_header(self):
        headers = (
            ["Epoch", "Train Loss (imp)", "Val Loss (imp)", "Test Loss (imp)", "Time", "Val Acc", "Test Acc", "LR"]
            if self.show_accuracy
            else ["Epoch", "Train Loss (imp)", "Val Loss (imp)", "Test Loss (imp)", "Time", "LR"]
        )
        header_str = " │ ".join(f"{h:>{w}}" for h, w in zip(headers, self.col_widths))

        self.logger.info(header_str)
        self.logger.info("─" * len(ANSI_ESCAPE.sub("", header_str)))

        b = self.baselines
        tr0, v0, te0 = b["train"][0], b["val"][0], b["test"][0]
        tr1, v1, te1 = b["train"][1], b["val"][1], b["test"][1]

        baseline_row = [
            "0",
            self._c(f"{tr0:.6f}   (0.0%)"),
            self._c(f"{v0:.6f}   (0.0%)"),
            self._c(f"{te0:.6f}   (0.0%)"),
            "",
            "",
            "",
            "(Repeat)",
        ]
        price_row = [
            "",
            self._c(f"{tr1:.6f}   (0.0%)"),
            self._c(f"{v1:.6f}   (0.0%)"),
            self._c(f"{te1:.6f}   (0.0%)"),
            "",
            "",
            "",
            "(Z-scaled close prices)",
        ]

        self.logger.info(self._format_row(baseline_row))
        self.logger.info(self._format_row(price_row))

    def _format_row(self, col_values):
        def fmt(text, width):
            visible_len = len(ANSI_ESCAPE.sub("", str(text)))
            return " " * max(0, width - visible_len) + str(text)

        return " │ ".join(fmt(v, w) for v, w in zip(col_values, self.col_widths))

    def _imp(self, loss, phase, idx=0):
        base = self.baselines[phase][idx]
        return (base - loss) / base * 100 if base != 0 else 0

    def _fmt_loss(self, loss, phase, idx=0):
        return self._c(f"{loss:.6f} ({self._imp(loss, phase, idx):+.1f}%)")

    def log_epoch(
        self,
        epoch,
        train_loss,
        val_loss,
        test_loss,
        train_ploss,
        val_ploss,
        test_ploss,
        val_acc,
        test_acc,
        elapsed,
        lr,
        improved,
        best_val,
    ):
        row = [
            f"{epoch + 1}",
            self._fmt_loss(train_loss, "train"),
            self._fmt_loss(val_loss, "val"),
            self._fmt_loss(test_loss, "test"),
            self._c(f"{elapsed:.1f}s"),
            self._c(f"{val_acc:.1%}") if self.show_accuracy else "",
            self._c(f"{test_acc:.1%}") if self.show_accuracy else "",
            self._c(f"{lr:.1e}"),
        ]

        price_row = [
            "",
            self._fmt_loss(train_ploss, "train", 1),
            self._fmt_loss(val_ploss, "val", 1),
            self._fmt_loss(test_ploss, "test", 1),
            "",
            "",
            "",
            "",
        ]

        row_str = self._format_row(row)
        price_row_str = self._format_row(price_row)

        if improved:
            self.logger.info(f"{row_str} * (Val {best_val:.6f} -> {val_loss:.6f})")
        else:
            self.logger.info(row_str)
        self.logger.info(price_row_str)
