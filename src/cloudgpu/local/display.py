"""Rich output helpers for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
error_console = Console(stderr=True)


def success(msg: str) -> None:
    console.print(f"[green]{msg}[/green]")


def error(msg: str) -> None:
    error_console.print(f"[red]{msg}[/red]")


def warn(msg: str) -> None:
    console.print(f"[yellow]{msg}[/yellow]")


def info(msg: str) -> None:
    console.print(msg)


def show_detection(detection: dict) -> None:
    """Display detection results in a formatted panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    persistent = detection.get("persistent_dir")
    table.add_row("Persistent Dir", persistent or "[red]Not found[/red]")

    gpu = detection.get("gpu", {})
    if gpu.get("available"):
        gpu_names = ", ".join(g["name"] for g in gpu.get("gpus", []))
        table.add_row("GPU", gpu_names)
        table.add_row("Driver", gpu.get("driver_version", "unknown"))
    else:
        table.add_row("GPU", "[red]Not detected[/red]")

    cuda = detection.get("cuda", {})
    table.add_row("CUDA", cuda.get("version") or "[red]Not detected[/red]")

    torch = detection.get("python_torch", {})
    if torch.get("available"):
        cuda_status = "[green]Yes[/green]" if torch.get("cuda_available") else "[red]No[/red]"
        table.add_row("PyTorch", f"{torch.get('torch_version')} (CUDA: {cuda_status})")
    else:
        table.add_row("PyTorch", "[red]Not available[/red]")

    console.print(Panel(table, title="Instance Detection", border_style="blue"))


def show_status(apps: dict) -> None:
    """Display installed apps status."""
    if not apps:
        info("No apps installed.")
        return

    table = Table(title="Installed Apps")
    table.add_column("App", style="bold")
    table.add_column("Status")
    table.add_column("Version")
    table.add_column("Path")

    for name, app in apps.items():
        status = app.get("status", "unknown")
        style = "green" if status == "installed" else "yellow"
        table.add_row(
            name,
            f"[{style}]{status}[/{style}]",
            app.get("version", "?"),
            app.get("app_dir", "?"),
        )

    console.print(table)
