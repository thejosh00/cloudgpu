"""Rich output helpers for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import capacity

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


def show_capacity_report(path: object, samples: list[dict], skipped: int, reports: list[dict]) -> None:
    """Print the capacity report as plain text: names in it come from files, so never as markup."""
    def line(text: str) -> None:
        console.print(text, markup=False, highlight=False, soft_wrap=True)

    def share(count: int, total: int) -> str:
        return f"{count} of {total} ({capacity.percent(count, total)})"

    stamps = [s["timestamp"] for s in samples]
    line(f"Capacity samples in {path}")
    line(f"  Samples: {len(samples)}, first {min(stamps)}, last {max(stamps)}")
    line(f"  Skipped: {skipped} unreadable lines, counted in no bucket")
    for report in reports:
        line("")
        line(f"{report['name']}: gpu {', '.join(report['gpu'])} = {', '.join(report['targets'])}")
        if "problem" in report:
            line(f"  No home figures: {report['problem']}")
            continue
        total, counts = report["total"], report["counts"]
        line(f"  Home region: {report['home']} ({report['home_source']})")
        line(f"  Home region had capacity:        {share(counts[capacity.HOME], total)}")
        line(f"  Only other regions had capacity: {share(counts[capacity.ELSEWHERE], total)}"
             "  <- the go/no-go reads this")
        line(f"  No region had capacity:          {share(counts[capacity.NONE], total)}")
        line(f"  Types not listed in the sample:  {share(counts[capacity.NOT_LISTED], total)}")
        line("  Regions by samples with capacity:" + ("" if report["regions"] else " none"))
        for region, count in report["regions"]:
            line(f"    {region}: {share(count, total)}")


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
    table.add_column("Service")
    table.add_column("Version")
    table.add_column("Path")

    for name, app in apps.items():
        status = app.get("status", "unknown")
        style = "green" if status == "installed" else "yellow"
        # Green only when running; "unknown" (systemctl couldn't say) is not "active".
        service = app.get("service") or "?"
        service_style = {"active": "green", "inactive": "red", "failed": "red"}.get(service, "yellow")
        table.add_row(
            name,
            f"[{style}]{status}[/{style}]",
            f"[{service_style}]{service}[/{service_style}]",
            app.get("version", "?"),
            app.get("app_dir", "?"),
        )

    console.print(table)


def _price(cents_per_hour: int | None) -> str:
    """Format Lambda's price (cents/hour) as a dollar string."""
    if cents_per_hour is None:
        return "?"
    return f"${cents_per_hour / 100:.2f}/hr"


def _gib(bytes_used: object) -> str:
    """Format a byte count as GiB (Lambda returns it as a string or int)."""
    try:
        return f"{int(bytes_used) / 1024 ** 3:.1f} GiB"
    except (TypeError, ValueError):
        return "?"


def price(cents_per_hour: int | None) -> str:
    """Public alias for formatting Lambda's price (cents/hour) as a dollar string."""
    return _price(cents_per_hour)


def show_filesystems(filesystems: list[dict]) -> None:
    """Display filesystems in a table."""
    if not filesystems:
        info("No filesystems found.")
        return

    table = Table(title="Filesystems")
    table.add_column("Name", style="bold")
    table.add_column("ID")
    table.add_column("Region")
    table.add_column("In Use")
    table.add_column("Used")
    table.add_column("Mount Point")

    for fs in filesystems:
        region = fs.get("region") or {}
        region_name = region.get("name") if isinstance(region, dict) else region
        in_use = fs.get("is_in_use")
        in_use_str = "[green]yes[/green]" if in_use else "no"
        table.add_row(
            fs.get("name", "?"),
            fs.get("id", "?"),
            region_name or "?",
            in_use_str,
            _gib(fs.get("bytes_used")),
            fs.get("mount_point", "?"),
        )

    console.print(table)


def show_instances(instances: list[dict]) -> None:
    """Display running instances in a table."""
    if not instances:
        info("No running instances.")
        return

    table = Table(title="Instances")
    table.add_column("Name", style="bold")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Type")
    table.add_column("Region")
    table.add_column("IP")

    for inst in instances:
        itype = inst.get("instance_type") or {}
        region = inst.get("region") or {}
        status = inst.get("status", "?")
        style = "green" if status == "active" else "yellow"
        table.add_row(
            inst.get("name") or "[dim]-[/dim]",
            inst.get("id", "?"),
            f"[{style}]{status}[/{style}]",
            itype.get("name", "?") if isinstance(itype, dict) else "?",
            region.get("name", "?") if isinstance(region, dict) else "?",
            inst.get("ip") or "[dim]-[/dim]",
        )

    console.print(table)


def show_instance_types(instance_types: dict, available_only: bool = False) -> None:
    """Display available instance types in a table.

    Args:
        instance_types: Map of name -> {instance_type, regions_with_capacity_available}.
        available_only: If True, only show types with capacity in some region.
    """
    if not instance_types:
        info("No instance types found.")
        return

    table = Table(title="Instance Types")
    table.add_column("Name", style="bold")
    table.add_column("GPU")
    table.add_column("vCPUs", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Price")
    table.add_column("Regions Available")

    for name in sorted(instance_types):
        entry = instance_types[name]
        spec = entry.get("instance_type", {})
        specs = spec.get("specs", {})
        regions = entry.get("regions_with_capacity_available", [])
        region_names = ", ".join(r.get("name", "?") for r in regions)

        if available_only and not regions:
            continue

        table.add_row(
            name,
            spec.get("gpu_description", "?"),
            str(specs.get("vcpus", "?")),
            f"{specs.get('memory_gib', '?')} GiB",
            _price(spec.get("price_cents_per_hour")),
            f"[green]{region_names}[/green]" if region_names else "[dim]none[/dim]",
        )

    console.print(table)
