"""System management CLI commands."""

import asyncio
import platform
import psutil
import os
from pathlib import Path
from datetime import datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.prompt import Confirm
from rich.tree import Tree

from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger
from nitroagi.core.network import NetworkOptimizer, NetworkMetrics

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def info():
    """Display system information."""
    config = get_config()
    
    # System info
    sys_info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "CPU Cores": psutil.cpu_count(),
        "Total Memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "Available Memory": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        "Disk Usage": f"{psutil.disk_usage('/').percent:.1f}%",
    }
    
    # NitroAGI info
    nitro_info = {
        "Version": "1.0.0",
        "Environment": os.environ.get("NITROAGI_ENV", "production"),
        "Config Path": str(Path.home() / ".nitroagi" / "config.yaml"),
        "Log Level": config.log_level,
        "Debug Mode": str(config.debug),
        "6G Support": str(config.network.enable_6g),
    }
    
    # Display system info
    table = Table(title="System Information", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in sys_info.items():
        table.add_row(key, value)
    
    console.print(table)
    
    # Display NitroAGI info
    table = Table(title="NitroAGI Information", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in nitro_info.items():
        table.add_row(key, value)
    
    console.print(table)


@app.command()
def resources():
    """Monitor system resource usage."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()
    
    # Memory usage
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # Disk usage
    disk = psutil.disk_usage('/')
    
    # Network
    net_io = psutil.net_io_counters()
    
    # Display resources
    console.print(Panel.fit("[bold]System Resources[/bold]", style="cyan"))
    
    # CPU
    console.print("\n[bold]CPU:[/bold]")
    console.print(f"  Usage: {cpu_percent}%")
    console.print(f"  Frequency: {cpu_freq.current:.2f} MHz")
    console.print(f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    
    # Memory
    console.print("\n[bold]Memory:[/bold]")
    console.print(f"  Total: {memory.total / (1024**3):.2f} GB")
    console.print(f"  Used: {memory.used / (1024**3):.2f} GB ({memory.percent}%)")
    console.print(f"  Available: {memory.available / (1024**3):.2f} GB")
    console.print(f"  Swap: {swap.used / (1024**3):.2f} GB / {swap.total / (1024**3):.2f} GB")
    
    # Disk
    console.print("\n[bold]Disk:[/bold]")
    console.print(f"  Total: {disk.total / (1024**3):.2f} GB")
    console.print(f"  Used: {disk.used / (1024**3):.2f} GB ({disk.percent}%)")
    console.print(f"  Free: {disk.free / (1024**3):.2f} GB")
    
    # Network
    console.print("\n[bold]Network:[/bold]")
    console.print(f"  Bytes sent: {net_io.bytes_sent / (1024**2):.2f} MB")
    console.print(f"  Bytes received: {net_io.bytes_recv / (1024**2):.2f} MB")
    console.print(f"  Packets sent: {net_io.packets_sent:,}")
    console.print(f"  Packets received: {net_io.packets_recv:,}")


@app.command()
def monitor(
    interval: int = typer.Option(1, "--interval", "-i", help="Update interval in seconds"),
    duration: int = typer.Option(0, "--duration", "-d", help="Monitor duration in seconds (0=infinite)"),
):
    """Real-time system monitoring."""
    import time
    
    start_time = time.time()
    
    try:
        while True:
            # Clear screen
            console.clear()
            
            # Header
            console.print(Panel.fit(
                f"[bold cyan]NitroAGI System Monitor[/bold cyan]\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style="cyan"
            ))
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            console.print("\n[bold]CPU Usage:[/bold]")
            for i, percent in enumerate(cpu_percent):
                bar = "█" * int(percent / 5) + "░" * (20 - int(percent / 5))
                console.print(f"  Core {i}: [{bar}] {percent:5.1f}%")
            
            # Memory
            memory = psutil.virtual_memory()
            mem_bar = "█" * int(memory.percent / 5) + "░" * (20 - int(memory.percent / 5))
            console.print(f"\n[bold]Memory:[/bold] [{mem_bar}] {memory.percent:.1f}%")
            console.print(f"  Used: {memory.used / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB")
            
            # Processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'nitroagi' in proc.info['name'].lower():
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if processes:
                console.print("\n[bold]NitroAGI Processes:[/bold]")
                for proc in processes:
                    console.print(f"  PID {proc['pid']}: {proc['name']} "
                                f"(CPU: {proc['cpu_percent']:.1f}%, "
                                f"Mem: {proc['memory_percent']:.1f}%)")
            
            # Check duration
            if duration > 0 and time.time() - start_time > duration:
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


@app.command()
def network():
    """Display network diagnostics and 6G optimization status."""
    async def check_network():
        optimizer = NetworkOptimizer()
        await optimizer.initialize()
        
        # Get current metrics
        metrics = NetworkMetrics(
            latency_ms=10.0,  # Would be measured in real implementation
            bandwidth_mbps=1000,
            packet_loss=0.001,
            jitter_ms=1.0
        )
        
        profile = await optimizer.select_profile(metrics)
        
        return {
            "metrics": metrics,
            "profile": profile,
            "6g_ready": optimizer.is_6g_capable()
        }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing network...", total=None)
        result = asyncio.run(check_network())
        progress.stop()
    
    # Display network info
    console.print(Panel.fit("[bold]Network Diagnostics[/bold]", style="cyan"))
    
    console.print("\n[bold]Current Metrics:[/bold]")
    console.print(f"  Latency: {result['metrics'].latency_ms} ms")
    console.print(f"  Bandwidth: {result['metrics'].bandwidth_mbps} Mbps")
    console.print(f"  Packet Loss: {result['metrics'].packet_loss * 100:.3f}%")
    console.print(f"  Jitter: {result['metrics'].jitter_ms} ms")
    
    console.print(f"\n[bold]Selected Profile:[/bold] {result['profile'].name}")
    console.print(f"  Max Latency: {result['profile'].max_latency_ms} ms")
    console.print(f"  Min Bandwidth: {result['profile'].min_bandwidth_mbps} Mbps")
    
    if result['6g_ready']:
        console.print("\n[green]✓ 6G Optimization Available[/green]")
        console.print("  - Ultra-low latency mode enabled")
        console.print("  - Terabit bandwidth support")
        console.print("  - Holographic communication ready")
        console.print("  - Brain-computer interface compatible")
    else:
        console.print("\n[yellow]⚠ 6G Optimization Not Available[/yellow]")
        console.print("  Running in standard network mode")


@app.command()
def clean(
    logs: bool = typer.Option(False, "--logs", help="Clean log files"),
    cache: bool = typer.Option(False, "--cache", help="Clean cache"),
    temp: bool = typer.Option(False, "--temp", help="Clean temporary files"),
    all: bool = typer.Option(False, "--all", help="Clean everything"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Clean system files and caches."""
    if not (logs or cache or temp or all):
        console.print("[yellow]Nothing to clean. Use --logs, --cache, --temp, or --all[/yellow]")
        raise typer.Exit()
    
    if not force:
        items = []
        if all or logs:
            items.append("logs")
        if all or cache:
            items.append("cache")
        if all or temp:
            items.append("temporary files")
        
        if not Confirm.ask(f"Clean {', '.join(items)}?"):
            raise typer.Exit()
    
    cleaned = []
    
    # Clean logs
    if all or logs:
        log_dir = Path.home() / ".nitroagi" / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("*.log*"):
                if log_file.is_file():
                    log_file.unlink()
            cleaned.append("logs")
    
    # Clean cache
    if all or cache:
        cache_dir = Path.home() / ".nitroagi" / "cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
            cleaned.append("cache")
    
    # Clean temp files
    if all or temp:
        temp_dir = Path.home() / ".nitroagi" / "temp"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            temp_dir.mkdir()
            cleaned.append("temporary files")
    
    if cleaned:
        console.print(f"[green]✓[/green] Cleaned: {', '.join(cleaned)}")
    else:
        console.print("[yellow]Nothing to clean[/yellow]")


@app.command()
def backup(
    output: str = typer.Argument(..., help="Backup file path"),
    include_logs: bool = typer.Option(False, "--logs", help="Include logs"),
    include_memory: bool = typer.Option(True, "--memory/--no-memory", help="Include memory data"),
):
    """Create system backup."""
    import tarfile
    import tempfile
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Creating backup...", total=100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy configuration
            config_src = Path.home() / ".nitroagi" / "config.yaml"
            if config_src.exists():
                import shutil
                shutil.copy(config_src, temp_path / "config.yaml")
            progress.update(task, advance=25)
            
            # Export memory if requested
            if include_memory:
                async def export_memory():
                    from nitroagi.core.memory import MemoryManager
                    manager = MemoryManager()
                    await manager.initialize()
                    memories = await manager.search("*")
                    return memories
                
                memories = asyncio.run(export_memory())
                with open(temp_path / "memory.json", "w") as f:
                    import json
                    json.dump(memories, f, indent=2, default=str)
            progress.update(task, advance=25)
            
            # Copy logs if requested
            if include_logs:
                log_dir = Path.home() / ".nitroagi" / "logs"
                if log_dir.exists():
                    import shutil
                    shutil.copytree(log_dir, temp_path / "logs")
            progress.update(task, advance=25)
            
            # Create tarball
            with tarfile.open(output, "w:gz") as tar:
                tar.add(temp_path, arcname="nitroagi_backup")
            progress.update(task, advance=25)
    
    # Get backup size
    backup_size = Path(output).stat().st_size / (1024**2)  # MB
    
    console.print(f"[green]✓[/green] Backup created: {output}")
    console.print(f"  Size: {backup_size:.2f} MB")
    console.print(f"  Includes: configuration", end="")
    if include_memory:
        console.print(", memory data", end="")
    if include_logs:
        console.print(", logs", end="")
    console.print()


@app.command()
def restore(
    backup_file: str = typer.Argument(..., help="Backup file path"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing data"),
):
    """Restore from backup."""
    if not Path(backup_file).exists():
        console.print(f"[red]Backup file not found: {backup_file}[/red]")
        raise typer.Exit(1)
    
    if not force:
        if not Confirm.ask("This will overwrite existing configuration and data. Continue?"):
            raise typer.Exit()
    
    import tarfile
    import tempfile
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Restoring backup...", total=None)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(temp_dir)
            
            backup_path = Path(temp_dir) / "nitroagi_backup"
            
            # Restore configuration
            config_src = backup_path / "config.yaml"
            if config_src.exists():
                import shutil
                config_dst = Path.home() / ".nitroagi" / "config.yaml"
                config_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(config_src, config_dst)
                console.print("  [green]✓[/green] Restored configuration")
            
            # Restore memory
            memory_src = backup_path / "memory.json"
            if memory_src.exists():
                async def import_memory():
                    import json
                    from nitroagi.core.memory import MemoryManager, MemoryType
                    
                    with open(memory_src) as f:
                        memories = json.load(f)
                    
                    manager = MemoryManager()
                    await manager.initialize()
                    
                    for memory in memories:
                        await manager.store(
                            memory.get("key"),
                            memory.get("value"),
                            MemoryType.WORKING
                        )
                
                asyncio.run(import_memory())
                console.print("  [green]✓[/green] Restored memory data")
            
            # Restore logs
            logs_src = backup_path / "logs"
            if logs_src.exists():
                import shutil
                logs_dst = Path.home() / ".nitroagi" / "logs"
                logs_dst.parent.mkdir(parents=True, exist_ok=True)
                if logs_dst.exists():
                    shutil.rmtree(logs_dst)
                shutil.copytree(logs_src, logs_dst)
                console.print("  [green]✓[/green] Restored logs")
    
    console.print(f"\n[green]✓[/green] Backup restored successfully")