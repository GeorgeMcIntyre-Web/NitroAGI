"""Server management CLI commands."""

import asyncio
import subprocess
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import psutil

from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def start(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="API host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="API port"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of worker processes"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon"),
):
    """Start the NitroAGI server."""
    config = get_config()
    
    host = host or config.api_host
    port = port or config.api_port
    
    console.print(f"[bold green]Starting NitroAGI server...[/bold green]")
    console.print(f"Host: {host}:{port}")
    console.print(f"Workers: {workers}")
    
    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "nitroagi.api.main:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers),
    ]
    
    if reload:
        cmd.append("--reload")
        console.print("[yellow]Auto-reload enabled[/yellow]")
    
    if daemon:
        # Run as background process
        import daemon
        with daemon.DaemonContext():
            process = subprocess.Popen(cmd)
            console.print(f"[green]✓[/green] Server started as daemon (PID: {process.pid})")
    else:
        # Run in foreground
        try:
            process = subprocess.Popen(cmd)
            console.print(f"[green]✓[/green] Server started (PID: {process.pid})")
            console.print("[dim]Press Ctrl+C to stop[/dim]")
            process.wait()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down server...[/yellow]")
            process.terminate()
            process.wait(timeout=5)
            console.print("[green]✓[/green] Server stopped")


@app.command()
def stop(
    force: bool = typer.Option(False, "--force", "-f", help="Force stop"),
):
    """Stop the NitroAGI server."""
    console.print("[yellow]Stopping NitroAGI server...[/yellow]")
    
    # Find server process
    server_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'nitroagi.api.main:app' in ' '.join(cmdline):
                server_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not server_processes:
        console.print("[red]No NitroAGI server process found[/red]")
        raise typer.Exit(1)
    
    # Stop processes
    for proc in server_processes:
        try:
            if force:
                proc.kill()
                console.print(f"[green]✓[/green] Force stopped server (PID: {proc.pid})")
            else:
                proc.terminate()
                proc.wait(timeout=5)
                console.print(f"[green]✓[/green] Stopped server (PID: {proc.pid})")
        except psutil.TimeoutExpired:
            console.print(f"[yellow]Process {proc.pid} did not terminate, forcing...[/yellow]")
            proc.kill()
        except Exception as e:
            console.print(f"[red]Error stopping process {proc.pid}: {e}[/red]")


@app.command()
def restart(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="API host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="API port"),
):
    """Restart the NitroAGI server."""
    console.print("[yellow]Restarting NitroAGI server...[/yellow]")
    
    # Stop server
    try:
        stop()
    except typer.Exit:
        pass  # Server might not be running
    
    # Wait a moment
    import time
    time.sleep(2)
    
    # Start server
    start(host=host, port=port)


@app.command()
def status():
    """Check server status."""
    config = get_config()
    
    # Check if server process is running
    server_running = False
    server_pids = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'nitroagi.api.main:app' in ' '.join(cmdline):
                server_running = True
                server_pids.append({
                    'pid': proc.pid,
                    'memory': proc.info['memory_info'].rss / 1024 / 1024,  # MB
                    'uptime': time.time() - proc.info['create_time'],
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Check API endpoint
    api_responding = False
    try:
        import requests
        response = requests.get(f"http://{config.api_host}:{config.api_port}/health", timeout=2)
        api_responding = response.status_code == 200
    except Exception:
        pass
    
    # Display status
    table = Table(title="Server Status", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Status", "🟢 Running" if server_running else "🔴 Stopped")
    table.add_row("API Endpoint", f"{config.api_host}:{config.api_port}")
    table.add_row("API Health", "✓ Responding" if api_responding else "✗ Not responding")
    
    if server_pids:
        for proc in server_pids:
            table.add_row(f"Process {proc['pid']}", f"Memory: {proc['memory']:.1f} MB")
            
            uptime = proc['uptime']
            if uptime > 3600:
                uptime_str = f"{uptime/3600:.1f} hours"
            elif uptime > 60:
                uptime_str = f"{uptime/60:.1f} minutes"
            else:
                uptime_str = f"{uptime:.0f} seconds"
            table.add_row("", f"Uptime: {uptime_str}")
    
    console.print(table)


@app.command()
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of lines to show"),
    level: str = typer.Option("INFO", "--level", "-l", help="Log level filter"),
):
    """View server logs."""
    log_file = Path.home() / ".nitroagi" / "logs" / "server.log"
    
    if not log_file.exists():
        console.print("[red]Log file not found[/red]")
        raise typer.Exit(1)
    
    if follow:
        # Follow logs in real-time
        import subprocess
        subprocess.run(["tail", "-f", str(log_file)])
    else:
        # Show last N lines
        with open(log_file) as f:
            lines_list = f.readlines()[-lines:]
            
            # Filter by level if specified
            if level != "ALL":
                lines_list = [l for l in lines_list if level in l]
            
            for line in lines_list:
                # Color code by level
                if "ERROR" in line:
                    console.print(line.strip(), style="red")
                elif "WARNING" in line:
                    console.print(line.strip(), style="yellow")
                elif "DEBUG" in line:
                    console.print(line.strip(), style="dim")
                else:
                    console.print(line.strip())


@app.command()
def metrics():
    """Display server metrics."""
    config = get_config()
    
    try:
        import requests
        response = requests.get(f"http://{config.api_host}:{config.api_port}/api/v1/system/metrics")
        
        if response.status_code == 200:
            metrics_data = response.json()
            
            table = Table(title="Server Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            
            table.add_row("CPU Usage", f"{metrics_data.get('cpu_percent', 0):.1f}%")
            table.add_row("Memory Usage", f"{metrics_data.get('memory_percent', 0):.1f}%")
            table.add_row("Disk Usage", f"{metrics_data.get('disk_usage', 0):.1f}%")
            table.add_row("Active Connections", str(metrics_data.get('active_connections', 0)))
            table.add_row("Request Rate", f"{metrics_data.get('request_rate', 0):.1f} req/s")
            table.add_row("Average Latency", f"{metrics_data.get('avg_latency', 0):.1f} ms")
            
            console.print(table)
        else:
            console.print("[red]Failed to retrieve metrics[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Server might not be running[/yellow]")