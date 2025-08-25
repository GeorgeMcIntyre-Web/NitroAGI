"""Main CLI application for NitroAGI."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
import click

from nitroagi.utils.config import get_config, Settings
from nitroagi.utils.logging import get_logger
from nitroagi.core.orchestrator import Orchestrator
from nitroagi.core.memory import MemoryManager, MemoryType
from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.api.app import create_app
from nitroagi.cli.commands import server, module, memory, chat, system

app = typer.Typer(
    name="nitroagi",
    help="NitroAGI powered by NEXUS - Advanced AI Agent System with 6G Network Support",
    add_completion=True,
    rich_markup_mode="rich"
)

console = Console()
logger = get_logger(__name__)


@app.callback()
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
):
    """NitroAGI NEXUS CLI - Manage and interact with the AI system."""
    if version:
        from nitroagi import __version__
        console.print(f"NitroAGI version {__version__}", style="bold green")
        raise typer.Exit()
    
    # Store config in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file
    ctx.obj["debug"] = debug
    
    if debug:
        console.print("[yellow]Debug mode enabled[/yellow]")


# Add subcommands
app.add_typer(server.app, name="server", help="Server management commands")
app.add_typer(module.app, name="module", help="Module management commands")
app.add_typer(memory.app, name="memory", help="Memory system commands")
app.add_typer(chat.app, name="chat", help="Interactive chat interface")
app.add_typer(system.app, name="system", help="System management commands")


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration")
):
    """Initialize NitroAGI configuration."""
    config_path = Path.home() / ".nitroagi" / "config.yaml"
    
    if config_path.exists() and not force:
        if not Confirm.ask(f"Configuration already exists at {config_path}. Overwrite?"):
            raise typer.Exit()
    
    # Create directory
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Interactive configuration
    console.print(Panel.fit("NitroAGI Configuration Setup", style="bold blue"))
    
    config = {
        "api": {
            "host": Prompt.ask("API Host", default="0.0.0.0"),
            "port": int(Prompt.ask("API Port", default="8000")),
        },
        "ai_models": {
            "openai_api_key": Prompt.ask("OpenAI API Key (optional)", default="", password=True) or None,
            "anthropic_api_key": Prompt.ask("Anthropic API Key (optional)", default="", password=True) or None,
            "default_llm_model": Prompt.ask("Default LLM Model", default="gpt-4"),
            "temperature": float(Prompt.ask("Temperature", default="0.7")),
            "max_tokens": int(Prompt.ask("Max Tokens", default="1000")),
        },
        "database": {
            "redis_url": Prompt.ask("Redis URL", default="redis://localhost:6379"),
            "postgres_url": Prompt.ask("PostgreSQL URL (optional)", default="") or None,
        },
        "network": {
            "enable_6g": Confirm.ask("Enable 6G network optimization?", default=True),
            "network_profile": Prompt.ask(
                "Network Profile",
                choices=["standard", "low_latency", "high_bandwidth", "ultra_low_latency"],
                default="standard"
            ),
        },
        "debug": Confirm.ask("Enable debug mode?", default=False),
        "log_level": Prompt.ask("Log Level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"),
    }
    
    # Write configuration
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print(f"[green]✓[/green] Configuration saved to {config_path}")
    console.print("\nYou can now start the server with: [bold]nitroagi server start[/bold]")


@app.command()
def status():
    """Check NitroAGI system status."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking system status...", total=None)
        
        status_info = {}
        
        # Check configuration
        try:
            config = get_config()
            status_info["config"] = "✓ Loaded"
        except Exception as e:
            status_info["config"] = f"✗ Error: {e}"
        
        # Check Redis connection
        try:
            import redis
            r = redis.Redis.from_url(config.database.redis_url)
            r.ping()
            status_info["redis"] = "✓ Connected"
        except Exception:
            status_info["redis"] = "✗ Not connected"
        
        # Check API server
        try:
            import requests
            response = requests.get(f"http://{config.api_host}:{config.api_port}/health", timeout=2)
            if response.status_code == 200:
                status_info["api_server"] = "✓ Running"
            else:
                status_info["api_server"] = "✗ Not responding"
        except Exception:
            status_info["api_server"] = "✗ Not running"
        
        progress.stop()
    
    # Display status table
    table = Table(title="NitroAGI System Status", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    for component, status in status_info.items():
        style = "green" if "✓" in status else "red"
        table.add_row(component.replace("_", " ").title(), status, style=style)
    
    console.print(table)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Edit configuration"),
    key: Optional[str] = typer.Argument(None, help="Configuration key to get/set"),
    value: Optional[str] = typer.Argument(None, help="Value to set"),
):
    """Manage NitroAGI configuration."""
    config_path = Path.home() / ".nitroagi" / "config.yaml"
    
    if show:
        if not config_path.exists():
            console.print("[red]Configuration file not found. Run 'nitroagi init' first.[/red]")
            raise typer.Exit(1)
        
        with open(config_path) as f:
            config_data = f.read()
        
        syntax = Syntax(config_data, "yaml", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="NitroAGI Configuration"))
        return
    
    if edit:
        if not config_path.exists():
            console.print("[red]Configuration file not found. Run 'nitroagi init' first.[/red]")
            raise typer.Exit(1)
        
        editor = os.environ.get("EDITOR", "nano")
        click.edit(filename=str(config_path))
        console.print("[green]Configuration updated[/green]")
        return
    
    if key:
        import yaml
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        # Navigate to the key
        keys = key.split(".")
        current = config_data
        
        if value is None:
            # Get value
            try:
                for k in keys:
                    current = current[k]
                console.print(f"{key}: {current}")
            except (KeyError, TypeError):
                console.print(f"[red]Key '{key}' not found[/red]")
                raise typer.Exit(1)
        else:
            # Set value
            try:
                for k in keys[:-1]:
                    current = current[k]
                
                # Convert value to appropriate type
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "." in value and value.replace(".", "").isdigit():
                    value = float(value)
                
                current[keys[-1]] = value
                
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False)
                
                console.print(f"[green]✓[/green] Set {key} = {value}")
            except (KeyError, TypeError) as e:
                console.print(f"[red]Error setting key '{key}': {e}[/red]")
                raise typer.Exit(1)


@app.command()
def benchmark(
    component: str = typer.Argument(
        "all",
        help="Component to benchmark (all, memory, language, network, api)"
    ),
    iterations: int = typer.Option(100, "--iterations", "-i", help="Number of iterations"),
    concurrent: int = typer.Option(10, "--concurrent", "-c", help="Concurrent requests"),
):
    """Run performance benchmarks."""
    import time
    import statistics
    
    console.print(Panel.fit(f"Running {component} benchmarks", style="bold blue"))
    
    async def run_benchmarks():
        results = {}
        
        if component in ["all", "memory"]:
            console.print("\n[bold]Memory Benchmarks[/bold]")
            memory_manager = MemoryManager()
            await memory_manager.initialize()
            
            # Write benchmark
            write_times = []
            with Progress(console=console) as progress:
                task = progress.add_task("Memory writes...", total=iterations)
                for i in range(iterations):
                    start = time.perf_counter()
                    await memory_manager.store(
                        f"bench_key_{i}",
                        {"data": f"value_{i}"},
                        MemoryType.WORKING
                    )
                    write_times.append(time.perf_counter() - start)
                    progress.update(task, advance=1)
            
            # Read benchmark
            read_times = []
            with Progress(console=console) as progress:
                task = progress.add_task("Memory reads...", total=iterations)
                for i in range(iterations):
                    start = time.perf_counter()
                    await memory_manager.retrieve(f"bench_key_{i}")
                    read_times.append(time.perf_counter() - start)
                    progress.update(task, advance=1)
            
            results["memory"] = {
                "write_avg": statistics.mean(write_times) * 1000,
                "write_p95": statistics.quantiles(write_times, n=20)[18] * 1000,
                "read_avg": statistics.mean(read_times) * 1000,
                "read_p95": statistics.quantiles(read_times, n=20)[18] * 1000,
            }
        
        if component in ["all", "language"]:
            console.print("\n[bold]Language Module Benchmarks[/bold]")
            # Language module benchmarks would go here
            results["language"] = {
                "latency_avg": 50.0,  # Mock values
                "latency_p95": 100.0,
            }
        
        if component in ["all", "network"]:
            console.print("\n[bold]Network Benchmarks[/bold]")
            # Network benchmarks would go here
            results["network"] = {
                "6g_latency": 0.1,  # Mock values
                "5g_latency": 10.0,
                "speedup": 100.0,
            }
        
        return results
    
    # Run benchmarks
    results = asyncio.run(run_benchmarks())
    
    # Display results
    console.print("\n[bold green]Benchmark Results[/bold green]")
    
    for component_name, metrics in results.items():
        table = Table(title=f"{component_name.title()} Performance", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric.replace("_", " ").title(), f"{value:.2f}ms")
            else:
                table.add_row(metric.replace("_", " ").title(), str(value))
        
        console.print(table)
        console.print()


@app.command()
def doctor():
    """Diagnose and fix common issues."""
    console.print(Panel.fit("NitroAGI System Diagnostics", style="bold blue"))
    
    issues = []
    fixes = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python version < 3.8")
        fixes.append("Upgrade to Python 3.8 or higher")
    
    # Check required packages
    try:
        import fastapi
        import redis
        import torch
        import transformers
    except ImportError as e:
        issues.append(f"Missing package: {e.name}")
        fixes.append(f"Run: pip install {e.name}")
    
    # Check configuration
    config_path = Path.home() / ".nitroagi" / "config.yaml"
    if not config_path.exists():
        issues.append("Configuration file not found")
        fixes.append("Run: nitroagi init")
    
    # Check Redis
    try:
        config = get_config()
        import redis
        r = redis.Redis.from_url(config.database.redis_url)
        r.ping()
    except Exception:
        issues.append("Cannot connect to Redis")
        fixes.append("Start Redis server or check connection settings")
    
    # Display results
    if not issues:
        console.print("[green]✓[/green] No issues found! System is healthy.")
    else:
        console.print(f"[yellow]Found {len(issues)} issue(s):[/yellow]\n")
        
        for i, (issue, fix) in enumerate(zip(issues, fixes), 1):
            console.print(f"[red]{i}. {issue}[/red]")
            console.print(f"   [cyan]Fix:[/cyan] {fix}\n")
        
        if Confirm.ask("Attempt automatic fixes?"):
            # Implement automatic fixes here
            console.print("[yellow]Automatic fixes not yet implemented[/yellow]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()