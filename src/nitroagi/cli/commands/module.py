"""Module management CLI commands."""

import asyncio
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from nitroagi.core.orchestrator import Orchestrator
from nitroagi.core.base import ModuleRegistry, ModuleCapability
from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def list():
    """List all available modules."""
    async def get_modules():
        orchestrator = Orchestrator()
        await orchestrator.initialize()
        
        modules = []
        for name in orchestrator.module_registry.list_modules():
            module = orchestrator.module_registry.get_module(name)
            if module:
                modules.append({
                    "name": module.config.name,
                    "version": module.config.version,
                    "status": module.status.value,
                    "capabilities": [cap.value for cap in module.config.capabilities],
                })
        
        await orchestrator.shutdown()
        return modules
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading modules...", total=None)
        modules = asyncio.run(get_modules())
        progress.stop()
    
    if not modules:
        console.print("[yellow]No modules found[/yellow]")
        return
    
    table = Table(title="Available Modules", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Capabilities", style="magenta")
    
    for module in modules:
        status_icon = "🟢" if module["status"] == "ready" else "🔴"
        table.add_row(
            module["name"],
            module["version"],
            f"{status_icon} {module['status']}",
            ", ".join(module["capabilities"])
        )
    
    console.print(table)


@app.command()
def info(
    module_name: str = typer.Argument(..., help="Module name"),
):
    """Show detailed information about a module."""
    async def get_module_info():
        orchestrator = Orchestrator()
        await orchestrator.initialize()
        
        module = orchestrator.module_registry.get_module(module_name)
        if not module:
            return None
        
        info = {
            "name": module.config.name,
            "version": module.config.version,
            "description": module.config.description,
            "status": module.status.value,
            "capabilities": [cap.value for cap in module.config.capabilities],
            "config": {
                "max_workers": module.config.max_workers,
                "timeout_seconds": module.config.timeout_seconds,
                "cache_enabled": module.config.cache_enabled,
                "cache_ttl_seconds": module.config.cache_ttl_seconds,
            },
            "metrics": module.get_metrics() if hasattr(module, 'get_metrics') else {},
        }
        
        await orchestrator.shutdown()
        return info
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Loading module {module_name}...", total=None)
        info = asyncio.run(get_module_info())
        progress.stop()
    
    if not info:
        console.print(f"[red]Module '{module_name}' not found[/red]")
        raise typer.Exit(1)
    
    # Display module information
    console.print(f"\n[bold cyan]Module: {info['name']}[/bold cyan]")
    console.print(f"Version: {info['version']}")
    console.print(f"Description: {info['description']}")
    console.print(f"Status: {info['status']}")
    console.print(f"Capabilities: {', '.join(info['capabilities'])}")
    
    # Configuration
    console.print("\n[bold]Configuration:[/bold]")
    for key, value in info['config'].items():
        console.print(f"  {key}: {value}")
    
    # Metrics
    if info['metrics']:
        console.print("\n[bold]Metrics:[/bold]")
        for key, value in info['metrics'].items():
            console.print(f"  {key}: {value}")


@app.command()
def enable(
    module_name: str = typer.Argument(..., help="Module name to enable"),
):
    """Enable a module."""
    async def enable_module():
        orchestrator = Orchestrator()
        await orchestrator.initialize()
        
        module = orchestrator.module_registry.get_module(module_name)
        if not module:
            return False
        
        await module.initialize()
        await orchestrator.shutdown()
        return True
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Enabling module {module_name}...", total=None)
        success = asyncio.run(enable_module())
        progress.stop()
    
    if success:
        console.print(f"[green]✓[/green] Module '{module_name}' enabled")
    else:
        console.print(f"[red]Failed to enable module '{module_name}'[/red]")
        raise typer.Exit(1)


@app.command()
def disable(
    module_name: str = typer.Argument(..., help="Module name to disable"),
):
    """Disable a module."""
    async def disable_module():
        orchestrator = Orchestrator()
        await orchestrator.initialize()
        
        module = orchestrator.module_registry.get_module(module_name)
        if not module:
            return False
        
        await module.shutdown()
        await orchestrator.shutdown()
        return True
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Disabling module {module_name}...", total=None)
        success = asyncio.run(disable_module())
        progress.stop()
    
    if success:
        console.print(f"[green]✓[/green] Module '{module_name}' disabled")
    else:
        console.print(f"[red]Failed to disable module '{module_name}'[/red]")
        raise typer.Exit(1)


@app.command()
def test(
    module_name: str = typer.Argument(..., help="Module name to test"),
    input_data: Optional[str] = typer.Option(None, "--input", "-i", help="Input data (JSON)"),
):
    """Test a module with sample input."""
    async def test_module():
        orchestrator = Orchestrator()
        await orchestrator.initialize()
        
        module = orchestrator.module_registry.get_module(module_name)
        if not module:
            return None
        
        # Prepare test data
        if input_data:
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                data = input_data  # Use as string if not JSON
        else:
            # Default test data based on module type
            if module_name == "language":
                data = "Hello, test the language module"
            else:
                data = {"test": "data"}
        
        # Create request
        from nitroagi.core.base import ModuleRequest, ModuleContext
        request = ModuleRequest(
            context=ModuleContext(
                request_id="test-request",
                user_id="test-user"
            ),
            data=data,
            required_capabilities=module.config.capabilities[:1] if module.config.capabilities else []
        )
        
        # Process request
        response = await module.process(request)
        
        await orchestrator.shutdown()
        return response
    
    console.print(f"[bold]Testing module: {module_name}[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing test request...", total=None)
        response = asyncio.run(test_module())
        progress.stop()
    
    if not response:
        console.print(f"[red]Module '{module_name}' not found[/red]")
        raise typer.Exit(1)
    
    # Display response
    console.print("\n[bold green]Test Results:[/bold green]")
    console.print(f"Status: {response.status}")
    console.print(f"Processing Time: {response.processing_time_ms:.2f}ms")
    console.print(f"Confidence: {response.confidence_score:.2f}")
    
    if response.error:
        console.print(f"[red]Error: {response.error}[/red]")
    else:
        console.print("\n[bold]Response Data:[/bold]")
        if isinstance(response.data, str):
            console.print(response.data)
        else:
            syntax = Syntax(
                json.dumps(response.data, indent=2),
                "json",
                theme="monokai"
            )
            console.print(syntax)


@app.command()
def install(
    module_path: str = typer.Argument(..., help="Path or URL to module"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Module name"),
):
    """Install a new module."""
    console.print(f"[bold]Installing module from: {module_path}[/bold]")
    
    # Module installation logic would go here
    # This is a placeholder for the actual implementation
    
    console.print("[yellow]Module installation not yet implemented[/yellow]")
    console.print("Modules can be added by:")
    console.print("1. Creating a new module class in src/nitroagi/modules/")
    console.print("2. Registering it in the module registry")
    console.print("3. Restarting the server")


@app.command()
def uninstall(
    module_name: str = typer.Argument(..., help="Module name to uninstall"),
    force: bool = typer.Option(False, "--force", "-f", help="Force uninstall"),
):
    """Uninstall a module."""
    if not force:
        if not Confirm.ask(f"Are you sure you want to uninstall '{module_name}'?"):
            raise typer.Exit()
    
    console.print(f"[bold]Uninstalling module: {module_name}[/bold]")
    
    # Module uninstallation logic would go here
    # This is a placeholder for the actual implementation
    
    console.print("[yellow]Module uninstallation not yet implemented[/yellow]")