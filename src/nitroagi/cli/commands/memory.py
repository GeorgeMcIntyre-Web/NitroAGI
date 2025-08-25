"""Memory system CLI commands."""

import asyncio
import json
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Confirm

from nitroagi.core.memory import MemoryManager, MemoryType
from nitroagi.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def store(
    key: str = typer.Argument(..., help="Memory key"),
    value: str = typer.Argument(..., help="Memory value (JSON or string)"),
    memory_type: str = typer.Option("working", "--type", "-t", help="Memory type (working/episodic/semantic)"),
    ttl: Optional[int] = typer.Option(None, "--ttl", help="Time to live in seconds"),
):
    """Store data in memory."""
    async def store_memory():
        manager = MemoryManager()
        await manager.initialize()
        
        # Parse value
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        
        # Convert memory type
        type_map = {
            "working": MemoryType.WORKING,
            "episodic": MemoryType.EPISODIC,
            "semantic": MemoryType.SEMANTIC,
        }
        mem_type = type_map.get(memory_type, MemoryType.WORKING)
        
        # Store with metadata
        metadata = {}
        if ttl:
            metadata["ttl"] = ttl
        
        memory_id = await manager.store(key, parsed_value, mem_type, metadata=metadata)
        return str(memory_id)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Storing memory...", total=None)
        memory_id = asyncio.run(store_memory())
        progress.stop()
    
    console.print(f"[green]✓[/green] Stored memory with ID: {memory_id}")
    console.print(f"  Key: {key}")
    console.print(f"  Type: {memory_type}")
    if ttl:
        console.print(f"  TTL: {ttl} seconds")


@app.command()
def get(
    key: str = typer.Argument(..., help="Memory key to retrieve"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json/text)"),
):
    """Retrieve data from memory."""
    async def get_memory():
        manager = MemoryManager()
        await manager.initialize()
        return await manager.retrieve(key)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Retrieving memory...", total=None)
        value = asyncio.run(get_memory())
        progress.stop()
    
    if value is None:
        console.print(f"[red]Memory key '{key}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Memory: {key}[/bold]")
    
    if format == "json" and not isinstance(value, str):
        syntax = Syntax(
            json.dumps(value, indent=2),
            "json",
            theme="monokai"
        )
        console.print(syntax)
    else:
        console.print(value)


@app.command()
def search(
    pattern: str = typer.Argument(..., help="Search pattern (supports wildcards)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Search memories by pattern."""
    async def search_memories():
        manager = MemoryManager()
        await manager.initialize()
        return await manager.search(pattern, limit=limit)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Searching for '{pattern}'...", total=None)
        results = asyncio.run(search_memories())
        progress.stop()
    
    if not results:
        console.print(f"[yellow]No memories found matching '{pattern}'[/yellow]")
        return
    
    table = Table(title=f"Search Results ({len(results)} found)", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Value Preview", style="white")
    table.add_column("Created", style="yellow")
    
    for result in results:
        value_preview = str(result.get("value", ""))[:50]
        if len(str(result.get("value", ""))) > 50:
            value_preview += "..."
        
        created = result.get("created_at", "")
        if isinstance(created, (int, float)):
            created = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            result.get("key", ""),
            result.get("type", ""),
            value_preview,
            created
        )
    
    console.print(table)


@app.command()
def delete(
    key: str = typer.Argument(..., help="Memory key to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a memory entry."""
    if not force:
        if not Confirm.ask(f"Delete memory '{key}'?"):
            raise typer.Exit()
    
    async def delete_memory():
        manager = MemoryManager()
        await manager.initialize()
        return await manager.delete(key)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Deleting memory...", total=None)
        deleted = asyncio.run(delete_memory())
        progress.stop()
    
    if deleted:
        console.print(f"[green]✓[/green] Deleted memory '{key}'")
    else:
        console.print(f"[red]Failed to delete memory '{key}'[/red]")


@app.command()
def clear(
    memory_type: Optional[str] = typer.Option(None, "--type", "-t", help="Clear specific memory type"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Clear memory storage."""
    if not force:
        msg = f"Clear {memory_type} memory?" if memory_type else "Clear ALL memory?"
        if not Confirm.ask(msg):
            raise typer.Exit()
    
    async def clear_memories():
        manager = MemoryManager()
        await manager.initialize()
        
        if memory_type:
            # Clear specific type
            type_map = {
                "working": MemoryType.WORKING,
                "episodic": MemoryType.EPISODIC,
                "semantic": MemoryType.SEMANTIC,
            }
            mem_type = type_map.get(memory_type)
            if mem_type:
                await manager.clear_type(mem_type)
        else:
            # Clear all
            await manager.clear_all()
        
        return True
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Clearing memory...", total=None)
        success = asyncio.run(clear_memories())
        progress.stop()
    
    if success:
        target = f"{memory_type} memory" if memory_type else "all memory"
        console.print(f"[green]✓[/green] Cleared {target}")


@app.command()
def stats():
    """Display memory statistics."""
    async def get_stats():
        manager = MemoryManager()
        await manager.initialize()
        return await manager.get_stats()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Gathering statistics...", total=None)
        stats = asyncio.run(get_stats())
        progress.stop()
    
    table = Table(title="Memory Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Total Items", str(stats.get("total_items", 0)))
    table.add_row("Working Memory", str(stats.get("working_memory_count", 0)))
    table.add_row("Episodic Memory", str(stats.get("episodic_memory_count", 0)))
    table.add_row("Semantic Memory", str(stats.get("semantic_memory_count", 0)))
    table.add_row("Memory Usage", f"{stats.get('memory_usage_mb', 0):.2f} MB")
    table.add_row("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1f}%")
    
    console.print(table)


@app.command()
def consolidate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview consolidation without executing"),
):
    """Consolidate and optimize memory storage."""
    async def run_consolidation():
        manager = MemoryManager()
        await manager.initialize()
        
        if dry_run:
            # Analyze what would be consolidated
            stats = await manager.get_stats()
            return {
                "would_consolidate": stats.get("fragmented_items", 0),
                "estimated_savings": stats.get("estimated_savings_mb", 0),
            }
        else:
            await manager.consolidate()
            return {"status": "completed"}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Consolidating memory...", total=None)
        result = asyncio.run(run_consolidation())
        progress.stop()
    
    if dry_run:
        console.print("[bold]Consolidation Preview:[/bold]")
        console.print(f"  Items to consolidate: {result['would_consolidate']}")
        console.print(f"  Estimated savings: {result['estimated_savings']:.2f} MB")
    else:
        console.print("[green]✓[/green] Memory consolidation completed")


@app.command()
def export(
    output_file: str = typer.Argument(..., help="Output file path"),
    memory_type: Optional[str] = typer.Option(None, "--type", "-t", help="Export specific memory type"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json/csv)"),
):
    """Export memory to file."""
    async def export_memories():
        manager = MemoryManager()
        await manager.initialize()
        
        # Get all memories
        if memory_type:
            pattern = f"{memory_type}_*"
        else:
            pattern = "*"
        
        memories = await manager.search(pattern, limit=None)
        return memories
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting memories...", total=None)
        memories = asyncio.run(export_memories())
        progress.stop()
    
    # Write to file
    if format == "json":
        with open(output_file, "w") as f:
            json.dump(memories, f, indent=2, default=str)
    elif format == "csv":
        import csv
        with open(output_file, "w", newline="") as f:
            if memories:
                writer = csv.DictWriter(f, fieldnames=memories[0].keys())
                writer.writeheader()
                writer.writerows(memories)
    
    console.print(f"[green]✓[/green] Exported {len(memories)} memories to {output_file}")


@app.command()
def import_(
    input_file: str = typer.Argument(..., help="Input file path"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing keys"),
):
    """Import memory from file."""
    async def import_memories():
        manager = MemoryManager()
        await manager.initialize()
        
        # Read file
        with open(input_file) as f:
            if input_file.endswith(".json"):
                memories = json.load(f)
            else:
                console.print("[red]Only JSON format is currently supported[/red]")
                return 0
        
        # Import memories
        imported = 0
        for memory in memories:
            key = memory.get("key")
            value = memory.get("value")
            mem_type = memory.get("type", "working")
            
            if key and value:
                # Check if exists
                existing = await manager.retrieve(key)
                if existing and not overwrite:
                    continue
                
                type_map = {
                    "working": MemoryType.WORKING,
                    "episodic": MemoryType.EPISODIC,
                    "semantic": MemoryType.SEMANTIC,
                }
                
                await manager.store(
                    key, value,
                    type_map.get(mem_type, MemoryType.WORKING)
                )
                imported += 1
        
        return imported
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Importing memories...", total=None)
        imported = asyncio.run(import_memories())
        progress.stop()
    
    console.print(f"[green]✓[/green] Imported {imported} memories from {input_file}")