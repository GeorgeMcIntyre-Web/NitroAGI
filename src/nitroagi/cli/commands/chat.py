"""Interactive chat CLI commands."""

import asyncio
import sys
from typing import Optional, List
import json
from datetime import datetime

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.core.base import ModuleRequest, ModuleContext, ModuleCapability
from nitroagi.core.memory import MemoryManager, MemoryType
from nitroagi.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


class ChatSession:
    """Interactive chat session manager."""
    
    def __init__(self, user_id: str = "cli-user"):
        self.user_id = user_id
        self.session_id = f"session-{datetime.now().timestamp()}"
        self.conversation_id = f"conv-{datetime.now().timestamp()}"
        self.history: List[dict] = []
        self.language_module = None
        self.memory_manager = None
    
    async def initialize(self):
        """Initialize chat session."""
        self.language_module = LanguageModule()
        await self.language_module.initialize()
        
        self.memory_manager = MemoryManager()
        await self.memory_manager.initialize()
    
    async def send_message(self, message: str, temperature: float = 0.7, max_tokens: int = 1000):
        """Send message and get response."""
        # Add to history
        self.history.append({"role": "user", "content": message})
        
        # Create request
        context = ModuleContext(
            request_id=f"chat-{datetime.now().timestamp()}",
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_id=self.conversation_id
        )
        
        request = ModuleRequest(
            context=context,
            data={
                "messages": self.history,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            required_capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        # Process request
        response = await self.language_module.process(request)
        
        if response.status == "success":
            self.history.append({"role": "assistant", "content": response.data})
            
            # Store in memory
            await self.memory_manager.store(
                f"chat_{self.conversation_id}_turn_{len(self.history)}",
                {
                    "user": message,
                    "assistant": response.data,
                    "timestamp": datetime.now().isoformat()
                },
                MemoryType.EPISODIC
            )
            
            return response.data
        else:
            return f"Error: {response.error}"
    
    async def shutdown(self):
        """Clean up session."""
        if self.language_module:
            await self.language_module.shutdown()


@app.command()
def interactive(
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Response temperature"),
    max_tokens: int = typer.Option(1000, "--max-tokens", "-m", help="Maximum response tokens"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Start interactive chat session."""
    console.print(Panel.fit(
        "[bold cyan]NitroAGI Interactive Chat[/bold cyan]\n"
        "Type 'exit' or 'quit' to end the session\n"
        "Type '/help' for commands",
        border_style="cyan"
    ))
    
    async def run_chat():
        session = ChatSession()
        await session.initialize()
        
        if system_prompt:
            session.history.append({"role": "system", "content": system_prompt})
        
        try:
            while True:
                # Get user input
                try:
                    user_input = Prompt.ask("\n[bold green]You[/bold green]")
                except (EOFError, KeyboardInterrupt):
                    break
                
                # Check for commands
                if user_input.lower() in ["exit", "quit"]:
                    break
                elif user_input == "/help":
                    console.print("\n[bold]Commands:[/bold]")
                    console.print("  /help - Show this help")
                    console.print("  /history - Show conversation history")
                    console.print("  /clear - Clear conversation history")
                    console.print("  /save <filename> - Save conversation")
                    console.print("  /temperature <value> - Set temperature")
                    console.print("  /tokens <value> - Set max tokens")
                    console.print("  exit/quit - End session")
                    continue
                elif user_input == "/history":
                    console.print("\n[bold]Conversation History:[/bold]")
                    for msg in session.history:
                        role = msg["role"].capitalize()
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        console.print(f"[cyan]{role}:[/cyan] {content}")
                    continue
                elif user_input == "/clear":
                    session.history = []
                    if system_prompt:
                        session.history.append({"role": "system", "content": system_prompt})
                    console.print("[yellow]Conversation history cleared[/yellow]")
                    continue
                elif user_input.startswith("/save "):
                    filename = user_input[6:].strip()
                    with open(filename, "w") as f:
                        json.dump(session.history, f, indent=2)
                    console.print(f"[green]Conversation saved to {filename}[/green]")
                    continue
                elif user_input.startswith("/temperature "):
                    try:
                        temperature = float(user_input[13:].strip())
                        console.print(f"[yellow]Temperature set to {temperature}[/yellow]")
                    except ValueError:
                        console.print("[red]Invalid temperature value[/red]")
                    continue
                elif user_input.startswith("/tokens "):
                    try:
                        max_tokens = int(user_input[8:].strip())
                        console.print(f"[yellow]Max tokens set to {max_tokens}[/yellow]")
                    except ValueError:
                        console.print("[red]Invalid token value[/red]")
                    continue
                
                # Send message with spinner
                with console.status("[bold cyan]Thinking...", spinner="dots"):
                    response = await session.send_message(user_input, temperature, max_tokens)
                
                # Display response
                console.print(f"\n[bold blue]NitroAGI:[/bold blue]")
                if "```" in response:
                    # Render as markdown if it contains code blocks
                    console.print(Markdown(response))
                else:
                    console.print(response)
        
        finally:
            await session.shutdown()
    
    # Run async chat
    asyncio.run(run_chat())
    console.print("\n[cyan]Chat session ended. Goodbye![/cyan]")


@app.command()
def single(
    message: str = typer.Argument(..., help="Message to send"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Response temperature"),
    max_tokens: int = typer.Option(1000, "--max-tokens", "-m", help="Maximum response tokens"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Send a single message and get response."""
    async def get_response():
        session = ChatSession()
        await session.initialize()
        
        response = await session.send_message(message, temperature, max_tokens)
        
        await session.shutdown()
        return response
    
    with console.status("[bold cyan]Processing...", spinner="dots"):
        response = asyncio.run(get_response())
    
    if json_output:
        output = {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        console.print(json.dumps(output, indent=2))
    else:
        console.print(f"\n[bold blue]Response:[/bold blue]")
        if "```" in response:
            console.print(Markdown(response))
        else:
            console.print(response)


@app.command()
def stream(
    message: str = typer.Argument(..., help="Message to send"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Response temperature"),
    max_tokens: int = typer.Option(1000, "--max-tokens", "-m", help="Maximum response tokens"),
):
    """Stream response in real-time."""
    async def stream_response():
        language_module = LanguageModule()
        await language_module.initialize()
        
        # Use streaming if provider supports it
        if hasattr(language_module.llm_provider, 'generate_stream'):
            console.print(f"\n[bold blue]NitroAGI:[/bold blue] ", end="")
            
            async for chunk in language_module.llm_provider.generate_stream(
                message,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                console.print(chunk, end="")
                sys.stdout.flush()
            
            console.print()  # New line at end
        else:
            # Fallback to non-streaming
            response = await language_module.llm_provider.generate(
                message,
                max_tokens=max_tokens,
                temperature=temperature
            )
            console.print(f"\n[bold blue]NitroAGI:[/bold blue] {response}")
        
        await language_module.shutdown()
    
    asyncio.run(stream_response())


@app.command()
def history(
    conversation_id: Optional[str] = typer.Argument(None, help="Conversation ID"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of messages to show"),
    format: str = typer.Option("text", "--format", "-f", help="Output format (text/json)"),
):
    """View chat history."""
    async def get_history():
        memory_manager = MemoryManager()
        await memory_manager.initialize()
        
        if conversation_id:
            pattern = f"chat_{conversation_id}_*"
        else:
            pattern = "chat_*"
        
        memories = await memory_manager.search(pattern, limit=limit)
        return memories
    
    memories = asyncio.run(get_history())
    
    if not memories:
        console.print("[yellow]No chat history found[/yellow]")
        return
    
    if format == "json":
        console.print(json.dumps(memories, indent=2, default=str))
    else:
        console.print(f"\n[bold]Chat History ({len(memories)} messages):[/bold]\n")
        
        for memory in memories:
            value = memory.get("value", {})
            timestamp = value.get("timestamp", "")
            
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            console.print(f"[dim]{timestamp}[/dim]")
            
            if "user" in value:
                console.print(f"[green]User:[/green] {value['user']}")
            
            if "assistant" in value:
                console.print(f"[blue]Assistant:[/blue] {value['assistant']}")
            
            console.print()  # Blank line between messages