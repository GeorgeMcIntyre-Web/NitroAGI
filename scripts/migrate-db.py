#!/usr/bin/env python3
"""Database migration script for NitroAGI."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

import asyncpg
from pydantic import BaseModel, Field
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nitroagi.utils.config import get_config

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)


class Migration(BaseModel):
    """Database migration model."""
    version: str
    name: str
    description: str
    sql_up: str
    sql_down: Optional[str] = None
    executed_at: Optional[datetime] = None


# Define migrations
MIGRATIONS: List[Migration] = [
    Migration(
        version="001",
        name="initial_schema",
        description="Create initial database schema",
        sql_up=open(Path(__file__).parent / "init-db.sql").read(),
        sql_down="DROP SCHEMA IF EXISTS core, modules, analytics CASCADE;"
    ),
    Migration(
        version="002",
        name="add_vector_search",
        description="Add vector search capability for semantic memory",
        sql_up="""
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Add embedding column to memory items
        ALTER TABLE core.memory_items 
        ADD COLUMN IF NOT EXISTS embedding vector(1536);
        
        -- Create index for vector similarity search
        CREATE INDEX IF NOT EXISTS idx_memory_embedding 
        ON core.memory_items 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        
        -- Function for semantic search
        CREATE OR REPLACE FUNCTION core.semantic_search(
            query_embedding vector,
            limit_count int DEFAULT 10,
            min_similarity float DEFAULT 0.7
        )
        RETURNS TABLE(
            id UUID,
            key VARCHAR,
            value JSONB,
            similarity float
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                m.id,
                m.key,
                m.value,
                1 - (m.embedding <=> query_embedding) as similarity
            FROM core.memory_items m
            WHERE m.embedding IS NOT NULL
            AND 1 - (m.embedding <=> query_embedding) > min_similarity
            ORDER BY m.embedding <=> query_embedding
            LIMIT limit_count;
        END;
        $$ LANGUAGE plpgsql;
        """,
        sql_down="""
        DROP FUNCTION IF EXISTS core.semantic_search;
        DROP INDEX IF EXISTS idx_memory_embedding;
        ALTER TABLE core.memory_items DROP COLUMN IF EXISTS embedding;
        DROP EXTENSION IF EXISTS vector;
        """
    ),
    Migration(
        version="003",
        name="add_audit_logging",
        description="Add audit logging for compliance",
        sql_up="""
        -- Create audit schema
        CREATE SCHEMA IF NOT EXISTS audit;
        
        -- Audit log table
        CREATE TABLE IF NOT EXISTS audit.logs (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            table_name VARCHAR(255) NOT NULL,
            operation VARCHAR(10) NOT NULL,
            user_id UUID,
            old_data JSONB,
            new_data JSONB,
            changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'
        );
        
        CREATE INDEX idx_audit_table ON audit.logs(table_name);
        CREATE INDEX idx_audit_operation ON audit.logs(operation);
        CREATE INDEX idx_audit_user ON audit.logs(user_id);
        CREATE INDEX idx_audit_changed ON audit.logs(changed_at DESC);
        
        -- Generic audit trigger function
        CREATE OR REPLACE FUNCTION audit.log_changes()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                INSERT INTO audit.logs(table_name, operation, old_data)
                VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD));
                RETURN OLD;
            ELSIF TG_OP = 'UPDATE' THEN
                INSERT INTO audit.logs(table_name, operation, old_data, new_data)
                VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW));
                RETURN NEW;
            ELSIF TG_OP = 'INSERT' THEN
                INSERT INTO audit.logs(table_name, operation, new_data)
                VALUES (TG_TABLE_NAME, TG_OP, row_to_json(NEW));
                RETURN NEW;
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Apply audit triggers to important tables
        CREATE TRIGGER audit_users AFTER INSERT OR UPDATE OR DELETE ON core.users
            FOR EACH ROW EXECUTE FUNCTION audit.log_changes();
        
        CREATE TRIGGER audit_memory_items AFTER INSERT OR UPDATE OR DELETE ON core.memory_items
            FOR EACH ROW EXECUTE FUNCTION audit.log_changes();
        """,
        sql_down="""
        DROP TRIGGER IF EXISTS audit_users ON core.users;
        DROP TRIGGER IF EXISTS audit_memory_items ON core.memory_items;
        DROP FUNCTION IF EXISTS audit.log_changes;
        DROP SCHEMA IF EXISTS audit CASCADE;
        """
    ),
    Migration(
        version="004",
        name="add_6g_network_tables",
        description="Add tables for 6G network optimization tracking",
        sql_up="""
        -- 6G network profiles table
        CREATE TABLE IF NOT EXISTS core.network_profiles (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name VARCHAR(255) UNIQUE NOT NULL,
            max_latency_ms FLOAT,
            min_bandwidth_mbps FLOAT,
            packet_loss_threshold FLOAT,
            jitter_ms FLOAT,
            priority INTEGER DEFAULT 5,
            is_6g_optimized BOOLEAN DEFAULT false,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'
        );
        
        -- Network optimization logs
        CREATE TABLE IF NOT EXISTS core.network_optimizations (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            profile_id UUID REFERENCES core.network_profiles(id),
            user_id UUID REFERENCES core.users(id),
            optimization_type VARCHAR(100),
            metrics_before JSONB,
            metrics_after JSONB,
            improvement_percent FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Insert default 6G profiles
        INSERT INTO core.network_profiles (name, max_latency_ms, min_bandwidth_mbps, is_6g_optimized)
        VALUES 
            ('real_time_ai', 1.0, 10000, true),
            ('holographic_communication', 0.5, 100000, true),
            ('brain_computer_interface', 0.1, 50000, true),
            ('digital_twin_sync', 2.0, 20000, true),
            ('massive_iot', 10.0, 1000, true)
        ON CONFLICT (name) DO NOTHING;
        """,
        sql_down="""
        DROP TABLE IF EXISTS core.network_optimizations;
        DROP TABLE IF EXISTS core.network_profiles;
        """
    )
]


class MigrationManager:
    """Manage database migrations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn: Optional[asyncpg.Connection] = None
    
    async def connect(self):
        """Connect to database."""
        self.conn = await asyncpg.connect(self.database_url)
        
        # Create migrations table if not exists
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version VARCHAR(10) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    async def disconnect(self):
        """Disconnect from database."""
        if self.conn:
            await self.conn.close()
    
    async def get_current_version(self) -> Optional[str]:
        """Get current migration version."""
        row = await self.conn.fetchrow(
            "SELECT version FROM migrations ORDER BY version DESC LIMIT 1"
        )
        return row["version"] if row else None
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations."""
        rows = await self.conn.fetch("SELECT version FROM migrations ORDER BY version")
        return [row["version"] for row in rows]
    
    async def apply_migration(self, migration: Migration):
        """Apply a single migration."""
        async with self.conn.transaction():
            # Execute migration SQL
            await self.conn.execute(migration.sql_up)
            
            # Record migration
            await self.conn.execute(
                """
                INSERT INTO migrations (version, name, description)
                VALUES ($1, $2, $3)
                """,
                migration.version,
                migration.name,
                migration.description
            )
    
    async def rollback_migration(self, migration: Migration):
        """Rollback a single migration."""
        if not migration.sql_down:
            raise ValueError(f"Migration {migration.version} cannot be rolled back")
        
        async with self.conn.transaction():
            # Execute rollback SQL
            await self.conn.execute(migration.sql_down)
            
            # Remove migration record
            await self.conn.execute(
                "DELETE FROM migrations WHERE version = $1",
                migration.version
            )
    
    async def migrate_to(self, target_version: Optional[str] = None):
        """Migrate to target version."""
        applied = await self.get_applied_migrations()
        
        if target_version is None:
            # Migrate to latest
            pending = [m for m in MIGRATIONS if m.version not in applied]
        else:
            # Migrate to specific version
            pending = [
                m for m in MIGRATIONS 
                if m.version not in applied and m.version <= target_version
            ]
        
        for migration in pending:
            console.print(f"Applying migration {migration.version}: {migration.name}")
            await self.apply_migration(migration)
            console.print(f"  [green]✓[/green] Applied successfully")
        
        return len(pending)
    
    async def rollback_to(self, target_version: str):
        """Rollback to target version."""
        applied = await self.get_applied_migrations()
        
        # Find migrations to rollback
        to_rollback = [
            m for m in reversed(MIGRATIONS)
            if m.version in applied and m.version > target_version
        ]
        
        for migration in to_rollback:
            console.print(f"Rolling back migration {migration.version}: {migration.name}")
            await self.rollback_migration(migration)
            console.print(f"  [green]✓[/green] Rolled back successfully")
        
        return len(to_rollback)


@app.command()
def migrate(
    version: Optional[str] = typer.Argument(None, help="Target version (latest if not specified)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing"),
):
    """Apply database migrations."""
    async def run_migrations():
        config = get_config()
        manager = MigrationManager(config.database.postgres_url)
        
        try:
            await manager.connect()
            
            current = await manager.get_current_version()
            console.print(f"Current version: {current or 'None'}")
            
            if dry_run:
                applied = await manager.get_applied_migrations()
                pending = [m for m in MIGRATIONS if m.version not in applied]
                
                if pending:
                    console.print("\n[bold]Pending migrations:[/bold]")
                    for m in pending:
                        console.print(f"  {m.version}: {m.name} - {m.description}")
                else:
                    console.print("[green]No pending migrations[/green]")
            else:
                count = await manager.migrate_to(version)
                if count > 0:
                    console.print(f"\n[green]✓[/green] Applied {count} migration(s)")
                else:
                    console.print("[yellow]No migrations to apply[/yellow]")
        
        finally:
            await manager.disconnect()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running migrations...", total=None)
        asyncio.run(run_migrations())
        progress.stop()


@app.command()
def rollback(
    version: str = typer.Argument(..., help="Target version to rollback to"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Rollback database migrations."""
    if not force:
        if not typer.confirm(f"Rollback to version {version}?"):
            raise typer.Exit()
    
    async def run_rollback():
        config = get_config()
        manager = MigrationManager(config.database.postgres_url)
        
        try:
            await manager.connect()
            count = await manager.rollback_to(version)
            
            if count > 0:
                console.print(f"\n[green]✓[/green] Rolled back {count} migration(s)")
            else:
                console.print("[yellow]No migrations to rollback[/yellow]")
        
        finally:
            await manager.disconnect()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Rolling back migrations...", total=None)
        asyncio.run(run_rollback())
        progress.stop()


@app.command()
def status():
    """Show migration status."""
    async def get_status():
        config = get_config()
        manager = MigrationManager(config.database.postgres_url)
        
        try:
            await manager.connect()
            
            current = await manager.get_current_version()
            applied = await manager.get_applied_migrations()
            
            return {
                "current": current,
                "applied": applied,
                "pending": [m for m in MIGRATIONS if m.version not in applied]
            }
        
        finally:
            await manager.disconnect()
    
    status = asyncio.run(get_status())
    
    # Display status table
    table = Table(title="Migration Status", show_header=True)
    table.add_column("Version", style="cyan")
    table.add_column("Name", style="yellow")
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")
    
    for migration in MIGRATIONS:
        status_text = "✓ Applied" if migration.version in status["applied"] else "⏳ Pending"
        status_style = "green" if migration.version in status["applied"] else "yellow"
        
        table.add_row(
            migration.version,
            migration.name,
            migration.description,
            status_text,
            style=status_style if migration.version in status["applied"] else None
        )
    
    console.print(table)
    console.print(f"\nCurrent version: [bold cyan]{status['current'] or 'None'}[/bold cyan]")


if __name__ == "__main__":
    app()