"""
River Trading System - Command Line Interface
============================================

Professional CLI with Google-level development experience.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import asyncio
import subprocess

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config, config_manager
from src.core.logging_config import setup_logging

# Initialize CLI
app = typer.Typer(
    name="river-trading",
    help="🌊 River Trading System - Enterprise AI Trading Platform",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()
logger = structlog.get_logger(__name__)


@app.command("run")
def run_system(
    mode: str = typer.Option("production", "--mode", "-m", help="Launch mode"),
    port: int = typer.Option(8501, "--port", "-p", help="UI port"),
    api_port: int = typer.Option(8000, "--api-port", help="API port"),
    host: str = typer.Option("localhost", "--host", help="Host address"),
    reload: bool = typer.Option(False, "--reload", help="Enable hot reload (dev mode)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """
    🚀 Run the River Trading System
    """
    try:
        # Load configuration
        config = get_config()
        
        if debug:
            config.debug = True
            setup_logging(level="DEBUG")
        
        # Update ports if specified
        config.ui.port = port
        config.api.port = api_port
        config.api.host = host
        config.ui.host = host
        
        console.print(Panel.fit(
            f"[bold blue]🌊 River Trading System[/bold blue]\n"
            f"Mode: [green]{mode}[/green]\n"
            f"UI: [cyan]http://{host}:{port}[/cyan]\n"
            f"API: [cyan]http://{host}:{api_port}[/cyan]",
            title="Starting System",
            border_style="blue"
        ))
        
        # Choose launcher based on mode
        if mode == "development":
            from src.launchers.beautiful_launcher import BeautifulLauncher
            launcher = BeautifulLauncher()
        else:
            from src.launchers.production_launcher import RiverTradingLauncher
            launcher = RiverTradingLauncher()
        
        launcher.ui_port = port
        launcher.api_port = api_port
        launcher.launch()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"[red]❌ Error: {exc}[/red]")
        sys.exit(1)


@app.command("dev")
def dev_mode(
    port: int = typer.Option(8501, "--port", "-p", help="UI port"),
    api_port: int = typer.Option(8000, "--api-port", help="API port"),
) -> None:
    """
    🔧 Start development environment with hot reload
    """
    run_system(
        mode="development",
        port=port,
        api_port=api_port,
        reload=True,
        debug=True
    )


@app.command("test")
def run_tests(
    pattern: Optional[str] = typer.Option(None, "--pattern", "-k", help="Test pattern"),
    coverage: bool = typer.Option(True, "--coverage/--no-coverage", help="Generate coverage report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    🧪 Run the test suite
    """
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if pattern:
        cmd.extend(["-k", pattern])
        
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    cmd.append("tests/")
    
    console.print(f"[blue]Running tests:[/blue] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


@app.command("lint")
def run_linting(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix issues where possible"),
) -> None:
    """
    🔍 Run code quality checks
    """
    checks = [
        ("Black (formatting)", ["black", "--check", "src/", "tests/"]),
        ("isort (imports)", ["isort", "--check-only", "src/", "tests/"]),
        ("flake8 (linting)", ["flake8", "src/", "tests/"]),
        ("mypy (typing)", ["mypy", "src/"]),
        ("bandit (security)", ["bandit", "-r", "src/"]),
    ]
    
    if fix:
        checks[0] = ("Black (formatting)", ["black", "src/", "tests/"])
        checks[1] = ("isort (imports)", ["isort", "src/", "tests/"])
    
    console.print("[blue]Running code quality checks...[/blue]")
    
    failed = []
    for name, cmd in checks:
        console.print(f"  🔍 {name}")
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            failed.append(name)
            console.print(f"    [red]❌ Failed[/red]")
        else:
            console.print(f"    [green]✅ Passed[/green]")
    
    if failed:
        console.print(f"\n[red]❌ {len(failed)} checks failed:[/red]")
        for check in failed:
            console.print(f"  - {check}")
        sys.exit(1)
    else:
        console.print("\n[green]✅ All checks passed![/green]")


@app.command("config")
def manage_config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
    create: bool = typer.Option(False, "--create", help="Create new configuration"),
) -> None:
    """
    ⚙️ Manage system configuration
    """
    if show:
        config = get_config()
        config_dict = config.to_dict()
        
        # Create configuration table
        table = Table(title="River Trading Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        def add_config_rows(data, prefix=""):
            for key, value in data.items():
                if isinstance(value, dict):
                    add_config_rows(value, f"{prefix}{key}.")
                else:
                    # Hide sensitive values
                    if any(sensitive in key.lower() for sensitive in ["password", "key", "secret"]):
                        value = "***HIDDEN***"
                    table.add_row(f"{prefix}{key}", str(value))
        
        add_config_rows(config_dict)
        console.print(table)
    
    elif validate:
        try:
            config = get_config()
            config.validate()
            console.print("[green]✅ Configuration is valid[/green]")
        except Exception as exc:
            console.print(f"[red]❌ Configuration validation failed: {exc}[/red]")
            sys.exit(1)
    
    elif create:
        console.print("[blue]Creating new configuration...[/blue]")
        
        # Interactive configuration creation
        environment = Prompt.ask(
            "Environment",
            choices=["development", "testing", "staging", "production"],
            default="development"
        )
        
        api_port = Prompt.ask("API Port", default="8000")
        ui_port = Prompt.ask("UI Port", default="8501")
        
        config_data = {
            "environment": environment,
            "api": {"port": int(api_port)},
            "ui": {"port": int(ui_port)}
        }
        
        # Save configuration
        config_path = Path("config/config.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        console.print(f"[green]✅ Configuration created: {config_path}[/green]")


@app.command("health")
def health_check() -> None:
    """
    🏥 Check system health
    """
    import requests
    
    config = get_config()
    api_url = f"http://{config.api.host}:{config.api.port}"
    ui_url = f"http://{config.ui.host}:{config.ui.port}"
    
    checks = [
        ("API Server", f"{api_url}/health"),
        ("UI Server", f"{ui_url}/_stcore/health"),
    ]
    
    table = Table(title="System Health Check")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Response Time", style="yellow")
    
    for service, url in checks:
        try:
            import time
            start = time.time()
            response = requests.get(url, timeout=5)
            duration = round((time.time() - start) * 1000, 2)
            
            if response.status_code == 200:
                status = "[green]✅ Healthy[/green]"
            else:
                status = f"[yellow]⚠️ Degraded ({response.status_code})[/yellow]"
                
            table.add_row(service, status, f"{duration}ms")
            
        except requests.exceptions.ConnectionError:
            table.add_row(service, "[red]❌ Down[/red]", "N/A")
        except requests.exceptions.Timeout:
            table.add_row(service, "[yellow]⚠️ Timeout[/yellow]", ">5000ms")
        except Exception as exc:
            table.add_row(service, f"[red]❌ Error[/red]", str(exc))
    
    console.print(table)


@app.command("docker")
def docker_commands(
    action: str = typer.Argument(..., help="Docker action: up, down, build, logs"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs"),
) -> None:
    """
    🐳 Docker operations
    """
    if action == "up":
        console.print("[blue]🐳 Starting Docker services...[/blue]")
        result = subprocess.run(["docker-compose", "up", "-d"])
        if result.returncode == 0:
            console.print("[green]✅ Services started successfully[/green]")
        else:
            console.print("[red]❌ Failed to start services[/red]")
            sys.exit(result.returncode)
    
    elif action == "down":
        console.print("[blue]🐳 Stopping Docker services...[/blue]")
        result = subprocess.run(["docker-compose", "down"])
        if result.returncode == 0:
            console.print("[green]✅ Services stopped successfully[/green]")
        else:
            console.print("[red]❌ Failed to stop services[/red]")
            sys.exit(result.returncode)
    
    elif action == "build":
        console.print("[blue]🐳 Building Docker images...[/blue]")
        result = subprocess.run(["docker-compose", "build"])
        if result.returncode == 0:
            console.print("[green]✅ Images built successfully[/green]")
        else:
            console.print("[red]❌ Failed to build images[/red]")
            sys.exit(result.returncode)
    
    elif action == "logs":
        cmd = ["docker-compose", "logs"]
        if follow:
            cmd.append("-f")
        subprocess.run(cmd)
    
    else:
        console.print(f"[red]❌ Unknown action: {action}[/red]")
        console.print("Available actions: up, down, build, logs")
        sys.exit(1)


@app.command("install")
def install_dependencies(
    dev: bool = typer.Option(False, "--dev", help="Install development dependencies"),
) -> None:
    """
    📦 Install system dependencies
    """
    console.print("[blue]📦 Installing dependencies...[/blue]")
    
    # Install production dependencies
    result = subprocess.run(["pip", "install", "-r", "requirements.txt"])
    if result.returncode != 0:
        console.print("[red]❌ Failed to install production dependencies[/red]")
        sys.exit(result.returncode)
    
    # Install development dependencies if requested
    if dev:
        result = subprocess.run(["pip", "install", "-r", "requirements-dev.txt"])
        if result.returncode != 0:
            console.print("[red]❌ Failed to install development dependencies[/red]")
            sys.exit(result.returncode)
    
    console.print("[green]✅ Dependencies installed successfully[/green]")
    
    # Setup pre-commit hooks if in dev mode
    if dev:
        console.print("[blue]🔧 Setting up pre-commit hooks...[/blue]")
        subprocess.run(["pre-commit", "install"])
        console.print("[green]✅ Pre-commit hooks installed[/green]")


def main() -> None:
    """Main CLI entry point"""
    app()


def dev_main() -> None:
    """Development CLI entry point"""
    dev_mode()


if __name__ == "__main__":
    main()