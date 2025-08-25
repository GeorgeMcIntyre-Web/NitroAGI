"""Main entry point for the NitroAGI API server."""

import asyncio
import signal
import sys
from pathlib import Path

import uvicorn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nitroagi.api.app import create_app
from nitroagi.utils.config import get_config
from nitroagi.utils.logging import setup_logging, get_logger


# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = create_app()

# Get configuration
config = get_config()


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal, gracefully stopping...")
    sys.exit(0)


def run():
    """Run the API server."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting NitroAGI API server on {config.api.host}:{config.api.port}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Debug mode: {config.debug}")
    
    # Run server
    uvicorn.run(
        "nitroagi.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload and config.is_development(),
        workers=config.api.workers if not config.api.reload else 1,
        log_level=config.log_level.lower(),
        access_log=config.debug,
        use_colors=True,
    )


if __name__ == "__main__":
    run()