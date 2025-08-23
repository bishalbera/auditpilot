#!/usr/bin/env python3
"""
Compliance Copilot - Application Startup Script
Enhanced for AgentHack 2025 Demo
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False

    # Check if .env exists
    if not os.path.exists(".env"):
        print("❌ .env file not found. Run setup.py first.")
        return False

    # Check critical imports
    try:
        # Check if packages are available without importing them
        import importlib.util

        packages = ["fastapi", "sqlalchemy", "pydantic"]
        for package in packages:
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"{package} not found")

        print("✅ Core dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def setup_environment():
    """Setup environment and create necessary directories"""

    # Create directories if they don't exist
    directories = ["templates", "static", "config", "logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    print("✅ Environment configured")


def initialize_database():
    """Initialize database with tables"""
    try:
        from src.models import create_database
        from src.config import settings

        create_database(settings.database_url)
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
        return True  # Continue anyway for demo


def create_demo_data():
    """Create sample data for demo purposes"""
    print("🎭 Creating demo data...")

    # This would create sample PR analyses for a better demo
    # For hackathon, we'll keep it simple and let the demo buttons handle it
    print("✅ Demo environment ready")


def start_application():
    """Start the FastAPI application"""
    try:
        from src.config import settings

        print(f"🚀 Starting Compliance Copilot on {settings.host}:{settings.port}")
        print(f"📊 Dashboard: http://localhost:{settings.port}")
        print(f"📚 API Docs: http://localhost:{settings.port}/docs")
        print("\n" + "=" * 50)
        print("🎪 AGENTHACK 2025 DEMO READY!")
        print("=" * 50)
        print("Demo Instructions:")
        print("1. Open dashboard in browser")
        print("2. Click demo buttons to trigger scenarios")
        print("3. Watch real-time compliance analysis")
        print("4. Show evidence bundles to judges")
        print("=" * 50)

        # Import and run the FastAPI app
        # from src.main import app
        from src.main import app

        print("Using mock Agent")

        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info" if not settings.debug else "debug",
        )

    except KeyboardInterrupt:
        print("\n👋 Compliance Copilot stopped")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)


def main():
    """Main startup function"""
    print("🛡️  Compliance Copilot - AgentHack 2025")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        print("Please run setup.py first or install missing dependencies")
        sys.exit(1)

    # Setup environment
    setup_environment()

    # Initialize database
    initialize_database()

    # Create demo data
    create_demo_data()

    # Start the application
    start_application()


if __name__ == "__main__":
    main()
