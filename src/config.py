import os
from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings):
    """Application configuration"""

    # App settings
    app_name: str = "Auditpilot"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Database 
    db_url: str = Field(
        default="sqlite:///./auditpilot.db",
        env="DATABASE_URL",
        description="Database connection URL"
    )

    # Github integration
    github_token: Optional[str] = None
    github_webhook_secret: Optional[str] = None

    # Portia configuration
    portia_api_key: Optional[str] = None
    portia_base_url: str = "https://api.portialabs.ai"

    # Slack integration
    slack_webhook_url: Optional[str] = None
    slack_bot_token: Optional[str] = None

    # Notion integration (optional)
    notion_token: Optional[str] = None
    notion_database_id: Optional[str] = None

    # Jira integration (optional)
    jira_url: Optional[str] = None
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None

    # Security
    secret_key: str = Field(default_factory=lambda: os.urandom(32).hex())

    class Config:
        env_file = ".env"
        case_sensitive = False

    
settings = Settings()

# Compliance frameworks configuration
COMPLIANCE_FRAMEWORKS = {
    "SOC2": {
        "name": "SOC 2 Type II",
        "version": "2017",
        "controls_file": "config/soc2_controls.yaml",
    },
    "ISO27001": {
        "name": "ISO 27001:2022",
        "version": "2022", 
        "controls_file": "config/iso27001_controls.yaml"
    },
    "GDPR": {
        "name": "General Data Protection Regulation",
        "version": "2018",
        "controls_file": "config/gdpr_controls.yaml"
    }
}

# Risk patterns for code analysis
RISK_PATTERNS = {
    "authentication": {
        "keywords": ["auth", "login", "password", "credentials", "token", "session", "oauth", "jwt"],
        "file_patterns": ["auth", "login", "session", "security"],
        "risk_level": "HIGH"
    },
    "data_access": {
        "keywords": ["database", "query", "sql", "select", "insert", "update", "delete", "connection"],
        "file_patterns": ["db", "database", "model", "repository", "dao"],
        "risk_level": "HIGH"
    },
    "encryption": {
        "keywords": ["encrypt", "decrypt", "crypto", "ssl", "tls", "certificate", "hash", "cipher"],
        "file_patterns": ["crypto", "encryption", "ssl", "cert"],
        "risk_level": "HIGH"
    },
    "logging": {
        "keywords": ["log", "audit", "trace", "monitor", "event", "alert"],
        "file_patterns": ["log", "audit", "monitor"],
        "risk_level": "MEDIUM"
    },
    "access_control": {
        "keywords": ["permission", "role", "rbac", "authorize", "access", "privilege", "acl"],
        "file_patterns": ["permission", "role", "auth", "access"],
        "risk_level": "HIGH"
    },
    "api_security": {
        "keywords": ["api", "endpoint", "route", "cors", "rate_limit", "middleware"],
        "file_patterns": ["api", "route", "endpoint", "middleware"],
        "risk_level": "MEDIUM"
    },
    "data_processing": {
        "keywords": ["personal", "pii", "gdpr", "privacy", "consent", "data_subject"],
        "file_patterns": ["privacy", "gdpr", "data"],
        "risk_level": "HIGH"
    }
}