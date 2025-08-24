# 🛡️ Compliance Copilot

> AI-Powered Compliance Automation for Development Workflows

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Overview

Compliance Copilot is an intelligent automation system that analyzes GitHub Pull Requests for compliance risks and automatically generates audit-ready evidence bundles. Built for modern development teams who need to maintain regulatory compliance (SOC 2, ISO 27001, GDPR) without slowing down their development velocity.

## 🎥 Demo Video

[![Compliance Copilot Demo](https://img.youtube.com/vi/dRIXyc8TmPo/0.jpg)](https://www.youtube.com/watch?v=dRIXyc8TmPo)

**[Watch the Demo on YouTube](https://www.youtube.com/watch?v=dRIXyc8TmPo)** 🎬

See Compliance Copilot in action! This demo showcases:
- Real-time PR analysis and compliance risk detection
- AI-powered evidence bundle generation
- Dashboard monitoring and compliance metrics
- Integration with GitHub webhooks and development workflows

### ✨ Key Features

- **🔍 Automated PR Analysis** - Real-time scanning of code changes for compliance risks
- **📊 Multi-Framework Support** - SOC 2, ISO 27001, GDPR compliance frameworks
- **🤖 AI-Powered Risk Assessment** - Intelligent pattern recognition and risk scoring
- **📝 Evidence Bundle Generation** - Audit-ready documentation with technical details
- **⚡ GitHub Webhook Integration** - Seamless CI/CD pipeline integration
- **📈 Real-time Dashboard** - Live monitoring and compliance metrics
- **🔗 Multi-Platform Integration** - Slack, Notion

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub PR     │───▶│ Compliance Agent │───▶│ Evidence Bundle │
│   Webhook       │    │  (Portia SDK)    │    │   Generator     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  FastAPI Web    │    │   Risk Pattern   │    │   Audit Trail   │
│   Dashboard     │    │    Analysis      │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

- **Compliance Agent** - Orchestrates analysis using Portia SDK for structured plan execution
- **PR Analyzer** - Scans code changes for compliance-relevant patterns
- **Control Mapper** - Maps code changes to specific compliance controls
- **Evidence Generator** - Creates comprehensive audit documentation
- **Risk Assessor** - Calculates compliance risk scores and levels

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GitHub Personal Access Token
- Github Webhook Secret
- Portia API Key (for AI agent orchestration)
- Gemini API Key (for AI analysis)

### Installation

#### Option 1: Local Development (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/compliance-copilot.git
   cd compliance-copilot
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

#### Option 2: Docker (Production/Isolated Environment)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/compliance-copilot.git
   cd compliance-copilot
   ```

2. **Run with Docker Compose**
   ```bash
   # Production mode
   docker-compose up --build

   # Development mode with live reloading
   docker-compose up --build  # Uses override automatically
   ```

   See [DOCKER.md](DOCKER.md) for detailed Docker instructions.

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Open dashboard**
   ```
   http://localhost:8000
   ```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Required
PORTIA_API_KEY=your_portia_api_key
GEMINI_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_token
NOTION_TOKEN=your_notion_token

# Webhook Security
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# Database (defaults to SQLite)
DATABASE_URL=sqlite:///./compliance.db
```

## 🔧 Usage

### GitHub Webhook Setup

1. Go to your repository's Settings → Webhooks
2. Add webhook URL: `https://your-domain.com/webhooks/github`
3. Select "Pull requests" events
4. Set content type to `application/json`
5. Add your webhook secret (optional but recommended)

### Manual Analysis

You can also trigger analysis manually through the API:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_name": "org/repository",
    "pr_number": 123,
    "frameworks": ["SOC2", "ISO27001"]
  }'
```

### Dashboard Features

- **Live Analysis Feed** - Real-time PR compliance analysis results
- **Framework Coverage** - Track compliance across different frameworks

## 📊 Compliance Frameworks

### SOC 2 Type II
- **CC6.1** - Logical Access Authentication
- **CC6.2** - User Authentication Controls
- **CC7.2** - System Operations Monitoring
- *More controls configurable in `config/soc2_controls.yaml`*

### ISO 27001:2022
- **A.9** - Access Control
- **A.12** - Operations Security
- **A.14** - System Acquisition
- *Configure in `config/iso27001_controls.yaml`*

### GDPR
- **Art. 25** - Data Protection by Design
- **Art. 32** - Security of Processing
- **Art. 35** - Data Protection Impact Assessment
- *Configure in `config/gdpr_controls.yaml`*

## 🔍 Risk Analysis Patterns

The system automatically detects compliance-relevant code patterns:

| Pattern | Risk Level | Examples |
|---------|------------|----------|
| Authentication | HIGH | login, auth, oauth, jwt |
| Data Access | HIGH | database, sql, query |
| Encryption | HIGH | crypto, ssl, tls, hash |
| Access Control | HIGH | permission, role, rbac |
| API Security | MEDIUM | endpoint, cors, middleware |
| Logging | MEDIUM | log, audit, monitor |
| Data Processing | HIGH | pii, gdpr, personal |

## 🛠️ Development

### Project Structure

```
compliance-copilot/
├── src/
│   ├── agent/              # Portia-powered compliance agent
│   ├── analysis/           # PR analysis logic
│   ├── evidence/           # Evidence bundle generation
│   ├── mapping/            # Control mapping algorithms
│   ├── integrations/       # Third-party integrations
│   └── models.py           # Data models
├── config/                 # Compliance framework definitions
├── templates/              # Dashboard HTML templates
├── requirements.txt        # Python dependencies
└── run.py                 # Application entry point
```


## 🎯 Use Cases

- **DevSecOps Teams** - Integrate compliance into CI/CD pipelines
- **Audit Preparation** - Generate evidence bundles for external auditors
- **Risk Management** - Continuous compliance risk monitoring
- **Regulatory Reporting** - Automated compliance documentation
- **Developer Education** - Real-time feedback on compliance implications

## 🚨 Security Considerations

- Store API keys securely using environment variables
- Use webhook secrets to verify GitHub authenticity
- Enable HTTPS in production environments
- Regularly rotate API tokens and credentials
- Review and audit compliance framework configurations

## 📚 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- Built with [Portia SDK](https://portialabs.ai) for AI agent orchestration
- Powered by [FastAPI](https://fastapi.tiangolo.com/) for high-performance APIs
- Uses [SQLAlchemy](https://www.sqlalchemy.org/) for database operations

---

**Built for AgentHack 2025** 🎪

*Making compliance automation intelligent, automated, and developer-friendly.*
