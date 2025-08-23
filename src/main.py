import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .config import settings
from .models import get_session, PRAnalysisDB, EvidenceBundleDB
from .agent.compliance_agent import ComplianceAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Compliance Copilot",
    description="AI-powered compliance automation for development workflows",
    version="1.0.0",
)

SessionLocal = get_session(settings.database_url)

compliance_agent = ComplianceAgent()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_github_signature(request: Request, payload: bytes) -> bool:
    """Verify GitHub webhook signature"""
    if not settings.github_webhook_secret:
        return True  # Skip verification if no secret configured

    signature_header = request.headers.get("X-Hub-Signature-256")
    if not signature_header:
        return False

    expected_signature = (
        "sha256="
        + hmac.new(
            settings.github_webhook_secret.encode(), payload, hashlib.sha256
        ).hexdigest()
    )

    return hmac.compare_digest(signature_header, expected_signature)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "title": "Compliance Copilot Dashboard"}
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agent_status": compliance_agent.get_agent_stats(),
    }


async def execute_compliance_analysis(
    repo_name: str,
    pr_number: int,
    policy_refs: Optional[list] = None,
    demo: bool = False,
):
    """
    Execute the Portia-based compliance analysis.

    Args:
        repo_name: Repository name (e.g., "org/repo")
        pr_number: Pull request number
        policy_refs: List of policy frameworks (SOC2, ISO27001, etc.)
        demo: Whether this is demo/test mode
    """
    try:
        logger.info(f"🚀 Starting compliance analysis for {repo_name}#{pr_number}")
        logger.info("⚙️ Using structured plan execution")

        plan_result = await compliance_agent.run_plan(
            repo_name=repo_name,
            pr_number=pr_number,
            policy_refs=policy_refs or [],
            demo=demo,
        )

        logger.info(f"✅ Compliance analysis completed for {repo_name}#{pr_number}")

        # Extract and log results
        await log_analysis_results(plan_result, repo_name, pr_number)

        return plan_result

    except Exception as e:
        logger.error(
            f"❌ Compliance analysis failed for {repo_name}#{pr_number}: {str(e)}",
            exc_info=True,
        )
        raise


async def log_analysis_results(plan_result, repo_name: str, pr_number: int):
    """Extract and log analysis results from plan execution"""
    try:
        # Try to extract results from different result formats
        if hasattr(plan_result, "final_output"):
            final_output = plan_result.final_output
            if isinstance(final_output, dict):
                analysis_info = final_output.get("analysis", {})
                risk_level = analysis_info.get("risk_level", "unknown")
                control_count = analysis_info.get("control_count", 0)
                logger.info(
                    f"📊 Analysis summary for {repo_name}#{pr_number}: Risk={risk_level}, Controls={control_count}"
                )

        elif hasattr(plan_result, "output"):
            output = plan_result.output
            logger.info(
                f"📊 Analysis output for {repo_name}#{pr_number}: {type(output).__name__}"
            )

        if hasattr(plan_result, "status"):
            logger.info(f"📈 Plan execution status: {plan_result.status}")

    except Exception as e:
        logger.warning(f"⚠️ Could not extract detailed results: {e}")


@app.post("/webhooks/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle GitHub webhook events by starting a Portia plan run.
    This is triggered automatically when PRs are opened, updated, or synchronized.
    """
    try:
        payload = await request.body()
        if not verify_github_signature(request, payload):
            raise HTTPException(status_code=401, detail="Invalid signature")

        webhook_data = json.loads(payload.decode())

        # Only process relevant PR actions
        action = webhook_data.get("action")
        if action not in ["opened", "synchronize", "edited", "reopened"]:
            logger.info(f"🔄 Ignoring PR action: {action}")
            return {"status": "ignored", "reason": f"Not processing action: {action}"}

        # Extract PR information
        repo_data = webhook_data.get("repository", {})
        pr_data = webhook_data.get("pull_request", {})

        repo_name = repo_data.get("full_name")
        pr_number = pr_data.get("number")
        pr_title = pr_data.get("title", "")
        pr_author = pr_data.get("user", {}).get("login", "")

        if not (repo_name and pr_number):
            raise HTTPException(
                status_code=400, detail="Missing repo or PR number in webhook payload"
            )

        logger.info(
            f"🔍 Processing PR webhook: {repo_name}#{pr_number} - {pr_title} by {pr_author}"
        )

        # Determine policy references based on repo or PR labels
        policy_refs = []

        # Check PR labels for compliance frameworks
        pr_labels = [
            label.get("name", "").lower() for label in pr_data.get("labels", [])
        ]
        if "soc2" in pr_labels or "soc-2" in pr_labels:
            policy_refs.append("SOC2")
        if "iso27001" in pr_labels or "iso-27001" in pr_labels:
            policy_refs.append("ISO27001")
        if "gdpr" in pr_labels:
            policy_refs.append("GDPR")

        # Default frameworks if none specified
        if not policy_refs:
            policy_refs = ["SOC2"]  # Default compliance framework

        # Check if this is a demo/test environment
        is_demo = (
            "demo" in repo_name.lower()
            or "test" in repo_name.lower()
            or "staging" in repo_name.lower()
            or any("demo" in label for label in pr_labels)
        )

        # Start Portia compliance analysis in background
        background_tasks.add_task(
            execute_compliance_analysis, repo_name, pr_number, policy_refs, is_demo
        )

        return {
            "status": "accepted",
            "message": "Compliance analysis started",
            "pr_id": f"{repo_name}#{pr_number}",
            "policy_frameworks": policy_refs,
            "demo_mode": is_demo,
            "action": action,
        }

    except json.JSONDecodeError:
        logger.error("❌ Invalid JSON in webhook payload")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"❌ Webhook processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Webhook processing failed: {str(e)}"
        )


# List recent analyses
@app.get("/analyses")
async def list_analyses(limit: int = 20, db: Session = Depends(get_db)):
    """List recent PR analyses with evidence bundle information"""
    try:
        # Join PR analyses with evidence bundles
        query = (
            db.query(PRAnalysisDB).order_by(PRAnalysisDB.created_at.desc()).limit(limit)
        )
        analyses = query.all()

        result = []
        for analysis in analyses:
            # Find corresponding evidence bundle
            evidence_bundle = (
                db.query(EvidenceBundleDB)
                .filter(EvidenceBundleDB.pr_id == analysis.pr_id)
                .first()
            )

            result.append(
                {
                    "pr_id": analysis.pr_id,
                    "pr_url": analysis.pr_url,
                    "title": analysis.title,
                    "author": analysis.author,
                    "risk_level": analysis.risk_level,
                    "risk_score": analysis.risk_score,
                    "control_count": (
                        len(analysis.control_mappings)
                        if analysis.control_mappings
                        else 0
                    ),
                    "has_evidence_bundle": evidence_bundle is not None,
                    "bundle_id": evidence_bundle.bundle_id if evidence_bundle else None,
                    "created_at": analysis.created_at.isoformat(),
                    "files_changed": analysis.files_changed,
                    "additions": analysis.additions,
                    "deletions": analysis.deletions,
                }
            )

        return {"analyses": result, "total_count": len(result)}

    except Exception as e:
        logger.error(f"Failed to list analyses: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list analyses: {str(e)}"
        )


# Agent stats and configuration
@app.get("/agent/stats")
async def get_agent_stats():
    """Get agent statistics and configuration"""
    stats = compliance_agent.get_agent_stats()
    stats.update(
        {
            "webhook_configured": bool(settings.github_webhook_secret),
            "database_url": (
                settings.database_url.split("@")[-1]
                if "@" in settings.database_url
                else "local"
            ),
        }
    )
    return stats


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
