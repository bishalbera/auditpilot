import json
import logging
from typing import Any, Dict, List, Optional

from portia import (
    Config,
    DefaultToolRegistry,
    InMemoryToolRegistry,
    LLMProvider,
    PlanRun,
    Portia,
    Tool,
    ToolRunContext,
    ToolHardError,
    MultipleChoiceClarification,
)
from pydantic import BaseModel, Field

from src.analysis.pr_analyzer import PRAnalyzer
from src.evidence.evidence_generator import EvidenceGenerator
from src.mapping.control_mapper import ControlMapper
from src.models import PRAnalysis
from src.config import settings
from portia import PlanBuilderV2, StepOutput, Input
from portia.end_user import EndUser

logger = logging.getLogger(__name__)


# ---------- Deterministic analysis and evidence tools ----------
class AnalyzePRArgs(BaseModel):
    repo_name: str = Field(..., description="The full repo name, e.g. org/repo")
    pr_number: int = Field(..., description="The pull request number")


class AnalyzePROutput(BaseModel):
    status: str
    pr_analysis: Dict[str, Any]
    control_count: int
    risk_level: str
    risk_score: float


class AnalyzePullRequestTool(Tool[AnalyzePROutput]):
    """Analyze a GitHub Pull Request for compliance risks and map to controls (deterministic)."""

    id: str = "analyze_pull_request"
    name: str = "Analyze Pull Request Tool"
    description: str = (
        "Analyze a GitHub Pull Request for compliance risks and map to controls"
    )
    args_schema: type[BaseModel] = AnalyzePRArgs
    output_schema: tuple[str, str] = (
        "AnalyzePROutput",
        "Analysis result with PR data and control mappings",
    )

    def run(
        self, ctx: ToolRunContext, repo_name: str, pr_number: int
    ) -> AnalyzePROutput:
        """Run the AnalyzePullRequestTool."""
        try:
            analyzer = PRAnalyzer()
            mapper = ControlMapper()

            logger.info(f"Analyzing PR: {repo_name}#{pr_number}")
            pr_analysis: PRAnalysis = analyzer.analyze_pr(repo_name, pr_number)
            logger.info(f"PR Analysis type: {type(pr_analysis)}, value: {pr_analysis}")

            control_mappings = mapper.map_controls(pr_analysis)
            pr_analysis.control_mappings = control_mappings

            pr_analysis_dict = pr_analysis.model_dump()
            logger.info(
                f"PR Analysis dict type: {type(pr_analysis_dict)}, keys: {pr_analysis_dict.keys() if isinstance(pr_analysis_dict, dict) else 'Not a dict'}"
            )

            result = AnalyzePROutput(
                status="success",
                pr_analysis=pr_analysis_dict,
                control_count=len(control_mappings),
                risk_level=pr_analysis.risk_level.value,
                risk_score=pr_analysis.risk_score,
            )
            logger.info(
                f"Returning result type: {type(result)}, pr_analysis field type: {type(result.pr_analysis)}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in analyze_pull_request: {e}", exc_info=True)
            raise


class GenerateEvidenceArgs(BaseModel):
    analyze_result: Dict[str, Any] = Field(
        ..., description="AnalyzePROutput result containing pr_analysis"
    )


class GenerateEvidenceOutput(BaseModel):
    status: str
    bundle_id: str
    evidence_bundle: Dict[str, Any]


class GenerateEvidenceBundleTool(Tool[GenerateEvidenceOutput]):
    """Generate compliance evidence bundle (markdown sections + audit trail) deterministically."""

    id: str = "generate_evidence_bundle"
    name: str = "Generate Evidence Bundle Tool"
    description: str = (
        "Generate compliance evidence bundle with markdown sections and audit trail"
    )
    args_schema: type[BaseModel] = GenerateEvidenceArgs
    output_schema: tuple[str, str] = (
        "GenerateEvidenceOutput",
        "Evidence bundle with generated compliance documentation",
    )

    def run(
        self, ctx: ToolRunContext, analyze_result: Dict[str, Any]
    ) -> GenerateEvidenceOutput:
        """Run the GenerateEvidenceBundleTool."""
        try:
            logger.info(
                f"GenerateEvidenceBundleTool received analyze_result type: {type(analyze_result)}"
            )
            logger.info(
                f"GenerateEvidenceBundleTool analyze_result content: {str(analyze_result)[:500]}..."
            )

            # Extract pr_analysis from the full analyze result
            if isinstance(analyze_result, str):
                try:
                    analyze_result = json.loads(analyze_result)
                except Exception as e:
                    logger.error(f"Invalid analyze_result string, expected JSON: {e}")
                    analyze_result = {}

            if not isinstance(analyze_result, dict):
                logger.error(
                    f"analyze_result is not a dict after processing: {type(analyze_result)}"
                )
                analyze_result = {}

            pr_analysis_dict = analyze_result.get("pr_analysis", {})
            logger.info(f"Extracted pr_analysis_dict type: {type(pr_analysis_dict)}")
            logger.info(
                f"pr_analysis_dict keys: {list(pr_analysis_dict.keys()) if isinstance(pr_analysis_dict, dict) else 'Not a dict'}"
            )

            if not pr_analysis_dict:
                logger.error("No pr_analysis found in analyze_result")
                raise ValueError("Missing pr_analysis data")

            pr_analysis = PRAnalysis(**pr_analysis_dict)
            logger.info(f"Successfully created PRAnalysis object: {pr_analysis.pr_id}")

            generator = EvidenceGenerator()
            bundle = generator.generate_bundle(pr_analysis)
            logger.info(f"Successfully generated evidence bundle: {bundle.bundle_id}")

            # Ensure bundle.model_dump() returns a dictionary
            evidence_bundle_dict = bundle.model_dump()
            if not isinstance(evidence_bundle_dict, dict):
                logger.warning(
                    "bundle.model_dump() did not return a dict, creating fallback"
                )
                # Fallback if model_dump() doesn't return a dict
                evidence_bundle_dict = {
                    "bundle_id": bundle.bundle_id,
                    "pr_analysis": pr_analysis_dict,
                    "executive_summary": getattr(
                        bundle, "executive_summary", "Generated evidence bundle"
                    ),
                    "technical_details": getattr(
                        bundle, "technical_details", "Technical analysis completed"
                    ),
                    "control_impact_assessment": getattr(
                        bundle, "control_impact_assessment", "Controls assessed"
                    ),
                    "risk_mitigation_plan": getattr(
                        bundle, "risk_mitigation_plan", "Risk mitigation planned"
                    ),
                    "audit_trail": getattr(bundle, "audit_trail", []),
                    "created_at": getattr(bundle, "created_at", None),
                }

            logger.info(
                f"Evidence bundle dict created successfully with keys: {list(evidence_bundle_dict.keys())}"
            )

            result = GenerateEvidenceOutput(
                status="success",
                bundle_id=bundle.bundle_id,
                evidence_bundle=evidence_bundle_dict,
            )
            logger.info(f"GenerateEvidenceOutput created successfully: {result.status}")
            return result

        except Exception as exc:
            logger.error(f"Error in GenerateEvidenceBundleTool: {exc}", exc_info=True)
            # Return an error response instead of raising to avoid breaking the pipeline
            raise ToolHardError(f"Something went wrong {exc}")


class PersistArgs(BaseModel):
    pr_analysis: Dict[str, Any]
    evidence_bundle: Dict[str, Any]


class PersistOutput(BaseModel):
    status: str
    pr_id: str
    bundle_id: str


class PersistResultsToDbTool(Tool[PersistOutput]):
    """Save analysis and evidence to database."""

    id: str = "persist_results_to_db"
    name: str = "Persist Results To Database Tool"
    description: str = "Save analysis and evidence to database"
    args_schema: type[BaseModel] = PersistArgs
    output_schema: tuple[str, str] = (
        "PersistOutput",
        "Database persistence status with IDs",
    )

    def run(
        self,
        ctx: ToolRunContext,
        pr_analysis: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
    ) -> PersistOutput:
        """Run the PersistResultsToDbTool."""
        # Lazy imports to avoid circulars
        from ..models import get_session, PRAnalysisDB, EvidenceBundleDB
        from ..config import settings

        logger.info(
            f"PersistResultsToDbTool - Received pr_analysis type: {type(pr_analysis)}"
        )
        logger.info(
            f"PersistResultsToDbTool - Received evidence_bundle type: {type(evidence_bundle)}"
        )

        # Handle JSON strings if needed
        if isinstance(pr_analysis, str):
            try:
                pr_analysis = json.loads(pr_analysis)
                logger.info("Successfully parsed pr_analysis JSON string")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse pr_analysis JSON: {e}")
                pr_analysis = {}

        if isinstance(evidence_bundle, str):
            try:
                evidence_bundle = json.loads(evidence_bundle)
                logger.info("Successfully parsed evidence_bundle JSON string")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse evidence_bundle JSON: {e}")
                evidence_bundle = {}

        session_factory = get_session(settings.database_url)
        db = session_factory()

        pr_id = None
        bundle_id = None

        try:
            # Extract data safely
            pr = pr_analysis if isinstance(pr_analysis, dict) else {}
            ev = evidence_bundle if isinstance(evidence_bundle, dict) else {}

            logger.info(
                f"Processing PR data: {list(pr.keys()) if pr else 'No PR data'}"
            )
            logger.info(
                f"Processing Evidence data: {list(ev.keys()) if ev else 'No Evidence data'}"
            )

            if pr and pr.get("pr_id"):
                pr_id = pr.get("pr_id")
                pr_row = PRAnalysisDB(
                    pr_id=pr_id,
                    pr_url=pr.get("pr_url"),
                    title=pr.get("title"),
                    description=pr.get("description"),
                    author=pr.get("author"),
                    files_changed=pr.get("files_changed", []),
                    additions=pr.get("additions", 0),
                    deletions=pr.get("deletions", 0),
                    risk_level=pr.get("risk_level"),
                    risk_score=pr.get("risk_score", 0.0),
                    risk_indicators=pr.get("risk_indicators", []),
                    control_mappings=pr.get("control_mappings", []),
                )

                # Check for existing record and replace
                existing = (
                    db.query(PRAnalysisDB).filter(PRAnalysisDB.pr_id == pr_id).first()
                )
                if existing:
                    logger.info(f"Replacing existing PR analysis for {pr_id}")
                    db.delete(existing)
                    db.commit()

                db.add(pr_row)
                logger.info(f"Added PR analysis to database: {pr_id}")

            if ev and ev.get("bundle_id"):
                bundle_id = ev.get("bundle_id")
                # Get pr_id from evidence bundle's pr_analysis if not set
                if not pr_id:
                    pr_analysis_in_ev = ev.get("pr_analysis", {})
                    pr_id = (
                        pr_analysis_in_ev.get("pr_id")
                        if isinstance(pr_analysis_in_ev, dict)
                        else None
                    )

                bundle_row = EvidenceBundleDB(
                    bundle_id=bundle_id,
                    pr_id=pr_id,
                    executive_summary=ev.get("executive_summary", ""),
                    technical_details=ev.get("technical_details", ""),
                    control_impact_assessment=ev.get("control_impact_assessment", ""),
                    risk_mitigation_plan=ev.get("risk_mitigation_plan", ""),
                    audit_trail=ev.get("audit_trail", []),
                )

                # Check for existing record and replace
                existing_b = (
                    db.query(EvidenceBundleDB)
                    .filter(EvidenceBundleDB.bundle_id == bundle_id)
                    .first()
                )
                if existing_b:
                    logger.info(f"Replacing existing evidence bundle for {bundle_id}")
                    db.delete(existing_b)
                    db.commit()

                db.add(bundle_row)
                logger.info(f"Added evidence bundle to database: {bundle_id}")

            db.commit()
            logger.info(
                f"Successfully persisted data - PR: {pr_id}, Bundle: {bundle_id}"
            )

            return PersistOutput(
                status="persisted",
                pr_id=pr_id or "unknown",
                bundle_id=bundle_id or "unknown",
            )

        except Exception as e:
            logger.error(f"Error persisting data: {e}", exc_info=True)
            db.rollback()
            # Return an error status instead of raising to avoid breaking the pipeline
            return PersistOutput(
                status="error",
                pr_id=pr_id or "unknown",
                bundle_id=bundle_id or "unknown",
            )
        finally:
            db.close()


class CreateApprovalClarificationArgs(BaseModel):
    pr_id: str = Field(..., description="The PR identifier")
    risk_level: str = Field(..., description="The risk level of the PR")
    evidence: Dict[str, Any] = Field(..., description="The evidence bundle data")


class CreateApprovalClarificationTool(Tool[str]):
    """Create a clarification request for human approval of medium/high risk PRs."""

    id: str = "create_approval_clarification"
    name: str = "Create Approval Clarification Tool"
    description: str = (
        "Create a clarification request for human approval of medium/high risk PRs"
    )
    args_schema: type[BaseModel] = CreateApprovalClarificationArgs
    output_schema: tuple[str, str] = (
        "str",
        "Clarification status message",
    )

    def run(
        self, ctx: ToolRunContext, pr_id: str, risk_level: str, evidence: Dict[str, Any]
    ) -> str:
        """Run the CreateApprovalClarificationTool."""
        try:
            # Log the clarification event
            logger.info(
                f"🎯 SLACK NOTIFICATION - Would send approval request for PR {pr_id} (risk: {risk_level})"
            )

            # Create a clarification for human review
            clarification = MultipleChoiceClarification(
                plan_run_id=ctx.plan_run.id,
                user_guidance=f"""
🚨 **Compliance Review Required**

**PR:** {pr_id}
**Risk Level:** {risk_level}

This pull request has been analyzed for compliance risks and requires human approval due to {risk_level.lower()} risk level.

**Key Findings:**
- Authentication-related code changes detected
- Multiple compliance controls impacted
- Evidence bundle generated and stored

**Action Required:** Please review the compliance analysis and choose your approval decision below.
                """,
                options=[
                    "✅ APPROVE - Low compliance risk, proceed with merge",
                    "⚠️ APPROVE WITH CONDITIONS - Medium risk, require additional testing",
                    "❌ REJECT - High compliance risk, changes required",
                    "🔍 REQUEST REVIEW - Need additional compliance team review",
                ],
            )

            # Raise the clarification to pause execution and wait for human input
            raise clarification

        except MultipleChoiceClarification:
            # Re-raise clarifications so Portia can handle them
            raise
        except Exception as e:
            logger.error(f"Error creating approval clarification: {e}")
            raise ToolHardError(f"Failed to create approval clarification: {e}")


class EvidenceItem(BaseModel):
    control_id: str
    summary: str
    artifacts: List[str] = Field(default_factory=list)
    risk: Optional[str] = None
    rationale: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class DraftEvidenceInput(BaseModel):
    pr_id: str
    repo: str
    analysis: Dict[str, Any]
    mapped_controls: List[Dict[str, Any]]
    policy_refs: List[str] = Field(default_factory=list)


class DraftEvidenceOutput(BaseModel):
    items: List[EvidenceItem]
    overall_summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class FinalOutput(BaseModel):
    analysis: Dict[str, Any]
    evidence: Dict[str, Any]
    drafted_evidence: DraftEvidenceOutput


class ComplianceAgent:
    config = Config.from_default(
        llm_provider=LLMProvider.GOOGLE,
        default_model="google/gemini-2.0-flash",
        google_api_key=settings.gemini_api_key,
    )
    tools = DefaultToolRegistry(
        config=config,
    ) + InMemoryToolRegistry.from_local_tools(
        [
            AnalyzePullRequestTool(),
            GenerateEvidenceBundleTool(),
            PersistResultsToDbTool(),
            CreateApprovalClarificationTool(),
        ]
    )
    if not tools.get_tool("portia:slack:bot:send_message"):
        raise ValueError(
            "Slack boy tool not found. Please install on Portia "
            "cloud by going to https://app.portialabs.ai/dashboard/tool-registry"
        )

    portia = Portia(config=config, tools=tools)

    def _extract_risk_level_from_data(self, pr_analysis) -> str:
        """Safely extract risk level from PR analysis data, handling both JSON strings and dicts."""
        try:
            # Handle JSON string
            if isinstance(pr_analysis, str):
                try:
                    data = json.loads(pr_analysis)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse pr_analysis JSON string")
                    return "LOW"
            else:
                data = pr_analysis

            # Extract risk_level from dict
            if isinstance(data, dict):
                risk_level = data.get("risk_level", "LOW")
                return str(risk_level).upper()

            logger.warning(f"Unexpected pr_analysis type: {type(data)}")
            return "LOW"

        except Exception as e:
            logger.error(f"Error extracting risk level: {e}")
            return "LOW"

    def build_text_plan(
        self,
        repo_name: str,
        pr_number: int,
        policy_refs: Optional[List[str]] = None,
        demo: bool = False,
    ) -> Any:
        """Build a Portia PlanBuilderV2 plan for compliance analysis, evidence generation, and notification."""

        import logging
        import json
        from datetime import datetime

        # Set up debug logger that writes to file
        debug_log_file = f"portia_debug_{repo_name.replace('/', '_')}_{pr_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        debug_logger = logging.getLogger("portia_debug")
        debug_logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        for handler in debug_logger.handlers[:]:
            debug_logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(debug_log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        debug_logger.addHandler(file_handler)

        def debug_log(message):
            """Helper to log debug messages"""
            debug_logger.debug(message)
            # Also print to console for immediate feedback
            print(f"DEBUG: {message}")

        def extract_pr_analysis(analysis_result):
            """Extract PR analysis from the analyze_pr step result"""
            debug_log(f"extract_pr_analysis - Input type: {type(analysis_result)}")
            debug_log(
                f"extract_pr_analysis - Input content: {str(analysis_result)[:1000]}..."
            )

            # Handle JSON string (most common case with Portia)
            if isinstance(analysis_result, str):
                debug_log("Input is JSON string, attempting to parse")
                try:
                    parsed_result = json.loads(analysis_result)
                    debug_log(f"Successfully parsed JSON, type: {type(parsed_result)}")
                    debug_log(
                        f"Parsed keys: {list(parsed_result.keys()) if isinstance(parsed_result, dict) else 'Not a dict'}"
                    )

                    if (
                        isinstance(parsed_result, dict)
                        and "pr_analysis" in parsed_result
                    ):
                        debug_log("Found pr_analysis in parsed JSON")
                        return parsed_result["pr_analysis"]

                except json.JSONDecodeError as e:
                    debug_log(f"Failed to parse JSON: {e}")

            # Handle AnalyzePROutput object
            if hasattr(analysis_result, "pr_analysis"):
                debug_log("Found pr_analysis attribute")
                result = analysis_result.pr_analysis
                debug_log(f"Extracted pr_analysis type: {type(result)}")
                return result

            # Handle dictionary from serialized AnalyzePROutput
            if isinstance(analysis_result, dict):
                debug_log(f"Input is dict with keys: {list(analysis_result.keys())}")
                if "pr_analysis" in analysis_result:
                    debug_log("Found pr_analysis key in dict")
                    return analysis_result["pr_analysis"]

                # Sometimes the entire dict IS the pr_analysis
                required_fields = {"pr_id", "title", "risk_level", "files_changed"}
                if required_fields.issubset(set(analysis_result.keys())):
                    debug_log("Dict appears to be pr_analysis itself")
                    return analysis_result

            debug_log("Returning empty dict - no valid data found")
            return {}

        def extract_evidence_bundle(evidence_result):
            """Extract evidence bundle from the generate_evidence step result"""
            debug_log(f"extract_evidence_bundle - Input type: {type(evidence_result)}")
            debug_log(
                f"extract_evidence_bundle - Input keys: {list(evidence_result.keys()) if isinstance(evidence_result, dict) else 'Not a dict'}"
            )

            # Handle JSON string (most common case with Portia)
            if isinstance(evidence_result, str):
                debug_log("Input is JSON string, attempting to parse")
                try:
                    parsed_result = json.loads(evidence_result)
                    debug_log(f"Successfully parsed JSON, type: {type(parsed_result)}")
                    debug_log(
                        f"Parsed keys: {list(parsed_result.keys()) if isinstance(parsed_result, dict) else 'Not a dict'}"
                    )

                    if (
                        isinstance(parsed_result, dict)
                        and "evidence_bundle" in parsed_result
                    ):
                        debug_log("Found evidence_bundle in parsed JSON")
                        return parsed_result["evidence_bundle"]

                except json.JSONDecodeError as e:
                    debug_log(f"Failed to parse JSON: {e}")

            # Handle GenerateEvidenceOutput object
            if hasattr(evidence_result, "evidence_bundle"):
                debug_log("Found evidence_bundle attribute")
                result = evidence_result.evidence_bundle
                debug_log(f"Extracted evidence_bundle type: {type(result)}")
                return result

            # Handle dictionary from serialized GenerateEvidenceOutput
            if isinstance(evidence_result, dict):
                debug_log(f"Input is dict with keys: {list(evidence_result.keys())}")
                if "evidence_bundle" in evidence_result:
                    debug_log("Found evidence_bundle key in dict")
                    return evidence_result["evidence_bundle"]

            debug_log("Returning empty dict - no valid evidence bundle found")
            return {}

        def extract_control_mappings(pr_analysis_data):
            """Extract control mappings from PR analysis data"""
            debug_log(
                f"extract_control_mappings - Input type: {type(pr_analysis_data)}"
            )
            debug_log(
                f"extract_control_mappings - Input content: {str(pr_analysis_data)[:500]}..."
            )

            # Handle JSON string (most common case with Portia)
            if isinstance(pr_analysis_data, str):
                debug_log("Input is JSON string, attempting to parse")
                try:
                    parsed_result = json.loads(pr_analysis_data)
                    debug_log(f"Successfully parsed JSON, type: {type(parsed_result)}")
                    debug_log(
                        f"Parsed keys: {list(parsed_result.keys()) if isinstance(parsed_result, dict) else 'Not a dict'}"
                    )

                    if (
                        isinstance(parsed_result, dict)
                        and "control_mappings" in parsed_result
                    ):
                        mappings = parsed_result["control_mappings"]
                        debug_log(
                            f"Found {len(mappings) if mappings else 0} control mappings in parsed JSON"
                        )
                        return mappings
                    else:
                        debug_log("No control_mappings key found in parsed JSON")

                except json.JSONDecodeError as e:
                    debug_log(f"Failed to parse JSON: {e}")

            # Handle dictionary
            if (
                isinstance(pr_analysis_data, dict)
                and "control_mappings" in pr_analysis_data
            ):
                mappings = pr_analysis_data["control_mappings"]
                debug_log(
                    f"Found {len(mappings) if mappings else 0} control mappings in dict"
                )
                return mappings

            debug_log("No control mappings found, returning empty list")
            return []

        debug_log(f"Starting plan build for {repo_name}#{pr_number}")
        debug_log(f"Debug log file: {debug_log_file}")

        builder = (
            PlanBuilderV2(f"Compliance analysis for {repo_name}#{pr_number}")
            .input(name="repo_name", description="The full repo name, e.g. org/repo")
            .input(name="pr_number", description="The pull request number")
            .input(
                name="policy_refs",
                description="List of policy/framework refs",
                default_value=policy_refs or [],
            )
            .input(
                name="demo",
                description="If true, mark Slack notification as demo",
                default_value=demo,
            )
            # Analyze PR
            .invoke_tool_step(
                tool="analyze_pull_request",
                args={
                    "repo_name": Input("repo_name"),
                    "pr_number": Input("pr_number"),
                },
                step_name="analyze_pr",
                output_schema=AnalyzePROutput,
            )
            # Generate evidence bundle using analysis from analyze_pr
            .invoke_tool_step(
                tool="generate_evidence_bundle",
                args={
                    "analyze_result": StepOutput("analyze_pr"),
                },
                step_name="generate_evidence",
                output_schema=GenerateEvidenceOutput,
            )
            # Extract PR analysis data with proper debugging
            .function_step(
                function=extract_pr_analysis,
                args={"analysis_result": StepOutput("analyze_pr")},
                step_name="extract_pr_analysis",
            )
            # Extract evidence bundle data with proper debugging
            .function_step(
                function=extract_evidence_bundle,
                args={"evidence_result": StepOutput("generate_evidence")},
                step_name="extract_evidence_bundle",
            )
            # Compute PR ID string
            .function_step(
                function=lambda repo_name, pr_number: f"{repo_name}#{pr_number}",
                args={"repo_name": Input("repo_name"), "pr_number": Input("pr_number")},
                step_name="compute_pr_id",
            )
            # Extract control mappings from PR analysis
            .function_step(
                function=extract_control_mappings,
                args={"pr_analysis_data": StepOutput("extract_pr_analysis")},
                step_name="extract_mapped_controls",
            )
            .invoke_tool_step(
                tool="persist_results_to_db",
                args={
                    "pr_analysis": StepOutput("extract_pr_analysis"),
                    "evidence_bundle": StepOutput("extract_evidence_bundle"),
                },
                step_name="persist_results",
                output_schema=PersistOutput,
            )
            .llm_step(
                task="""
                You are a Compliance Evidence Assistant analyzing a pull request for compliance violations.

                Context:
                - PR ID: {compute_pr_id}
                - Repository: {repo_name}
                - PR Analysis Data: {extract_pr_analysis}
                - Mapped Controls: {extract_mapped_controls}
                - Policy Frameworks: {policy_refs}

                For each mapped compliance control, you must generate evidence items that strictly conform to the DraftEvidenceOutput JSON schema.

                For each control, analyze:
                1. **summary**: How well the control is being met (met/partially met/not met)
                2. **artifacts**: List specific files, logs, or evidence (e.g., PR files, code changes)
                3. **risk**: Any residual compliance risks from the PR changes
                4. **rationale**: Why your assessment is justified based on the code changes
                5. **confidence**: Float 0-1 indicating your confidence in the assessment

                Important:
                - Always reference specific files and line numbers in artifacts
                - Be specific about what the code changes actually do

                Output must be valid DraftEvidenceOutput JSON with:
                - items: List[EvidenceItem] (one per control)
                - overall_summary: String summarizing compliance impact
                - confidence: Float 0-1 for overall confidence
                """,
                output_schema=DraftEvidenceOutput,
                inputs=[
                    StepOutput("compute_pr_id"),
                    Input("repo_name"),
                    StepOutput("extract_pr_analysis"),
                    StepOutput("extract_mapped_controls"),
                    Input("policy_refs"),
                ],
                step_name="draft_evidence",
            )
            # Extract risk level for conditional logic
            .function_step(
                step_name="extract_risk_level",
                function=self._extract_risk_level_from_data,
                args={"pr_analysis": StepOutput("extract_pr_analysis")},
            )
            # Conditional branch: if risk is MEDIUM or above, require human approval
            .if_(
                condition=lambda risk_level: str(risk_level)
                in ["MEDIUM", "HIGH", "CRITICAL"],
                args={"risk_level": StepOutput("extract_risk_level")},
            )
            # Send Slack notification for compliance team review
            .invoke_tool_step(
                tool="portia:slack:bot:send_message",
                args={
                    "target_id": "compliance-alerts",
                    "message": f"""🚨 Compliance Review Required - {StepOutput('compute_pr_id')}
                                Risk level - {StepOutput('extract_risk_level')}.
                               """,
                },
                step_name="send_slack_notification",
            )
            # Raise clarification for human approval
            .invoke_tool_step(
                step_name="request_approval",
                tool="create_approval_clarification",
                args={
                    "pr_id": StepOutput("compute_pr_id"),
                    "risk_level": StepOutput("extract_risk_level"),
                    "evidence": StepOutput("extract_evidence_bundle"),
                },
            )
            .endif()
            .final_output(output_schema=FinalOutput, summarize=True)
        )
        return builder.build()

    async def run_plan(
        self,
        repo_name: str,
        pr_number: int,
        policy_refs: Optional[List[str]] = None,
        demo: bool = False,
    ) -> PlanRun:
        """
        Execute using PlanBuilderV2 approach for more structured control.
        This method builds an explicit plan and then runs it.
        """
        plan = self.build_text_plan(
            repo_name=repo_name,
            pr_number=pr_number,
            policy_refs=policy_refs or [],
            demo=demo,
        )

        # Prepare inputs for the plan
        inputs = {
            "repo_name": repo_name,
            "pr_number": pr_number,
            "policy_refs": policy_refs or [],
            "demo": demo,
        }

        # Create proper EndUser object
        end_user = EndUser(external_id="compliance_system", name="Compliance System")

        # Execute the plan - use run_builder_plan directly (async method)
        try:
            return await self.portia.arun_plan(
                plan=plan, plan_run_inputs=inputs, end_user=end_user
            )
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            raise
