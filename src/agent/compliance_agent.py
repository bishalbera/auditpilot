import json
import logging
from typing import Any, Dict, List, Optional

from portia import (
    Config,
    DefaultToolRegistry,
    InMemoryToolRegistry,
    McpToolRegistry,
    LLMProvider,
    PlanRun,
    Portia,
    Tool,
    ToolRunContext,
    ToolHardError,
)
from pydantic import BaseModel, Field

from src.analysis.pr_analyzer import PRAnalyzer
from src.evidence.evidence_generator import EvidenceGenerator
from src.mapping.control_mapper import ControlMapper
from src.models import PRAnalysis
from src.config import settings
from portia import PlanBuilderV2, StepOutput, Input
from portia.end_user import EndUser
from src.tools import CustomNotionPageCreatorTool, CreateNotionPageOutput

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


class CheckApprovalNeededArgs(BaseModel):
    risk_level: str = Field(..., description="The risk level of the PR")


class ApprovalNeededOutput(BaseModel):
    approval_needed: bool = Field(
        ..., description="True if approval is needed, False otherwise"
    )


class CheckApprovalNeededTool(Tool[ApprovalNeededOutput]):
    """Check if human approval is needed based on risk level."""

    id: str = "check_approval_needed"
    name: str = "Check Approval Needed Tool"
    description: str = "Check if human approval is needed based on risk level"
    args_schema: type[BaseModel] = CheckApprovalNeededArgs
    output_schema: tuple[str, str] = (
        "ApprovalNeededOutput",
        "Result indicating if approval is needed",
    )

    def run(self, ctx: ToolRunContext, risk_level: str) -> ApprovalNeededOutput:
        """Check if approval is needed based on risk level."""
        approval_needed = str(risk_level).upper() in ["MEDIUM", "HIGH", "CRITICAL"]
        logger.info(f"Risk level: {risk_level}, Approval needed: {approval_needed}")
        return ApprovalNeededOutput(approval_needed=approval_needed)


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
    approval_decision: Optional[str] = None


class ComplianceAgent:
    config = Config.from_default(
        llm_provider=LLMProvider.GOOGLE,
        default_model="google/gemini-2.0-flash",
        google_api_key=settings.gemini_api_key,
    )

    # Create Notion MCP tool registry
    notion_tools = McpToolRegistry.from_stdio_connection(
        server_name="notionApi",
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={
            "OPENAPI_MCP_HEADERS": f'{{"Authorization": "Bearer {settings.notion_token}", "Notion-Version": "2022-06-28"}}'
        },
    )

    tools = (
        DefaultToolRegistry(
            config=config,
        )
        + InMemoryToolRegistry.from_local_tools(
            [
                AnalyzePullRequestTool(),
                GenerateEvidenceBundleTool(),
                PersistResultsToDbTool(),
                CheckApprovalNeededTool(),
                CustomNotionPageCreatorTool(),
            ]
        )
        + notion_tools
    )

    if not tools.get_tool("portia:slack:bot:send_message"):
        raise ValueError(
            "Slack boy tool not found. Please install on Portia "
            "cloud by going to https://app.portialabs.ai/dashboard/tool-registry"
        )
    if not tools.get_tool("portia:mcp:mcp.notion.com:notion_create_pages"):
        raise ValueError("notion tool not available")

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

    def _ensure_pages_list(self, pages_data):
        """Ensure pages data is a proper list, not a JSON string."""
        logger.info(f"_ensure_pages_list - Input type: {type(pages_data)}")
        logger.info(
            f"_ensure_pages_list - Input content preview: {str(pages_data)[:200]}..."
        )

        try:
            # If it's already a list, return it as-is
            if isinstance(pages_data, list):
                logger.info(f"Data is already a list with {len(pages_data)} items")
                return pages_data

            # If it's a string, try to parse it as JSON
            if isinstance(pages_data, str):
                logger.info("Data is a string, attempting to parse as JSON")
                try:
                    parsed_data = json.loads(pages_data)
                    if isinstance(parsed_data, list):
                        logger.info(
                            f"Successfully parsed JSON string to list with {len(parsed_data)} items"
                        )
                        return parsed_data
                    else:
                        logger.warning(
                            f"Parsed JSON is not a list, got: {type(parsed_data)}"
                        )
                        # If it's a single object, wrap it in a list
                        return [parsed_data]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse pages_data as JSON: {e}")
                    # Return empty list as fallback
                    return []

            # If it's some other type, try to convert to list
            logger.warning(
                f"Unexpected pages_data type: {type(pages_data)}, attempting to convert to list"
            )
            if hasattr(pages_data, "__iter__") and not isinstance(
                pages_data, (str, bytes)
            ):
                return list(pages_data)
            else:
                # Wrap single item in list
                return [pages_data]

        except Exception as e:
            logger.error(f"Error in _ensure_pages_list: {e}")
            # Return empty list as ultimate fallback
            return []

    def build_text_plan(
        self,
        repo_name: str,
        pr_number: int,
        policy_refs: Optional[List[str]] = None,
        demo: bool = False,
    ) -> Any:
        """Build a Portia PlanBuilderV2 plan for compliance analysis with proper clarification handling."""

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

        def create_evidence_summary(evidence_bundle_data):
            """Create a concise summary of evidence for approval requests."""
            debug_log(
                f"create_evidence_summary - Input type: {type(evidence_bundle_data)}"
            )
            debug_log(
                f"create_evidence_summary - Input keys: {list(evidence_bundle_data.keys()) if isinstance(evidence_bundle_data, dict) else 'Not a dict'}"
            )

            if not isinstance(evidence_bundle_data, dict):
                debug_log("Input is not a dictionary, returning default message")
                return "Evidence bundle generated with compliance analysis."

            # Try to extract executive summary directly from the data
            summary = evidence_bundle_data.get("executive_summary", "")
            debug_log(f"Found executive_summary: {bool(summary)}")
            if summary:
                # Clean up the summary text - remove markdown headers and extra formatting
                cleaned_summary = (
                    summary.replace("\n# ", " ")
                    .replace("\n## ", " ")
                    .replace("\n", " ")
                )
                cleaned_summary = cleaned_summary.strip()
                result = (
                    cleaned_summary[:300] + "..."
                    if len(cleaned_summary) > 300
                    else cleaned_summary
                )
                debug_log(f"Returning executive summary (length: {len(result)})")
                return result

            # Fallback to technical details
            technical = evidence_bundle_data.get("technical_details", "")
            debug_log(f"Found technical_details: {bool(technical)}")
            if technical:
                # Clean up technical details - remove markdown and extract key info
                cleaned_technical = (
                    technical.replace("\n# ", " ")
                    .replace("\n## ", " ")
                    .replace("\n", " ")
                )
                cleaned_technical = cleaned_technical.strip()
                result = (
                    cleaned_technical[:300] + "..."
                    if len(cleaned_technical) > 300
                    else cleaned_technical
                )
                debug_log(f"Returning technical details (length: {len(result)})")
                return result

            # Try to extract PR analysis info as fallback
            pr_analysis = evidence_bundle_data.get("pr_analysis", {})
            if isinstance(pr_analysis, dict):
                title = pr_analysis.get("title", "")
                risk_level = pr_analysis.get("risk_level", "")
                files_changed = pr_analysis.get("files_changed", [])
                if title or risk_level:
                    result = f"PR '{title}' - Risk: {risk_level.upper() if risk_level else 'Unknown'}, Files: {len(files_changed) if isinstance(files_changed, list) else 0}"
                    debug_log(f"Returning PR analysis summary: {result}")
                    return result

        def create_notion_page_data(
            pr_id, evidence_bundle, risk_level, pr_analysis, control_mappings
        ):
            """Create Notion page data with proper data extraction and return as a plain list."""
            debug_log(
                f"create_notion_page_data - evidence_bundle type: {type(evidence_bundle)}"
            )
            debug_log(
                f"create_notion_page_data - pr_analysis type: {type(pr_analysis)}"
            )
            debug_log(
                f"create_notion_page_data - control_mappings type: {type(control_mappings)}"
            )

            # Helper function to safely parse JSON strings
            def safe_parse_json(data, fallback=None):
                if isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        debug_log(
                            f"Successfully parsed JSON data of type: {type(parsed)}"
                        )
                        return parsed
                    except json.JSONDecodeError as e:
                        debug_log(f"Failed to parse JSON: {e}")
                        return fallback or {}
                elif isinstance(data, dict):
                    return data
                else:
                    debug_log(f"Unexpected data type: {type(data)}, returning fallback")
                    return fallback or {}

            # Safe extraction helper
            def safe_get(data, key, default="Not available"):
                if isinstance(data, dict):
                    return data.get(key, default)
                return default

            # Parse all input parameters
            evidence_bundle_dict = safe_parse_json(evidence_bundle, {})
            pr_analysis_dict = safe_parse_json(pr_analysis, {})
            control_mappings_list = safe_parse_json(control_mappings, [])

            debug_log(
                f"Parsed evidence_bundle keys: {list(evidence_bundle_dict.keys()) if isinstance(evidence_bundle_dict, dict) else 'Not a dict'}"
            )
            debug_log(
                f"Parsed pr_analysis keys: {list(pr_analysis_dict.keys()) if isinstance(pr_analysis_dict, dict) else 'Not a dict'}"
            )
            debug_log(
                f"Parsed control_mappings length: {len(control_mappings_list) if isinstance(control_mappings_list, list) else 'Not a list'}"
            )

            # Extract data safely from the evidence bundle
            executive_summary = safe_get(
                evidence_bundle_dict,
                "executive_summary",
                "Analysis completed successfully",
            )
            technical_details = safe_get(
                evidence_bundle_dict,
                "technical_details",
                "Technical analysis completed",
            )
            control_impact = safe_get(
                evidence_bundle_dict,
                "control_impact_assessment",
                "Control impact assessed",
            )
            risk_mitigation = safe_get(
                evidence_bundle_dict,
                "risk_mitigation_plan",
                "Risk mitigation plan prepared",
            )
            audit_trail = safe_get(evidence_bundle_dict, "audit_trail", [])

            # Extract PR analysis data safely
            pr_title = safe_get(pr_analysis_dict, "title", "Unknown PR")
            pr_url = safe_get(pr_analysis_dict, "pr_url", "#")
            pr_author = safe_get(pr_analysis_dict, "author", "Unknown")
            files_changed = safe_get(pr_analysis_dict, "files_changed", [])
            files_count = (
                len(files_changed) if isinstance(files_changed, list) else "N/A"
            )
            additions = safe_get(pr_analysis_dict, "additions", 0)
            deletions = safe_get(pr_analysis_dict, "deletions", 0)
            risk_score = safe_get(pr_analysis_dict, "risk_score", 0.0)

            # Extract control mappings data safely
            control_count = (
                len(control_mappings_list)
                if isinstance(control_mappings_list, list)
                else 0
            )

            # Generate control list
            if isinstance(control_mappings_list, list) and control_mappings_list:
                control_items = []
                for control_mapping in control_mappings_list:
                    if isinstance(control_mapping, dict):
                        control_info = control_mapping.get("control", {})
                        if isinstance(control_info, dict):
                            control_id = control_info.get("control_id", "Unknown")
                            control_title = control_info.get("title", "No title")
                            framework = control_info.get("framework", "Unknown")
                            confidence = control_mapping.get("confidence", 0)
                            control_items.append(
                                f"- **{framework} {control_id}**: {control_title} (Confidence: {confidence:.0%})"
                            )
                control_list = (
                    "\n".join(control_items)
                    if control_items
                    else "No control mappings found."
                )
            else:
                control_list = "No control mappings found."

            debug_log(
                f"Generated control_list with {len(control_list.split(chr(10)))} items"
            )

            # Format audit trail as JSON (handle both list and string cases)
            if isinstance(audit_trail, list):
                audit_trail_json = (
                    json.dumps(audit_trail, indent=2) if audit_trail else "[]"
                )
            elif isinstance(audit_trail, str):
                # It might already be a JSON string
                try:
                    # Try to parse and re-format
                    parsed_trail = json.loads(audit_trail)
                    audit_trail_json = json.dumps(parsed_trail, indent=2)
                except json.JSONDecodeError:
                    audit_trail_json = audit_trail  # Use as-is if not valid JSON
            else:
                audit_trail_json = "[]"

            # Clean up text content for Notion (remove excessive markdown formatting)
            def clean_text(text):
                if not isinstance(text, str):
                    return str(text)
                # Remove excessive markdown headers and clean up formatting
                cleaned = (
                    text.replace("\n# ", "\n## ")
                    .replace("\n## ", "\n**")
                    .replace("**\n", "**\n\n")
                )
                # Remove multiple consecutive newlines
                import re

                cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
                return cleaned.strip()

            executive_summary_clean = clean_text(executive_summary)
            technical_details_clean = clean_text(technical_details)
            control_impact_clean = clean_text(control_impact)
            risk_mitigation_clean = clean_text(risk_mitigation)

            # Create the page data structure - NOTE: Return raw list, not JSON string
            page_data = {
                "properties": {"title": f"🛡️ Compliance Analysis: {pr_id}"},
                "content": f"""# 🛡️ Compliance Analysis Report

**PR ID**: [{pr_id}]({pr_url})
**Title**: {pr_title}
**Author**: {pr_author}
**Risk Level**: {risk_level.upper()}
**Risk Score**: {risk_score:.2f}/1.0
**Analysis Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Change Summary

**Files Changed**: {files_count}
**Lines Added**: {additions}
**Lines Deleted**: {deletions}
**Control Mappings**: {control_count}

## 📋 Executive Summary

{executive_summary_clean}

## ⚠️ Risk Assessment

> **Risk Level**: {risk_level.upper()}
>
> **Overall Risk Score**: {risk_score:.2f}/1.0

## 🎯 Compliance Controls

{control_list}

## 🔧 Technical Analysis

{technical_details_clean}

## 🏗️ Control Impact Assessment

{control_impact_clean}

## 🛠️ Risk Mitigation Plan

{risk_mitigation_clean}

## 📝 Audit Trail

```json
{audit_trail_json}
```

---
*Generated by Compliance Copilot - Automated compliance analysis for development workflows*""",
            }

            # CRITICAL: Return the list directly, don't JSON serialize it
            # The function step should return the actual Python list object
            result = [page_data]
            debug_log(f"Returning raw Python list with {len(result)} pages")
            debug_log(
                f"Result type: {type(result)}, first item type: {type(result[0])}"
            )

            return result

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
            # Step 1: Analyze PR
            .invoke_tool_step(
                tool="analyze_pull_request",
                args={
                    "repo_name": Input("repo_name"),
                    "pr_number": Input("pr_number"),
                },
                step_name="analyze_pr",
                output_schema=AnalyzePROutput,
            )
            # Step 2: Generate evidence bundle
            .invoke_tool_step(
                tool="generate_evidence_bundle",
                args={
                    "analyze_result": StepOutput("analyze_pr"),
                },
                step_name="generate_evidence",
                output_schema=GenerateEvidenceOutput,
            )
            # Step 3: Extract data for processing
            .function_step(
                function=extract_pr_analysis,
                args={"analysis_result": StepOutput("analyze_pr")},
                step_name="extract_pr_analysis",
            )
            .function_step(
                function=extract_evidence_bundle,
                args={"evidence_result": StepOutput("generate_evidence")},
                step_name="extract_evidence_bundle",
            )
            .function_step(
                function=lambda repo_name, pr_number: f"{repo_name}#{pr_number}",
                args={"repo_name": Input("repo_name"), "pr_number": Input("pr_number")},
                step_name="compute_pr_id",
            )
            .function_step(
                function=extract_control_mappings,
                args={"pr_analysis_data": StepOutput("extract_pr_analysis")},
                step_name="extract_mapped_controls",
            )
            # Step 4: Persist to database
            .invoke_tool_step(
                tool="persist_results_to_db",
                args={
                    "pr_analysis": StepOutput("extract_pr_analysis"),
                    "evidence_bundle": StepOutput("extract_evidence_bundle"),
                },
                step_name="persist_results",
                output_schema=PersistOutput,
            )
            # Step 5: Generate evidence draft with LLM
            .llm_step(
                task="""
    You are a Compliance Evidence Assistant analyzing a pull request for compliance violations.

    Context:
    - PR ID: {compute_pr_id}
    - Repository: {repo_name}
    - PR Analysis Data: {extract_pr_analysis}
    - Mapped Controls: {extract_mapped_controls}
    - Policy Frameworks: {policy_refs}

    For each mapped compliance control, generate evidence items that strictly conform to the DraftEvidenceOutput JSON schema.

    For each control, analyze:
    1. **summary**: How well the control is being met (met/partially met/not met)
    2. **artifacts**: List specific files, logs, or evidence (e.g., PR files, code changes)
    3. **risk**: Any residual compliance risks from the PR changes
    4. **rationale**: Why your assessment is justified based on the code changes
    5. **confidence**: Float 0-1 indicating your confidence in the assessment

    Important:
    - Always reference specific files and line numbers in artifacts
    - Be specific about what the code changes actually do
    - Return valid JSON matching DraftEvidenceOutput schema

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
            # Step 6: Extract risk level for approval check
            .function_step(
                step_name="extract_risk_level",
                function=self._extract_risk_level_from_data,
                args={"pr_analysis": StepOutput("extract_pr_analysis")},
            )
            # Step 7: Check if approval is needed (separate from raising clarification)
            .invoke_tool_step(
                tool="check_approval_needed",
                args={"risk_level": StepOutput("extract_risk_level")},
                step_name="check_approval_needed",
                output_schema=ApprovalNeededOutput,
            )
            # Step 8: Create Notion page data and then create the page using custom tool
            .function_step(
                function=create_notion_page_data,
                args={
                    "pr_id": StepOutput("compute_pr_id"),
                    "evidence_bundle": StepOutput("extract_evidence_bundle"),
                    "risk_level": StepOutput("extract_risk_level"),
                    "pr_analysis": StepOutput("extract_pr_analysis"),
                    "control_mappings": StepOutput("extract_mapped_controls"),
                },
                step_name="prepare_notion_page_data",
            )
            .invoke_tool_step(
                tool="custom_notion_page_creator",
                args={"pages": StepOutput("prepare_notion_page_data")},
                step_name="create_notion_page",
                output_schema=CreateNotionPageOutput,
            )
            # Step 9: Send notification for medium/high risk PRs
            .invoke_tool_step(
                tool="portia:slack:bot:send_message",
                args={
                    "target_id": "general",
                    "message": f"🚨 Compliance Analysis Complete - {StepOutput('compute_pr_id')} (Risk: {StepOutput('extract_risk_level')}). Notion documentation created. Review required for medium/high risk.",
                },
                step_name="send_slack_notification",
            )
            # Step 10: Create evidence summary for approval
            .function_step(
                function=create_evidence_summary,
                args={"evidence_bundle_data": StepOutput("extract_evidence_bundle")},
                step_name="create_evidence_summary",
            )
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
        try:
            return await self.portia.arun_plan(
                plan=plan, plan_run_inputs=inputs, end_user=end_user
            )
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            raise
