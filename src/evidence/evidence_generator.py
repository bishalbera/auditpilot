import uuid
from datetime import datetime
from typing import Dict, List, Any
from jinja2 import Environment, DictLoader

from ..models import PRAnalysis, EvidenceBundle, RiskLevel


class EvidenceGenerator:
    """Generates audit-ready evidence bundles"""

    def __init__(self):
        self.templates = self._load_templates()
        self.jinja_env = Environment(loader=DictLoader(self.templates))

    def _load_templates(self) -> Dict[str, str]:
        """Load evidence templates"""
        return {
            "executive_summary": """
# Executive Summary - PR {{ pr_id }}

## Overview
This document provides a compliance analysis summary for Pull Request #{{ pr_number }} in repository {{ repo_name }}.

**Risk Assessment:** {{ risk_level }} ({{ risk_score }}/1.0)
**Controls Impacted:** {{ control_count }} compliance controls across {{ framework_count }} frameworks
**Analysis Date:** {{ analysis_date }}
**Analyst:** Compliance Copilot Agent

## Key Findings
- **{{ risk_indicators|length }} risk indicators** detected in code changes
- **{{ high_confidence_indicators }} high-confidence** security patterns identified
- **{{ critical_controls }} critical controls** require immediate attention
- **Approval Status:** {{ approval_status }}

## Risk Summary
{{ risk_summary }}

## Recommended Actions
{{ recommendations }}
""",
            "technical_details": """
# Technical Analysis - PR {{ pr_id }}

## Change Summary
**Files Modified:** {{ files_changed|length }} files
**Lines Added:** {{ additions }}
**Lines Deleted:** {{ deletions }}
**Author:** {{ author }}

### Files Changed
{% for file in files_changed %}
- `{{ file }}`
{% endfor %}

## Risk Indicators Detected
{% for indicator in risk_indicators %}
### {{ indicator.pattern_type|title }} Risk - {{ indicator.confidence|round(2) }} Confidence
**File:** `{{ indicator.file_path }}`
{% if indicator.line_number %}**Line:** {{ indicator.line_number }}{% endif %}
**Pattern:** {{ indicator.matched_text }}
**Description:** {{ indicator.description }}

{% endfor %}

## Code Analysis Details
{{ code_analysis_details }}
""",
            "control_impact_assessment": """
# Control Impact Assessment - PR {{ pr_id }}

## Compliance Framework Coverage

{% for framework, controls in frameworks.items() %}
### {{ framework }} Controls ({{ controls|length }} impacted)

{% for mapping in controls %}
#### {{ mapping.control.control_id }}: {{ mapping.control.title }}
**Risk Level:** {{ mapping.control.risk_level }}
**Confidence:** {{ mapping.confidence|round(2) }}
**Category:** {{ mapping.control.category }}

**Description:** {{ mapping.control.description }}

**Impact Reasoning:** {{ mapping.reasoning }}

**Evidence:**
{% for indicator in mapping.risk_indicators[:3] %}
- {{ indicator.description }} ({{ indicator.confidence|round(2) }} confidence)
{% endfor %}

---
{% endfor %}
{% endfor %}

## Control Coverage Summary
{{ coverage_summary }}
""",
            "risk_mitigation_plan": """
# Risk Mitigation Plan - PR {{ pr_id }}

## Risk Level: {{ risk_level }}

{% if risk_level in ['HIGH', 'CRITICAL'] %}
## Immediate Actions Required
{{ immediate_actions }}

## Pre-Deployment Checklist
{{ pre_deployment_checklist }}
{% endif %}

## Ongoing Monitoring Requirements
{{ monitoring_requirements }}

## Compliance Validation Steps
{{ validation_steps }}

## Approval Workflow
{{ approval_workflow }}

## Post-Deployment Actions
{{ post_deployment_actions }}
""",
        }

    def generate_bundle(self, pr_analysis: PRAnalysis) -> EvidenceBundle:
        """Generate complete evidence bundle (synchronous)."""
        bundle_id = str(uuid.uuid4())

        # Prepare template context
        context = self._prepare_template_context(pr_analysis)

        # Generate each section
        executive_summary = self._render_template("executive_summary", context)
        technical_details = self._render_template("technical_details", context)
        control_impact_assessment = self._render_template(
            "control_impact_assessment", context
        )
        risk_mitigation_plan = self._render_template("risk_mitigation_plan", context)

        # Generate audit trail
        audit_trail = self._generate_audit_trail(pr_analysis)

        return EvidenceBundle(
            bundle_id=bundle_id,
            pr_analysis=pr_analysis,
            executive_summary=executive_summary,
            technical_details=technical_details,
            control_impact_assessment=control_impact_assessment,
            risk_mitigation_plan=risk_mitigation_plan,
            audit_trail=audit_trail,
        )

    def _prepare_template_context(self, pr_analysis: PRAnalysis) -> Dict[str, Any]:
        """Prepare context variables for templates"""
        # Parse PR ID
        repo_name, pr_number = pr_analysis.pr_id.split("#")

        # Group controls by framework
        frameworks = {}
        for mapping in pr_analysis.control_mappings:
            framework = mapping.control.framework.value
            if framework not in frameworks:
                frameworks[framework] = []
            frameworks[framework].append(mapping)

        # Risk analysis
        high_confidence_indicators = len(
            [r for r in pr_analysis.risk_indicators if r.confidence > 0.7]
        )
        critical_controls = len(
            [
                m
                for m in pr_analysis.control_mappings
                if m.control.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ]
        )

        context = {
            # Basic info
            "pr_id": pr_analysis.pr_id,
            "repo_name": repo_name,
            "pr_number": pr_number,
            "pr_url": pr_analysis.pr_url,
            "title": pr_analysis.title,
            "author": pr_analysis.author,
            "analysis_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            # Change details
            "files_changed": pr_analysis.files_changed,
            "additions": pr_analysis.additions,
            "deletions": pr_analysis.deletions,
            # Risk assessment
            "risk_level": pr_analysis.risk_level.value.upper(),
            "risk_score": f"{pr_analysis.risk_score:.2f}",
            "risk_indicators": pr_analysis.risk_indicators,
            "high_confidence_indicators": high_confidence_indicators,
            # Control mapping
            "control_mappings": pr_analysis.control_mappings,
            "control_count": len(pr_analysis.control_mappings),
            "framework_count": len(frameworks),
            "frameworks": frameworks,
            "critical_controls": critical_controls,
            # Status
            "approval_status": (
                "Pending Review"
                if pr_analysis.risk_level.value in ["HIGH", "CRITICAL"]
                else "Auto-Approved"
            ),
            # Generated content
            "risk_summary": self._generate_risk_summary(pr_analysis),
            "recommendations": self._generate_recommendations(pr_analysis),
            "code_analysis_details": self._generate_code_analysis_details(pr_analysis),
            "coverage_summary": self._generate_coverage_summary(frameworks),
            "immediate_actions": self._generate_immediate_actions(pr_analysis),
            "pre_deployment_checklist": self._generate_pre_deployment_checklist(
                pr_analysis
            ),
            "monitoring_requirements": self._generate_monitoring_requirements(
                pr_analysis
            ),
            "validation_steps": self._generate_validation_steps(pr_analysis),
            "approval_workflow": self._generate_approval_workflow(pr_analysis),
            "post_deployment_actions": self._generate_post_deployment_actions(
                pr_analysis
            ),
        }

        return context

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with context"""
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)

    def _generate_risk_summary(self, pr_analysis: PRAnalysis) -> str:
        """Generate risk summary narrative"""
        risk_patterns = {}
        for indicator in pr_analysis.risk_indicators:
            pattern = indicator.pattern_type
            if pattern not in risk_patterns:
                risk_patterns[pattern] = 0
            risk_patterns[pattern] += 1

        if not risk_patterns:
            return "No significant compliance risks detected in this change."

        summary_parts = []

        if pr_analysis.risk_level == RiskLevel.CRITICAL:
            summary_parts.append(
                "🔴 **CRITICAL RISK**: This change requires immediate compliance review."
            )
        elif pr_analysis.risk_level == RiskLevel.HIGH:
            summary_parts.append(
                "🟠 **HIGH RISK**: This change impacts critical security controls."
            )
        elif pr_analysis.risk_level == RiskLevel.MEDIUM:
            summary_parts.append(
                "🟡 **MEDIUM RISK**: This change requires compliance validation."
            )
        else:
            summary_parts.append(
                "🟢 **LOW RISK**: This change has minimal compliance impact."
            )

        pattern_summary = []
        for pattern, count in sorted(
            risk_patterns.items(), key=lambda x: x[1], reverse=True
        ):
            pattern_summary.append(
                f"{pattern.replace('_', ' ').title()} ({count} indicators)"
            )

        summary_parts.append(f"Primary risk areas: {', '.join(pattern_summary)}.")

        return " ".join(summary_parts)

    def _generate_recommendations(self, pr_analysis: PRAnalysis) -> str:
        """Generate actionable recommendations"""
        recommendations = []

        if pr_analysis.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append(
                "1. **Mandatory Review**: Compliance team review required before deployment."
            )
            recommendations.append(
                "2. **Security Testing**: Execute additional security test scenarios."
            )

        # Pattern-specific recommendations
        risk_patterns = set(
            indicator.pattern_type for indicator in pr_analysis.risk_indicators
        )

        if "authentication" in risk_patterns:
            recommendations.append(
                "3. **Auth Testing**: Validate authentication flows and session management."
            )

        if "data_access" in risk_patterns:
            recommendations.append(
                "4. **Data Validation**: Review data access patterns and authorization controls."
            )

        if "encryption" in risk_patterns:
            recommendations.append(
                "5. **Crypto Review**: Validate encryption implementation and key management."
            )

        if not recommendations:
            recommendations.append(
                "1. **Standard Process**: Follow standard deployment procedures."
            )

        return "\n".join(recommendations)

    def _generate_code_analysis_details(self, pr_analysis: PRAnalysis) -> str:
        """Generate detailed code analysis"""
        details = []

        # File type analysis
        file_types = {}
        for file_path in pr_analysis.files_changed:
            ext = file_path.split(".")[-1] if "." in file_path else "no_extension"
            file_types[ext] = file_types.get(ext, 0) + 1

        if file_types:
            details.append("**File Types Modified:**")
            for ext, count in sorted(file_types.items()):
                details.append(f"- {ext}: {count} files")
            details.append("")

        # Risk distribution
        pattern_confidence = {}
        for indicator in pr_analysis.risk_indicators:
            pattern = indicator.pattern_type
            if pattern not in pattern_confidence:
                pattern_confidence[pattern] = []
            pattern_confidence[pattern].append(indicator.confidence)

        if pattern_confidence:
            details.append("**Risk Pattern Analysis:**")
            for pattern, confidences in pattern_confidence.items():
                avg_conf = sum(confidences) / len(confidences)
                details.append(
                    f"- {pattern.replace('_', ' ').title()}: {len(confidences)} indicators (avg confidence: {avg_conf:.2f})"
                )

        return (
            "\n".join(details)
            if details
            else "Standard code changes with minimal compliance implications."
        )

    def _generate_coverage_summary(self, frameworks: Dict[str, List]) -> str:
        """Generate control coverage summary"""
        if not frameworks:
            return "No compliance controls impacted by this change."

        summary = []
        total_controls = sum(len(controls) for controls in frameworks.values())

        summary.append(
            f"**Total Impact:** {total_controls} controls across {len(frameworks)} frameworks"
        )
        summary.append("")

        for framework, controls in frameworks.items():
            high_risk = len(
                [
                    c
                    for c in controls
                    if c.control.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                ]
            )
            avg_confidence = (
                sum(c.confidence for c in controls) / len(controls) if controls else 0
            )

            summary.append(
                f"**{framework}:** {len(controls)} controls ({high_risk} high-risk, avg confidence: {avg_confidence:.2f})"
            )

        return "\n".join(summary)

    def _generate_immediate_actions(self, pr_analysis: PRAnalysis) -> str:
        """Generate immediate actions for high-risk changes"""
        actions = []

        if pr_analysis.risk_level == RiskLevel.CRITICAL:
            actions.extend(
                [
                    "1. **STOP DEPLOYMENT** - Do not deploy until compliance review is complete",
                    "2. **Escalate to Security Team** - Immediate security team notification required",
                    "3. **Document Justification** - Business justification must be documented",
                ]
            )
        elif pr_analysis.risk_level == RiskLevel.HIGH:
            actions.extend(
                [
                    "1. **Compliance Review** - Schedule compliance team review within 24 hours",
                    "2. **Security Testing** - Execute comprehensive security test suite",
                    "3. **Stakeholder Approval** - Obtain written approval from security stakeholders",
                ]
            )

        # Add pattern-specific actions
        risk_patterns = set(
            indicator.pattern_type for indicator in pr_analysis.risk_indicators
        )

        if "authentication" in risk_patterns:
            actions.append(
                "4. **Auth Flow Testing** - Validate all authentication and authorization paths"
            )

        if "data_access" in risk_patterns:
            actions.append(
                "5. **Data Access Audit** - Review data access logs and permissions"
            )

        return (
            "\n".join(actions)
            if actions
            else "No immediate actions required for this risk level."
        )

    def _generate_pre_deployment_checklist(self, pr_analysis: PRAnalysis) -> str:
        """Generate pre-deployment checklist"""
        checklist = [
            "- [ ] Code review completed by authorized reviewer",
            "- [ ] Security testing executed successfully",
            "- [ ] Compliance impact assessment reviewed",
            "- [ ] Required approvals obtained",
        ]

        risk_patterns = set(
            indicator.pattern_type for indicator in pr_analysis.risk_indicators
        )

        if "authentication" in risk_patterns:
            checklist.extend(
                [
                    "- [ ] Authentication flows tested",
                    "- [ ] Session management validated",
                ]
            )

        if "data_access" in risk_patterns:
            checklist.extend(
                [
                    "- [ ] Data access permissions verified",
                    "- [ ] Database security controls tested",
                ]
            )

        if "encryption" in risk_patterns:
            checklist.extend(
                [
                    "- [ ] Encryption implementation validated",
                    "- [ ] Key management procedures followed",
                ]
            )

        return "\n".join(checklist)

    def _generate_monitoring_requirements(self, pr_analysis: PRAnalysis) -> str:
        """Generate monitoring requirements"""
        requirements = [
            "- Monitor application performance post-deployment",
            "- Track error rates and system stability",
        ]

        risk_patterns = set(
            indicator.pattern_type for indicator in pr_analysis.risk_indicators
        )

        if "authentication" in risk_patterns:
            requirements.extend(
                [
                    "- Monitor authentication success/failure rates",
                    "- Track session anomalies and timeout events",
                ]
            )

        if "data_access" in risk_patterns:
            requirements.extend(
                [
                    "- Monitor database access patterns",
                    "- Track unauthorized access attempts",
                ]
            )

        if "logging" in risk_patterns:
            requirements.extend(
                [
                    "- Verify log generation and retention",
                    "- Monitor audit trail completeness",
                ]
            )

        return "\n".join(requirements)

    def _generate_validation_steps(self, pr_analysis: PRAnalysis) -> str:
        """Generate compliance validation steps"""
        steps = [
            "1. **Control Testing**: Validate that all impacted controls function as designed",
            "2. **Documentation Review**: Ensure all compliance documentation is updated",
            "3. **Evidence Collection**: Gather evidence of control effectiveness",
        ]

        if len(pr_analysis.control_mappings) > 5:
            steps.append(
                "4. **Comprehensive Audit**: Schedule detailed compliance audit"
            )

        return "\n".join(steps)

    def _generate_approval_workflow(self, pr_analysis: PRAnalysis) -> str:
        """Generate approval workflow description"""
        if pr_analysis.risk_level == RiskLevel.CRITICAL:
            return """
**CRITICAL APPROVAL REQUIRED**
1. Security Team Lead approval
2. Compliance Officer approval
3. CISO sign-off required
4. Documentation of business justification
"""
        elif pr_analysis.risk_level == RiskLevel.HIGH:
            return """
**HIGH-RISK APPROVAL REQUIRED**
1. Security Team review and approval
2. Compliance team validation
3. Technical lead sign-off
"""
        elif pr_analysis.risk_level == RiskLevel.MEDIUM:
            return """
**STANDARD APPROVAL**
1. Peer code review
2. Compliance checklist completion
3. Team lead approval
"""
        else:
            return "Standard approval process - peer review and team lead approval sufficient."

    def _generate_post_deployment_actions(self, pr_analysis: PRAnalysis) -> str:
        """Generate post-deployment actions"""
        actions = [
            "1. **Monitor Metrics**: Track key performance and security metrics",
            "2. **Validate Controls**: Confirm compliance controls are functioning",
            "3. **Update Documentation**: Update compliance documentation as needed",
        ]

        if pr_analysis.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            actions.extend(
                [
                    "4. **Security Review**: Conduct post-deployment security assessment",
                    "5. **Compliance Validation**: Validate compliance posture post-change",
                ]
            )

        return "\n".join(actions)

    def _generate_audit_trail(self, pr_analysis: PRAnalysis) -> List[Dict[str, Any]]:
        """Generate audit trail entries"""
        trail = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "pr_analysis_initiated",
                "details": f"Started compliance analysis for PR {pr_analysis.pr_id}",
                "actor": "Compliance Copilot Agent",
                "pr_id": pr_analysis.pr_id,
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "risk_assessment_completed",
                "details": f"Risk level assessed as {pr_analysis.risk_level.value} with score {pr_analysis.risk_score:.3f}",
                "actor": "Compliance Copilot Agent",
                "pr_id": pr_analysis.pr_id,
                "risk_level": pr_analysis.risk_level.value,
                "risk_score": pr_analysis.risk_score,
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "controls_mapped",
                "details": f"Mapped {len(pr_analysis.control_mappings)} compliance controls",
                "actor": "Compliance Copilot Agent",
                "pr_id": pr_analysis.pr_id,
                "control_count": len(pr_analysis.control_mappings),
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "evidence_bundle_generated",
                "details": "Complete evidence bundle generated for audit purposes",
                "actor": "Compliance Copilot Agent",
                "pr_id": pr_analysis.pr_id,
            },
        ]

        return trail
