from typing import Dict, List
from src.models import (
    ComplianceControl,
    ComplianceFramework,
    ControlMapping,
    PRAnalysis,
    RiskIndicator,
    RiskLevel,
)


class ControlMapper:
    """Maps code changes to compliance controls"""

    def __init__(self):
        self.control_mappings = self._load_control_mappings()

    def _load_control_mappings(self) -> dict:
        """Load control mapping rules"""
        return {
            "SOC2": {
                "authentication": [
                    {
                        "control_id": "CC6.1",
                        "description": "The entity implements logical access security software, infrastructure, and architectures over protected information assets to protect them from security events.",
                        "title": "Logical Access - Authentication",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "CC6.2",
                        "description": "Prior to issuing system credentials and granting system access, the entity registers and authorizes new internal and external users.",
                        "title": "Logical Access - User Authentication",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "CC6.3",
                        "description": "The entity authorizes, modifies, or removes access to data, software, functions, and other protected information assets.",
                        "title": "Logical Access - Authentication Protocols",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                ],
                "data_access": [
                    {
                        "control_id": "CC6.4",
                        "description": "The entity restricts logical access to information and system resources associated with confidential information.",
                        "title": "Logical Access - Data Access",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "CC7.1",
                        "description": "To meet its objectives, the entity uses detection and monitoring procedures to identify system security events.",
                        "title": "System Operations - Data Transmission",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.MEDIUM,
                    },
                ],
                "encryption": [
                    {
                        "control_id": "CC6.1",
                        "description": "The entity implements logical access security software, infrastructure, and architectures over protected information assets.",
                        "title": "Logical Access - Encryption",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "CC6.7",
                        "description": "The entity restricts the transmission, movement and removal of information to authorized internal and external users.",
                        "title": "Logical Access - Data Encryption",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                ],
                "logging": [
                    {
                        "control_id": "CC7.2",
                        "description": "The entity monitors system components and the operation of controls to detect anomalies.",
                        "title": "System Operations - System Monitoring",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.MEDIUM,
                    },
                    {
                        "control_id": "CC7.3",
                        "description": "The entity evaluates security events to determine whether they could have resulted in a failure of controls.",
                        "title": "System Operations - Logging and Monitoring",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.MEDIUM,
                    },
                ],
                "access_control": [
                    {
                        "control_id": "CC6.2",
                        "description": "Prior to issuing system credentials and granting system access, the entity registers and authorizes new internal and external users.",
                        "title": "Logical Access - Access Control",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "CC6.3",
                        "description": "The entity authorizes, modifies, or removes access to data, software, functions, and other protected information assets.",
                        "title": "Logical Access - Authorization",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    },
                ],
                "api_security": [
                    {
                        "control_id": "CC6.6",
                        "description": "The entity implements network security measures to protect against network-based threats.",
                        "title": "Logical Access - Network Security",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.MEDIUM,
                    }
                ],
                "data_processing": [
                    {
                        "control_id": "CC6.4",
                        "description": "The entity restricts logical access to information and system resources associated with confidential information.",
                        "title": "Logical Access - Data Protection",
                        "category": "Common Criteria",
                        "risk_level": RiskLevel.HIGH,
                    }
                ],
            },
            "ISO27001": {
                "authentication": [
                    {
                        "control_id": "A.9.1.1",
                        "title": "Access Control Policy",
                        "description": "An access control policy should be established, documented and reviewed based on business and information security requirements.",
                        "category": "Access Control",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "A.9.2.1",
                        "title": "User Registration and De-registration",
                        "description": "A formal user registration and de-registration process should be implemented to enable assignment of access rights.",
                        "category": "Access Control",
                        "risk_level": RiskLevel.HIGH,
                    },
                ],
                "encryption": [
                    {
                        "control_id": "A.10.1.1",
                        "title": "Cryptographic Controls Policy",
                        "description": "A policy on the use of cryptographic controls for protection of information should be developed and implemented.",
                        "category": "Cryptography",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "A.10.1.2",
                        "title": "Key Management",
                        "description": "A policy on the use, protection and lifetime of cryptographic keys should be developed and implemented.",
                        "category": "Cryptography",
                        "risk_level": RiskLevel.HIGH,
                    },
                ],
                "data_access": [
                    {
                        "control_id": "A.9.4.1",
                        "title": "Information Access Restriction",
                        "description": "Access to information and application system functions should be restricted in accordance with the access control policy.",
                        "category": "Access Control",
                        "risk_level": RiskLevel.HIGH,
                    }
                ],
                "logging": [
                    {
                        "control_id": "A.12.4.1",
                        "title": "Event Logging",
                        "description": "Event logs recording user activities, exceptions, faults and information security events should be produced, kept and regularly reviewed.",
                        "category": "Operations Security",
                        "risk_level": RiskLevel.MEDIUM,
                    }
                ],
            },
            "GDPR": {
                "data_processing": [
                    {
                        "control_id": "Art.25",
                        "title": "Data Protection by Design and by Default",
                        "description": "Taking into account the nature, scope, context and purposes of processing as well as the risks for the rights and freedoms of natural persons posed by processing.",
                        "category": "Data Protection Principles",
                        "risk_level": RiskLevel.HIGH,
                    },
                    {
                        "control_id": "Art.32",
                        "title": "Security of Processing",
                        "description": "Taking into account the state of the art, the costs of implementation and the nature, scope, context and purposes of processing.",
                        "category": "Security",
                        "risk_level": RiskLevel.HIGH,
                    },
                ],
                "access_control": [
                    {
                        "control_id": "Art.32",
                        "title": "Security of Processing - Access Controls",
                        "description": "Implementing appropriate technical and organisational measures to ensure a level of security appropriate to the risk.",
                        "category": "Security",
                        "risk_level": RiskLevel.HIGH,
                    }
                ],
                "encryption": [
                    {
                        "control_id": "Art.32",
                        "title": "Security of Processing - Encryption",
                        "description": "The pseudonymisation and encryption of personal data as appropriate technical measures.",
                        "category": "Security",
                        "risk_level": RiskLevel.HIGH,
                    }
                ],
            },
        }

    def map_controls(self, pr_analysis: PRAnalysis) -> List[ControlMapping]:
        """Map PR analysis to compliance controls (synchronous)."""
        control_mappings = []

        # Group risk indicators by pattern type
        indicators_by_type = self._group_indicators_by_type(pr_analysis.risk_indicators)

        # Map each pattern type to controls for each framework
        for pattern_type, indicators in indicators_by_type.items():
            for framework in [
                ComplianceFramework.SOC2,
                ComplianceFramework.ISO27001,
                ComplianceFramework.GDPR,
            ]:
                mappings = self._map_pattern_to_controls(
                    pattern_type, indicators, framework
                )
                control_mappings.extend(mappings)

        # Remove duplicate mappings and sort by confidence
        unique_mappings = self._remove_duplicate_mappings(control_mappings)
        unique_mappings.sort(key=lambda x: x.confidence, reverse=True)

        return unique_mappings

    def _group_indicators_by_type(
        self, risk_indicators: List[RiskIndicator]
    ) -> Dict[str, List[RiskIndicator]]:
        """Group risk indicators by pattern type"""
        grouped = {}
        for indicator in risk_indicators:
            if indicator.pattern_type not in grouped:
                grouped[indicator.pattern_type] = []
            grouped[indicator.pattern_type].append(indicator)
        return grouped

    def _map_pattern_to_controls(
        self,
        pattern_type: str,
        indicators: List[RiskIndicator],
        framework: ComplianceFramework,
    ) -> List[ControlMapping]:
        """Map a pattern type to controls for a specific compliance framework"""
        mappings = []

        framework_mappings = self.control_mappings.get(framework.value, {})
        control_configs = framework_mappings.get(pattern_type, [])

        for control_config in control_configs:
            # Calculate confidence based on indicators
            confidence = self._calculate_mapping_confidence(
                indicators, control_config, pattern_type
            )

            if confidence > 0.3:  # Minimum confidence threshold
                control = ComplianceControl(
                    framework=framework,
                    control_id=control_config["control_id"],
                    title=control_config["title"],
                    description=control_config["description"],
                    category=control_config["category"],
                    risk_level=control_config["risk_level"],
                )

                reasoning = self._generate_mapping_reasoning(
                    indicators, control, pattern_type
                )

                mapping = ControlMapping(
                    control=control,
                    confidence=confidence,
                    reasoning=reasoning,
                    risk_indicators=indicators,
                )

                mappings.append(mapping)

        return mappings

    def _calculate_mapping_confidence(
        self, indicators: List[RiskIndicator], control_config: Dict, pattern_type: str
    ) -> float:
        """Calculate confidence score for control mapping"""
        if not indicators:
            return 0.0

        # Base confidence from indicator strengths
        avg_indicator_confidence = sum(ind.confidence for ind in indicators) / len(
            indicators
        )

        # Boost confidence for high-risk controls
        risk_boost = {
            RiskLevel.CRITICAL: 0.2,
            RiskLevel.HIGH: 0.1,
            RiskLevel.MEDIUM: 0.05,
            RiskLevel.LOW: 0.0,
        }

        control_risk_level = control_config.get("risk_level", RiskLevel.MEDIUM)
        confidence = avg_indicator_confidence + risk_boost.get(control_risk_level, 0.0)

        # Boost confidence for multiple indicators of the same type
        if len(indicators) > 1:
            confidence += min(0.2, (len(indicators) - 1) * 0.05)

        pattern_boost = {
            "authentication": 0.1,
            "encryption": 0.05,
            "data_access": 0.1,
        }

        confidence += pattern_boost.get(pattern_type, 0.0)

        return min(1.0, max(0.0, confidence))

    def _generate_mapping_reasoning(
        self,
        indicators: List[RiskIndicator],
        control: ComplianceControl,
        pattern_type: str,
    ) -> str:
        """Generate human-readable reasoning for control mapping"""
        indicator_count = len(indicators)
        high_confidence_count = len([i for i in indicators if i.confidence > 0.7])

        reasoning_parts = []

        # Main reason
        reasoning_parts.append(
            f"This PR contains {indicator_count} {pattern_type}-related changes that fall under {control.framework.value} control {control.control_id}."
        )

        # Confidence justification
        if high_confidence_count > 0:
            reasoning_parts.append(
                f"{high_confidence_count} high-confidence indicators were detected."
            )

        # Specific examples
        top_indicators = sorted(indicators, key=lambda x: x.confidence, reverse=True)[
            :2
        ]
        for indicator in top_indicators:
            if (
                indicator.file_path != "commit_message"
                and indicator.file_path != "pr_description"
            ):
                reasoning_parts.append(
                    f"File '{indicator.file_path}' contains {pattern_type} patterns."
                )

        # Control relevance
        reasoning_parts.append(
            f"Control {control.control_id} ({control.title}) directly addresses {pattern_type} security requirements."
        )

        return " ".join(reasoning_parts)

    def _remove_duplicate_mappings(
        self, mappings: List[ControlMapping]
    ) -> List[ControlMapping]:
        """Remove duplicate control mappings"""
        seen = set()
        unique_mappings = []

        for mapping in mappings:
            key = (mapping.control.framework.value, mapping.control.control_id)
            if key not in seen:
                seen.add(key)
                unique_mappings.append(mapping)

        return unique_mappings

    def get_framework_coverage(self, mappings: List[ControlMapping]) -> Dict[str, Dict]:
        """Get coverage statistics by framework"""
        coverage = {}

        for framework in [
            ComplianceFramework.SOC2,
            ComplianceFramework.ISO27001,
            ComplianceFramework.GDPR,
        ]:
            framework_mappings = [
                m for m in mappings if m.control.framework == framework
            ]

            if framework_mappings:
                total_confidence = sum(m.confidence for m in framework_mappings)
                avg_confidence = total_confidence / len(framework_mappings)
                high_risk_count = len(
                    [
                        m
                        for m in framework_mappings
                        if m.control.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                    ]
                )

                coverage[framework.value] = {
                    "control_count": len(framework_mappings),
                    "avg_confidence": avg_confidence,
                    "high_risk_count": high_risk_count,
                    "controls": [
                        {
                            "control_id": m.control.control_id,
                            "title": m.control.title,
                            "risk_level": m.control.risk_level.value,
                            "confidence": round(m.confidence, 3),
                        }
                        for m in framework_mappings
                    ],
                }

        return coverage
