from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    Float,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


class ComplianceFramework(str, Enum):
    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    GDPR = "GDPR"


# Api models
class ComplianceControl(BaseModel):
    """Compliance control definition"""

    framework: ComplianceFramework
    control_id: str
    title: str
    description: str
    risk_level: RiskLevel
    category: str


class RiskIndicator(BaseModel):
    """Risk indicator found in code"""

    pattern_type: str
    matched_text: str
    file_path: str
    line_number: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)
    description: str


class ControlMapping(BaseModel):
    """Mapping between code changes and compliance controls"""

    control: ComplianceControl
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_indicators: List[RiskIndicator]


class PRAnalysis(BaseModel):
    """Complete analysis of a pull request"""

    pr_id: str
    title: str
    pr_url: str
    description: Optional[str] = None
    author: str
    files_changed: List[str]
    additions: int
    deletions: int
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_indicators: List[RiskIndicator]
    control_mappings: List[ControlMapping]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvidenceBundle(BaseModel):
    """Generated evidence bundle for audit"""

    bundle_id: str
    pr_analysis: PRAnalysis
    executive_summary: str
    technical_details: str
    control_impact_assessment: str
    risk_mitigation_plan: str
    audit_trail: List[Dict[str, Any]]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ApprovalRequest(BaseModel):
    """Request for human approval"""

    request_id: str
    pr_analysis: PRAnalysis
    evidence_bundle: EvidenceBundle
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    comments: Optional[str] = None


# SQLAlchemy models for db
class PRAnalysisDB(Base):
    __tablename__ = "pr_analysis"

    id = Column(Integer, primary_key=True, index=True)
    pr_id = Column(String, unique=True, index=True)
    pr_url = Column(String)
    title = Column(String)
    description = Column(Text)
    author = Column(String)
    files_changed = Column(JSON)
    additions = Column(Integer)
    deletions = Column(Integer)
    risk_level = Column(String)
    risk_score = Column(Float)
    risk_indicators = Column(JSON)
    control_mappings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvidenceBundleDB(Base):
    __tablename__ = "evidence_bundles"

    id = Column(Integer, primary_key=True, index=True)
    bundle_id = Column(String, unique=True, index=True)
    pr_id = Column(String, index=True)
    executive_summary = Column(Text)
    technical_details = Column(Text)
    control_impact_assessment = Column(Text)
    risk_mitigation_plan = Column(Text)
    audit_trail = Column(JSON)
    generated_at = Column(DateTime, default=datetime.utcnow)


class ApprovalRequestDB(Base):
    __tablename__ = "approval_requests"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True)
    pr_id = Column(String, index=True)
    bundle_id = Column(String, index=True)
    status = Column(String, default=ApprovalStatus.PENDING)
    requested_at = Column(DateTime, default=datetime.utcnow)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    comments = Column(Text, nullable=True)


class ComplianceControlDB(Base):
    __tablename__ = "compliance_controls"

    id = Column(Integer, primary_key=True, index=True)
    framework = Column(String, index=True)
    control_id = Column(String, index=True)
    title = Column(String)
    description = Column(Text)
    risk_level = Column(String)
    category = Column(String)


# Database setup
def create_database(database_url: str):
    """Create the database and tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return engine


def get_session(database_url: str):
    """Get SQLAlchemy session"""
    engine = create_database(database_url)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
