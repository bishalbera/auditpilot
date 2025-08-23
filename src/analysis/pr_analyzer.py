from typing import List, Dict, Tuple
from github import Github
from src.models import PRAnalysis, RiskIndicator, RiskLevel
from src.config import RISK_PATTERNS, settings


class PRAnalyzer:
    """Analyzes Pull Requests for compliance risks"""

    def __init__(self, github_token: str = None):
        self.github = (
            Github(github_token or settings.github_token)
            if (github_token or settings.github_token)
            else None
        )
        self.risk_patterns = RISK_PATTERNS

    def analyze_pr(self, repo_name: str, pr_number: int) -> PRAnalysis:
        """Main method to analyze a Pull Request (synchronous).
        Falls back to a synthetic demo analysis when GitHub is unavailable or the repo is a known demo repo.
        """
        # Demo repos we intentionally synthesize
        demo_repos = {
            "acme-corp/user-service": "authentication",
            "acme-corp/data-service": "data_access",
            "acme-corp/crypto-service": "encryption",
        }

        # If no GitHub token configured or this is a demo repo, use demo analysis
        use_demo = (self.github is None) or (repo_name in demo_repos)

        if not use_demo:
            try:
                repo = self.github.get_repo(repo_name)
                pr = repo.get_pull(pr_number)

                # Analyze changed files
                files_analysis = self._analyze_changed_files(pr)

                # Detect risk indicators
                risk_indicators = self._detect_risk_indicators(files_analysis, pr)

                # Calculate risk score and level
                risk_score, risk_level = self._calculate_risk_score(risk_indicators)

                return PRAnalysis(
                    pr_id=f"{repo_name}#{pr_number}",
                    pr_url=pr.html_url,
                    title=pr.title,
                    description=pr.body,
                    author=pr.user.login,
                    files_changed=[f["filename"] for f in files_analysis],
                    additions=pr.additions,
                    deletions=pr.deletions,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    risk_indicators=risk_indicators,
                    control_mappings=[],  # Will be populated by ControlMapper
                )
            except Exception:
                # Fallback to demo if GitHub lookups fail (e.g., 404 for non-existent repo)
                use_demo = True

        if use_demo:
            theme = demo_repos.get(repo_name, "authentication")
            return self._analyze_demo(repo_name, pr_number, theme)

    def _extract_pr_data(self, pr) -> Dict:
        """Extract basic data from PR"""
        return {
            "title": pr.title,
            "body": pr.body or "",
            "author": pr.user.login,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "commits": pr.commits,
        }

    def _analyze_demo(self, repo_name: str, pr_number: int, theme: str) -> PRAnalysis:
        """Produce a synthetic PRAnalysis for demo scenarios without calling GitHub."""
        # Fabricate a PR-like object
        title_map = {
            "authentication": "Refactor auth middleware to support MFA",
            "data_access": "Introduce repository layer and tighten DB queries",
            "encryption": "Upgrade TLS config and rotate encryption keys",
        }
        title = title_map.get(theme, "Update service components and configs")
        author = "demo-bot"
        html_url = f"https://github.com/{repo_name}/pull/{pr_number}"

        # Create synthetic changed files with patches including keywords
        if theme == "authentication":
            files = [
                {
                    "filename": "src/auth/middleware.py",
                    "patch": "+ def verify_token(token):\n+    # TODO: add MFA\n",
                },
                {
                    "filename": "src/routes/login.py",
                    "patch": "+ return create_session(user)\n",
                },
            ]
        elif theme == "data_access":
            files = [
                {
                    "filename": "src/db/repository.py",
                    "patch": "+ def query_users():\n+    return db.execute('SELECT * FROM users')\n",
                },
                {
                    "filename": "src/services/user_service.py",
                    "patch": "+ user = repo.get_user(id)\n",
                },
            ]
        else:  # encryption
            files = [
                {
                    "filename": "src/crypto/ssl_config.py",
                    "patch": "+ context.set_ciphers('TLS_AES_256_GCM_SHA384')\n",
                },
                {
                    "filename": "src/crypto/keys.py",
                    "patch": "+ def rotate_keys():\n+    # rotate encryption keys\n",
                },
            ]

        # Convert to RiskIndicators by reusing the code paths
        class _PseudoPR:
            def __init__(
                self, title, body, user_login, additions=42, deletions=7, html_url=""
            ):
                self.title = title
                self.body = body
                self.user = type("U", (), {"login": user_login})
                self.additions = additions
                self.deletions = deletions
                self.html_url = html_url

            def get_files(self):
                class F:
                    def __init__(self, filename, patch):
                        self.filename = filename
                        self.status = "modified"
                        self.additions = 10
                        self.deletions = 2
                        self.changes = 12
                        self.patch = patch

                return [F(f["filename"], f.get("patch", "")) for f in files]

            def get_commits(self):
                class C:
                    def __init__(self, msg):
                        self.commit = type("CM", (), {"message": msg})

                msgs = {
                    "authentication": ["add auth check", "fix login flow"],
                    "data_access": ["optimize SQL query", "add repository pattern"],
                    "encryption": ["update TLS", "rotate keys"],
                }[theme]
                return [C(m) for m in msgs]

        pseudo_pr = _PseudoPR(title, f"Demo PR for {theme}", author, html_url=html_url)
        files_analysis = [
            {
                "filename": f["filename"],
                "status": "modified",
                "additions": 10,
                "deletions": 2,
                "changes": 12,
                "patch": f.get("patch", ""),
                "contents": f.get("patch", ""),
            }
            for f in files
        ]

        risk_indicators = []
        risk_indicators.extend(self._analyze_file_paths(files_analysis))
        risk_indicators.extend(self._analyze_code_content(files_analysis))
        risk_indicators.extend(self._analyze_commit_messages(pseudo_pr))
        risk_indicators.extend(self._analyze_pr_text(pseudo_pr))

        risk_score, risk_level = self._calculate_risk_score(risk_indicators)

        return PRAnalysis(
            pr_id=f"{repo_name}#{pr_number}",
            pr_url=html_url,
            title=title,
            description=f"Demo analysis for {theme}",
            author=author,
            files_changed=[f["filename"] for f in files],
            additions=42,
            deletions=7,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_indicators=risk_indicators,
            control_mappings=[],
        )

    def _analyze_changed_files(self, pr) -> List[Dict]:
        """Analyze all changed files in the PR"""
        files_analysis = []

        for file in pr.get_files():
            file_analysis = {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch if hasattr(file, "patch") else "",
                "contents": self._get_file_contents(file),
            }
            files_analysis.append(file_analysis)

        return files_analysis

    def _get_file_contents(self, file) -> str:
        """Get file contents for analysis"""
        try:
            if hasattr(file, "patch") and file.patch:
                return file.patch
            return ""
        except Exception as e:
            print(f"Error getting file contents: {e}")
            return ""

    def _detect_risk_indicators(
        self, files_analysis: List[Dict], pr
    ) -> List[RiskIndicator]:
        """Detect risk indicators in the PR"""
        risk_indicators = []

        # Analyze file names and paths
        risk_indicators.extend(self._analyze_file_paths(files_analysis))

        # Analyze code content
        risk_indicators.extend(self._analyze_code_content(files_analysis))

        # Analyze commit messages
        risk_indicators.extend(self._analyze_commit_messages(pr))

        # Analyze PR title and description
        risk_indicators.extend(self._analyze_pr_text(pr))

        return risk_indicators

    def _analyze_file_paths(self, files_analysis: List[Dict]) -> List[RiskIndicator]:
        """Analyze file paths for risk patterns"""
        indicators = []

        for file_data in files_analysis:
            filename = file_data["filename"].lower()

            for pattern_type, pattern_config in self.risk_patterns.items():
                for file_pattern in pattern_config["file_patterns"]:
                    if file_pattern in filename:
                        indicators.append(
                            RiskIndicator(
                                pattern_type=pattern_type,
                                matched_text=f"File path contains '{file_pattern}'",
                                file_path=file_data["filename"],
                                confidence=0.7,
                                description=f"File path indicates {pattern_type} related changes",
                            )
                        )

        return indicators

    def _analyze_code_content(self, files_analysis: List[Dict]) -> List[RiskIndicator]:
        """Analyze code content for risk patterns"""
        indicators = []

        for file_data in files_analysis:
            content = file_data.get("contents", "") + file_data.get("patch", "")
            if not content:
                continue

            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()

                for pattern_type, pattern_config in self.risk_patterns.items():
                    for keyword in pattern_config["keywords"]:
                        if keyword in line_lower:
                            # Calculate confidence based on context
                            confidence = self._calculate_keyword_confidence(
                                line, keyword, pattern_type
                            )

                            if confidence > 0.3:  # Minimum threshold
                                indicators.append(
                                    RiskIndicator(
                                        pattern_type=pattern_type,
                                        matched_text=(
                                            line.strip()[:100] + "..."
                                            if len(line) > 100
                                            else line.strip()
                                        ),
                                        file_path=file_data["filename"],
                                        line_number=line_num,
                                        confidence=confidence,
                                        description=f"Code contains {pattern_type} keyword: '{keyword}'",
                                    )
                                )

        return indicators

    def _calculate_keyword_confidence(
        self, line: str, keyword: str, pattern_type: str
    ) -> float:
        """Calculate confidence score for keyword matches"""
        base_confidence = 0.5

        # Boost confidence for function/method definitions
        if any(
            token in line.lower() for token in ["def ", "function ", "class ", "method"]
        ):
            base_confidence += 0.2

        # Boost confidence for imports/declarations
        if any(
            token in line.lower() for token in ["import", "from", "require", "include"]
        ):
            base_confidence += 0.3

        # Context-specific adjustments
        context_boosts = {
            "authentication": ["login", "signin", "auth", "token", "session"],
            "encryption": ["encrypt", "decrypt", "hash", "crypto", "ssl"],
            "data_access": ["database", "query", "select", "insert", "update"],
        }

        if pattern_type in context_boosts:
            for context_word in context_boosts[pattern_type]:
                if context_word in line.lower() and context_word != keyword:
                    base_confidence += 0.1

        return min(1.0, max(0.0, base_confidence))

    def _analyze_commit_messages(self, pr) -> List[RiskIndicator]:
        """Analyze commit messages for risk indicators"""
        indicators = []

        for commit in pr.get_commits():
            message = commit.commit.message.lower()

            for pattern_type, pattern_config in self.risk_patterns.items():
                for keyword in pattern_config["keywords"]:
                    if keyword in message:
                        indicators.append(
                            RiskIndicator(
                                pattern_type=pattern_type,
                                matched_text=(
                                    commit.commit.message[:100] + "..."
                                    if len(commit.commit.message) > 100
                                    else commit.commit.message
                                ),
                                file_path="commit_message",
                                confidence=0.6,
                                description=f"Commit message mentions {pattern_type}: '{keyword}'",
                            )
                        )

        return indicators

    def _analyze_pr_text(self, pr) -> List[RiskIndicator]:
        """Analyze PR title and description"""
        indicators = []
        text_content = f"{pr.title} {pr.body or ''}".lower()

        for pattern_type, pattern_config in self.risk_patterns.items():
            for keyword in pattern_config["keywords"]:
                if keyword in text_content:
                    indicators.append(
                        RiskIndicator(
                            pattern_type=pattern_type,
                            matched_text=f"PR text mentions: '{keyword}'",
                            file_path="pr_description",
                            confidence=0.5,
                            description=f"PR title/description mentions {pattern_type}",
                        )
                    )

        return indicators

    def _calculate_risk_score(
        self, risk_indicators: List[RiskIndicator]
    ) -> Tuple[float, RiskLevel]:
        """Calculate overall risk score and level"""
        if not risk_indicators:
            return 0.0, RiskLevel.LOW

        # Weight indicators by confidence and pattern type
        weighted_score = 0.0
        total_weight = 0.0

        pattern_weights = {
            "authentication": 1.0,
            "data_access": 0.9,
            "encryption": 0.9,
            "access_control": 0.8,
            "data_processing": 0.8,
            "api_security": 0.6,
            "logging": 0.4,
        }

        for indicator in risk_indicators:
            weight = pattern_weights.get(indicator.pattern_type, 0.5)
            weighted_score += indicator.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0, RiskLevel.LOW

        final_score = min(1.0, weighted_score / total_weight)

        # Determine risk level
        if final_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif final_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif final_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return final_score, risk_level
