"""Golden dataset: 20 annotated code review examples for regression testing.

Золотой датасет: 20 аннотированных примеров с ground truth для регрессионного
тестирования качества LLM code review. Разбит на три домена:
- security (8 примеров): SQL injection, XSS, hardcoded secrets, etc.
- correctness (8 примеров): off-by-one, null deref, race condition, etc.
- clean (4 примера): корректный код без реальных проблем (тест FPR).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenExample:
    """Аннотированный пример для оценки качества code review.

    Annotated ground-truth example: diff + known issues + expected keywords
    that should appear in a high-quality review comment.
    """

    id: str
    diff: str
    domain: str  # "security" | "correctness" | "clean"
    ground_truth_issues: list[dict] = field(default_factory=list)
    is_clean: bool = False
    description: str = ""


def get_golden_dataset() -> list[GoldenExample]:
    """Return the curated golden dataset of 20 annotated examples."""
    return list(_GOLDEN_DATASET)


_GOLDEN_DATASET: list[GoldenExample] = [
    # ── Security: 8 examples ──────────────────────────────────────────────
    GoldenExample(
        id="sec_001_sqli",
        domain="security",
        description="SQL injection via f-string formatting",
        diff="""\
--- a/db.py
+++ b/db.py
@@ -10,6 +10,7 @@
 def get_user(conn, username: str) -> dict | None:
-    row = conn.execute("SELECT * FROM users WHERE name = ?", (username,)).fetchone()
+    query = f"SELECT * FROM users WHERE name = '{username}'"
+    row = conn.execute(query).fetchone()
     return dict(row) if row else None""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "sql injection",
                    "sqli",
                    "f-string",
                    "parameterized",
                    "prepared statement",
                    "injection",
                ],
            }
        ],
    ),
    GoldenExample(
        id="sec_002_hardcoded_secret",
        domain="security",
        description="Hardcoded credentials in source code",
        diff="""\
--- a/config.py
+++ b/config.py
@@ -1,3 +1,5 @@
+DB_PASSWORD = "sup3r_s3cr3t_passw0rd"
+JWT_SECRET = "hardcoded-jwt-signing-key-do-not-use"
 DEBUG = True
 DATABASE_URL = "sqlite:///./test.db\"""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "hardcoded",
                    "secret",
                    "credential",
                    "environment",
                    "vault",
                    "rotate",
                    "password",
                ],
            }
        ],
    ),
    GoldenExample(
        id="sec_003_path_traversal",
        domain="security",
        description="Path traversal via unsanitized user input",
        diff="""\
--- a/files.py
+++ b/files.py
@@ -5,6 +5,4 @@
 def read_file(filename: str) -> str:
-    base = Path("/srv/uploads")
-    safe = base / filename
-    safe.resolve().relative_to(base)
-    return safe.read_text()
+    return open(filename).read()""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "path traversal",
                    "directory traversal",
                    "sanitize",
                    "resolve",
                    "arbitrary file",
                ],
            }
        ],
    ),
    GoldenExample(
        id="sec_004_cmd_injection",
        domain="security",
        description="Command injection via subprocess with shell=True",
        diff="""\
--- a/runner.py
+++ b/runner.py
@@ -3,5 +3,5 @@
 import subprocess
 def run_test(test_name: str) -> str:
-    result = subprocess.run(["pytest", test_name], capture_output=True, text=True)
+    result = subprocess.run(f"pytest {test_name}", shell=True, capture_output=True, text=True)
     return result.stdout""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "command injection",
                    "shell=true",
                    "shell injection",
                    "subprocess",
                    "shlex",
                ],
            }
        ],
    ),
    GoldenExample(
        id="sec_005_xss",
        domain="security",
        description="XSS via unescaped user input in HTML template",
        diff="""\
--- a/views.py
+++ b/views.py
@@ -8,5 +8,4 @@
 def render_profile(username: str) -> str:
-    safe = html.escape(username)
-    return f"<h1>Hello, {safe}</h1>"
+    return f"<h1>Hello, {username}</h1>\"""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "xss",
                    "cross-site scripting",
                    "escape",
                    "sanitize",
                    "html.escape",
                    "injection",
                ],
            }
        ],
    ),
    GoldenExample(
        id="sec_006_weak_crypto",
        domain="security",
        description="Weak MD5 hashing for passwords",
        diff="""\
--- a/auth.py
+++ b/auth.py
@@ -5,6 +5,5 @@
 import hashlib
 def hash_password(password: str) -> str:
-    salt = os.urandom(16)
-    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000).hex()
+    return hashlib.md5(password.encode()).hexdigest()""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "md5",
                    "weak",
                    "hash",
                    "bcrypt",
                    "argon2",
                    "pbkdf2",
                    "salt",
                    "password",
                ],
            }
        ],
    ),
    GoldenExample(
        id="sec_007_ssrf",
        domain="security",
        description="SSRF via unvalidated URL parameter",
        diff="""\
--- a/proxy.py
+++ b/proxy.py
@@ -4,4 +4,7 @@
 import requests
+def fetch_url(url: str) -> bytes:
+    resp = requests.get(url, timeout=10)
+    return resp.content
+
 def health() -> dict:
     return {"status": "ok"}""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "major",
                "keywords": [
                    "ssrf",
                    "server-side request forgery",
                    "validate",
                    "allowlist",
                    "internal",
                    "metadata",
                ],
            }
        ],
    ),
    GoldenExample(
        id="sec_008_insecure_deserialize",
        domain="security",
        description="Insecure pickle deserialization from untrusted input",
        diff="""\
--- a/cache.py
+++ b/cache.py
@@ -3,4 +3,7 @@
 import pickle
+def load_model(data: bytes):
+    return pickle.loads(data)
+
 def save_model(model, path: str) -> None:
     with open(path, "wb") as f:
         pickle.dump(model, f)""",
        ground_truth_issues=[
            {
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "pickle",
                    "deserialization",
                    "arbitrary code",
                    "untrusted",
                    "rce",
                    "remote code execution",
                ],
            }
        ],
    ),
    # ── Correctness: 8 examples ─────────────────────────────────────────────
    GoldenExample(
        id="bug_001_off_by_one",
        domain="correctness",
        description="Off-by-one error in loop boundary",
        diff="""\
--- a/process.py
+++ b/process.py
@@ -6,5 +6,5 @@
 def process_items(items: list) -> list:
     results = []
-    for i in range(len(items)):
+    for i in range(len(items) + 1):
         results.append(transform(items[i]))
     return results""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "critical",
                "keywords": ["off-by-one", "index", "out of range", "indexerror", "boundary"],
            }
        ],
    ),
    GoldenExample(
        id="bug_002_null_deref",
        domain="correctness",
        description="Null dereference without None check",
        diff="""\
--- a/user.py
+++ b/user.py
@@ -8,7 +8,5 @@
 def get_user_email(user_id: int) -> str:
     user = db.find_user(user_id)
-    if user is None:
-        return ""
     return user.email""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "critical",
                "keywords": ["none", "null", "attributeerror", "dereference", "check", "guard"],
            }
        ],
    ),
    GoldenExample(
        id="bug_003_mutable_default",
        domain="correctness",
        description="Mutable default argument bug in Python",
        diff="""\
--- a/utils.py
+++ b/utils.py
@@ -2,6 +2,4 @@
-def append_item(item: str, lst: list | None = None) -> list:
-    if lst is None:
-        lst = []
+def append_item(item: str, lst: list = []) -> list:
     lst.append(item)
     return lst""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "mutable default",
                    "shared state",
                    "default argument",
                    "list",
                    "persists",
                ],
            }
        ],
    ),
    GoldenExample(
        id="bug_004_race_condition",
        domain="correctness",
        description="Race condition: removed thread lock",
        diff="""\
--- a/counter.py
+++ b/counter.py
@@ -5,9 +5,6 @@
 class Counter:
     def __init__(self):
-        self._lock = threading.Lock()
         self._value = 0

     def increment(self):
-        with self._lock:
-            self._value += 1
+        self._value += 1""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "critical",
                "keywords": [
                    "race condition",
                    "thread",
                    "lock",
                    "concurrent",
                    "atomic",
                    "synchronize",
                ],
            }
        ],
    ),
    GoldenExample(
        id="bug_005_exception_swallowed",
        domain="correctness",
        description="Exception silently swallowed — error masked",
        diff="""\
--- a/io.py
+++ b/io.py
@@ -3,8 +3,7 @@
 def read_config(path: str) -> dict:
     try:
         with open(path) as f:
             return json.load(f)
-    except Exception as e:
-        logger.error("Config load failed: %s", e)
-        raise
+    except Exception:
+        pass""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "exception",
                    "silenced",
                    "swallowed",
                    "masked",
                    "pass",
                    "bare except",
                    "error handling",
                ],
            }
        ],
    ),
    GoldenExample(
        id="bug_006_integer_division",
        domain="correctness",
        description="Integer division truncation in Python 3 average",
        diff="""\
--- a/stats.py
+++ b/stats.py
@@ -2,4 +2,4 @@
 def average(numbers: list[int]) -> float:
-    return sum(numbers) / len(numbers)
+    return sum(numbers) // len(numbers)""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "integer division",
                    "truncation",
                    "//",
                    "floor division",
                    "precision",
                ],
            }
        ],
    ),
    GoldenExample(
        id="bug_007_memory_leak",
        domain="correctness",
        description="File handle not closed — resource leak",
        diff="""\
--- a/reader.py
+++ b/reader.py
@@ -2,5 +2,4 @@
 def read_all(path: str) -> str:
-    with open(path) as f:
-        return f.read()
+    f = open(path)
+    return f.read()""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "resource leak",
                    "file handle",
                    "close",
                    "context manager",
                    "with",
                    "finally",
                ],
            }
        ],
    ),
    GoldenExample(
        id="bug_008_wrong_comparison",
        domain="correctness",
        description="Identity check instead of equality for string comparison",
        diff="""\
--- a/validate.py
+++ b/validate.py
@@ -4,4 +4,4 @@
 def is_admin(user: dict) -> bool:
-    return user.get("role") == "admin"
+    return user.get("role") is "admin\"""",
        ground_truth_issues=[
            {
                "category": "bug",
                "severity": "major",
                "keywords": ["is", "==", "identity", "equality", "string", "interning"],
            }
        ],
    ),
    # ── Clean code: 4 examples ────────────────────────────────────────────
    GoldenExample(
        id="clean_001_refactor",
        domain="clean",
        description="Simple clean refactor — no real issues",
        is_clean=True,
        diff="""\
--- a/utils.py
+++ b/utils.py
@@ -5,7 +5,5 @@
-def double(x: int) -> int:
-    result = x * 2
-    return result
+def double(x: int) -> int:
+    return x * 2""",
        ground_truth_issues=[],
    ),
    GoldenExample(
        id="clean_002_docstring",
        domain="clean",
        description="Docstring addition — no functional change",
        is_clean=True,
        diff="""\
--- a/math_utils.py
+++ b/math_utils.py
@@ -1,4 +1,8 @@
 def clamp(value: float, low: float, high: float) -> float:
+    \"\"\"Clamp value to [low, high] range.
+
+    Clips the input to ensure it falls within [low, high].
+    \"\"\"
     return max(low, min(high, value))""",
        ground_truth_issues=[],
    ),
    GoldenExample(
        id="clean_003_type_hint",
        domain="clean",
        description="Type annotation improvement — no real issues",
        is_clean=True,
        diff="""\
--- a/api.py
+++ b/api.py
@@ -1,4 +1,4 @@
-def process(items):
+def process(items: list[str]) -> list[str]:
     return [item.strip() for item in items if item]""",
        ground_truth_issues=[],
    ),
    GoldenExample(
        id="clean_004_logging",
        domain="clean",
        description="Adding structured logging — correct implementation",
        is_clean=True,
        diff="""\
--- a/service.py
+++ b/service.py
@@ -5,5 +5,9 @@
+import logging
+logger = logging.getLogger(__name__)
+
 def process_request(request_id: str, payload: dict) -> dict:
+    logger.info("Processing request", extra={"request_id": request_id})
     result = _handle(payload)
+    logger.info("Request completed", extra={"request_id": request_id, "success": True})
     return result""",
        ground_truth_issues=[],
    ),
]
