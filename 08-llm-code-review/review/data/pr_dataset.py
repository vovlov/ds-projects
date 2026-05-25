"""Synthetic PR dataset for LoRA fine-tuning demonstration.

Генерирует реалистичные GitHub PR дифф примеры для демонстрации
domain-specific fine-tuning (LoRA). Охватывает Python, JavaScript,
SQL, YAML — типичный стек ML-инженера.

Sources:
- GitHub Code Review Best Practices 2026
- OWASP Top 10 Web Application Security Risks 2021
- Google Engineering Practices (code review standards)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PRExample:
    """Аннотированный пример PR для fine-tuning.

    Annotated PR example: diff + category label + domain metadata.
    """

    id: str
    diff: str
    category: str  # bug / security / performance / style / documentation
    domain: str  # python / javascript / sql / yaml / generic
    severity: str  # critical / major / minor
    review_comment: str


def get_pr_dataset() -> list[PRExample]:
    """Возвращает 32 аннотированных PR примера для fine-tuning.

    Returns 32 annotated PR examples across 5 categories and 4 languages.
    Designed to simulate the distribution found in real open-source repos.
    """
    return (
        _SECURITY_EXAMPLES + _BUG_EXAMPLES + _PERFORMANCE_EXAMPLES + _STYLE_EXAMPLES + _DOC_EXAMPLES
    )


def get_pr_dataset_by_category(category: str) -> list[PRExample]:
    """Фильтрует датасет по категории для domain-specific fine-tuning."""
    return [ex for ex in get_pr_dataset() if ex.category == category]


def get_pr_stats() -> dict:
    """Сводная статистика датасета.

    Summary statistics for the PR dataset.
    """
    examples = get_pr_dataset()
    by_category: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for ex in examples:
        by_category[ex.category] = by_category.get(ex.category, 0) + 1
        by_domain[ex.domain] = by_domain.get(ex.domain, 0) + 1
        by_severity[ex.severity] = by_severity.get(ex.severity, 0) + 1
    return {
        "total": len(examples),
        "by_category": by_category,
        "by_domain": by_domain,
        "by_severity": by_severity,
        "categories": sorted(by_category.keys()),
    }


# ── Security examples (8) ─────────────────────────────────────────────────────

_SECURITY_EXAMPLES: list[PRExample] = [
    PRExample(
        id="sec-001",
        diff=(
            "--- a/app/views.py\n+++ b/app/views.py\n"
            "@@ -14,7 +14,7 @@\n"
            " def get_user(request):\n"
            "-    uid = int(request.GET['id'])\n"
            "+    uid = request.GET['id']\n"
            "     user = User.objects.raw(f'SELECT * FROM users WHERE id = {uid}')\n"
            "     return JsonResponse({'user': user[0].to_dict()})\n"
        ),
        category="security",
        domain="python",
        severity="critical",
        review_comment=(
            "SQL injection: the raw f-string query allows an attacker to inject arbitrary SQL. "
            "Use parameterized queries: `User.objects.raw('SELECT * FROM users WHERE id = %s', [uid])`. "  # noqa: E501
            "Also validate that `id` is an integer before using it."
        ),
    ),
    PRExample(
        id="sec-002",
        diff=(
            "--- a/src/routes/files.js\n+++ b/src/routes/files.js\n"
            "@@ -8,5 +8,6 @@\n"
            " router.get('/download', (req, res) => {\n"
            "+    const filePath = path.join(__dirname, req.query.name);\n"
            "+    res.sendFile(filePath);\n"
            " });\n"
        ),
        category="security",
        domain="javascript",
        severity="critical",
        review_comment=(
            "Path traversal vulnerability: `req.query.name` can contain `../../etc/passwd`. "
            "Sanitize with `path.basename()` and validate the resolved path stays within "
            "the allowed directory. Consider a whitelist of allowed filenames."
        ),
    ),
    PRExample(
        id="sec-003",
        diff=(
            "--- a/config/settings.py\n+++ b/config/settings.py\n"
            "@@ -5,4 +5,5 @@\n"
            " SECRET_KEY = os.environ.get('SECRET_KEY', 'changeme')\n"
            "+DB_PASSWORD = 'prod_p@ssw0rd_2024'\n"
            "+AWS_SECRET_KEY = 'AKIAIOSFODNN7EXAMPLE'\n"
        ),
        category="security",
        domain="python",
        severity="critical",
        review_comment=(
            "Hardcoded credentials committed to source control. "
            "Rotate both secrets immediately. Use `os.environ.get('DB_PASSWORD')` "
            "and store secrets in a vault (HashiCorp Vault, AWS Secrets Manager). "
            "Add `*.env` and `secrets.py` to `.gitignore`."
        ),
    ),
    PRExample(
        id="sec-004",
        diff=(
            "--- a/src/template.html\n+++ b/src/template.html\n"
            "@@ -12,3 +12,3 @@\n"
            "-<p>Hello {{ user.name | e }}</p>\n"
            "+<p>Hello {{ user.name }}</p>\n"
            "+<script>var msg = '{{ message }}';</script>\n"
        ),
        category="security",
        domain="javascript",
        severity="critical",
        review_comment=(
            "XSS vulnerability: the `| e` (escape) filter was removed from `user.name`. "
            "Restore `| e` on all user-controlled template variables. "
            "The new `<script>` block injects `message` without escaping — use `| tojson` "
            "to safely embed data in JavaScript contexts."
        ),
    ),
    PRExample(
        id="sec-005",
        diff=(
            "--- a/api/upload.py\n+++ b/api/upload.py\n"
            "@@ -10,6 +10,8 @@\n"
            " @app.post('/upload')\n"
            " async def upload(file: UploadFile):\n"
            "+    content = await file.read()\n"
            "+    subprocess.run(['convert', file.filename, 'output.png'])\n"
            "     return {'status': 'ok'}\n"
        ),
        category="security",
        domain="python",
        severity="critical",
        review_comment=(
            "Command injection: `file.filename` is passed directly to `subprocess.run` "
            "without sanitization. An attacker can inject `;rm -rf /`. "
            "Use a generated UUID filename, validate mime type before conversion, "
            "and pass arguments as a list (already done) but also sanitize the filename."
        ),
    ),
    PRExample(
        id="sec-006",
        diff=(
            "--- a/auth/tokens.py\n+++ b/auth/tokens.py\n"
            "@@ -8,5 +8,5 @@\n"
            " import hashlib\n"
            "-token = secrets.token_hex(32)\n"
            "+token = hashlib.md5(username.encode()).hexdigest()\n"
        ),
        category="security",
        domain="python",
        severity="critical",
        review_comment=(
            "Cryptographic weakness: MD5 is broken for security purposes and deterministic "
            "(same username → same token). Revert to `secrets.token_hex(32)` which generates "
            "a cryptographically secure random 256-bit token. Never use MD5/SHA1 for tokens."
        ),
    ),
    PRExample(
        id="sec-007",
        diff=(
            "--- a/src/auth/jwt.js\n+++ b/src/auth/jwt.js\n"
            "@@ -5,5 +5,5 @@\n"
            "-const token = jwt.verify(req.headers.authorization, SECRET);\n"
            "+const token = jwt.decode(req.headers.authorization);\n"
            " req.user = token.payload;\n"
        ),
        category="security",
        domain="javascript",
        severity="critical",
        review_comment=(
            "`jwt.decode()` does not verify the signature — any attacker can forge tokens "
            "by crafting a payload with `alg: none`. Revert to `jwt.verify(token, SECRET)`. "
            "This is an authentication bypass vulnerability."
        ),
    ),
    PRExample(
        id="sec-008",
        diff=(
            "--- a/infra/k8s/deployment.yaml\n+++ b/infra/k8s/deployment.yaml\n"
            "@@ -18,3 +18,5 @@\n"
            " spec:\n"
            "   containers:\n"
            "+    - securityContext:\n"
            "+        privileged: true\n"
            "+        runAsRoot: true\n"
        ),
        category="security",
        domain="yaml",
        severity="critical",
        review_comment=(
            "Container runs as root with `privileged: true` — a container escape gives full "
            "host access. Set `runAsNonRoot: true`, `runAsUser: 1000`, "
            "`allowPrivilegeEscalation: false`, and `readOnlyRootFilesystem: true`. "
            "Privileged mode is almost never needed in application containers."
        ),
    ),
]

# ── Bug examples (10) ─────────────────────────────────────────────────────────

_BUG_EXAMPLES: list[PRExample] = [
    PRExample(
        id="bug-001",
        diff=(
            "--- a/src/billing.py\n+++ b/src/billing.py\n"
            "@@ -22,6 +22,6 @@\n"
            " def charge_customer(amount: float, currency: str) -> Receipt:\n"
            "     if amount < 0:\n"
            "-        raise ValueError('Amount must be positive')\n"
            "+        return None\n"
            "     return _process_charge(amount, currency)\n"
        ),
        category="bug",
        domain="python",
        severity="major",
        review_comment=(
            "Silently returning `None` instead of raising hides billing errors. "
            "Callers expecting a `Receipt` will get `AttributeError` downstream. "
            "Keep the `ValueError` or return a typed `Optional[Receipt]` and document "
            "that `None` signals rejection."
        ),
    ),
    PRExample(
        id="bug-002",
        diff=(
            "--- a/src/cache.py\n+++ b/src/cache.py\n"
            "@@ -18,6 +18,6 @@\n"
            " def get_cached(key: str) -> dict | None:\n"
            "     result = cache.get(key)\n"
            "-    if result is None:\n"
            "+    if not result:\n"
            "         return None\n"
            "     return json.loads(result)\n"
        ),
        category="bug",
        domain="python",
        severity="major",
        review_comment=(
            "`not result` treats empty string `''`, `0`, `False`, and `{}` as cache misses. "
            "For Redis, an empty string is a valid cached value. Keep the strict `is None` check."
        ),
    ),
    PRExample(
        id="bug-003",
        diff=(
            "--- a/services/order.js\n+++ b/services/order.js\n"
            "@@ -30,6 +30,7 @@\n"
            " async function processOrders(orders) {\n"
            "-  for (const order of orders) {\n"
            "-    await processOne(order);\n"
            "-  }\n"
            "+  orders.forEach(order => processOne(order));\n"
            " }\n"
        ),
        category="bug",
        domain="javascript",
        severity="major",
        review_comment=(
            "`forEach` ignores the returned Promise — errors in `processOne` are silently "
            "swallowed and execution proceeds without waiting. Use `for...of` with `await` "
            "or `await Promise.all(orders.map(processOne))` for parallel processing."
        ),
    ),
    PRExample(
        id="bug-004",
        diff=(
            "--- a/models/user.py\n+++ b/models/user.py\n"
            "@@ -45,5 +45,5 @@\n"
            " class User:\n"
            "     def deactivate(self) -> None:\n"
            "         self.is_active = False\n"
            "-        db.session.commit()\n"
            "+        db.session.flush()\n"
        ),
        category="bug",
        domain="python",
        severity="major",
        review_comment=(
            "`flush()` sends SQL to the DB but does NOT commit the transaction. "
            "If the request fails after this point, the change will be rolled back. "
            "Keep `commit()` to persist the deactivation, or document why flush-only "
            "is intentional (e.g., in a larger transaction unit of work)."
        ),
    ),
    PRExample(
        id="bug-005",
        diff=(
            "--- a/src/retry.py\n+++ b/src/retry.py\n"
            "@@ -12,6 +12,6 @@\n"
            " def retry(fn, max_attempts=3):\n"
            "     for attempt in range(max_attempts):\n"
            "         try:\n"
            "             return fn()\n"
            "-        except Exception:\n"
            "+        except Exception as e:\n"
            "+            if attempt == max_attempts:\n"
            "                 pass\n"
            "     raise RuntimeError('Max retries exceeded')\n"
        ),
        category="bug",
        domain="python",
        severity="major",
        review_comment=(
            "Off-by-one: `range(max_attempts)` yields 0, 1, 2 — so `attempt` never equals "
            "`max_attempts` (3). The condition is always False. Use "
            "`attempt == max_attempts - 1` or restructure with `else` on the for loop."
        ),
    ),
    PRExample(
        id="bug-006",
        diff=(
            "--- a/db/migrations/0012.sql\n+++ b/db/migrations/0012.sql\n"
            "@@ -1,4 +1,6 @@\n"
            " ALTER TABLE orders ADD COLUMN discount_pct NUMERIC(5,2);\n"
            "+UPDATE orders SET discount_pct = 0;\n"
            "+ALTER TABLE orders ALTER COLUMN discount_pct SET NOT NULL;\n"
        ),
        category="bug",
        domain="sql",
        severity="major",
        review_comment=(
            "Race condition in migration: between `UPDATE` and `ALTER ... NOT NULL`, "
            "concurrent inserts can create rows with NULL `discount_pct`. "
            "Set the DEFAULT inline: `ADD COLUMN discount_pct NUMERIC(5,2) DEFAULT 0 NOT NULL`. "
            "This is atomic and safe under concurrent load."
        ),
    ),
    PRExample(
        id="bug-007",
        diff=(
            "--- a/src/pricing.py\n+++ b/src/pricing.py\n"
            "@@ -28,5 +28,5 @@\n"
            " def apply_promo(price: float, promo_code: str) -> float:\n"
            "     discount = PROMOS.get(promo_code, 0)\n"
            "-    return round(price * (1 - discount), 2)\n"
            "+    return price * (1 - discount)\n"
        ),
        category="bug",
        domain="python",
        severity="minor",
        review_comment=(
            "Removing `round(..., 2)` causes floating-point drift: `100 * 0.9 = 89.99999999...`. "
            "Financial calculations must round to the display precision. "
            "Better: use `decimal.Decimal` for monetary values throughout."
        ),
    ),
    PRExample(
        id="bug-008",
        diff=(
            "--- a/src/api/health.js\n+++ b/src/api/health.js\n"
            "@@ -5,6 +5,6 @@\n"
            " app.get('/health', async (req, res) => {\n"
            "   try {\n"
            "     await db.ping();\n"
            "-    res.status(200).json({ status: 'ok' });\n"
            "+    res.json({ status: 'ok' });\n"
            "   } catch (e) {\n"
            "     res.status(500).json({ status: 'error' });\n"
        ),
        category="bug",
        domain="javascript",
        severity="minor",
        review_comment=(
            "Missing explicit 200 status on the happy path is fine (default is 200), "
            "but the inconsistency with the 500 path is confusing. "
            "More importantly: the health check doesn't time out — a slow DB can hang the probe. "
            "Add `await Promise.race([db.ping(), timeout(2000)])` to bound response time."
        ),
    ),
    PRExample(
        id="bug-009",
        diff=(
            "--- a/pipeline/transform.py\n+++ b/pipeline/transform.py\n"
            "@@ -14,6 +14,6 @@\n"
            " def normalize_features(df: pd.DataFrame) -> pd.DataFrame:\n"
            "     for col in df.select_dtypes('number').columns:\n"
            "-        df[col] = (df[col] - df[col].mean()) / df[col].std()\n"
            "+        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)\n"
        ),
        category="bug",
        domain="python",
        severity="major",
        review_comment=(
            "Good fix adding `+ 1e-8` to prevent division by zero for constant features. "
            "However, the normalization is computed on the full DataFrame including the test set — "
            "this causes data leakage. Fit the scaler on train only, then transform train+test."
        ),
    ),
    PRExample(
        id="bug-010",
        diff=(
            "--- a/infra/docker-compose.yml\n+++ b/infra/docker-compose.yml\n"
            "@@ -12,4 +12,5 @@\n"
            " services:\n"
            "   app:\n"
            "     restart: always\n"
            "+    healthcheck:\n"
            "+      test: ['CMD', 'curl', 'http://localhost:8080/health']\n"
        ),
        category="bug",
        domain="yaml",
        severity="minor",
        review_comment=(
            "Healthcheck is missing `interval`, `timeout`, and `retries`. "
            "Without these, Docker uses defaults (30s/30s/3) which may be too slow for "
            "fast-failing services. Also add `start_period: 10s` to avoid false failures "
            "during startup. Example: `interval: 10s`, `timeout: 5s`, `retries: 3`."
        ),
    ),
]

# ── Performance examples (8) ──────────────────────────────────────────────────

_PERFORMANCE_EXAMPLES: list[PRExample] = [
    PRExample(
        id="perf-001",
        diff=(
            "--- a/models/recommendation.py\n+++ b/models/recommendation.py\n"
            "@@ -35,7 +35,9 @@\n"
            " def get_recommendations(user_id: int) -> list[Item]:\n"
            "     user = User.objects.get(id=user_id)\n"
            "+    items = []\n"
            "+    for item_id in user.liked_item_ids:\n"
            "+        item = Item.objects.get(id=item_id)  # N queries\n"
            "+        items.append(item)\n"
            "+    return items\n"
        ),
        category="performance",
        domain="python",
        severity="major",
        review_comment=(
            "N+1 query problem: one DB query per liked item. "
            "Use `Item.objects.filter(id__in=user.liked_item_ids)` for a single query. "
            "With 1000 liked items this is the difference between 1ms and 1s response time."
        ),
    ),
    PRExample(
        id="perf-002",
        diff=(
            "--- a/src/search.py\n+++ b/src/search.py\n"
            "@@ -18,8 +18,10 @@\n"
            " def find_users_by_email(email_domain: str) -> list[User]:\n"
            "-    return User.objects.filter(email__endswith=email_domain)\n"
            "+    all_users = User.objects.all()\n"
            "+    return [u for u in all_users if u.email.endswith(email_domain)]\n"
        ),
        category="performance",
        domain="python",
        severity="major",
        review_comment=(
            "Loading all users into memory for a Python-level filter. "
            "The original ORM query pushes the filter to the database and uses an index. "
            "This change makes the endpoint O(n) in memory and CPU. Revert to the queryset filter."
        ),
    ),
    PRExample(
        id="perf-003",
        diff=(
            "--- a/src/utils/matrix.js\n+++ b/src/utils/matrix.js\n"
            "@@ -6,8 +6,10 @@\n"
            " function dotProduct(a, b) {\n"
            "+  let result = [];\n"
            "+  for (let i = 0; i < a.length; i++) {\n"
            "+    result.push(a[i] * b[i]);\n"
            "+  }\n"
            "+  return result.reduce((sum, x) => sum + x, 0);\n"
            " }\n"
        ),
        category="performance",
        domain="javascript",
        severity="minor",
        review_comment=(
            "Creating an intermediate `result` array allocates memory unnecessarily. "
            "Compute the dot product in a single pass: "
            "`return a.reduce((sum, ai, i) => sum + ai * b[i], 0)`. "
            "For large vectors, consider TypedArrays (Float32Array) for 4x memory savings."
        ),
    ),
    PRExample(
        id="perf-004",
        diff=(
            "--- a/db/queries.sql\n+++ b/db/queries.sql\n"
            "@@ -1,4 +1,5 @@\n"
            " SELECT u.id, u.name, COUNT(o.id) as order_count\n"
            " FROM users u\n"
            "+LEFT JOIN orders o ON o.user_id = u.id\n"
            " WHERE u.created_at > NOW() - INTERVAL '30 days'\n"
            " GROUP BY u.id, u.name\n"
        ),
        category="performance",
        domain="sql",
        severity="major",
        review_comment=(
            "Missing index: `orders.user_id` is used in the JOIN condition but likely "
            "unindexed after the schema change. Add `CREATE INDEX CONCURRENTLY idx_orders_user_id "
            "ON orders(user_id)`. Without it, this query does a full table scan on `orders` "
            "for every new user."
        ),
    ),
    PRExample(
        id="perf-005",
        diff=(
            "--- a/training/pipeline.py\n+++ b/training/pipeline.py\n"
            "@@ -42,6 +42,8 @@\n"
            " for epoch in range(n_epochs):\n"
            "     for batch in dataloader:\n"
            "-        loss = model(batch)\n"
            "+        predictions = [model.predict_one(x) for x in batch]\n"
            "+        loss = compute_loss(predictions, batch.labels)\n"
        ),
        category="performance",
        domain="python",
        severity="major",
        review_comment=(
            "Replacing vectorized batch inference with a Python loop kills GPU utilization. "
            "The whole point of batch processing is to run operations in parallel on tensors. "
            "Revert to `model(batch)` which processes the entire batch in one forward pass. "
            "This change likely causes a 10-100x training slowdown."
        ),
    ),
    PRExample(
        id="perf-006",
        diff=(
            "--- a/src/api/events.js\n+++ b/src/api/events.js\n"
            "@@ -25,5 +25,7 @@\n"
            " async function getEvents(userId) {\n"
            "-  const events = await cache.get(`events:${userId}`);\n"
            "-  if (events) return JSON.parse(events);\n"
            "+  // cache disabled for debugging\n"
            "   const data = await db.query('SELECT * FROM events WHERE user_id = $1', [userId]);\n"
            "   return data.rows;\n"
        ),
        category="performance",
        domain="javascript",
        severity="major",
        review_comment=(
            "Disabled cache committed to production. This endpoint will now hit the DB on "
            "every request. Re-enable caching before merging. If debugging cache behavior, "
            "add a feature flag or env var: `if (!process.env.DISABLE_CACHE) { ... }`."
        ),
    ),
    PRExample(
        id="perf-007",
        diff=(
            "--- a/preprocessing/features.py\n+++ b/preprocessing/features.py\n"
            "@@ -55,6 +55,8 @@\n"
            " def build_feature_matrix(records: list[dict]) -> np.ndarray:\n"
            "-    return np.array([extract_features(r) for r in records])\n"
            "+    mat = np.zeros((len(records), N_FEATURES))\n"
            "+    for i, r in enumerate(records):\n"
            "+        mat[i] = extract_features(r)\n"
            "+    return mat\n"
        ),
        category="performance",
        domain="python",
        severity="minor",
        review_comment=(
            "Pre-allocating with `np.zeros` and filling in a loop is the right pattern "
            "to avoid repeated array copies. Good change. "
            "If `extract_features` is CPU-bound and records > 10k, consider "
            "`joblib.Parallel(n_jobs=-1)` to parallelize across cores."
        ),
    ),
    PRExample(
        id="perf-008",
        diff=(
            "--- a/infra/k8s/hpa.yaml\n+++ b/infra/k8s/hpa.yaml\n"
            "@@ -14,4 +14,4 @@\n"
            " spec:\n"
            "   minReplicas: 2\n"
            "-  maxReplicas: 10\n"
            "+  maxReplicas: 100\n"
        ),
        category="performance",
        domain="yaml",
        severity="major",
        review_comment=(
            "Increasing maxReplicas to 100 without a corresponding cost/quota review "
            "risks runaway scaling on traffic spike or HPA misconfiguration. "
            "Set a realistic ceiling based on DB connection pool limits and cost budget. "
            "Also add `scaleDown.stabilizationWindowSeconds: 300` to prevent oscillation."
        ),
    ),
]

# ── Style examples (4) ────────────────────────────────────────────────────────

_STYLE_EXAMPLES: list[PRExample] = [
    PRExample(
        id="style-001",
        diff=(
            "--- a/src/data_processor.py\n+++ b/src/data_processor.py\n"
            "@@ -10,8 +10,10 @@\n"
            "+def processData(inputDF, colName, doNorm, thresh=0.5):\n"
            "+    DF2 = inputDF.copy()\n"
            "+    if doNorm == True:\n"
            "+        DF2[colName] = (DF2[colName] - DF2[colName].min()) / (DF2[colName].max() - DF2[colName].min())\n"  # noqa: E501
            "+    DF2 = DF2[DF2[colName] > thresh]\n"
            "+    return DF2\n"
        ),
        category="style",
        domain="python",
        severity="minor",
        review_comment=(
            "PEP 8 violations: use `snake_case` for function and variable names "
            "(`process_data`, `input_df`, `col_name`, `do_normalize`, `df_filtered`). "
            "Replace `if do_normalize == True:` with `if do_normalize:`. "
            "The long normalization line exceeds 88 chars — split it."
        ),
    ),
    PRExample(
        id="style-002",
        diff=(
            "--- a/src/config.js\n+++ b/src/config.js\n"
            "@@ -3,5 +3,7 @@\n"
            "+const T = 30000;\n"
            "+const R = 3;\n"
            "+const B = 1.5;\n"
            "+const M = 10;\n"
            " module.exports = { T, R, B, M };\n"
        ),
        category="style",
        domain="javascript",
        severity="minor",
        review_comment=(
            "Single-letter constants are impossible to grep or understand. "
            "Rename to `TIMEOUT_MS`, `MAX_RETRIES`, `BACKOFF_FACTOR`, `MAX_CONNECTIONS`. "
            "Add a comment documenting the units for TIMEOUT_MS (milliseconds vs seconds "
            "is a common source of bugs)."
        ),
    ),
    PRExample(
        id="style-003",
        diff=(
            "--- a/ml/model.py\n+++ b/ml/model.py\n"
            "@@ -20,12 +20,22 @@\n"
            "+def train(X, y, cfg):\n"
            "+    if cfg.get('normalize'):\n"
            "+        if X.shape[1] > 0:\n"
            "+            if not np.isnan(X).any():\n"
            "+                scaler = StandardScaler()\n"
            "+                X = scaler.fit_transform(X)\n"
            "+                if cfg.get('save_scaler'):\n"
            "+                    joblib.dump(scaler, cfg['scaler_path'])\n"
            "+    model = LogisticRegression()\n"
            "+    model.fit(X, y)\n"
            "+    return model\n"
        ),
        category="style",
        domain="python",
        severity="minor",
        review_comment=(
            "Deeply nested conditionals (4 levels) make the control flow hard to follow. "
            "Use guard clauses: `if not cfg.get('normalize') or X.shape[1] == 0: return`. "
            "Extract scaler logic into `_normalize_features(X, cfg) -> tuple[np.ndarray, ...]`. "
            "Also add type hints: `def train(X: np.ndarray, y: np.ndarray, cfg: dict) -> LogisticRegression:`."  # noqa: E501
        ),
    ),
    PRExample(
        id="style-004",
        diff=(
            "--- a/services/etl.py\n+++ b/services/etl.py\n"
            "@@ -15,4 +15,4 @@\n"
            " def run_pipeline():\n"
            "     # step 1: load\n"
            "-    data = load_from_s3(bucket, prefix, start_date, end_date, file_format)\n"
            "+    data = load_from_s3(bucket, prefix, start_date, end_date, file_format, True, False, None, 512)\n"  # noqa: E501
        ),
        category="style",
        domain="python",
        severity="minor",
        review_comment=(
            "Positional boolean arguments are unreadable — `True, False, None, 512` require "
            "reading the function signature to understand. Use keyword arguments: "
            "`load_from_s3(..., use_cache=True, strict_schema=False, schema=None, chunk_size=512)`. "  # noqa: E501
            "Consider a `LoadConfig` dataclass if there are more than 4 options."
        ),
    ),
]

# ── Documentation examples (4) ────────────────────────────────────────────────

_DOC_EXAMPLES: list[PRExample] = [
    PRExample(
        id="doc-001",
        diff=(
            "--- a/ml/calibration.py\n+++ b/ml/calibration.py\n"
            "@@ -8,6 +8,12 @@\n"
            "+def calibrate(scores, labels, method='isotonic', n_bins=10):\n"
            "+    if method == 'isotonic':\n"
            "+        reg = IsotonicRegression(out_of_bounds='clip')\n"
            "+        reg.fit(scores, labels)\n"
            "+        return reg\n"
            "+    elif method == 'platt':\n"
            "+        lr = LogisticRegression()\n"
            "+        lr.fit(scores.reshape(-1, 1), labels)\n"
            "+        return lr\n"
        ),
        category="documentation",
        domain="python",
        severity="minor",
        review_comment=(
            "Missing docstring: what does `n_bins` do (it's unused)? "
            "What does the function return — a fitted calibrator object? What interface does it expose? "  # noqa: E501
            "Add: Args (scores shape, labels range, method options), Returns (calibrator with `.predict_proba()`), "  # noqa: E501
            "Raises (ValueError for unknown method), and a usage example."
        ),
    ),
    PRExample(
        id="doc-002",
        diff=(
            "--- a/infra/k8s/configmap.yaml\n+++ b/infra/k8s/configmap.yaml\n"
            "@@ -8,4 +8,6 @@\n"
            " data:\n"
            "+  FEATURE_FLAG_NEW_RANKER: 'true'\n"
            "+  SHADOW_MODE: 'false'\n"
            "+  ROLLOUT_PCT: '10'\n"
        ),
        category="documentation",
        domain="yaml",
        severity="minor",
        review_comment=(
            "New config keys need comments explaining their purpose and valid values. "
            "Add inline comments: what does `ROLLOUT_PCT: 10` mean (percentage of traffic?). "
            "Also document in `docs/configuration.md` or a README section so ops knows "
            "how to change them safely."
        ),
    ),
    PRExample(
        id="doc-003",
        diff=(
            "--- a/api/endpoints.py\n+++ b/api/endpoints.py\n"
            "@@ -30,5 +30,12 @@\n"
            "+@router.post('/predict')\n"
            "+async def predict(request: PredictRequest) -> PredictResponse:\n"
            "+    result = model.predict(request.features)\n"
            "+    return PredictResponse(score=result.score, label=result.label)\n"
        ),
        category="documentation",
        domain="python",
        severity="minor",
        review_comment=(
            "Public API endpoint without OpenAPI documentation. Add a docstring that becomes "
            "the endpoint description in Swagger UI: expected input format, score range (0-1?), "
            "error codes (422 on invalid features, 503 if model not loaded). "
            "Also add `response_description` to the decorator for client-facing clarity."
        ),
    ),
    PRExample(
        id="doc-004",
        diff=(
            "--- a/README.md\n+++ b/README.md\n"
            "@@ -1,3 +1,5 @@\n"
            " # My Service\n"
            "+Updated the service to use the new algorithm.\n"
            "+See CHANGELOG for details.\n"
        ),
        category="documentation",
        domain="generic",
        severity="minor",
        review_comment=(
            "README update doesn't explain what 'the new algorithm' is or how it affects users. "
            "Document: what changed, why it's better (latency improvement? accuracy gain?), "
            "any migration steps for existing users, and link to the relevant PR/issue. "
            "Also update the CHANGELOG with the version and date."
        ),
    ),
]
