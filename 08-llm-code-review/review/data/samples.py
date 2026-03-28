"""Sample code review dataset for training and demonstration."""

from __future__ import annotations

CATEGORIES = ("bug", "security", "style", "performance", "documentation")


def get_sample_reviews() -> list[dict]:
    """Return sample code review entries: {code_diff, review_comment, category}."""
    return [
        # ── bug ──────────────────────────────────────────────────────────────
        {
            "code_diff": (
                "--- a/src/payment.py\n"
                "+++ b/src/payment.py\n"
                "@@ -12,7 +12,7 @@\n"
                " def process_refund(order_id: int, amount: float) -> bool:\n"
                "-    if amount > 0:\n"
                "+    if amount >= 0:\n"
                "         ledger.credit(order_id, amount)\n"
                "         return True\n"
                "     return False\n"
            ),
            "review_comment": (
                "Allowing `amount == 0` means a zero-dollar refund will hit the ledger "
                "and send a confirmation email for nothing. Keep the strict `> 0` check."
            ),
            "category": "bug",
        },
        {
            "code_diff": (
                "--- a/src/cart.py\n"
                "+++ b/src/cart.py\n"
                "@@ -30,6 +30,8 @@\n"
                " def apply_discount(cart: Cart, code: str) -> Cart:\n"
                "+    discount = discounts.get(code)\n"
                "+    cart.total = cart.total - discount.value\n"
                "     return cart\n"
            ),
            "review_comment": (
                "`discounts.get(code)` returns `None` when the code is invalid, so "
                "`discount.value` will raise `AttributeError`. Add a None-check or "
                "use `discounts[code]` with a try/except KeyError."
            ),
            "category": "bug",
        },
        {
            "code_diff": (
                "--- a/src/inventory.py\n"
                "+++ b/src/inventory.py\n"
                "@@ -18,7 +18,7 @@\n"
                " def reserve_stock(sku: str, qty: int) -> bool:\n"
                "     available = db.get_stock(sku)\n"
                "-    if available >= qty:\n"
                "+    if available > qty:\n"
                "         db.decrement(sku, qty)\n"
                "         return True\n"
                "     return False\n"
            ),
            "review_comment": (
                "Off-by-one: if `available == qty` we should still allow the reservation. "
                "The original `>=` was correct; this change breaks the exact-match case."
            ),
            "category": "bug",
        },
        # ── security ─────────────────────────────────────────────────────────
        {
            "code_diff": (
                "--- a/src/auth.py\n"
                "+++ b/src/auth.py\n"
                "@@ -5,8 +5,7 @@\n"
                " def authenticate(username: str, password: str) -> User | None:\n"
                '-    query = "SELECT * FROM users WHERE name = ? AND pw_hash = ?"\n'
                "-    row = db.execute(query, (username, hash_pw(password)))\n"
                "+    query = f\"SELECT * FROM users WHERE name = '{username}' "
                "AND pw_hash = '{hash_pw(password)}'\"\n"
                "+    row = db.execute(query)\n"
                "     return User.from_row(row) if row else None\n"
            ),
            "review_comment": (
                "SQL injection vulnerability: the parameterized query was replaced with "
                "an f-string. An attacker can craft a username like `admin' OR '1'='1` "
                "to bypass authentication. Revert to parameterized queries."
            ),
            "category": "security",
        },
        {
            "code_diff": (
                "--- a/src/api/users.py\n"
                "+++ b/src/api/users.py\n"
                "@@ -22,6 +22,8 @@\n"
                " @router.post('/upload-avatar')\n"
                " async def upload_avatar(file: UploadFile):\n"
                "+    path = Path('/uploads') / file.filename\n"
                "+    path.write_bytes(await file.read())\n"
                "     return {'status': 'ok'}\n"
            ),
            "review_comment": (
                "Path traversal risk: `file.filename` can contain `../../etc/passwd`. "
                "Sanitize the filename with `secure_filename()` or generate a UUID-based "
                "name. Also validate the content type and set a max file size."
            ),
            "category": "security",
        },
        # ── style ────────────────────────────────────────────────────────────
        {
            "code_diff": (
                "--- a/src/utils.py\n"
                "+++ b/src/utils.py\n"
                "@@ -1,9 +1,7 @@\n"
                "+def calcPrice(basePrice, tax_rate, Discount, is_member):\n"
                "+    FinalPrice = basePrice * (1 + tax_rate) - Discount\n"
                "+    if is_member == True:\n"
                "+        FinalPrice = FinalPrice * 0.95\n"
                "+    return FinalPrice\n"
            ),
            "review_comment": (
                "Naming convention issues: function and variables should be snake_case per "
                "PEP 8 (`calc_price`, `final_price`). Also prefer `if is_member:` over "
                "`if is_member == True:`."
            ),
            "category": "style",
        },
        {
            "code_diff": (
                "--- a/src/report.py\n"
                "+++ b/src/report.py\n"
                "@@ -15,6 +15,12 @@\n"
                "+def get_data(x):\n"
                "+    d = db.query(x)\n"
                "+    r = []\n"
                "+    for i in d:\n"
                "+        r.append(i['val'])\n"
                "+    return r\n"
            ),
            "review_comment": (
                "Function and variable names are too terse — `get_data`, `x`, `d`, `r` "
                "tell the reader nothing. Rename to something descriptive like "
                "`fetch_monthly_values(metric_name)`. The loop can be a list comprehension."
            ),
            "category": "style",
        },
        # ── performance ──────────────────────────────────────────────────────
        {
            "code_diff": (
                "--- a/src/search.py\n"
                "+++ b/src/search.py\n"
                "@@ -8,9 +8,9 @@\n"
                " def find_duplicates(items: list[str]) -> list[str]:\n"
                "     duplicates = []\n"
                "     for i, a in enumerate(items):\n"
                "         for j, b in enumerate(items):\n"
                "-            if i != j and a == b and a not in duplicates:\n"
                "+            if i < j and a == b and a not in duplicates:\n"
                "                 duplicates.append(a)\n"
                "     return duplicates\n"
            ),
            "review_comment": (
                "Still O(n^2) with a linear `not in` scan on `duplicates`. Use a "
                "`collections.Counter` — `[k for k, v in Counter(items).items() if v > 1]` "
                "runs in O(n) and is much clearer."
            ),
            "category": "performance",
        },
        {
            "code_diff": (
                "--- a/src/etl.py\n"
                "+++ b/src/etl.py\n"
                "@@ -20,6 +20,10 @@\n"
                " def load_user_orders(user_ids: list[int]) -> list[Order]:\n"
                "+    orders = []\n"
                "+    for uid in user_ids:\n"
                "+        rows = db.execute('SELECT * FROM orders WHERE user_id = ?', (uid,))\n"
                "+        orders.extend(Order.from_row(r) for r in rows)\n"
                "+    return orders\n"
            ),
            "review_comment": (
                "N+1 query problem: this fires one SQL query per user_id. Batch them into "
                "a single `WHERE user_id IN (...)` query. With 10k users this will hammer "
                "the DB."
            ),
            "category": "performance",
        },
        # ── documentation ────────────────────────────────────────────────────
        {
            "code_diff": (
                "--- a/src/billing.py\n"
                "+++ b/src/billing.py\n"
                "@@ -1,5 +1,9 @@\n"
                "+def prorate_charge(\n"
                "+    plan_price: float,\n"
                "+    days_remaining: int,\n"
                "+    billing_cycle_days: int = 30,\n"
                "+) -> float:\n"
                "+    return plan_price * (days_remaining / billing_cycle_days)\n"
            ),
            "review_comment": (
                "Missing docstring: it's unclear what happens when `days_remaining > "
                "billing_cycle_days` (does the caller owe more than `plan_price`?). "
                "Document the expected range, return value semantics, and add a note about "
                "rounding to cents."
            ),
            "category": "documentation",
        },
        {
            "code_diff": (
                "--- a/src/config.py\n"
                "+++ b/src/config.py\n"
                "@@ -10,6 +10,8 @@\n"
                "+RETRY_BACKOFF = 1.5\n"
                "+MAX_RETRIES = 5\n"
                "+CIRCUIT_BREAKER_THRESHOLD = 10\n"
                "+HALF_OPEN_TIMEOUT = 30\n"
            ),
            "review_comment": (
                "Magic numbers without any comment. What are the units — seconds? "
                "milliseconds? Add inline comments explaining each constant and consider "
                "moving them to a config file or environment variables."
            ),
            "category": "documentation",
        },
        {
            "code_diff": (
                "--- a/src/middleware.py\n"
                "+++ b/src/middleware.py\n"
                "@@ -4,6 +4,11 @@\n"
                "+class RateLimiter:\n"
                "+    def __init__(self, rpm: int = 60):\n"
                "+        self._rpm = rpm\n"
                "+        self._window: dict[str, list[float]] = {}\n"
                "+\n"
                "+    def allow(self, client_id: str) -> bool:\n"
                "+        now = time.time()\n"
                "+        hits = [t for t in self._window.get(client_id, []) if now - t < 60]\n"
                "+        self._window[client_id] = hits\n"
                "+        if len(hits) >= self._rpm:\n"
                "+            return False\n"
                "+        hits.append(now)\n"
                "+        return True\n"
            ),
            "review_comment": (
                "The `_window` dict grows unboundedly — entries for clients who stop "
                "sending requests are never cleaned up. Add a periodic cleanup or use a "
                "TTL cache. In production, consider Redis-based rate limiting instead of "
                "in-memory state."
            ),
            "category": "bug",
        },
    ]
