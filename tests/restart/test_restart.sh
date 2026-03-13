#!/usr/bin/env bash
#
# End-to-end restart consistency test for melder live mode.
#
# Tests that after shutdown and restart:
#   1. API-added records survive (WAL replay)
#   2. Crossmap pairs survive (crossmap flush + WAL replay)
#   3. Coverage stats are consistent
#   4. Blocking index is rebuilt for WAL-replayed records
#   5. Vector index cache is reused (skip re-encoding)
#
# Requires: melder built with --features usearch
#           curl, jq
#
# Usage: bash tests/restart/test_restart.sh
#        (run from the project root)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT=19090
BASE_URL="http://localhost:${PORT}/api/v1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BINARY="${PROJECT_ROOT}/target/release/meld"
TMPDIR_BASE=""
CONFIG=""
SERVER_PID=""
PASS=0
FAIL=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TMPDIR_BASE}" && -d "${TMPDIR_BASE}" ]]; then
        rm -rf "${TMPDIR_BASE}"
    fi
}
trap cleanup EXIT

fail() {
    echo "  FAIL: $1"
    FAIL=$((FAIL + 1))
}

pass() {
    echo "  PASS: $1"
    PASS=$((PASS + 1))
}

assert_eq() {
    local label="$1" actual="$2" expected="$3"
    if [[ "${actual}" == "${expected}" ]]; then
        pass "${label}"
    else
        fail "${label} (expected '${expected}', got '${actual}')"
    fi
}

assert_ge() {
    local label="$1" actual="$2" minimum="$3"
    if [[ "${actual}" -ge "${minimum}" ]]; then
        pass "${label}"
    else
        fail "${label} (expected >= ${minimum}, got ${actual})"
    fi
}

assert_contains() {
    local label="$1" haystack="$2" needle="$3"
    if echo "${haystack}" | grep -q "${needle}"; then
        pass "${label}"
    else
        fail "${label} (expected to contain '${needle}')"
    fi
}

wait_for_server() {
    local max_wait=60
    local waited=0
    while ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [[ ${waited} -ge ${max_wait} ]]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            if [[ -n "${SERVER_PID}" ]]; then
                echo "Server stderr (last 20 lines):"
                tail -20 "${TMPDIR_BASE}/server.log" 2>/dev/null || true
            fi
            exit 1
        fi
    done
    echo "  Server ready after ${waited}s"
}

start_server() {
    echo "Starting server on port ${PORT}..."
    "${BINARY}" serve --config "${CONFIG}" --port "${PORT}" \
        > "${TMPDIR_BASE}/server.log" 2>&1 &
    SERVER_PID=$!
    wait_for_server
}

stop_server() {
    echo "Stopping server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    echo "  Server stopped"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

echo "============================================"
echo "  melder restart consistency test"
echo "============================================"
echo ""

echo "Step 0: Build release binary with usearch..."
(cd "${PROJECT_ROOT}" && cargo build --release --features usearch 2>&1 | tail -3)

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Binary not found at ${BINARY}"
    exit 1
fi
echo "  Binary: ${BINARY}"
echo ""

# ---------------------------------------------------------------------------
# Setup temp directory and config
# ---------------------------------------------------------------------------

echo "Step 1: Setup temp directory..."
TMPDIR_BASE="$(mktemp -d)"
mkdir -p "${TMPDIR_BASE}/cache"
echo "  Temp dir: ${TMPDIR_BASE}"

# Generate config from template
CONFIG="${TMPDIR_BASE}/config.yaml"
sed "s|__TMPDIR__|${TMPDIR_BASE}|g" "${SCRIPT_DIR}/config.template.yaml" > "${CONFIG}"
echo "  Config: ${CONFIG}"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: First run — start, add records, snapshot, stop
# ---------------------------------------------------------------------------

echo "Step 2: First run — start server..."
start_server
echo ""

echo "Step 3: Add 5 A-side records via API..."
A_ADD_IDS=("ENT-101" "ENT-102" "ENT-103" "ENT-104" "ENT-105")
A_ADD_NAMES=("Mitsubishi Heavy Industries" "Samsung Electronics Co" "Petrobras SA" "Tata Consultancy Services" "BHP Group Limited")
A_ADD_SHORT=("Mitsubishi Heavy" "Samsung Electronics" "Petrobras" "Tata Consulting" "BHP Group")
A_ADD_COUNTRY=("JP" "KR" "BR" "IN" "AU")

for i in 0 1 2 3 4; do
    resp=$(curl -sf -X POST "${BASE_URL}/a/add" \
        -H "Content-Type: application/json" \
        -d "{\"record\": {
            \"entity_id\": \"${A_ADD_IDS[$i]}\",
            \"legal_name\": \"${A_ADD_NAMES[$i]}\",
            \"short_name\": \"${A_ADD_SHORT[$i]}\",
            \"country_code\": \"${A_ADD_COUNTRY[$i]}\",
            \"lei\": \"\"
        }}")
    status=$(echo "${resp}" | jq -r '.status')
    echo "  Added ${A_ADD_IDS[$i]}: status=${status}"
done
echo ""

echo "Step 4: Add 5 B-side records via API..."
B_ADD_IDS=("CP-101" "CP-102" "CP-103" "CP-104" "CP-105")
B_ADD_NAMES=("Mitsubishi Heavy Ind" "Samsung Electr Co" "Petrobras" "Tata Consulting Svcs" "BHP Group Ltd")
B_ADD_COUNTRY=("JP" "KR" "BR" "IN" "AU")

for i in 0 1 2 3 4; do
    resp=$(curl -sf -X POST "${BASE_URL}/b/add" \
        -H "Content-Type: application/json" \
        -d "{\"record\": {
            \"counterparty_id\": \"${B_ADD_IDS[$i]}\",
            \"counterparty_name\": \"${B_ADD_NAMES[$i]}\",
            \"domicile\": \"${B_ADD_COUNTRY[$i]}\",
            \"lei_code\": \"\"
        }}")
    status=$(echo "${resp}" | jq -r '.status')
    echo "  Added ${B_ADD_IDS[$i]}: status=${status}"
done
echo ""

echo "Step 5: Confirm a crossmap pair (ENT-101 <-> CP-101)..."
confirm_resp=$(curl -sf -X POST "${BASE_URL}/crossmap/confirm" \
    -H "Content-Type: application/json" \
    -d '{"a_id": "ENT-101", "b_id": "CP-101"}')
echo "  Response: ${confirm_resp}"
echo ""

# Give crossmap flush a moment
sleep 2

echo "Step 6: Snapshot state before shutdown..."
STATS_BEFORE=$(curl -sf "${BASE_URL}/crossmap/stats")
PAIRS_BEFORE=$(curl -sf "${BASE_URL}/crossmap/pairs")
UNMATCHED_A_BEFORE=$(curl -sf "${BASE_URL}/a/unmatched")
UNMATCHED_B_BEFORE=$(curl -sf "${BASE_URL}/b/unmatched")

records_a_before=$(echo "${STATS_BEFORE}" | jq '.records_a')
records_b_before=$(echo "${STATS_BEFORE}" | jq '.records_b')
pairs_before=$(echo "${STATS_BEFORE}" | jq '.crossmap_pairs')
unmatched_a_before=$(echo "${STATS_BEFORE}" | jq '.unmatched_a')
unmatched_b_before=$(echo "${STATS_BEFORE}" | jq '.unmatched_b')

echo "  records_a=${records_a_before} records_b=${records_b_before}"
echo "  crossmap_pairs=${pairs_before}"
echo "  unmatched_a=${unmatched_a_before} unmatched_b=${unmatched_b_before}"

# Query an API-added A record
QUERY_A101=$(curl -sf "${BASE_URL}/a/query?id=ENT-101")
echo "  ENT-101 query: $(echo "${QUERY_A101}" | jq -c '{id, side, crossmap: .crossmap.status}')"

# Query an API-added B record
QUERY_B102=$(curl -sf "${BASE_URL}/b/query?id=CP-102")
echo "  CP-102 query: $(echo "${QUERY_B102}" | jq -c '{id, side}')"
echo ""

echo "Step 7: Shutdown..."
stop_server
echo ""

# Verify WAL and cache files were written
echo "Step 8: Verify shutdown artifacts..."
wal_count=$(find "${TMPDIR_BASE}" -name "wal*.ndjson" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "  WAL files: ${wal_count}"
assert_ge "WAL file exists" "${wal_count}" 1

# usearch saves to .usearchdb directories; flat saves to .index files
cache_count=$(find "${TMPDIR_BASE}/cache" \( -name "*.usearchdb" -type d -o -name "*.index" -type f \) 2>/dev/null | wc -l | tr -d ' ')
echo "  Index cache entries: ${cache_count}"
assert_ge "Index cache files exist" "${cache_count}" 1

crossmap_exists=0
if [[ -f "${TMPDIR_BASE}/crossmap.csv" ]]; then crossmap_exists=1; fi
assert_eq "Crossmap CSV exists" "${crossmap_exists}" "1"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Restart — verify consistency
# ---------------------------------------------------------------------------

echo "Step 9: Second run — restart server..."
start_server

# Check for skip message in server log
echo "  Checking for vector cache reuse..."
if grep -q "skipped (cached)" "${TMPDIR_BASE}/server.log"; then
    skip_line=$(grep "skipped (cached)" "${TMPDIR_BASE}/server.log")
    pass "Vector index cache reused: ${skip_line}"
else
    fail "No 'skipped (cached)' message in server log — vectors were re-encoded"
fi
echo ""

echo "Step 10: Verify state consistency after restart..."

# 10a. Stats match
STATS_AFTER=$(curl -sf "${BASE_URL}/crossmap/stats")
records_a_after=$(echo "${STATS_AFTER}" | jq '.records_a')
records_b_after=$(echo "${STATS_AFTER}" | jq '.records_b')
pairs_after=$(echo "${STATS_AFTER}" | jq '.crossmap_pairs')
unmatched_a_after=$(echo "${STATS_AFTER}" | jq '.unmatched_a')
unmatched_b_after=$(echo "${STATS_AFTER}" | jq '.unmatched_b')

assert_eq "records_a matches" "${records_a_after}" "${records_a_before}"
assert_eq "records_b matches" "${records_b_after}" "${records_b_before}"
assert_eq "crossmap_pairs matches" "${pairs_after}" "${pairs_before}"
assert_eq "unmatched_a matches" "${unmatched_a_after}" "${unmatched_a_before}"
assert_eq "unmatched_b matches" "${unmatched_b_after}" "${unmatched_b_before}"

# 10b. Crossmap pairs match
PAIRS_AFTER=$(curl -sf "${BASE_URL}/crossmap/pairs")
pairs_list_before=$(echo "${PAIRS_BEFORE}" | jq -c '[.pairs[] | {a_id, b_id}] | sort')
pairs_list_after=$(echo "${PAIRS_AFTER}" | jq -c '[.pairs[] | {a_id, b_id}] | sort')
assert_eq "crossmap pairs identical" "${pairs_list_after}" "${pairs_list_before}"

# 10c. API-added records survive
for id in "${A_ADD_IDS[@]}"; do
    resp=$(curl -sf "${BASE_URL}/a/query?id=${id}" || echo '{"error":"not_found"}')
    found_id=$(echo "${resp}" | jq -r '.id // "not_found"')
    assert_eq "A record ${id} survives restart" "${found_id}" "${id}"
done

for id in "${B_ADD_IDS[@]}"; do
    resp=$(curl -sf "${BASE_URL}/b/query?id=${id}" || echo '{"error":"not_found"}')
    found_id=$(echo "${resp}" | jq -r '.id // "not_found"')
    assert_eq "B record ${id} survives restart" "${found_id}" "${id}"
done

# 10d. Confirmed pair survives
lookup=$(curl -sf "${BASE_URL}/crossmap/lookup?id=ENT-101&side=a")
paired_id=$(echo "${lookup}" | jq -r '.paired_id // "none"')
assert_eq "ENT-101 still paired with CP-101" "${paired_id}" "CP-101"
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Blocking index test — WAL-replayed records are findable
# ---------------------------------------------------------------------------

echo "Step 11: Blocking index test — add new B record matching WAL-replayed A..."

# Add a B record that should match ENT-102 (Samsung Electronics, KR)
# via blocking (country_code=KR <-> domicile=KR) and embedding similarity.
blocking_resp=$(curl -sf -X POST "${BASE_URL}/b/add" \
    -H "Content-Type: application/json" \
    -d '{
        "record": {
            "counterparty_id": "CP-200",
            "counterparty_name": "Samsung Electronics Corporation",
            "domicile": "KR",
            "lei_code": ""
        }
    }')

echo "  CP-200 add response:"
echo "    status: $(echo "${blocking_resp}" | jq -r '.status')"
echo "    matches: $(echo "${blocking_resp}" | jq -c '[.matches[]? | {id, score}]')"

# ENT-102 (Samsung Electronics, KR) was added via API in phase 1.
# If blocking index was rebuilt correctly, CP-200 should find ENT-102 as a match.
match_ids=$(echo "${blocking_resp}" | jq -r '[.matches[]?.id] | join(",")')
if echo "${match_ids}" | grep -q "ENT-102"; then
    pass "Blocking finds WAL-replayed ENT-102 for new CP-200"
else
    fail "Blocking did NOT find WAL-replayed ENT-102 for CP-200 (got: ${match_ids})"
fi

# Also test the reverse: add a new A record that should match a WAL-replayed B.
blocking_resp2=$(curl -sf -X POST "${BASE_URL}/a/add" \
    -H "Content-Type: application/json" \
    -d '{
        "record": {
            "entity_id": "ENT-200",
            "legal_name": "Tata Consulting Services Ltd",
            "short_name": "Tata Consulting",
            "country_code": "IN",
            "lei": ""
        }
    }')

echo ""
echo "  ENT-200 add response:"
echo "    status: $(echo "${blocking_resp2}" | jq -r '.status')"
echo "    matches: $(echo "${blocking_resp2}" | jq -c '[.matches[]? | {id, score}]')"

match_ids2=$(echo "${blocking_resp2}" | jq -r '[.matches[]?.id] | join(",")')
if echo "${match_ids2}" | grep -q "CP-104"; then
    pass "Blocking finds WAL-replayed CP-104 for new ENT-200"
else
    fail "Blocking did NOT find WAL-replayed CP-104 for ENT-200 (got: ${match_ids2})"
fi
echo ""

# ---------------------------------------------------------------------------
# Cleanup and results
# ---------------------------------------------------------------------------

echo "Step 12: Shutdown..."
stop_server
echo ""

echo "============================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "============================================"

if [[ ${FAIL} -gt 0 ]]; then
    echo ""
    echo "FAILED — see above for details"
    exit 1
else
    echo ""
    echo "ALL TESTS PASSED"
    exit 0
fi
