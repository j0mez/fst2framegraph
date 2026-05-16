#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

FST2FRAMEGRAPH_BIN="${FST2FRAMEGRAPH:-}"
if [[ -z "$FST2FRAMEGRAPH_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/fst2framegraph" ]]; then
    FST2FRAMEGRAPH_BIN="$ROOT/.venv/bin/fst2framegraph"
  else
    FST2FRAMEGRAPH_BIN="fst2framegraph"
  fi
fi

TMP="${TMPDIR:-/tmp}/fst2framegraph_smoke_$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

"$PYTHON_BIN" - <<'PY'
import fst2framegraph
print(f"fst2framegraph {fst2framegraph.__version__}")
PY

"$FST2FRAMEGRAPH_BIN" inspect --input examples/flat_only_old_fst.csv > "$TMP/flat_inspect.json"
"$PYTHON_BIN" - "$TMP/flat_inspect.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "flat_only", report
assert not report["graph_ready"], report
PY

"$FST2FRAMEGRAPH_BIN" inspect --input examples/fst_like.jsonl > "$TMP/jsonl_inspect.json"
"$PYTHON_BIN" - "$TMP/jsonl_inspect.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "convertible", report
assert report["convertible"], report
PY

"$FST2FRAMEGRAPH_BIN" convert --input examples/fst_like.jsonl --out "$TMP/fst_clean" > "$TMP/convert.json"
"$FST2FRAMEGRAPH_BIN" materialise --run-dir "$TMP/fst_clean" > "$TMP/materialise.json"

if find "$TMP/fst_clean" \( -name '*.pkl' -o -name '*.pickle' \) -print -quit | grep -q .; then
  echo "Smoke failed: pickle files were created" >&2
  exit 1
fi

mkdir -p "$TMP/framebase"
cat > "$TMP/framebase/FrameBase_schema_core.ttl" <<'TTL'
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://framebase.org/frame/Assistance> rdfs:label "Assistance" .
<http://framebase.org/frame/Cause_change_of_position_on_a_scale> rdfs:label "Cause_change_of_position_on_a_scale" .
<http://framebase.org/frame/Change_position_on_a_scale> rdfs:label "Change_position_on_a_scale" .
<http://framebase.org/fe/Assistance.has_helper> rdfs:label "Helper" .
<http://framebase.org/fe/Assistance.has_benefited_party> rdfs:label "Benefited_party" .
<http://framebase.org/fe/Assistance.has_goal> rdfs:label "Goal" .
<http://framebase.org/fe/Cause_change_of_position_on_a_scale.has_agent> rdfs:label "Agent" .
<http://framebase.org/fe/Cause_change_of_position_on_a_scale.has_entity> rdfs:label "Entity" .
<http://framebase.org/fe/Change_position_on_a_scale.has_item> rdfs:label "Item" .
<http://framebase.org/fe/Change_position_on_a_scale.has_circumstances> rdfs:label "Circumstances" .
TTL

cat > "$TMP/framebase/FrameBase_schema_dbps.ttl" <<'TTL'
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://framebase.org/dbp/Assistance.goal> rdfs:label "goal" .
TTL

"$FST2FRAMEGRAPH_BIN" build-framebase-index \
  --framebase-dir "$TMP/framebase" \
  --index "$TMP/framebase/framebase_index.sqlite" \
  --overwrite > "$TMP/index.json"

"$FST2FRAMEGRAPH_BIN" build \
  --input "$TMP/fst_clean/frame_elements_long.csv" \
  --out "$TMP/graph" \
  --framebase-index "$TMP/framebase/framebase_index.sqlite" \
  --no-rdf \
  --no-graphml > "$TMP/build.log"

"$PYTHON_BIN" - "$TMP/graph/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["input_rows"] == 7, summary
assert summary["frame_instances"] == 3, summary
assert summary["frame_elements"] == 7, summary
assert summary["nested_edges"] >= 1, summary
for key in [
    "framebase_validated_frames",
    "framebase_unmatched_frames",
    "framebase_validated_frame_elements",
    "framebase_unmatched_frame_elements",
    "warnings",
]:
    assert key in summary, summary
PY

echo "fst2framegraph smoke test passed"
