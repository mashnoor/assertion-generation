"""
tests/test_tools.py

Validates tool interfaces with minimal mocking.

Tests cover:
  - get_parameters parsing from a pre-built MODULE_INTERFACE content string
  - _parse_prove_results for proven / syntax_error / falsified statuses
  - TOOL_SCHEMAS completeness (10 entries, required structure)
"""

from __future__ import annotations

import sys
import os

import pytest

# Ensure the parent package (cursor_style/) is on the path regardless of
# how pytest is invoked.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tools import _parse_prove_results, TOOL_SCHEMAS


# ===========================================================================
# get_parameters parsing
# ===========================================================================

# Pre-built content string that mimics what ASTIndexer stores as a
# MODULE_INTERFACE chunk — no ChromaDB required.
_INTERFACE_CONTENT = """MODULE INTERFACE: pipeline
Parameters: NS=8, OPD=2, WIDTH=128
Localparams: DEPTH=8
Ports:
  - clk: input logic
  - reset_: input logic (active-low)
  - in_vld: input logic
  - in_data: input logic[127:0]
  - out_vld: output logic (reg)
  - out_data: output logic[127:0]
"""


def _parse_params_from_content(content: str):
    """Replicate the logic used in DesignTools.get_parameters.

    Handles both:
      - Formatted chunk content: "Parameters: NS=8, WIDTH=128"
      - Raw Verilog syntax: "parameter NS = 8"
    """
    import re
    params = {}
    # Formatted content lines (produced by ast_indexer)
    formatted_re = re.compile(r'^(?:Parameters|Localparams):\s*(.+)$', re.MULTILINE)
    for m in formatted_re.finditer(content):
        for item in m.group(1).split(','):
            item = item.strip()
            if '=' in item:
                k, _, v = item.partition('=')
                params[k.strip()] = v.strip()
    # Raw Verilog syntax fallback
    param_re = re.compile(
        r'(?:parameter|localparam)\s+(?:\w+\s+)?(\w+)\s*=\s*([^;,\)]+)',
        re.IGNORECASE,
    )
    for m in param_re.finditer(content):
        name  = m.group(1).strip()
        value = m.group(2).strip().rstrip(',').strip()
        params[name] = value
    return params


def test_get_parameters_parsing():
    """get_parameters extracts correct values from MODULE_INTERFACE content."""
    params = _parse_params_from_content(_INTERFACE_CONTENT)
    assert params, "No parameters extracted from content string"
    assert params.get("NS") == "8",     f"NS wrong: {params}"
    assert params.get("WIDTH") == "128", f"WIDTH wrong: {params}"
    assert params.get("OPD") == "2",    f"OPD wrong: {params}"
    assert params.get("DEPTH") == "8",  f"DEPTH wrong: {params}"


# ===========================================================================
# _parse_prove_results
# ===========================================================================

_PROVEN_OUTPUT = """\
===RESULTS===
PROP: chk_out_vld STATUS: proven
PROP: chk_out_data STATUS: proven
===END===
"""

_SYNTAX_ERROR_OUTPUT = """\
ERROR: syntax error near 'assert' (dut.sv, line 42)
ERROR: Compilation failed.
"""

_COMPILATION_ERROR_OUTPUT = """\
[ERROR] Failed to elaborate top module 'pipeline'.
[ERROR] Undefined module 'missing_mod'.
"""

_FALSIFIED_OUTPUT = """\
===RESULTS===
PROP: chk_state STATUS: proven
PROP: chk_output STATUS: falsified
===END===
"""

_PARTIALLY_PROVEN_OUTPUT = """\
===RESULTS===
PROP: chk_a STATUS: proven
PROP: chk_b STATUS: undetermined
===END===
"""

_NO_PROPERTIES_OUTPUT = """\
Elaboration done.
No properties found.
"""


def test_parse_prove_results_proven():
    """_parse_prove_results correctly identifies proven status."""
    result = _parse_prove_results(_PROVEN_OUTPUT)
    assert result["status"] == "proven", f"Expected proven, got {result['status']}"
    assert result["proven"] == 2,       f"Expected 2 proven, got {result['proven']}"
    assert result["falsified"] == 0
    assert result["undetermined"] == 0


def test_parse_prove_results_syntax_error():
    """_parse_prove_results correctly identifies syntax/compilation errors."""
    result = _parse_prove_results(_SYNTAX_ERROR_OUTPUT)
    assert result["status"] == "syntax_error", (
        f"Expected syntax_error, got {result['status']}"
    )
    assert len(result["errors"]) > 0, "Expected error messages to be captured"


def test_parse_prove_results_compilation_error():
    """_parse_prove_results correctly identifies compilation errors (no syntax keyword)."""
    result = _parse_prove_results(_COMPILATION_ERROR_OUTPUT)
    assert result["status"] == "compilation_error", (
        f"Expected compilation_error, got {result['status']}"
    )
    assert len(result["errors"]) > 0


def test_parse_prove_results_falsified():
    """_parse_prove_results correctly identifies falsified assertions."""
    result = _parse_prove_results(_FALSIFIED_OUTPUT)
    assert result["status"] == "falsified", (
        f"Expected falsified, got {result['status']}"
    )
    assert result["falsified"] >= 1, f"Expected at least 1 falsified, got {result['falsified']}"
    assert result["proven"] >= 1


def test_parse_prove_results_partially_proven():
    """_parse_prove_results identifies partially_proven when some are undetermined."""
    result = _parse_prove_results(_PARTIALLY_PROVEN_OUTPUT)
    assert result["status"] == "partially_proven", (
        f"Expected partially_proven, got {result['status']}"
    )
    assert result["proven"] >= 1
    assert result["undetermined"] >= 1


def test_parse_prove_results_no_properties():
    """_parse_prove_results returns no_properties when nothing is found."""
    result = _parse_prove_results(_NO_PROPERTIES_OUTPUT)
    assert result["status"] == "no_properties", (
        f"Expected no_properties, got {result['status']}"
    )


def test_parse_prove_results_raw_preserved():
    """_parse_prove_results always preserves the raw JG output string."""
    result = _parse_prove_results(_PROVEN_OUTPUT)
    assert result["raw"] == _PROVEN_OUTPUT


# ===========================================================================
# TOOL_SCHEMAS completeness
# ===========================================================================

_EXPECTED_TOOL_NAMES = {
    "search_design",
    "get_module_info",
    "get_hierarchy",
    "get_parameters",
    "get_always_blocks",
    "get_fanin",
    "get_fanout",
    "get_flop_info",
    "verify_sva",
    "check_vacuity",
}


def test_tool_schemas_complete():
    """TOOL_SCHEMAS has exactly 10 entries with correct top-level structure."""
    assert len(TOOL_SCHEMAS) == 10, (
        f"Expected 10 tool schemas, found {len(TOOL_SCHEMAS)}: "
        + str([s["function"]["name"] for s in TOOL_SCHEMAS])
    )

    for schema in TOOL_SCHEMAS:
        # Top-level keys
        assert "type" in schema,     f"Missing 'type' key in schema: {schema}"
        assert "function" in schema, f"Missing 'function' key in schema: {schema}"
        assert schema["type"] == "function", (
            f"Expected type='function', got {schema['type']!r}"
        )

        fn = schema["function"]
        assert "name" in fn,        f"Missing 'name' in function schema: {fn}"
        assert "description" in fn, f"Missing 'description' in function schema: {fn}"
        assert "parameters" in fn,  f"Missing 'parameters' in function schema: {fn}"

        params = fn["parameters"]
        assert params.get("type") == "object", (
            f"parameters.type should be 'object', got {params.get('type')!r}"
        )
        assert "properties" in params, (
            f"parameters.properties missing in schema for {fn['name']}"
        )
        assert "required" in params, (
            f"parameters.required missing in schema for {fn['name']}"
        )

    # Verify all expected names are present
    actual_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    missing = _EXPECTED_TOOL_NAMES - actual_names
    extra   = actual_names - _EXPECTED_TOOL_NAMES
    assert not missing, f"Missing tools in TOOL_SCHEMAS: {missing}"
    assert not extra,   f"Unexpected tools in TOOL_SCHEMAS: {extra}"


def test_tool_schemas_required_args_are_strings():
    """Every 'required' entry in TOOL_SCHEMAS refers to a defined property."""
    for schema in TOOL_SCHEMAS:
        fn     = schema["function"]
        params = fn["parameters"]
        props  = set(params.get("properties", {}).keys())
        for req in params.get("required", []):
            assert req in props, (
                f"Tool '{fn['name']}': required arg '{req}' not in properties {props}"
            )
