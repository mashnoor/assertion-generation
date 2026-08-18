"""
ast_indexer.py — RTL AST indexer for hardware formal verification.

Uses pyslang to parse SystemVerilog/Verilog RTL and extract semantic chunks
(module interfaces, always blocks, sub-module instantiations, continuous
assigns) for storage in a ChromaDB collection.

Chunk types
-----------
MODULE_INTERFACE  — one per module: parameters, ports
ALWAYS_BLOCK      — one per always/always_ff/always_comb/always_latch block
INSTANCE          — one per sub-module instantiation
ASSIGN            — one per continuous-assign group (capped at 20 per module)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy pyslang import — failure here means the whole module is unusable but
# we still export the public symbols so callers can guard themselves.
# ---------------------------------------------------------------------------
try:
    import pyslang  # flat namespace — everything lives under pyslang.*
    _PYSLANG_AVAILABLE = True
except ImportError:  # pragma: no cover
    pyslang = None  # type: ignore[assignment]
    _PYSLANG_AVAILABLE = False
    logger.warning("pyslang not available; AST indexer will use regex fallback only")

# ---------------------------------------------------------------------------
# ChromaDB / LangChain imports
# ---------------------------------------------------------------------------
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ===========================================================================
# Low-level extraction helpers
# ===========================================================================

def _type_str(sym_type: Any) -> str:
    """Convert a pyslang type object to a concise string representation."""
    try:
        return str(sym_type)
    except Exception:
        return "unknown"


def _direction_str(direction: Any) -> str:
    """Convert ArgumentDirection enum to 'input' / 'output' / 'inout'."""
    s = str(direction)
    if "In" in s and "Out" not in s:
        return "input"
    if "Out" in s and "In" not in s:
        return "output"
    if "InOut" in s:
        return "inout"
    return s.lower()


def _is_active_low_reset(signal_name: str) -> bool:
    """Heuristic: trailing underscore or 'n' suffix → active-low."""
    return signal_name.endswith("_") or signal_name.endswith("n") or signal_name.lower() in {"resetn", "rst_n", "reset_n"}


def _collect_driven_signals(proc_sym: Any) -> List[str]:
    """
    Walk the AST of a ProceduralBlockSymbol and collect names of every signal
    that appears on the left-hand side of an assignment.

    Returns a deduplicated, sorted list of signal names.
    """
    driven: set[str] = set()

    def _visitor(node: Any) -> Any:
        if isinstance(node, pyslang.AssignmentExpression):
            lhs = node.left
            _extract_lhs_name(lhs, driven)
        return pyslang.VisitAction.Advance

    try:
        proc_sym.visit(_visitor)
    except Exception as exc:
        logger.debug("Visitor error collecting driven signals: %s", exc)

    return sorted(driven)


def _extract_lhs_name(lhs: Any, driven: set) -> None:
    """Recursively extract the base signal name from an LHS expression."""
    try:
        if isinstance(lhs, pyslang.NamedValueExpression):
            sr = lhs.getSymbolReference()
            if sr and hasattr(sr, "name"):
                driven.add(sr.name)
            return
        # Array element select: mem[i] → 'mem'
        if isinstance(lhs, pyslang.ElementSelectExpression):
            if lhs.syntax:
                base = str(lhs.syntax).strip().split("[")[0].strip()
                if base:
                    driven.add(base)
            return
        # Range select or member access — try syntax
        if hasattr(lhs, "syntax") and lhs.syntax:
            text = str(lhs.syntax).strip()
            base = re.split(r"[\[\.]", text)[0].strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base):
                driven.add(base)
    except Exception as exc:
        logger.debug("LHS name extraction error: %s", exc)


def _parse_sensitivity_list(timed_stmt: Any) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Parse sensitivity list from a TimedStatement.

    Returns:
        events   — list of (edge_str, signal_name) pairs, e.g. [("posedge", "clk"), ...]
        implicit — True if sensitivity is @(*) / implicit
    """
    events: List[Tuple[str, str]] = []
    implicit = False

    try:
        timing = timed_stmt.timing
        if isinstance(timing, pyslang.ImplicitEventControl):
            implicit = True
            return events, implicit

        def _process_signal_event(ev: Any) -> None:
            if not isinstance(ev, pyslang.SignalEventControl):
                return
            edge = ev.edge
            if str(edge) == "EdgeKind.PosEdge":
                edge_str = "posedge"
            elif str(edge) == "EdgeKind.NegEdge":
                edge_str = "negedge"
            else:
                edge_str = "edge"
            sig_name = "?"
            expr = ev.expr
            if isinstance(expr, pyslang.NamedValueExpression):
                sr = expr.getSymbolReference()
                if sr and hasattr(sr, "name"):
                    sig_name = sr.name
            events.append((edge_str, sig_name))

        if isinstance(timing, pyslang.EventListControl):
            for ev in timing.events:
                _process_signal_event(ev)
        elif isinstance(timing, pyslang.SignalEventControl):
            _process_signal_event(timing)

    except Exception as exc:
        logger.debug("Sensitivity list parse error: %s", exc)

    return events, implicit


# ===========================================================================
# Raw-RTL helper utilities (no pyslang dependency)
# ===========================================================================

def _extract_always_blocks_raw(rtl_code: str) -> List[str]:
    """
    Extract always/always_ff/always_comb/always_latch block source text from
    raw RTL by tracking begin/end nesting depth.

    Returns a list of source snippets (one per block), in document order.
    """
    blocks: List[str] = []
    lines = rtl_code.splitlines()
    n = len(lines)
    i = 0

    # Regex matches the start of an always block (at word boundary)
    _ALWAYS_RE = re.compile(r"\b(always_ff|always_comb|always_latch|always)\b")

    while i < n:
        if _ALWAYS_RE.search(lines[i]):
            start = i
            depth = 0
            found_begin = False
            j = i
            while j < n:
                # Count begin/end tokens in this line
                tokens = re.findall(r"\b(begin|end)\b", lines[j])
                for tok in tokens:
                    if tok == "begin":
                        depth += 1
                        found_begin = True
                    elif tok == "end":
                        depth -= 1
                if found_begin and depth == 0:
                    snippet = "\n".join(lines[start : j + 1])
                    blocks.append(snippet)
                    i = j + 1
                    break
                j += 1
            else:
                # Did not find matching end; skip this line
                i += 1
        else:
            i += 1

    return blocks


def _extract_assign_lines_raw(rtl_code: str) -> List[str]:
    """Extract 'assign ...' lines from raw RTL (regex, no pyslang)."""
    return [
        line.strip()
        for line in rtl_code.splitlines()
        if re.match(r"\s*assign\b", line)
    ]


def _extract_module_names_raw(rtl_code: str) -> List[str]:
    """Return list of module names found in raw RTL via regex."""
    return re.findall(r"\bmodule\s+(\w+)", rtl_code)


def _extract_ports_raw(rtl_code: str, module_name: str) -> List[str]:
    """
    Extract port declarations from the module header via regex.

    This is used as a fallback when pyslang is unavailable or fails.
    Returns lines like 'input clk', 'output reg [7:0] dout', etc.
    """
    # Try to find the module header block
    pattern = re.compile(
        r"\bmodule\s+" + re.escape(module_name) + r"\b.*?\)",
        re.DOTALL,
    )
    m = pattern.search(rtl_code)
    if not m:
        return []
    header = m.group(0)
    ports: List[str] = []
    for line in header.splitlines():
        stripped = line.strip()
        if re.match(r"\b(input|output|inout)\b", stripped):
            # Remove trailing comma
            ports.append(stripped.rstrip(","))
    return ports


def _extract_params_raw(rtl_code: str, module_name: str) -> List[str]:
    """
    Extract parameter declarations from raw RTL via regex.
    Returns lines like 'parameter WIDTH=8', 'localparam DEPTH=4'.
    """
    params: List[str] = []
    in_module = False
    for line in rtl_code.splitlines():
        if re.search(r"\bmodule\s+" + re.escape(module_name) + r"\b", line):
            in_module = True
        if not in_module:
            continue
        stripped = line.strip()
        if re.match(r"\b(parameter|localparam)\b", stripped):
            params.append(stripped.rstrip(",;"))
        if re.match(r"\bendmodule\b", stripped):
            break
    return params


# ===========================================================================
# Per-module chunk builders (AST path)
# ===========================================================================

def _build_interface_chunk_ast(
    inst: Any,
    design_id: str,
    module_name: str,
    design_type: str,
) -> Dict:
    """
    Build a MODULE_INTERFACE chunk from a pyslang InstanceBodySymbol.
    """
    params: List[str] = []
    localparams: List[str] = []
    ports: List[str] = []

    for sym in inst.body:
        if isinstance(sym, pyslang.ParameterSymbol):
            try:
                value_str = str(sym.value)
            except Exception:
                value_str = "?"
            entry = f"{sym.name}={value_str}"
            if sym.isLocalParam:
                localparams.append(entry)
            else:
                params.append(entry)

        elif isinstance(sym, pyslang.PortSymbol):
            dir_str = _direction_str(sym.direction)
            type_str = _type_str(sym.type)
            # Annotate likely active-low reset signals
            note = ""
            if "reset" in sym.name.lower() and _is_active_low_reset(sym.name):
                note = " (active-low)"
            ports.append(f"  - {sym.name}: {dir_str} {type_str}{note}")

    lines = [f"MODULE INTERFACE: {module_name}"]
    if params:
        lines.append(f"Parameters: {', '.join(params)}")
    if localparams:
        lines.append(f"Localparams: {', '.join(localparams)}")
    if ports:
        lines.append("Ports:")
        lines.extend(ports)

    return {
        "chunk_type": "MODULE_INTERFACE",
        "module_name": module_name,
        "design_id": design_id,
        "design_type": design_type,
        "content": "\n".join(lines),
    }


def _build_always_chunk_ast(
    proc_sym: Any,
    block_idx: int,
    module_name: str,
    design_id: str,
    design_type: str,
    raw_rtl: str,
    raw_block_idx: int,
) -> Dict:
    """
    Build an ALWAYS_BLOCK chunk from a pyslang ProceduralBlockSymbol.
    """
    kind = proc_sym.procedureKind
    kind_str = str(kind)

    # Classify block type
    if "AlwaysFF" in kind_str:
        block_type = "always_ff (sequential)"
    elif "AlwaysComb" in kind_str:
        block_type = "always_comb (combinational)"
    elif "AlwaysLatch" in kind_str:
        block_type = "always_latch (latch)"
    else:
        block_type = "always (inferred)"

    # Sensitivity list
    clock_sig: Optional[str] = None
    reset_sig: Optional[str] = None
    reset_active: Optional[str] = None
    sensitivity_parts: List[str] = []
    implicit = False

    b = proc_sym.body
    if isinstance(b, pyslang.TimedStatement):
        events, implicit = _parse_sensitivity_list(b)
        for edge_str, sig_name in events:
            sensitivity_parts.append(f"{edge_str} {sig_name}")
            # Identify clock (posedge non-reset signal)
            if edge_str == "posedge" and "reset" not in sig_name.lower() and "rst" not in sig_name.lower():
                if clock_sig is None:
                    clock_sig = sig_name
            # Identify reset (signal with reset/rst in name, or negedge)
            if "reset" in sig_name.lower() or "rst" in sig_name.lower():
                reset_sig = sig_name
                if edge_str == "negedge":
                    reset_active = "active-low, negedge"
                else:
                    reset_active = "active-high, posedge"

    if implicit:
        sensitivity_str = "implicit (*)"
    elif sensitivity_parts:
        sensitivity_str = ", ".join(sensitivity_parts)
    else:
        sensitivity_str = "unknown"

    # Driven signals
    driven = _collect_driven_signals(proc_sym)

    # Source snippet from raw RTL
    raw_blocks = _extract_always_blocks_raw(raw_rtl)
    snippet = ""
    if raw_block_idx < len(raw_blocks):
        raw_text = raw_blocks[raw_block_idx]
        # Limit snippet to 40 lines to avoid bloating the chunk
        snippet_lines = raw_text.splitlines()[:40]
        if len(raw_text.splitlines()) > 40:
            snippet_lines.append("    ... (truncated)")
        snippet = "\n".join("  " + ln for ln in snippet_lines)

    lines = [
        f"ALWAYS BLOCK {block_idx} in module: {module_name}",
        f"Type: {block_type}",
        f"Sensitivity: {sensitivity_str}",
    ]
    if clock_sig:
        lines.append(f"Clock: {clock_sig} (posedge)")
    if reset_sig:
        lines.append(f"Reset: {reset_sig} ({reset_active})")
    if driven:
        lines.append(f"Driven signals: {', '.join(driven)}")
    if snippet:
        lines.append("Source snippet:")
        lines.append(snippet)

    return {
        "chunk_type": "ALWAYS_BLOCK",
        "module_name": module_name,
        "design_id": design_id,
        "design_type": design_type,
        "block_index": block_idx,
        "content": "\n".join(lines),
    }


def _build_instance_chunk_ast(
    inst_sym: Any,
    parent_module: str,
    design_id: str,
    design_type: str,
    raw_rtl: str,
) -> Dict:
    """
    Build an INSTANCE chunk from a pyslang InstanceSymbol.
    """
    inst_name = inst_sym.name
    child_name = inst_sym.definition.name if hasattr(inst_sym, "definition") else "unknown"

    # Build port connection list
    port_connections: List[str] = []
    try:
        for pc in inst_sym.portConnections:
            port_name = pc.port.name if pc.port else None
            expr = pc.expression
            expr_str = None

            if expr is not None:
                # NamedValueExpression → get symbol name directly (input ports)
                if isinstance(expr, pyslang.NamedValueExpression):
                    sr = expr.getSymbolReference()
                    if sr and hasattr(sr, "name"):
                        expr_str = sr.name
                # AssignmentExpression → output port: parent signal is the LHS
                elif isinstance(expr, pyslang.AssignmentExpression):
                    lhs = expr.left
                    if isinstance(lhs, pyslang.NamedValueExpression):
                        sr = lhs.getSymbolReference()
                        if sr and hasattr(sr, "name"):
                            expr_str = sr.name
                    if expr_str is None:
                        try:
                            sn = lhs.syntax
                            if sn:
                                expr_str = str(sn).strip()
                        except Exception:
                            pass
                # Fallback: try .syntax attribute on the expression itself
                if expr_str is None:
                    try:
                        sn = expr.syntax
                        if sn:
                            expr_str = str(sn).strip()
                    except Exception:
                        pass
                if expr_str is None:
                    expr_str = "<expr>"

            if port_name and expr_str is not None:
                port_connections.append(f".{port_name}({expr_str})")
    except Exception as exc:
        logger.debug("Port connection extraction error for %s: %s", inst_name, exc)

    # Fallback: regex scan of raw RTL for the instantiation line
    if not port_connections:
        port_connections = _extract_port_connections_regex(raw_rtl, inst_name, child_name)

    lines = [
        f"INSTANCE {inst_name} of module {child_name} in {parent_module}",
        "Port connections visible in parent:",
        "  " + ", ".join(port_connections) if port_connections else "  (none found)",
    ]

    return {
        "chunk_type": "INSTANCE",
        "module_name": parent_module,
        "design_id": design_id,
        "design_type": design_type,
        "instance_name": inst_name,
        "child_module": child_name,
        "content": "\n".join(lines),
    }


def _extract_port_connections_regex(rtl_code: str, inst_name: str, child_name: str) -> List[str]:
    """
    Fallback: extract port connections via regex when pyslang gives no data.
    Looks for patterns like:  child_mod u_inst (.a(b), .c(d));
    """
    # Match the instantiation block
    pattern = re.compile(
        r"\b" + re.escape(child_name) + r"\s+" + re.escape(inst_name) + r"\s*\("
        r"(.*?)\)\s*;",
        re.DOTALL,
    )
    m = pattern.search(rtl_code)
    if not m:
        return []
    port_block = m.group(1)
    connections = re.findall(r"\.\s*(\w+)\s*\(([^)]*)\)", port_block)
    return [f".{p}({c.strip()})" for p, c in connections]


def _extract_instance_chunks_regex(
    rtl_code: str,
    parent_module: str,
    design_id: str,
    design_type: str,
) -> List[Dict]:
    """Regex-based fallback to extract INSTANCE chunks.

    Catches cases pyslang misses: instantiations inside generate blocks.
    Handles ``child_mod #(.PARAM(val)) inst_name (...)`` by stripping
    ``#(...)`` sections (one level of nesting) before matching.

    Pattern matched:  <child_module> [#(...)] <inst_name> (...)
    """
    _KEYWORDS = {
        "module", "endmodule", "always", "initial", "assign", "if", "else",
        "begin", "end", "case", "casez", "casex", "endcase", "for", "while",
        "forever", "generate", "endgenerate", "function", "task",
        "posedge", "negedge", "input", "output", "inout", "reg", "wire",
        "logic", "integer", "parameter", "localparam", "genvar",
    }

    # Strip #(...) parameter overrides — handles one level of nested parens
    # e.g. #(.WIDTH(WIDTH)) → '' so the simple word-word-( pattern can match
    stripped = re.sub(r'#\s*\([^()]*(?:\([^()]*\)[^()]*)*\)', '', rtl_code)

    # Simple pattern: word whitespace word (  → child_module inst_name (
    inst_pattern = re.compile(r'\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(')

    chunks: List[Dict] = []
    seen_child_modules: set = set()
    for m in inst_pattern.finditer(stripped):
        child_mod = m.group(1)
        inst_name = m.group(2)
        if child_mod in _KEYWORDS or inst_name in _KEYWORDS:
            continue
        if child_mod == parent_module:
            continue  # skip self-reference
        if child_mod in seen_child_modules:
            continue
        seen_child_modules.add(child_mod)
        content = (
            f"INSTANCE {inst_name} of module {child_mod} in {parent_module} "
            f"(regex-extracted, may be in generate block)\n"
            f"Child module: {child_mod}\n"
            f"Instance name: {inst_name}\n"
            f"Parent module: {parent_module}"
        )
        chunks.append({
            "chunk_type": "INSTANCE",
            "content": content,
            "design_id": design_id,
            "module_name": parent_module,
            "child_module": child_mod,
            "instance_name": inst_name,
            "design_type": design_type,
            "extraction_method": "regex",
        })
    return chunks


def _build_assign_chunk_ast(
    assign_lines: List[str],
    module_name: str,
    design_id: str,
    design_type: str,
) -> Dict:
    """
    Build an ASSIGN chunk from a list of 'assign ...' lines.
    """
    lines = [f"CONTINUOUS ASSIGN in module: {module_name}", "Assignments (from RTL scan):"]
    for line in assign_lines[:20]:
        lines.append(f"  {line}")

    return {
        "chunk_type": "ASSIGN",
        "module_name": module_name,
        "design_id": design_id,
        "design_type": design_type,
        "content": "\n".join(lines),
    }


# ===========================================================================
# Fallback chunk builders (pure regex, no pyslang)
# ===========================================================================

def _build_interface_chunk_regex(
    module_name: str,
    rtl_code: str,
    design_id: str,
    design_type: str,
) -> Dict:
    """
    Build a MODULE_INTERFACE chunk using regex extraction only.
    Used when pyslang is unavailable or fails to parse.
    """
    param_lines = _extract_params_raw(rtl_code, module_name)
    port_lines = _extract_ports_raw(rtl_code, module_name)

    params = []
    localparams = []
    for p in param_lines:
        if re.match(r"localparam\b", p):
            m = re.search(r"localparam\s+(\w+)\s*=\s*(\S+)", p)
            if m:
                localparams.append(f"{m.group(1)}={m.group(2)}")
        else:
            m = re.search(r"parameter\s+(?:\w+\s+)?(\w+)\s*=\s*(\S+)", p)
            if m:
                params.append(f"{m.group(1)}={m.group(2)}")

    lines = [f"MODULE INTERFACE: {module_name} (regex fallback)"]
    if params:
        lines.append(f"Parameters: {', '.join(params)}")
    if localparams:
        lines.append(f"Localparams: {', '.join(localparams)}")
    if port_lines:
        lines.append("Ports:")
        for pl in port_lines:
            lines.append(f"  - {pl}")

    return {
        "chunk_type": "MODULE_INTERFACE",
        "module_name": module_name,
        "design_id": design_id,
        "design_type": design_type,
        "content": "\n".join(lines),
    }


# ===========================================================================
# Per-module RTL splitter (regex)
# ===========================================================================

def _split_rtl_by_module(rtl_code: str) -> Dict[str, str]:
    """
    Split concatenated RTL into a dict of {module_name: module_rtl}.

    Uses a simple regex scan that finds 'module <name>' and 'endmodule'
    pairs. Works for flat (non-generate) designs.
    """
    modules: Dict[str, str] = {}
    pattern = re.compile(r"\bmodule\s+(\w+)\b")
    endmod_re = re.compile(r"\bendmodule\b")

    lines = rtl_code.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        m = pattern.search(lines[i])
        if m:
            module_name = m.group(1)
            start = i
            # Scan forward for endmodule
            depth = 0
            j = i
            while j < len(lines):
                if re.search(r"\bmodule\b", lines[j]) and not re.search(r"\bendmodule\b", lines[j]):
                    depth += 1
                if endmod_re.search(lines[j]):
                    depth -= 1
                    if depth <= 0:
                        module_rtl = "".join(lines[start : j + 1])
                        modules[module_name] = module_rtl
                        i = j + 1
                        break
                j += 1
            else:
                i += 1
        else:
            i += 1

    return modules


# ===========================================================================
# Core extraction: extract_chunks_from_rtl
# ===========================================================================

def extract_chunks_from_rtl(
    design_id: str,
    rtl_code: str,
    design_type: str = "pipeline",
) -> List[Dict]:
    """
    Pure function — parse RTL and return a list of chunk dicts (no ChromaDB).

    Each chunk dict has at minimum:
        chunk_type   : str  — MODULE_INTERFACE | ALWAYS_BLOCK | INSTANCE | ASSIGN
        module_name  : str
        design_id    : str
        design_type  : str
        content      : str  — human-readable semantic text for embedding

    Additional keys depend on chunk_type (e.g. block_index, instance_name).

    If pyslang is not available or fails, falls back to regex extraction.
    Raises nothing — always returns at least one chunk per module.
    """
    chunks: List[Dict] = []

    if not _PYSLANG_AVAILABLE:
        logger.warning(
            "pyslang unavailable; using regex fallback for design %s", design_id
        )
        return _extract_chunks_regex_fallback(design_id, rtl_code, design_type)

    # -----------------------------------------------------------------------
    # Attempt full AST parse
    # -----------------------------------------------------------------------
    try:
        tree = pyslang.SyntaxTree.fromText(rtl_code)
        comp = pyslang.Compilation()
        comp.addSyntaxTree(tree)

        # Log hard errors (not warnings)
        diags = comp.getAllDiagnostics()
        error_count = sum(1 for i in range(len(diags)) if diags[i].isError())
        if error_count > 0:
            logger.warning(
                "design %s: %d parse error(s) from pyslang; continuing with partial AST",
                design_id,
                error_count,
            )

        root = comp.getRoot()
        top_instances = root.topInstances

        if not top_instances:
            logger.warning(
                "design %s: no top instances found; using regex fallback", design_id
            )
            return _extract_chunks_regex_fallback(design_id, rtl_code, design_type)

    except Exception as exc:
        logger.warning(
            "design %s: pyslang raised exception (%s); using regex fallback",
            design_id,
            exc,
        )
        return _extract_chunks_regex_fallback(design_id, rtl_code, design_type)

    # -----------------------------------------------------------------------
    # Split raw RTL by module so we can pass the right snippet to each builder
    # -----------------------------------------------------------------------
    module_rtl_map = _split_rtl_by_module(rtl_code)

    # -----------------------------------------------------------------------
    # Walk top-level instances AND recursively discovered child instances.
    # seen_modules tracks which definition names have been processed to avoid
    # duplicating chunks when the same module is instantiated multiple times.
    # child_instances is a list of (InstanceSymbol, parent_module_name) pairs
    # discovered while walking body symbols.
    # -----------------------------------------------------------------------
    seen_modules: set[str] = set()
    child_instances_to_process: List[Tuple[Any, str]] = []

    def _process_body(inst_obj: Any, module_name: str) -> None:
        """
        Walk inst_obj.body and append chunks for procedural blocks, child
        instances, and continuous assigns in *module_name*.
        Discovered child InstanceSymbols are queued in child_instances_to_process.
        """
        raw_rtl_for_mod = module_rtl_map.get(module_name, rtl_code)
        always_idx = 0
        raw_always_idx = 0
        assign_lines: List[str] = []

        for sym in inst_obj.body:

            # --- ALWAYS_BLOCK ---
            if isinstance(sym, pyslang.ProceduralBlockSymbol):
                try:
                    ab_chunk = _build_always_chunk_ast(
                        sym,
                        always_idx,
                        module_name,
                        design_id,
                        design_type,
                        raw_rtl_for_mod,
                        raw_always_idx,
                    )
                    chunks.append(ab_chunk)
                except Exception as exc:
                    logger.warning(
                        "AlwaysBlock chunk error for %s block %d: %s",
                        module_name, always_idx, exc
                    )
                always_idx += 1
                raw_always_idx += 1

            # --- INSTANCE ---
            elif isinstance(sym, pyslang.InstanceSymbol):
                try:
                    inst_chunk = _build_instance_chunk_ast(
                        sym, module_name, design_id, design_type, raw_rtl_for_mod
                    )
                    chunks.append(inst_chunk)
                except Exception as exc:
                    logger.warning(
                        "Instance chunk error for %s in %s: %s",
                        getattr(sym, "name", "?"), module_name, exc
                    )
                # Queue child for its own interface/body processing
                child_def_name = sym.definition.name if hasattr(sym, "definition") else None
                if child_def_name and child_def_name not in seen_modules:
                    child_instances_to_process.append((sym, child_def_name))
                    seen_modules.add(child_def_name)

            # --- ASSIGN (collect lines, emit one chunk per module) ---
            elif isinstance(sym, pyslang.ContinuousAssignSymbol):
                try:
                    sn = sym.syntax
                    if sn:
                        line = str(sn).strip()
                        if not line.startswith("assign"):
                            line = "assign " + line
                        assign_lines.append(line)
                except Exception as exc:
                    logger.debug("ContinuousAssign syntax error: %s", exc)

        # Flush assign lines (up to 20) as one chunk
        if not assign_lines:
            assign_lines = _extract_assign_lines_raw(raw_rtl_for_mod)
        if assign_lines:
            try:
                chunks.append(
                    _build_assign_chunk_ast(assign_lines, module_name, design_id, design_type)
                )
            except Exception as exc:
                logger.warning("Assign chunk error for %s: %s", module_name, exc)

        # If no INSTANCE chunks were produced for this module (e.g. all instantiations
        # are inside generate blocks which pyslang surfaces as GenerateBlockArraySymbol
        # rather than plain InstanceSymbol), fall back to a regex scan of the raw RTL.
        module_instance_chunks = [
            c for c in chunks
            if c.get("chunk_type") == "INSTANCE" and c.get("module_name") == module_name
        ]
        if not module_instance_chunks:
            inst_chunks_regex = _extract_instance_chunks_regex(
                raw_rtl_for_mod, module_name, design_id, design_type
            )
            chunks.extend(inst_chunks_regex)

    # --- Process top instances ---
    for top_inst in top_instances:
        module_name = top_inst.name
        seen_modules.add(module_name)
        raw_rtl_for_mod = module_rtl_map.get(module_name, rtl_code)

        # MODULE_INTERFACE
        try:
            chunks.append(
                _build_interface_chunk_ast(top_inst, design_id, module_name, design_type)
            )
        except Exception as exc:
            logger.warning("Interface chunk error for %s: %s", module_name, exc)
            chunks.append(
                _build_interface_chunk_regex(module_name, raw_rtl_for_mod, design_id, design_type)
            )

        _process_body(top_inst, module_name)

    # --- Process child instances discovered above (BFS order) ---
    i = 0
    while i < len(child_instances_to_process):
        child_inst_sym, child_mod_name = child_instances_to_process[i]
        i += 1
        raw_rtl_for_mod = module_rtl_map.get(child_mod_name, rtl_code)

        # MODULE_INTERFACE from child body
        try:
            chunks.append(
                _build_interface_chunk_ast(child_inst_sym, design_id, child_mod_name, design_type)
            )
        except Exception as exc:
            logger.warning("Interface chunk error for child %s: %s", child_mod_name, exc)
            chunks.append(
                _build_interface_chunk_regex(child_mod_name, raw_rtl_for_mod, design_id, design_type)
            )

        _process_body(child_inst_sym, child_mod_name)

    # -----------------------------------------------------------------------
    # Any module in the raw RTL that was never reached via the AST walk
    # (e.g. unreferenced modules) — use regex fallback for those only.
    # -----------------------------------------------------------------------
    for mod_name, mod_rtl in module_rtl_map.items():
        if mod_name in seen_modules:
            continue
        logger.debug(
            "design %s: module %s unreachable from AST; using regex fallback",
            design_id,
            mod_name,
        )
        sub_chunks = _extract_chunks_regex_fallback(design_id, mod_rtl, design_type)
        chunks.extend(sub_chunks)
        seen_modules.add(mod_name)

    return chunks


# ===========================================================================
# Regex fallback for complete extraction without pyslang
# ===========================================================================

def _extract_chunks_regex_fallback(
    design_id: str,
    rtl_code: str,
    design_type: str,
) -> List[Dict]:
    """
    Extract chunks using only regex. Called when pyslang is unavailable or
    fails.  Produces MODULE_INTERFACE and ASSIGN chunks at minimum; also
    produces ALWAYS_BLOCK snippets (without AST analysis) and INSTANCE chunks
    via pattern matching.
    """
    chunks: List[Dict] = []
    module_names = _extract_module_names_raw(rtl_code)

    if not module_names:
        # Nothing we can do; return a minimal placeholder
        chunks.append({
            "chunk_type": "MODULE_INTERFACE",
            "module_name": "unknown",
            "design_id": design_id,
            "design_type": design_type,
            "content": f"MODULE INTERFACE: unknown\n(regex fallback — no module found in design {design_id})",
        })
        return chunks

    module_rtl_map = _split_rtl_by_module(rtl_code)

    for mod_name in module_names:
        mod_rtl = module_rtl_map.get(mod_name, rtl_code)

        # MODULE_INTERFACE
        chunks.append(
            _build_interface_chunk_regex(mod_name, mod_rtl, design_id, design_type)
        )

        # ALWAYS_BLOCK snippets (source only, no AST analysis)
        raw_blocks = _extract_always_blocks_raw(mod_rtl)
        for idx, snippet in enumerate(raw_blocks):
            snippet_lines = snippet.splitlines()[:40]
            if len(snippet.splitlines()) > 40:
                snippet_lines.append("  ... (truncated)")
            indented = "\n".join("  " + ln for ln in snippet_lines)
            chunks.append({
                "chunk_type": "ALWAYS_BLOCK",
                "module_name": mod_name,
                "design_id": design_id,
                "design_type": design_type,
                "block_index": idx,
                "content": (
                    f"ALWAYS BLOCK {idx} in module: {mod_name} (regex fallback)\n"
                    f"Source snippet:\n{indented}"
                ),
            })

        # INSTANCE chunks via regex
        inst_pattern = re.compile(
            r"\b(\w+)\s+(\w+)\s*\((.*?)\)\s*;", re.DOTALL
        )
        keywords = {
            "module", "input", "output", "inout", "wire", "reg", "logic",
            "assign", "always", "begin", "end", "if", "else", "case",
            "endcase", "for", "while", "integer", "localparam", "parameter",
            "endmodule",
        }
        for m in inst_pattern.finditer(mod_rtl):
            child_name = m.group(1)
            inst_name = m.group(2)
            port_block = m.group(3)
            if child_name in keywords or inst_name in keywords:
                continue
            connections = re.findall(r"\.\s*(\w+)\s*\(([^)]*)\)", port_block)
            if not connections:
                continue
            conn_str = ", ".join(f".{p}({c.strip()})" for p, c in connections)
            chunks.append({
                "chunk_type": "INSTANCE",
                "module_name": mod_name,
                "design_id": design_id,
                "design_type": design_type,
                "instance_name": inst_name,
                "child_module": child_name,
                "content": (
                    f"INSTANCE {inst_name} of module {child_name} in {mod_name}\n"
                    f"Port connections visible in parent:\n  {conn_str}"
                ),
            })

        # ASSIGN chunk
        assign_lines = _extract_assign_lines_raw(mod_rtl)
        if assign_lines:
            chunks.append(
                _build_assign_chunk_ast(assign_lines, mod_name, design_id, design_type)
            )

    return chunks


# ===========================================================================
# ASTIndexer class
# ===========================================================================

class ASTIndexer:
    """
    Indexes RTL designs as semantic chunks in a ChromaDB collection.

    Each design is split into typed chunks (MODULE_INTERFACE, ALWAYS_BLOCK,
    INSTANCE, ASSIGN) and stored as LangChain Documents with rich metadata.
    Supports incremental indexing (skip already-indexed designs via
    ``is_design_indexed``).

    Parameters
    ----------
    db_path:         Path to the ChromaDB persist directory.
    embeddings:      A LangChain embeddings object (e.g. HuggingFaceEmbeddings
                     or OllamaEmbeddings).
    collection_name: ChromaDB collection name (default: "rtl_ast_chunks").
    """

    def __init__(
        self,
        db_path: str,
        embeddings: Any,
        collection_name: str = "rtl_ast_chunks",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=db_path,
        )

    # ------------------------------------------------------------------
    def is_design_indexed(self, design_id: str) -> bool:
        """Return True if at least one chunk for *design_id* already exists."""
        try:
            result = self.vector_store.get(
                where={"design_id": design_id},
                limit=1,
            )
            return len(result.get("ids", [])) > 0
        except Exception as exc:
            logger.debug("is_design_indexed check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    def index_design(
        self,
        design_id: str,
        rtl_code: str,
        design_type: str = "pipeline",
    ) -> int:
        """
        Parse *rtl_code* and index all extracted chunks into ChromaDB.

        Skips silently if the design is already indexed.

        Returns
        -------
        int : number of chunks added (0 if skipped).
        """
        if self.is_design_indexed(design_id):
            logger.debug("design %s already indexed — skipping", design_id)
            return 0

        chunks = extract_chunks_from_rtl(design_id, rtl_code, design_type)
        if not chunks:
            logger.warning("design %s produced no chunks", design_id)
            return 0

        documents: List[Document] = []
        for chunk in chunks:
            content = chunk.pop("content")
            doc = Document(page_content=content, metadata=chunk)
            documents.append(doc)

        try:
            self.vector_store.add_documents(documents)
            logger.info(
                "design %s: indexed %d chunks into '%s'",
                design_id,
                len(documents),
                self.collection_name,
            )
        except Exception as exc:
            logger.error("Failed to add documents to ChromaDB for design %s: %s", design_id, exc)
            raise

        return len(documents)

    # ------------------------------------------------------------------
    def get_all_chunks(self, design_id: str) -> List[Dict]:
        """
        Retrieve all stored chunks for *design_id*.

        Returns a list of dicts with keys ``content`` and all metadata fields.
        Useful for validation and inspection.
        """
        try:
            result = self.vector_store.get(
                where={"design_id": design_id},
            )
        except Exception as exc:
            logger.error("get_all_chunks failed for design %s: %s", design_id, exc)
            return []

        chunks: List[Dict] = []
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        for doc_id, content, meta in zip(ids, docs, metas):
            entry = {"id": doc_id, "content": content}
            if meta:
                entry.update(meta)
            chunks.append(entry)

        return chunks
