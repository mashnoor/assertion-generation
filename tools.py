"""
tools.py

Tool registry for the Cursor-style RTL assertion generation agent.
Each tool gathers context on demand during assertion generation.

Tools fall into two tiers:
  - Fast tools: ChromaDB semantic search (milliseconds).
  - JasperGold tools: authoritative formal analysis over SSH (seconds).

The DesignTools class wraps both tiers and logs every call for introspection.
"""

import os
import re
import json
import subprocess
import time
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# JasperGold SSH helper
# ---------------------------------------------------------------------------

def _parse_delimited(raw: str) -> Dict[str, str]:
    """Split JG output by ===DELIM_<KEY>=== markers into a dict."""
    sections: Dict[str, str] = {}
    current_key = None
    lines_buf: List[str] = []

    for line in raw.splitlines():
        if "===DELIM_" in line and "===" in line.split("===DELIM_", 1)[1]:
            if current_key is not None:
                sections[current_key] = "\n".join(lines_buf).strip()
            key = line.split("===DELIM_", 1)[1].split("===", 1)[0]
            current_key = key
            lines_buf = []
        elif current_key is not None:
            lines_buf.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(lines_buf).strip()
    return sections


# JasperGold SSH target and remote scratch directory are environment-specific;
# configure a host alias in ~/.ssh/config (recommended, key-based auth) rather
# than hardcoding credentials here. See README.md "Setup".
JG_SSH_HOST = os.environ.get("JASPERGOLD_SSH_HOST", "jaspergold")
JG_REMOTE_DIR = os.environ.get("JASPERGOLD_REMOTE_DIR", "~/proofloop_jg_work")


def _ssh_base_cmd() -> List[str]:
    """Build the SSH invocation used for all JasperGold calls.

    Prefers key-based auth (set up JASPERGOLD_SSH_HOST as an alias in
    ~/.ssh/config with an IdentityFile). Password auth is only used if the
    SSHPASS env var is explicitly set, and is read from the environment
    rather than a command-line argument.
    """
    cmd: List[str] = []
    if os.environ.get("SSHPASS"):
        cmd += ['sshpass', '-e']
    cmd += [
        'ssh',
        '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=10',
        '-o', 'ServerAliveCountMax=3',
        '-o', 'StrictHostKeyChecking=no',
        JG_SSH_HOST, 'bash',
    ]
    return cmd


def _run_jg_ssh(module_name: str, rtl_code: str, tcl_body: str, tag: str,
                timeout: int = 120) -> str:
    """Run a JasperGold TCL script on the configured JasperGold host via SSH.

    Creates a temporary directory on the remote, writes dut.sv and run.tcl,
    runs JasperGold in batch mode, then cleans up.

    Returns raw stdout from JasperGold (comment lines filtered out).
    Raises subprocess.TimeoutExpired on timeout; all other failures raise RuntimeError.
    """
    remote_dir = f"{JG_REMOTE_DIR}/tmp_jg_{module_name}_{tag}_{os.getpid()}"

    bash_script = f'''
mkdir -p {remote_dir}
cd {remote_dir}

cat > dut.sv << 'PYEOF'
{rtl_code}
PYEOF

cat > run.tcl << 'PYEOF'
{tcl_body}
PYEOF

jg -allow_unsupported_OS -batch run.tcl 2>&1 | grep -v "^#"

cd {JG_REMOTE_DIR}
rm -rf {remote_dir}
'''

    ssh_cmd = _ssh_base_cmd()

    try:
        proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(input=bash_script, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            raise  # re-raise TimeoutExpired for callers
    except subprocess.TimeoutExpired:
        raise
    except Exception as e:
        raise RuntimeError(f"SSH execution failed: {type(e).__name__}: {e}") from e

    raw = (stdout + "\n" + stderr).strip()
    lines = raw.splitlines()
    prompt_start = next((i for i, ln in enumerate(lines) if not ln.startswith('%')), 0)
    return "\n".join(lines[prompt_start:])


# ---------------------------------------------------------------------------
# Clock / Reset detection
# ---------------------------------------------------------------------------

def _detect_clock_reset(rtl_code: str) -> dict:
    """Detect clock and reset signal names from RTL port declarations.

    Returns dict with keys: clock, reset_name, reset_active_low,
    reset_tcl (ready-to-use TCL catch lines), disable_iff (SVA expression).
    """
    clock = "clk"
    reset_name = ""
    reset_active_low = True

    for pat, name in [
        (r'\binput\b[^;]*\bclk\b', 'clk'),
        (r'\binput\b[^;]*\bCLK\b', 'CLK'),
        (r'\binput\b[^;]*\bclock\b', 'clock'),
    ]:
        if re.search(pat, rtl_code):
            clock = name
            break

    for pat, active_low in [
        (r'\binput\b[^;]*\b(reset_)\b', True),
        (r'\binput\b[^;]*\b(rst_n)\b', True),
        (r'\binput\b[^;]*\b(reset_n)\b', True),
        (r'\binput\b[^;]*\b(resetn)\b', True),
        (r'\binput\b[^;]*\b(rst)\b', False),
        (r'\binput\b[^;]*\b(reset)\b', False),
    ]:
        m = re.search(pat, rtl_code)
        if m:
            reset_name = m.group(1)
            reset_active_low = active_low
            break

    tcl_lines = f'catch {{clock {clock}}}\n'
    if reset_name:
        if reset_active_low:
            tcl_lines += f'catch {{reset -expression {{!{reset_name}}}}}\n'
        else:
            tcl_lines += f'catch {{reset {reset_name}}}\n'
    else:
        tcl_lines += 'catch {reset -expression {!reset_}}\n'
        tcl_lines += 'catch {reset rst}\n'

    disable_iff = (
        f"!{reset_name}" if reset_name and reset_active_low
        else reset_name if reset_name
        else "!reset_"
    )

    return {
        "clock": clock,
        "reset_name": reset_name,
        "reset_active_low": reset_active_low,
        "reset_tcl": tcl_lines,
        "disable_iff": disable_iff,
    }


# ---------------------------------------------------------------------------
# SVA extraction / injection helpers
# ---------------------------------------------------------------------------

def _extract_sva_code(text: str) -> str:
    """Strip markdown fenced code blocks if present, returning bare SVA."""
    pattern = re.compile(r"```(?:systemverilog|sv|verilog)?\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _inject_sva(rtl_code: str, sva_code: str, top_module: str = "") -> str:
    """Inject SVA text before the endmodule of the top-level module.

    If top_module is given, finds that module's endmodule specifically.
    Otherwise injects before the LAST endmodule (top module is typically
    last in concatenated RTL).
    """
    injection = "\n// --- Injected SVA assertions ---\n" + sva_code + "\n\n"

    if top_module:
        # Find 'module <top_module>' then scan for its closing endmodule
        mod_pattern = re.compile(
            r'\bmodule\s+' + re.escape(top_module) + r'\b', re.MULTILINE
        )
        mod_match = mod_pattern.search(rtl_code)
        if mod_match:
            search_start = mod_match.end()
            depth = 1
            pos = search_start
            while pos < len(rtl_code):
                next_mod = re.search(r'\bmodule\s+\w+', rtl_code[pos:])
                next_end = re.search(r'\bendmodule\b', rtl_code[pos:])
                if next_end is None:
                    break
                end_abs = pos + next_end.start()
                if next_mod and (pos + next_mod.start()) < end_abs:
                    depth += 1
                    pos = pos + next_mod.end()
                    continue
                depth -= 1
                if depth == 0:
                    return rtl_code[:end_abs] + injection + rtl_code[end_abs:]
                pos = end_abs + len("endmodule")

    # Fallback: inject before the LAST endmodule
    pos = rtl_code.rfind("endmodule")
    if pos != -1:
        return rtl_code[:pos] + injection + rtl_code[pos:]
    return rtl_code + "\n" + sva_code


def _parse_prove_results(raw: str) -> Dict:
    """Parse JasperGold prove output into a structured result dict.

    Primary method: parse structured PROP/STATUS lines after ===RESULTS=== marker.
    Fallback: regex scan for STATUS: keywords (handles non-standard output).
    """
    result: Dict[str, Any] = {
        "status": "unknown",
        "proven": 0,
        "falsified": 0,
        "undetermined": 0,
        "properties": [],   # list of {"name": ..., "status": ...}
        "errors": [],
        "raw": raw,
    }

    # --- Early exit: syntax errors ---
    if re.search(r"syntax error", raw, re.IGNORECASE):
        result["status"] = "syntax_error"
        result["errors"] = re.findall(r"(?:ERROR|syntax error)[^\n]*", raw, re.IGNORECASE)[:10]
        return result

    # --- Early exit: compilation errors (no RESULTS marker reached) ---
    all_errors = re.findall(r"\[?ERROR[^\n]*", raw, re.IGNORECASE)
    if all_errors and not re.search(r"===RESULTS===", raw):
        result["status"] = "compilation_error"
        result["errors"] = all_errors[:10]
        return result

    # --- Primary: parse structured PROP: <name> STATUS: <status> lines ---
    results_section = ""
    results_match = re.search(r"===RESULTS===(.*?)(?:===END===|$)", raw, re.DOTALL)
    if results_match:
        results_section = results_match.group(1)

    prop_lines = re.findall(r"PROP:\s*(\S+)\s+STATUS:\s*(\S+)", results_section, re.IGNORECASE)
    for prop_name, prop_status in prop_lines:
        st = prop_status.strip().lower()
        result["properties"].append({"name": prop_name, "status": st})
        if st == "proven":
            result["proven"] += 1
        elif st in ("falsified", "cex"):
            result["falsified"] += 1
        else:
            result["undetermined"] += 1

    # --- Fallback: if no structured lines found, scan for STATUS: keywords ---
    if not prop_lines:
        result["proven"]       = len(re.findall(r"STATUS:\s*proven",            raw, re.IGNORECASE))
        result["falsified"]    = len(re.findall(r"STATUS:\s*(?:falsified|cex)", raw, re.IGNORECASE))
        result["undetermined"] = len(re.findall(r"STATUS:\s*undetermined",      raw, re.IGNORECASE))

    # --- Determine overall status ---
    total = result["proven"] + result["falsified"] + result["undetermined"]
    if total > 0 and result["falsified"] == 0:
        result["status"] = "proven" if result["undetermined"] == 0 else "partially_proven"
    elif result["falsified"] > 0:
        result["status"] = "falsified"
    else:
        result["status"] = "no_properties"

    return result


def _extract_defines(rtl_code: str) -> str:
    """Extract `define values from RTL and build JasperGold +define+ string.

    Reads lines like '`define WIDTH 128' from the RTL and returns
    '+define+WIDTH=128+DEPTH=13+NS=8+OPD=2' for the analyze command.
    Falls back to safe defaults for any missing define.
    """
    defaults = {"WIDTH": "128", "DEPTH": "8", "NS": "8", "OPD": "2"}
    found = {}
    for m in re.finditer(r"^\s*`define\s+(\w+)\s+(\d+)", rtl_code, re.MULTILINE):
        name, value = m.group(1), m.group(2)
        if name in defaults:
            found[name] = value
    merged = {**defaults, **found}
    # Build single +define+ string: +define+K1=V1+K2=V2+...
    kv_parts = "+".join(f"{k}={v}" for k, v in sorted(merged.items()))
    return f"+define+{kv_parts}"


# ---------------------------------------------------------------------------
# DesignTools
# ---------------------------------------------------------------------------

class DesignTools:
    """On-demand context tools for the Cursor-style assertion generation agent.

    Two tiers:
      - Fast (ChromaDB): search_design, get_module_info, get_hierarchy,
                         get_parameters, get_always_blocks
      - JasperGold (SSH): get_fanin, get_fanout, get_flop_info,
                          verify_sva, check_vacuity

    Every public method appends a log entry to _tool_call_log recording the
    tool name, arguments, result length, and elapsed time.
    """

    def __init__(self, vector_store, design_id: str, design_dir: str, llm=None):
        """
        Args:
            vector_store: Chroma instance on the ``rtl_ast_chunks`` collection.
            design_id:    Design identifier (e.g. "pipeline_0").
            design_dir:   Absolute path to ``<debug_dir>/<design_id>/``.
            llm:          Optional LLM instance (not used by tools directly).
        """
        self.vector_store = vector_store
        self.design_id    = design_id
        self.design_dir   = design_dir
        self.llm          = llm
        self._tool_call_log: List[Dict] = []
        # Pre-compute JG define string from design RTL
        self._jg_defines = self._compute_defines()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_defines(self) -> str:
        """Extract `define values from design RTL for JasperGold compilation."""
        # Check defines.sv first (saved by pipeline_v4 from original RTL)
        defines_path = os.path.join(self.design_dir, "defines.sv")
        if os.path.exists(defines_path):
            with open(defines_path) as f:
                return _extract_defines(f.read())
        all_rtl = self._collect_all_rtl()
        return _extract_defines(all_rtl)

    def _log(self, tool: str, args: Dict, result: str, elapsed: float):
        self._tool_call_log.append({
            "tool":       tool,
            "args":       args,
            "result_len": len(result),
            "elapsed_s":  round(elapsed, 3),
        })

    def _find_module_rtl(self, module_name: str) -> Optional[str]:
        """Return RTL source for module_name, or None if not found."""
        # Primary location: design_dir/<module_name>/rtl.sv
        primary = os.path.join(self.design_dir, module_name, "rtl.sv")
        if os.path.exists(primary):
            with open(primary) as f:
                return f.read()
        # Fallback: scan all subdirs of design_dir for a matching rtl.sv
        try:
            for entry in os.scandir(self.design_dir):
                if entry.is_dir():
                    candidate = os.path.join(entry.path, "rtl.sv")
                    if os.path.exists(candidate):
                        with open(candidate) as f:
                            content = f.read()
                        # Check if module_name appears in module declaration
                        if re.search(r'\bmodule\s+' + re.escape(module_name) + r'\b', content):
                            return content
        except OSError:
            pass
        return None

    def _collect_all_rtl(self) -> str:
        """Concatenate all module RTLs from design_dir subdirs."""
        parts: List[str] = []
        try:
            for entry in sorted(os.scandir(self.design_dir), key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                rtl_path = os.path.join(entry.path, "rtl.sv")
                if os.path.exists(rtl_path):
                    with open(rtl_path) as f:
                        parts.append(f.read())
        except OSError:
            pass
        return "\n\n".join(parts)

    def _top_module_name(self) -> str:
        """Infer the top module name from design_graph.json or fallback."""
        graph_path = os.path.join(self.design_dir, "design_graph.json")
        if os.path.exists(graph_path):
            try:
                with open(graph_path) as f:
                    graph = json.load(f)
                sorted_mods = graph.get("sorted_modules", [])
                if sorted_mods:
                    return sorted_mods[0]
            except Exception:
                pass
        return self.design_id

    # ------------------------------------------------------------------
    # Clock/Reset helper
    # ------------------------------------------------------------------

    def get_clock_reset_info(self) -> dict:
        """Return detected clock/reset info for the design."""
        all_rtl = self._collect_all_rtl()
        return _detect_clock_reset(all_rtl)

    # ------------------------------------------------------------------
    # Fast tools (ChromaDB)
    # ------------------------------------------------------------------

    def search_design(self, query: str, k: int = 5) -> str:
        """Semantic search over all AST chunks for this design.

        Returns a formatted string of the top-k matching chunks including
        their metadata (chunk_type, module_name) and page content.

        Args:
            query: Natural language or signal-name search query.
            k:     Number of results to return (default 5).
        """
        t0 = time.monotonic()
        try:
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter={"design_id": self.design_id},
            )
            lines: List[str] = []
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                chunk_type  = meta.get("chunk_type", "UNKNOWN")
                module_name = meta.get("module_name", "?")
                lines.append(f"### Result {i} — {chunk_type} in `{module_name}`")
                lines.append(doc.page_content)
                lines.append("")
            result = "\n".join(lines).strip() if lines else "No results found."
        except Exception as e:
            result = f"[search_design error] {e}"

        self._log("search_design", {"query": query, "k": k}, result, time.monotonic() - t0)
        return result

    def get_module_info(self, module_name: str) -> str:
        """Return the MODULE_INTERFACE chunk for module_name.

        Attempts an exact metadata filter on module_name + chunk_type first;
        falls back to a semantic search if no exact match is found.

        Args:
            module_name: Name of the Verilog module (e.g. ``exec_unit_0``).
        """
        t0 = time.monotonic()
        try:
            docs = self.vector_store.similarity_search(
                f"module interface ports {module_name}",
                k=5,
                filter={
                    "$and": [
                        {"design_id": self.design_id},
                        {"module_name": module_name},
                        {"chunk_type": "MODULE_INTERFACE"},
                    ]
                },
            )
            if not docs:
                # Fallback: any chunk from this module
                docs = self.vector_store.similarity_search(
                    f"module {module_name} ports parameters",
                    k=3,
                    filter={
                        "$and": [
                            {"design_id": self.design_id},
                            {"module_name": module_name},
                        ]
                    },
                )
            if docs:
                meta = docs[0].metadata
                header = (
                    f"### MODULE_INTERFACE: `{module_name}`\n"
                    f"chunk_type={meta.get('chunk_type','?')} | "
                    f"design_id={meta.get('design_id','?')}\n\n"
                )
                result = header + docs[0].page_content
            else:
                result = f"No MODULE_INTERFACE chunk found for module `{module_name}`."
        except Exception as e:
            result = f"[get_module_info error] {e}"

        self._log("get_module_info", {"module_name": module_name}, result, time.monotonic() - t0)
        return result

    def get_hierarchy(self) -> str:
        """Return the full module dependency tree as a formatted string.

        Reads from ``design_dir/design_graph.json`` when available; otherwise
        falls back to a ChromaDB search for HIERARCHY chunks.
        """
        t0 = time.monotonic()
        try:
            graph_path = os.path.join(self.design_dir, "design_graph.json")
            if os.path.exists(graph_path):
                with open(graph_path) as f:
                    graph = json.load(f)
                sorted_mods = graph.get("sorted_modules", [])
                adj         = graph.get("adjacency_list", {})
                lines = ["### Design Hierarchy", ""]
                lines.append(f"**Topological order** (top → leaf):")
                for i, mod in enumerate(sorted_mods):
                    lines.append(f"  {i+1}. `{mod}`")
                lines.append("")
                lines.append("**Parent → Children:**")
                for parent, children in adj.items():
                    child_str = ", ".join(f"`{c}`" for c in children) if children else "*(leaf)*"
                    lines.append(f"  `{parent}` → {child_str}")
                result = "\n".join(lines)
            else:
                # ChromaDB fallback
                docs = self.vector_store.similarity_search(
                    "module hierarchy dependency tree",
                    k=3,
                    filter={
                        "$and": [
                            {"design_id": self.design_id},
                            {"chunk_type": "HIERARCHY"},
                        ]
                    },
                )
                if docs:
                    result = docs[0].page_content
                else:
                    result = f"No hierarchy data found for design `{self.design_id}`."
        except Exception as e:
            result = f"[get_hierarchy error] {e}"

        self._log("get_hierarchy", {}, result, time.monotonic() - t0)
        return result

    def get_parameters(self, module_name: str) -> str:
        """Return parameter and localparam values for a module.

        Parses the MODULE_INTERFACE chunk for parameter/localparam lines and
        also checks the RTL file directly if available.

        Returns a compact string like ``NS=8, WIDTH=128, DEPTH=10, OPD=2``.

        Args:
            module_name: Name of the Verilog module.
        """
        t0 = time.monotonic()
        params: Dict[str, str] = {}
        try:
            # 1. Try ChromaDB MODULE_INTERFACE chunk
            docs = self.vector_store.similarity_search(
                f"parameters localparam {module_name}",
                k=3,
                filter={
                    "$and": [
                        {"design_id": self.design_id},
                        {"module_name": module_name},
                        {"chunk_type": "MODULE_INTERFACE"},
                    ]
                },
            )
            source_text = ""
            if docs:
                source_text = docs[0].page_content

            # 2. Also try RTL file
            rtl = self._find_module_rtl(module_name)
            if rtl:
                source_text += "\n" + rtl

            # Parse formatted chunk content: "Parameters: NS=8, WIDTH=128"
            formatted_re = re.compile(
                r'^(?:Parameters|Localparams):\s*(.+)$', re.MULTILINE
            )
            for m in formatted_re.finditer(source_text):
                for item in m.group(1).split(','):
                    item = item.strip()
                    if '=' in item:
                        k, _, v = item.partition('=')
                        params[k.strip()] = v.strip()

            # Also parse raw Verilog parameter/localparam declarations
            param_re = re.compile(
                r'(?:parameter|localparam)\s+(?:\w+\s+)?(\w+)\s*=\s*([^;,\)]+)',
                re.IGNORECASE,
            )
            for match in param_re.finditer(source_text):
                name  = match.group(1).strip()
                value = match.group(2).strip().rstrip(',').strip()
                params[name] = value

            if params:
                result = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
            else:
                result = f"No parameters found for module `{module_name}`."
        except Exception as e:
            result = f"[get_parameters error] {e}"

        self._log("get_parameters", {"module_name": module_name}, result, time.monotonic() - t0)
        return result

    def get_always_blocks(self, signal_name: str, module_name: str) -> str:
        """Return ALWAYS_BLOCK chunks that drive signal_name in module_name.

        Searches ChromaDB for ALWAYS_BLOCK chunks filtered by module_name,
        then ranks results by relevance to signal_name.

        Args:
            signal_name: RTL signal identifier to look for (e.g. ``state``).
            module_name: Module containing the signal.
        """
        t0 = time.monotonic()
        try:
            docs = self.vector_store.similarity_search(
                f"always block {signal_name} assignment sequential combinational",
                k=8,
                filter={
                    "$and": [
                        {"design_id": self.design_id},
                        {"module_name": module_name},
                        {"chunk_type": "ALWAYS_BLOCK"},
                    ]
                },
            )
            if not docs:
                # Fallback: any chunk from this module mentioning the signal
                docs = self.vector_store.similarity_search(
                    f"{signal_name} always @",
                    k=5,
                    filter={
                        "$and": [
                            {"design_id": self.design_id},
                            {"module_name": module_name},
                        ]
                    },
                )

            if docs:
                lines = [
                    f"### ALWAYS_BLOCK chunks driving `{signal_name}` in `{module_name}`", ""
                ]
                for i, doc in enumerate(docs, 1):
                    meta = doc.metadata
                    lines.append(
                        f"#### Block {i} "
                        f"(chunk_type={meta.get('chunk_type','?')}, "
                        f"sensitivity={meta.get('sensitivity','')})"
                    )
                    lines.append(doc.page_content)
                    lines.append("")
                result = "\n".join(lines).strip()
            else:
                result = (
                    f"No ALWAYS_BLOCK chunks found for signal `{signal_name}` "
                    f"in module `{module_name}`."
                )
        except Exception as e:
            result = f"[get_always_blocks error] {e}"

        self._log(
            "get_always_blocks",
            {"signal_name": signal_name, "module_name": module_name},
            result,
            time.monotonic() - t0,
        )
        return result

    # ------------------------------------------------------------------
    # JasperGold tools (SSH)
    # ------------------------------------------------------------------

    def get_fanin(self, signal: str, module: str) -> str:
        """Run JasperGold get_fanin for signal in module via SSH.

        Reads RTL from ``design_dir/<module>/rtl.sv``.  Returns a newline-
        separated list of fanin signals, or an error message.

        Args:
            signal: Signal name (e.g. ``count``, ``state[2:0]``).
            module: Module containing the signal.
        """
        t0 = time.monotonic()
        rtl = self._find_module_rtl(module)
        if rtl is None:
            result = f"[get_fanin error] RTL not found for module `{module}`."
            self._log("get_fanin", {"signal": signal, "module": module}, result,
                      time.monotonic() - t0)
            return result

        tcl = (
            f"analyze -sv {{{self._jg_defines}}} dut.sv\n"
            f"elaborate -top {module}\n"
            f"puts \"===DELIM_FANIN===\"\n"
            f"catch {{get_fanin {signal}}}\n"
            f"puts \"===DELIM_END===\"\n"
            f"exit\n"
        )
        try:
            raw = _run_jg_ssh(module, rtl, tcl, tag=f"fanin_{signal[:20]}", timeout=120)
            sections = _parse_delimited(raw)
            fanin_raw = sections.get("FANIN", "").strip()
            result = fanin_raw if fanin_raw else "(no fanin results)"
        except subprocess.TimeoutExpired:
            result = "[get_fanin error] SSH timeout."
        except Exception as e:
            result = f"[get_fanin error] {e}"

        self._log("get_fanin", {"signal": signal, "module": module}, result,
                  time.monotonic() - t0)
        return result

    def get_fanout(self, signal: str, module: str) -> str:
        """Run JasperGold get_fanout for signal in module via SSH.

        Args:
            signal: Signal name.
            module: Module containing the signal.
        """
        t0 = time.monotonic()
        rtl = self._find_module_rtl(module)
        if rtl is None:
            result = f"[get_fanout error] RTL not found for module `{module}`."
            self._log("get_fanout", {"signal": signal, "module": module}, result,
                      time.monotonic() - t0)
            return result

        tcl = (
            f"analyze -sv {{{self._jg_defines}}} dut.sv\n"
            f"elaborate -top {module}\n"
            f"puts \"===DELIM_FANOUT===\"\n"
            f"catch {{get_fanout {signal}}}\n"
            f"puts \"===DELIM_END===\"\n"
            f"exit\n"
        )
        try:
            raw = _run_jg_ssh(module, rtl, tcl, tag=f"fanout_{signal[:20]}", timeout=120)
            sections = _parse_delimited(raw)
            fanout_raw = sections.get("FANOUT", "").strip()
            result = fanout_raw if fanout_raw else "(no fanout results)"
        except subprocess.TimeoutExpired:
            result = "[get_fanout error] SSH timeout."
        except Exception as e:
            result = f"[get_fanout error] {e}"

        self._log("get_fanout", {"signal": signal, "module": module}, result,
                  time.monotonic() - t0)
        return result

    def get_flop_info(self, signal: str, module: str) -> str:
        """Run JasperGold get_flop_info + get_signal_info for signal in module.

        Returns a formatted string: clock, reset pin, reset type, reset value,
        direction, and fanin list (up to 10 entries).

        Args:
            signal: Signal name (exact RTL identifier).
            module: Module containing the signal.
        """
        t0 = time.monotonic()
        rtl = self._find_module_rtl(module)
        if rtl is None:
            result = f"[get_flop_info error] RTL not found for module `{module}`."
            self._log("get_flop_info", {"signal": signal, "module": module}, result,
                      time.monotonic() - t0)
            return result

        tcl = (
            f"analyze -sv {{{self._jg_defines}}} dut.sv\n"
            f"elaborate -top {module}\n"
            f"puts \"===DELIM_FLOPINFO===\"\n"
            f"catch {{get_flop_info {{{signal}}}}}\n"
            f"puts \"===DELIM_FANIN===\"\n"
            f"catch {{get_fanin {signal}}}\n"
            f"puts \"===DELIM_SIGNALINFO===\"\n"
            f"catch {{get_signal_info {signal}}}\n"
            f"puts \"===DELIM_END===\"\n"
            f"exit\n"
        )
        try:
            raw = _run_jg_ssh(module, rtl, tcl, tag=f"flop_{signal[:20]}", timeout=120)
            sections = _parse_delimited(raw)
            flop_raw    = sections.get("FLOPINFO", "")
            fanin_raw   = sections.get("FANIN", "").strip()
            dir_raw     = sections.get("SIGNALINFO", "").strip()

            info: Dict[str, str] = {
                "signal":    signal,
                "module":    module,
                "is_flop":   "yes" if "Flop:" in flop_raw else "no",
                "clock":     "",
                "data":      "",
                "reset_pin": "",
                "reset_value_pin": "",
                "reset_type": "",
                "direction": dir_raw if dir_raw in ("input", "output", "internal") else "",
                "fanin":     ", ".join(fanin_raw.split()[:10]) if fanin_raw else "",
            }
            patterns = {
                "clock":           r"Clock:[ \t]*(.+)",
                "data":            r"Data:[ \t]*(.+)",
                "reset_pin":       r"Reset pin:[ \t]*(.+)",
                "reset_value_pin": r"Reset value pin:[ \t]*(.+)",
                "reset_type":      r"Reset type:[ \t]*(.+)",
            }
            for key, pat in patterns.items():
                m = re.search(pat, flop_raw)
                if m:
                    info[key] = m.group(1).strip()

            lines = [f"Flop info for `{signal}` in `{module}`:", ""]
            for k, v in info.items():
                if k not in ("signal", "module"):
                    lines.append(f"  {k}: {v if v else '(none)'}")
            result = "\n".join(lines)
        except subprocess.TimeoutExpired:
            result = "[get_flop_info error] SSH timeout."
        except Exception as e:
            result = f"[get_flop_info error] {e}"

        self._log("get_flop_info", {"signal": signal, "module": module}, result,
                  time.monotonic() - t0)
        return result

    def verify_sva(self, sva_code: str, top_module: str = "", all_rtl: str = "") -> str:
        """Inject SVA into DUT RTL and run JasperGold prove -all -time_limit 1m.

        Returns a structured string reporting:
        - status (proven/falsified/syntax_error/compilation_error/no_properties)
        - proven / falsified / undetermined counts
        - error messages (if any)

        Args:
            sva_code:   SVA property and assert statements (may include markdown fences).
            top_module: Top-level module name for ``elaborate -top``.  Inferred
                        from design_graph.json if not provided.
            all_rtl:    Combined RTL source.  If empty, all module RTLs in
                        design_dir are concatenated automatically.
        """
        t0 = time.monotonic()
        if not top_module:
            top_module = self._top_module_name()
        if not all_rtl:
            all_rtl = self._collect_all_rtl()

        sva_clean = _extract_sva_code(sva_code)
        combined  = _inject_sva(all_rtl, sva_clean, top_module=top_module)

        cr = _detect_clock_reset(all_rtl)
        tcl = (
            f"analyze -sv {{{self._jg_defines}}} dut.sv\n"
            f"elaborate -top {top_module}\n"
            f"{cr['reset_tcl']}"
            f"prove -all -time_limit 1m\n"
            f"puts \"===RESULTS===\"\n"
            f"foreach p [get_property_list] {{\n"
            f"    set st [get_status $p]\n"
            f"    puts \"PROP: $p STATUS: $st\"\n"
            f"}}\n"
            f"puts \"===END===\"\n"
            f"exit\n"
        )

        try:
            raw     = _run_jg_ssh(top_module, combined, tcl, tag="verify", timeout=180)
            parsed  = _parse_prove_results(raw)
            # Return JSON so agent can parse reliably
            result = json.dumps({
                "status": parsed["status"],
                "proven": parsed["proven"],
                "falsified": parsed["falsified"],
                "undetermined": parsed["undetermined"],
                "errors": parsed["errors"],
                "properties": parsed.get("properties", []),
                "raw": parsed["raw"][-2000:] if len(parsed.get("raw", "")) > 2000 else parsed.get("raw", ""),
            })
        except subprocess.TimeoutExpired:
            result = json.dumps({"status": "timeout", "proven": 0, "falsified": 0,
                                 "undetermined": 0, "errors": ["SSH timeout (180s)"], "raw": ""})
        except Exception as e:
            result = json.dumps({"status": "unknown", "proven": 0, "falsified": 0,
                                 "undetermined": 0, "errors": [str(e)], "raw": ""})

        self._log(
            "verify_sva",
            {"sva_len": len(sva_code), "top_module": top_module},
            result,
            time.monotonic() - t0,
        )
        return result

    def check_vacuity(self, sva_code: str, top_module: str = "", all_rtl: str = "") -> str:
        """Run JasperGold check_vacuity on SVA properties.

        First proves all properties (prove -all -time_limit 1m), then runs
        check_vacuity -all.  Returns per-property vacuity results.

        Args:
            sva_code:   SVA code to check.
            top_module: Top-level module (inferred if empty).
            all_rtl:    Combined RTL (collected automatically if empty).
        """
        t0 = time.monotonic()
        if not top_module:
            top_module = self._top_module_name()
        if not all_rtl:
            all_rtl = self._collect_all_rtl()

        sva_clean = _extract_sva_code(sva_code)
        combined  = _inject_sva(all_rtl, sva_clean, top_module=top_module)

        cr = _detect_clock_reset(all_rtl)
        tcl = (
            f"analyze -sv {{{self._jg_defines}}} dut.sv\n"
            f"elaborate -top {top_module}\n"
            f"{cr['reset_tcl']}"
            f"prove -all -time_limit 1m\n"
            f"puts \"===VACUITY_START===\"\n"
            f"catch {{check_vacuity -all}}\n"
            f"puts \"===VACUITY_END===\"\n"
            f"exit\n"
        )

        try:
            raw    = _run_jg_ssh(top_module, combined, tcl, tag="vacuity", timeout=180)
            # Extract vacuity section
            vac_match = re.search(
                r"===VACUITY_START===(.*?)===VACUITY_END===",
                raw,
                re.DOTALL,
            )
            vac_raw = vac_match.group(1).strip() if vac_match else raw

            # Parse per-property vacuity lines (typical format: "prop_name : <result>")
            prop_lines = re.findall(r"\S+\s*:\s*\S+[^\n]*", vac_raw)
            if prop_lines:
                lines = ["check_vacuity results:"]
                for ln in prop_lines:
                    lines.append(f"  {ln.strip()}")
                result = "\n".join(lines)
            elif vac_raw:
                result = "check_vacuity output:\n" + vac_raw
            else:
                result = "check_vacuity: no output parsed."
        except subprocess.TimeoutExpired:
            result = "[check_vacuity error] SSH timeout (180s)."
        except Exception as e:
            result = f"[check_vacuity error] {e}"

        self._log(
            "check_vacuity",
            {"sva_len": len(sva_code), "top_module": top_module},
            result,
            time.monotonic() - t0,
        )
        return result

    # ------------------------------------------------------------------
    # Log accessor
    # ------------------------------------------------------------------

    def get_tool_log(self) -> List[Dict]:
        """Return the list of tool call log entries.

        Each entry is a dict with keys:
          ``tool``, ``args``, ``result_len`` (chars), ``elapsed_s``.
        """
        return list(self._tool_call_log)


# ---------------------------------------------------------------------------
# TOOL_SCHEMAS — OpenAI / Ollama function-calling format
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_design",
            "description": (
                "Semantic search over AST chunks for this design. "
                "Use to find relevant modules, signals, behavioral context, "
                "or any design information by natural language query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. 'state machine transitions', 'output valid signal').",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_module_info",
            "description": (
                "Return the MODULE_INTERFACE chunk for a module: port list, "
                "parameter declarations, and module signature. "
                "Use before writing assertions to understand the module's interface."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": "Exact name of the Verilog/SV module (e.g. 'exec_unit_0').",
                    },
                },
                "required": ["module_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hierarchy",
            "description": (
                "Return the full module dependency tree for this design as a "
                "formatted string: topological order and parent→child edges. "
                "Use to understand the design structure before selecting signals."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_parameters",
            "description": (
                "Return all parameter and localparam values for a module. "
                "Returns a compact string like 'NS=8, WIDTH=128, DEPTH=10'. "
                "Use to find exact numeric constants needed in assertions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": "Name of the Verilog module.",
                    },
                },
                "required": ["module_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_always_blocks",
            "description": (
                "Return ALWAYS_BLOCK chunks that drive a signal in a module. "
                "Use to understand the clocking, reset behaviour, and assignment "
                "logic for a specific signal before writing temporal assertions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_name": {
                        "type": "string",
                        "description": "RTL signal identifier to look for (e.g. 'state', 'count').",
                    },
                    "module_name": {
                        "type": "string",
                        "description": "Module that contains the signal.",
                    },
                },
                "required": ["signal_name", "module_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fanin",
            "description": (
                "Run JasperGold get_fanin for a signal in a module via SSH. "
                "Returns the list of signals that feed into this signal (cone of influence). "
                "Authoritative but slower (~5–15s). Use when you need to trace data origins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "string",
                        "description": "Signal name (exact RTL identifier, e.g. 'out_data', 'state[2:0]').",
                    },
                    "module": {
                        "type": "string",
                        "description": "Module containing the signal.",
                    },
                },
                "required": ["signal", "module"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fanout",
            "description": (
                "Run JasperGold get_fanout for a signal in a module via SSH. "
                "Returns the list of signals driven by this signal. "
                "Use to understand what downstream logic depends on this signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "string",
                        "description": "Signal name (exact RTL identifier).",
                    },
                    "module": {
                        "type": "string",
                        "description": "Module containing the signal.",
                    },
                },
                "required": ["signal", "module"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flop_info",
            "description": (
                "Run JasperGold get_flop_info + get_signal_info for a signal via SSH. "
                "Returns: is_flop, clock, reset pin, reset type, reset value, "
                "signal direction (input/output/internal), and fanin list. "
                "Use to determine exact clock/reset for temporal assertions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "string",
                        "description": "Signal name (exact RTL identifier).",
                    },
                    "module": {
                        "type": "string",
                        "description": "Module containing the signal.",
                    },
                },
                "required": ["signal", "module"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_sva",
            "description": (
                "Inject SVA into the DUT RTL and run JasperGold prove -all -time_limit 1m. "
                "Returns status (proven/falsified/syntax_error/compilation_error/no_properties), "
                "proven/falsified/undetermined counts, and any error messages. "
                "Use to validate a candidate assertion before finalising it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sva_code": {
                        "type": "string",
                        "description": "SVA property and assert statements. Markdown fenced blocks are accepted.",
                    },
                    "top_module": {
                        "type": "string",
                        "description": "Top-level module name for 'elaborate -top'. Inferred automatically if omitted.",
                        "default": "",
                    },
                    "all_rtl": {
                        "type": "string",
                        "description": "Combined RTL source. Concatenated from design_dir automatically if omitted.",
                        "default": "",
                    },
                },
                "required": ["sva_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_vacuity",
            "description": (
                "Run JasperGold check_vacuity -all on SVA properties. "
                "Returns per-property vacuity results indicating whether the "
                "antecedent of each assertion is ever triggered. "
                "Use to detect assertions that are trivially true due to an "
                "unreachable antecedent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sva_code": {
                        "type": "string",
                        "description": "SVA code to check. Markdown fenced blocks are accepted.",
                    },
                    "top_module": {
                        "type": "string",
                        "description": "Top-level module name. Inferred automatically if omitted.",
                        "default": "",
                    },
                    "all_rtl": {
                        "type": "string",
                        "description": "Combined RTL source. Collected automatically if omitted.",
                        "default": "",
                    },
                },
                "required": ["sva_code"],
            },
        },
    },
]
