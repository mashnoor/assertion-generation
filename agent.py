"""
agent.py

ReAct (Reason + Act) agent for SystemVerilog Assertion (SVA) generation.

Architecture: Two-phase agent
  Phase A — Context Gathering (up to max_context_rounds=6 tool calls):
      Agent reasons about what it needs and calls tools to query the RTL design.
  Phase B — Generation + Verification (up to max_verify_rounds=5):
      Agent generates SVA, calls verify_sva, iteratively fixes on failure.

LLM: qwen3.5:35b via Ollama native API (flat /api/chat endpoint with tools parameter).
Tools: DesignTools class from tools.py, TOOL_SCHEMAS list for the Ollama tools API.
"""

import json
import os
import re
import signal
import time
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests as _http_client


# ---------------------------------------------------------------------------
# MinimalLLM — copied from pipeline_v3.py, extended for native tool calling
# ---------------------------------------------------------------------------

class _LLMResponse:
    """Wraps an LLM text response to mirror langchain AIMessage interface."""
    def __init__(self, content: str):
        self.content = content


class MinimalLLM:
    """Minimal requests-based LLM supporting Ollama native API and OpenAI-compatible APIs.

    Replaces langchain_openai.ChatOpenAI to avoid openai/httpx import hangs on HPC.
    """
    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 temperature: float = 0.1, timeout: int = 420,
                 native_ollama: bool = False):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.native_ollama = native_ollama

    def _msgs_to_dicts(self, messages):
        result = []
        for m in messages:
            if isinstance(m, dict):
                result.append(m)
            elif hasattr(m, 'type'):
                role = {"ai": "assistant", "system": "system"}.get(m.type, "user")
                result.append({"role": role, "content": m.content})
            else:
                result.append({"role": "user", "content": str(m)})
        return result

    def _call(self, messages, structured_schema=None):
        msg_dicts = self._msgs_to_dicts(messages)
        if self.native_ollama:
            body = {
                "model": self.model,
                "messages": msg_dicts,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            if structured_schema is not None:
                body["format"] = structured_schema.model_json_schema()
            headers = {"Content-Type": "application/json"}
            endpoint = f"{self.base_url}/api/chat"
            resp = _http_client.post(endpoint, json=body, headers=headers,
                                     timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        else:
            body = {
                "model": self.model,
                "messages": msg_dicts,
                "stream": False,
                "temperature": self.temperature,
            }
            if structured_schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured_schema.__name__,
                        "schema": structured_schema.model_json_schema(),
                    },
                }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            endpoint = f"{self.base_url}/chat/completions"
            resp = _http_client.post(endpoint, json=body, headers=headers,
                                     timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def invoke(self, messages):
        return _LLMResponse(self._call(messages))

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """Call Ollama native /api/chat with tools parameter. Returns raw message dict."""
        if not self.native_ollama:
            raise RuntimeError("chat_with_tools requires native_ollama=True")
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.base_url}/api/chat"
        resp = _http_client.post(endpoint, json=body, headers=headers,
                                 timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class AgentTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise AgentTimeoutError("Agent run exceeded timeout")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SVAResult:
    spec_id: str
    sva_code: str               # Final SVA
    status: str                 # proven | partially_proven | falsified | syntax_error | unknown | timeout
    proven: int
    falsified: int
    undetermined: int
    total: int
    vacuity_status: str         # vacuous | non_vacuous | not_checked | error
    tool_calls: List[Dict]      # log of all tool calls made
    jg_iterations: int          # how many verify rounds ran
    wall_time_s: float
    error_message: str          # populated if run failed/timed out
    context_rounds: int         # Phase A tool call rounds used


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are an expert SystemVerilog formal verification engineer. Your task is to \
generate correct, provable SVA assertions for a hardware design.

You have access to tools to query the RTL design. Use them to:
1. Find the exact signal names, widths, and array dimensions \
(CRITICAL: wrong array indices cause failures)
2. Understand the clock and reset behavior
3. Identify the temporal relationships needed for the assertion

STRATEGY:
- Start with get_parameters() to get exact NS, DEPTH, OPD, WIDTH values \
- this prevents index hallucination
- Use get_module_info() to see exact port types and widths
- Use get_always_blocks() to understand reset values and clock behavior
- Only call get_fanin()/get_flop_info() for signals you're uncertain about

When you have enough context, generate SVA. The assertions are injected at the \
top-level module scope.
SVA format: property declarations + assert statements only, no module wrapper.
"""

REACT_TOOL_INSTRUCTIONS = """\

To call a tool, output EXACTLY:
TOOL_CALL: {"tool": "tool_name", "args": {"param": "value"}}

To generate the final SVA, output:
SVA_OUTPUT:
<sva code here>
"""


# ---------------------------------------------------------------------------
# SVAAgent
# ---------------------------------------------------------------------------

class SVAAgent:
    """Two-phase ReAct agent for SVA generation and verification.

    Phase A: Context gathering via tool calls (up to max_context_rounds).
    Phase B: SVA generation + JasperGold verification loop (up to max_verify_rounds).
    """

    def __init__(self, llm: MinimalLLM, tools, spec_id: str,
                 debug_dir: Optional[str] = None,
                 max_context_rounds: int = 6,
                 max_verify_rounds: int = 5,
                 disabled_tools: Optional[set] = None):
        self.llm = llm
        self.tools = tools          # DesignTools instance from tools.py
        self.spec_id = spec_id
        self.debug_dir = debug_dir
        self.max_context_rounds = max_context_rounds
        self.max_verify_rounds = max_verify_rounds
        self.disabled_tools = disabled_tools or set()

        self.conversation: List[Dict] = []
        self.context_gathered: Dict[str, str] = {}  # tool_name -> result
        self._tool_call_log: List[Dict] = []
        self._jg_iterations: int = 0

        # Import TOOL_SCHEMAS lazily to avoid circular import issues
        try:
            from tools import TOOL_SCHEMAS
            if self.disabled_tools:
                self._tool_schemas = [
                    s for s in TOOL_SCHEMAS
                    if s.get("function", {}).get("name") not in self.disabled_tools
                ]
            else:
                self._tool_schemas = TOOL_SCHEMAS
        except ImportError:
            self._tool_schemas = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, spec_text: str, design_id: str, top_module: str = "",
            timeout_s: int = 480) -> SVAResult:
        """Run the full two-phase agent loop. Returns SVAResult."""
        start = time.time()

        # Set up signal-based timeout
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)

        try:
            result = self._run_inner(spec_text, design_id, top_module)
        except AgentTimeoutError:
            result = SVAResult(
                spec_id=self.spec_id,
                sva_code="",
                status="timeout",
                proven=0, falsified=0, undetermined=0, total=0,
                vacuity_status="not_checked",
                tool_calls=self._tool_call_log,
                jg_iterations=self._jg_iterations,
                wall_time_s=time.time() - start,
                error_message=f"Agent timed out after {timeout_s}s",
                context_rounds=0,
            )
        except Exception as e:
            result = SVAResult(
                spec_id=self.spec_id,
                sva_code="",
                status="unknown",
                proven=0, falsified=0, undetermined=0, total=0,
                vacuity_status="not_checked",
                tool_calls=self._tool_call_log,
                jg_iterations=self._jg_iterations,
                wall_time_s=time.time() - start,
                error_message=str(e),
                context_rounds=0,
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        result.wall_time_s = time.time() - start

        if self.debug_dir:
            self._save_debug(result)

        return result

    # ------------------------------------------------------------------
    # Inner run (no timeout wrapping)
    # ------------------------------------------------------------------

    def _build_design_context(self, design_id: str, top_module: str) -> str:
        """Pre-gather essential design context so the agent starts informed.

        Returns a formatted string with hierarchy, clock/reset, parameters,
        and top module interface — things the agent would otherwise waste
        tool-call rounds discovering.
        """
        parts = []

        # 1. Design hierarchy from design_graph.json
        if self.tools and hasattr(self.tools, 'design_dir'):
            graph_path = os.path.join(self.tools.design_dir, "design_graph.json")
            if os.path.exists(graph_path):
                try:
                    with open(graph_path) as f:
                        graph = json.load(f)
                    sorted_mods = graph.get("sorted_modules", [])
                    adj = graph.get("adjacency_list", {})
                    if sorted_mods:
                        parts.append(f"Module hierarchy (topological): {' → '.join(sorted_mods)}")
                    if adj:
                        deps = "; ".join(f"{k} instantiates {', '.join(v)}" for k, v in adj.items() if v)
                        if deps:
                            parts.append(f"Dependencies: {deps}")
                except Exception:
                    pass

        # 2. Clock/reset detection
        if self.tools and hasattr(self.tools, 'get_clock_reset_info'):
            try:
                cr = self.tools.get_clock_reset_info()
                self._clock_reset_info = cr  # cache for later prompts
                clock = cr.get("clock", "clk")
                reset_name = cr.get("reset_name", "")
                active_low = cr.get("reset_active_low", True)
                disable_iff = cr.get("disable_iff", "!reset_")
                parts.append(
                    f"Clock: {clock}  |  Reset: {reset_name or '(not detected)'}"
                    f"  (active {'low' if active_low else 'high'})"
                    f"  →  use `disable iff ({disable_iff})` in properties"
                )
            except Exception:
                self._clock_reset_info = {}
        else:
            self._clock_reset_info = {}

        # 3. Top module interface (ports + parameters) via get_module_info
        if self.tools and top_module:
            try:
                iface = self.tools.get_module_info(top_module)
                if iface and "error" not in iface.lower() and "no results" not in iface.lower():
                    # Truncate if huge
                    if len(iface) > 2000:
                        iface = iface[:2000] + "\n... (truncated)"
                    parts.append(f"Top module interface ({top_module}):\n{iface}")
            except Exception:
                pass

        # 4. Parameters via get_parameters
        if self.tools and top_module:
            try:
                params = self.tools.get_parameters(top_module)
                if params and "error" not in params.lower() and "no results" not in params.lower():
                    parts.append(f"Parameters:\n{params}")
            except Exception:
                pass

        return "\n\n".join(parts) if parts else ""

    def _run_inner(self, spec_text: str, design_id: str, top_module: str) -> SVAResult:
        start = time.time()

        # Pre-gather design context so agent starts with key facts
        design_context = self._build_design_context(design_id, top_module)

        # Initialize conversation with system prompt
        self.conversation = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        ]

        # Inject spec as first user message with pre-loaded context
        user_msg = (
            f"Design ID: {design_id}\n"
            f"Top module: {top_module or 'unknown'}\n\n"
        )
        if design_context:
            user_msg += f"=== Pre-loaded design context ===\n{design_context}\n=== End pre-loaded context ===\n\n"
        user_msg += (
            f"Assertion specification:\n{spec_text}\n\n"
            "Use the pre-loaded context above to guide your tool calls. "
            "Focus on finding the specific signals and behaviors needed for this assertion."
        )
        self.conversation.append({"role": "user", "content": user_msg})

        # ---- Phase A: Context Gathering ----
        context_rounds = 0
        for _ in range(self.max_context_rounds):
            msg = self._llm_call_with_tools()
            role_msg = {"role": "assistant", "content": msg.get("content") or ""}
            # Preserve tool_calls in assistant message if present
            if msg.get("tool_calls"):
                role_msg["tool_calls"] = msg["tool_calls"]
            self.conversation.append(role_msg)

            tool_calls = msg.get("tool_calls") or []
            # Fallback: try to parse tool calls from text content
            if not tool_calls and msg.get("content"):
                tool_calls = self._try_parse_tool_calls(msg["content"])

            if not tool_calls:
                # Model decided no more tools needed — exit Phase A
                break

            context_rounds += 1
            for tc in tool_calls:
                tool_result = self._execute_tool_call(tc)
                # Append tool result as tool role message
                self.conversation.append({
                    "role": "tool",
                    "content": tool_result,
                })

            # Check if model signaled SVA output in content
            if msg.get("content") and "SVA_OUTPUT:" in msg["content"]:
                break

        # ---- Phase B: Generation + Verification ----
        sva_code = self._generate_sva(spec_text)

        final_status = "unknown"
        proven = falsified = undetermined = total = 0
        vacuity_status = "not_checked"
        error_message = ""

        # If max_verify_rounds == 0 (ablation: no verification), skip Phase B
        if self.max_verify_rounds == 0:
            # Save the SVA and return without verification
            if self.debug_dir:
                sv_path = os.path.join(self.debug_dir, "sva_assertion.sv")
                with open(sv_path, "w") as f:
                    f.write(sva_code)
            return SVAResult(
                spec_id=self.spec_id,
                sva_code=sva_code,
                status="no_verification",
                proven=0, falsified=0, undetermined=0, total=0,
                vacuity_status="not_checked",
                tool_calls=self._tool_call_log,
                jg_iterations=0,
                wall_time_s=time.time() - start,
                error_message="ablation: verification disabled",
                context_rounds=context_rounds,
            )

        for verify_round in range(1, self.max_verify_rounds + 1):
            self._jg_iterations = verify_round

            t0 = time.time()
            jg_result = self._call_verify_sva(sva_code)
            elapsed = time.time() - t0

            self._tool_call_log.append({
                "phase": "B",
                "round": verify_round,
                "tool": "verify_sva",
                "args": {"sva_length": len(sva_code)},
                "result_summary": {
                    "status": jg_result.get("status"),
                    "proven": jg_result.get("proven", 0),
                    "falsified": jg_result.get("falsified", 0),
                    "undetermined": jg_result.get("undetermined", 0),
                },
                "elapsed_s": round(elapsed, 2),
            })

            # Save verification round output
            if self.debug_dir:
                spec_dir = self.debug_dir
                os.makedirs(spec_dir, exist_ok=True)
                vpath = os.path.join(spec_dir, f"verification_v{verify_round}.json")
                with open(vpath, "w") as f:
                    json.dump(jg_result, f, indent=2)
                sv_path = os.path.join(spec_dir, f"sva_assertion_v{verify_round}.sv")
                with open(sv_path, "w") as f:
                    f.write(sva_code)

            status = jg_result.get("status", "unknown")
            proven = jg_result.get("proven", 0)
            falsified = jg_result.get("falsified", 0)
            undetermined = jg_result.get("undetermined", 0)
            total = proven + falsified + undetermined
            final_status = status

            if status in ("proven", "partially_proven"):
                # Check vacuity
                vacuity_status = self._call_check_vacuity(sva_code)
                break

            if verify_round == self.max_verify_rounds:
                error_message = "; ".join(jg_result.get("errors", []))
                break

            # Request fix from LLM
            error_info = "\n".join(jg_result.get("errors") or [jg_result.get("raw", "")[-2000:]])
            sva_code = self._fix_sva(sva_code, error_info, spec_text, jg_result)

        return SVAResult(
            spec_id=self.spec_id,
            sva_code=sva_code,
            status=final_status,
            proven=proven,
            falsified=falsified,
            undetermined=undetermined,
            total=total,
            vacuity_status=vacuity_status,
            tool_calls=self._tool_call_log,
            jg_iterations=self._jg_iterations,
            wall_time_s=time.time() - start,
            error_message=error_message,
            context_rounds=context_rounds,
        )

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _llm_call_with_tools(self) -> Dict:
        """Call the LLM with tool schemas. Returns raw message dict from Ollama."""
        if self._tool_schemas and getattr(self.llm, 'native_ollama', False):
            try:
                return self.llm.chat_with_tools(self.conversation, self._tool_schemas)
            except Exception:
                pass
        # Fallback: plain call with ReAct instructions injected into last user message
        augmented_conv = list(self.conversation)
        # Inject ReAct instructions if not already present
        if augmented_conv and augmented_conv[-1]["role"] == "user":
            augmented_conv[-1] = dict(augmented_conv[-1])
            if "TOOL_CALL:" not in augmented_conv[-1]["content"]:
                augmented_conv[-1]["content"] += REACT_TOOL_INSTRUCTIONS
        resp = self.llm.invoke(augmented_conv)
        return {"content": resp.content, "tool_calls": []}

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tc: Dict) -> str:
        """Dispatch a tool call (from native tool_calls or parsed fallback)."""
        # Normalize: tc may be {"function": {"name": ..., "arguments": ...}} (Ollama native)
        # or {"tool": ..., "args": ...} (ReAct fallback)
        if "function" in tc:
            tool_name = tc["function"].get("name", "")
            args_raw = tc["function"].get("arguments", {})
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {}
            else:
                args = args_raw or {}
        else:
            tool_name = tc.get("tool", "")
            args = tc.get("args", {})

        # Block disabled tools (ablation)
        if tool_name in self.disabled_tools:
            return f"Tool '{tool_name}' is not available in this configuration."

        t0 = time.time()
        result = self._dispatch_tool(tool_name, args)
        elapsed = time.time() - t0

        self.context_gathered[tool_name] = result
        self._tool_call_log.append({
            "phase": "A",
            "tool": tool_name,
            "args": args,
            "result_length": len(result),
            "elapsed_s": round(elapsed, 2),
        })
        return result

    def _dispatch_tool(self, tool_name: str, args: dict) -> str:
        """Call the named tool with given args, return result string."""
        if self.tools is None:
            return f"[Error: no tools configured for tool '{tool_name}']"
        try:
            method = getattr(self.tools, tool_name, None)
            if method is None:
                return f"[Error: unknown tool '{tool_name}']"
            result = method(**args)
            if not isinstance(result, str):
                result = json.dumps(result, indent=2)
            return result
        except Exception as e:
            return f"[Tool error in '{tool_name}': {e}]"

    # ------------------------------------------------------------------
    # ReAct text fallback parser
    # ------------------------------------------------------------------

    def _try_parse_tool_calls(self, content: str) -> List[Dict]:
        """Fallback: parse tool calls from text content when native function calling failed.

        Looks for patterns like:
            TOOL_CALL: {"tool": "tool_name", "args": {...}}
        """
        calls = []
        # Match TOOL_CALL: {...} lines
        for match in re.finditer(r'TOOL_CALL:\s*(\{.*?\})', content, re.DOTALL):
            try:
                parsed = json.loads(match.group(1))
                if "tool" in parsed:
                    calls.append(parsed)
            except Exception:
                pass
        return calls

    # ------------------------------------------------------------------
    # SVA generation
    # ------------------------------------------------------------------

    def _extract_reset_signal_hint(self) -> str:
        """Try to infer the reset signal name from design RTL files.

        Scans rtl.sv files in design_dir subdirs for input reset signal patterns.
        Returns a hint string for the generation prompt, or empty string if not found.
        """
        if not self.tools or not hasattr(self.tools, 'design_dir'):
            return ""
        design_dir = self.tools.design_dir
        # Priority-ordered (most specific first): reset_, rst_n, rst, reset
        reset_patterns = [
            (r'\binput\b[^\n]*\breset_\b', 'reset_'),
            (r'\binput\b[^\n]*\brst_n\b',  'rst_n'),
            (r'\binput\b[^\n]*\brst\b',    'rst'),
            (r'\binput\b[^\n]*\breset\b',  'reset'),
        ]
        try:
            for entry in sorted(os.scandir(design_dir), key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                rtl_path = os.path.join(entry.path, "rtl.sv")
                if not os.path.exists(rtl_path):
                    continue
                with open(rtl_path) as f:
                    rtl = f.read()
                for pattern, name in reset_patterns:
                    if re.search(pattern, rtl):
                        active_low = name.endswith('_') or name.endswith('_n')
                        polarity = f"!{name}" if active_low else name
                        return (
                            f"Reset signal: {name}  "
                            f"(use `disable iff ({polarity})` in properties)\n\n"
                        )
        except Exception:
            pass
        return ""

    def _generate_sva(self, spec_text: str) -> str:
        """Ask LLM to generate SVA based on gathered context."""
        context_summary = self._format_context_summary()

        # Build clock/reset directive from cached detection
        cr = getattr(self, '_clock_reset_info', {})
        clock = cr.get("clock", "clk")
        disable_iff = cr.get("disable_iff", "!reset_")
        cr_directive = (
            f"Clock: {clock}  |  disable iff: ({disable_iff})\n"
            f"Use `@(posedge {clock}) disable iff ({disable_iff})` in ALL properties.\n"
        )

        prompt = (
            f"Based on all the context gathered, generate the SVA assertion for:\n\n"
            f"{spec_text}\n\n"
            f"Gathered context:\n{context_summary}\n\n"
            f"=== Clock/Reset (MUST use these exact signals) ===\n{cr_directive}\n"
            "RULES:\n"
            "1. MUST include at least one `assert property(name);` statement.\n"
            "2. Use ONLY signal names that appear in the gathered context — never guess.\n"
            "3. Match array widths/indices exactly to parameter values from context.\n"
            "4. Assert EXACTLY what the spec says — no more, no less. "
            "If the spec says 'unconditionally', do NOT add input conditions. "
            "If the spec says 'when X', only check condition X.\n"
            "5. No module wrapper, no explanations — output ONLY property declarations "
            "and assert statements.\n"
        )
        self.conversation.append({"role": "user", "content": prompt})

        resp = self.llm.invoke(self.conversation)
        sva_code = resp.content
        self.conversation.append({"role": "assistant", "content": sva_code})

        # Strip markdown fences if present
        sva_code = _extract_sva_code(sva_code)

        # Fallback: if no assert property found, auto-insert for last declared property
        if "assert property" not in sva_code:
            prop_match = re.search(r'property\s+(\w+)', sva_code)
            if prop_match:
                prop_name = prop_match.group(1)
                sva_code += f"\nassert property ({prop_name});"

        return sva_code

    def _fix_sva(self, sva_code: str, error_info: str, spec_text: str,
                 jg_result: Dict) -> str:
        """Ask LLM to fix SVA given JG error feedback.

        May also call get_always_blocks or get_fanin for failing signals before
        generating the fix, following the Phase B strategy.
        """
        status = jg_result.get("status", "unknown")
        errors = jg_result.get("errors", [])

        # For falsified/undetermined, try to pull extra signal context first
        if status in ("falsified", "undetermined") and self.tools is not None:
            failing_signals = _extract_signal_names_from_errors(error_info)
            for sig in failing_signals[:2]:  # limit to 2 extra lookups
                extra = self._dispatch_tool("get_always_blocks", {"signal": sig})
                if extra and "Error" not in extra:
                    self.context_gathered[f"get_always_blocks({sig})"] = extra
                    self._tool_call_log.append({
                        "phase": "B_fix",
                        "tool": "get_always_blocks",
                        "args": {"signal": sig},
                        "result_length": len(extra),
                    })

        # Build clock/reset directive from cached detection
        cr = getattr(self, '_clock_reset_info', {})
        clock = cr.get("clock", "clk")
        disable_iff = cr.get("disable_iff", "!reset_")
        cr_directive = (
            f"Clock: {clock}  |  disable iff: ({disable_iff})\n"
            f"Use `@(posedge {clock}) disable iff ({disable_iff})` in ALL properties.\n"
        )

        fix_prompt = (
            f"The SVA assertion failed JasperGold verification.\n\n"
            f"Specification:\n{spec_text}\n\n"
            f"Failed SVA:\n```systemverilog\n{sva_code}\n```\n\n"
            f"JasperGold status: {status}\n"
            f"Errors/feedback:\n{error_info}\n\n"
            f"=== Clock/Reset (MUST use these exact signals) ===\n{cr_directive}\n"
            "Analyze the error and produce a corrected SVA.\n"
            "Use ONLY signal names from the gathered context — never invent signals.\n"
            "Output ONLY the corrected property declarations and assert statements."
        )
        self.conversation.append({"role": "user", "content": fix_prompt})
        resp = self.llm.invoke(self.conversation)
        fixed = resp.content
        self.conversation.append({"role": "assistant", "content": fixed})
        return _extract_sva_code(fixed)

    # ------------------------------------------------------------------
    # Verification tool calls (delegated to DesignTools)
    # ------------------------------------------------------------------

    def _call_verify_sva(self, sva_code: str) -> Dict:
        """Invoke verify_sva tool and return parsed JG result dict."""
        result_str = self._dispatch_tool("verify_sva", {"sva_code": sva_code})
        try:
            return json.loads(result_str)
        except Exception:
            # Parse from raw string if JSON failed
            return _parse_prove_results_minimal(result_str)

    def _call_check_vacuity(self, sva_code: str) -> str:
        """Invoke check_vacuity tool and return status string."""
        result_str = self._dispatch_tool("check_vacuity", {"sva_code": sva_code})
        try:
            data = json.loads(result_str)
            return data.get("vacuity_status", "not_checked")
        except Exception:
            if "non_vacuous" in result_str.lower() or "non-vacuous" in result_str.lower():
                return "non_vacuous"
            if "vacuous" in result_str.lower():
                return "vacuous"
            return "not_checked"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_context_summary(self) -> str:
        """Format all gathered context into a readable string."""
        if not self.context_gathered:
            return "(no tool context gathered)"
        parts = []
        for tool_name, result in self.context_gathered.items():
            parts.append(f"### {tool_name}\n{result}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------

    def _save_debug(self, result: SVAResult):
        """Save debug artifacts to debug_dir/ (caller already scoped to spec)."""
        spec_dir = self.debug_dir
        os.makedirs(spec_dir, exist_ok=True)

        # agent_conversation.json — full message history
        conv_path = os.path.join(spec_dir, "agent_conversation.json")
        with open(conv_path, "w") as f:
            json.dump(self.conversation, f, indent=2, default=str)

        # tool_call_log.json — all tool calls with timing
        log_path = os.path.join(spec_dir, "tool_call_log.json")
        with open(log_path, "w") as f:
            json.dump(self._tool_call_log, f, indent=2)

        # sva_assertion.sv — final SVA
        sv_path = os.path.join(spec_dir, "sva_assertion.sv")
        with open(sv_path, "w") as f:
            f.write(result.sva_code)

        # verification_log.json — all JG verification attempts
        vlog_path = os.path.join(spec_dir, "verification_log.json")
        # Extract verification entries from tool_call_log
        verify_entries = [
            entry for entry in self._tool_call_log
            if entry.get("tool") == "verify_sva"
        ]
        with open(vlog_path, "w") as f:
            json.dump(verify_entries, f, indent=2)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _extract_sva_code(text: str) -> str:
    """Extract SVA from markdown code blocks or SVA_OUTPUT section if present."""
    # Check for SVA_OUTPUT: marker (ReAct fallback)
    sva_out_match = re.search(r'SVA_OUTPUT:\s*\n(.*)', text, re.DOTALL)
    if sva_out_match:
        return sva_out_match.group(1).strip()
    # Strip markdown fences
    fence_match = re.compile(
        r'```(?:systemverilog|sv|verilog)?\s*\n(.*?)```', re.DOTALL
    ).search(text)
    if fence_match:
        return fence_match.group(1).strip()
    # Return as-is, filtering comment-only lines that aren't assertions
    lines = text.strip().splitlines()
    sva_lines = [l for l in lines if not l.strip().startswith("//") or "assert" in l.lower()]
    return "\n".join(sva_lines) if sva_lines else text.strip()


def _extract_signal_names_from_errors(error_text: str) -> List[str]:
    """Extract likely signal names from JasperGold error messages."""
    # Look for quoted identifiers or identifier-like tokens near "error" or "undetermined"
    candidates = re.findall(r"'([a-zA-Z_][a-zA-Z0-9_\[\]\.]*)'", error_text)
    candidates += re.findall(r'"([a-zA-Z_][a-zA-Z0-9_\[\]\.]*)"', error_text)
    # Deduplicate, keep order, skip common keywords
    _keywords = {"property", "assert", "assume", "cover", "sequence",
                 "endproperty", "module", "endmodule", "input", "output",
                 "logic", "reg", "wire", "always", "begin", "end"}
    seen = set()
    result = []
    for c in candidates:
        if c not in _keywords and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _parse_prove_results_minimal(raw: str) -> Dict:
    """Minimal JasperGold prove output parser — mirrors _parse_prove_results in tools.py."""
    result = {
        "status": "unknown",
        "proven": 0, "falsified": 0, "undetermined": 0,
        "errors": [], "raw": raw,
    }

    if re.search(r"syntax error", raw, re.IGNORECASE):
        result["status"] = "syntax_error"
        result["errors"] = re.findall(r"(?:ERROR|syntax error)[^\n]*", raw, re.IGNORECASE)[:10]
        return result

    all_errors = re.findall(r"\[?ERROR[^\n]*", raw, re.IGNORECASE)
    if all_errors and not re.search(r"===RESULTS===", raw):
        result["status"] = "compilation_error"
        result["errors"] = all_errors[:10]
        return result

    # Primary: parse structured PROP: <name> STATUS: <status> lines
    results_match = re.search(r"===RESULTS===(.*?)(?:===END===|$)", raw, re.DOTALL)
    if results_match:
        for _, prop_status in re.findall(r"PROP:\s*(\S+)\s+STATUS:\s*(\S+)",
                                          results_match.group(1), re.IGNORECASE):
            st = prop_status.strip().lower()
            if st == "proven":
                result["proven"] += 1
            elif st in ("falsified", "cex"):
                result["falsified"] += 1
            else:
                result["undetermined"] += 1

    # Fallback if no structured lines
    if result["proven"] + result["falsified"] + result["undetermined"] == 0:
        result["proven"] = len(re.findall(r"STATUS:\s*proven", raw, re.IGNORECASE))
        result["falsified"] = len(re.findall(r"STATUS:\s*(?:falsified|cex)", raw, re.IGNORECASE))
        result["undetermined"] = len(re.findall(r"STATUS:\s*undetermined", raw, re.IGNORECASE))

    total = result["proven"] + result["falsified"] + result["undetermined"]
    if total > 0:
        if result["falsified"] == 0:
            result["status"] = "proven" if result["undetermined"] == 0 else "partially_proven"
        else:
            result["status"] = "falsified"
    else:
        result["status"] = "no_properties"
    return result
