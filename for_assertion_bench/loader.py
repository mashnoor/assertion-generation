"""
loader.py — Load AssertionBench dataset for assertion generation experiments.

Scans verified_assertions/ and Jasper_results/ for test designs with ground truth.
Each test design has: .v files, FPV_*.tcl (clock/reset), property.sva.tar.gz (ground truth).
Also loads 7 ICL examples (small designs with proven assertions) for few-shot prompting.
"""

import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # assertion-generation/
DATA_DIR = os.path.join(_PROJECT_DIR, "assertion_data_for_LLM")

ICL_NAMES = ["Arbiter", "full_adder", "full_subtractor", "half_adder",
             "half_subtractor", "jk_ff", "t_ff"]


# ---------------------------------------------------------------------------
# TCL parsing
# ---------------------------------------------------------------------------

def parse_tcl(tcl_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (module_name, clock, reset) from FPV_*.tcl.

    Returns None for clock/reset if they use -infer/-none.
    """
    module_name = clock = reset = None
    with open(tcl_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"elaborate\s+-top\s+(\w+)", line)
            if m:
                module_name = m.group(1)
            m = re.match(r"clock\s+(\S+)", line)
            if m and m.group(1) != "-infer":
                clock = m.group(1)
            m = re.match(r"reset\s+(.+)", line)
            if m:
                val = m.group(1).strip()
                if val == "-none":
                    reset = None
                elif val.startswith("-expression"):
                    expr_m = re.search(r"\{!?(\w+)\}", val)
                    if expr_m:
                        reset = expr_m.group(1)
                else:
                    reset = val.split()[0]
    return module_name, clock, reset


# ---------------------------------------------------------------------------
# RTL loading
# ---------------------------------------------------------------------------

def read_rtl_files(design_dir: str) -> str:
    """Read and concatenate all .v files in directory, stripping `include directives."""
    v_files = sorted(f for f in os.listdir(design_dir) if f.endswith(".v"))
    if not v_files:
        return ""
    parts = []
    for vf in v_files:
        with open(os.path.join(design_dir, vf)) as f:
            content = f.read()
        # Remove include directives since we concatenate everything
        content = re.sub(r'`include\s+"[^"]*"', "", content)
        parts.append(content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth_sva(tar_path: str) -> str:
    """Extract the full SVA content from property.sva.tar.gz."""
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".sva"):
                f = tar.extractfile(member)
                if f:
                    return f.read().decode("utf-8", errors="replace")
    return ""


def count_assertions(sva_text: str) -> int:
    """Count assert property(...) statements in SVA text."""
    return len(re.findall(r"assert\s+property\s*\(", sva_text))


# ---------------------------------------------------------------------------
# Design scanner
# ---------------------------------------------------------------------------

def _find_designs_in(base_dir: str, skip_icl: bool = True) -> Dict[str, dict]:
    """Scan a directory (verified_assertions or Jasper_results) for test designs."""
    designs = {}
    if not os.path.isdir(base_dir):
        return designs

    for category in sorted(os.listdir(base_dir)):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue
        if skip_icl and category in ICL_NAMES:
            continue
        # Skip the append_exit.py script
        if category.endswith(".py"):
            continue

        # Check if category itself is a design (has property.sva.tar.gz)
        _try_add(designs, cat_path, category, category)

        # Check subdirectories
        for item in sorted(os.listdir(cat_path)):
            item_path = os.path.join(cat_path, item)
            if not os.path.isdir(item_path):
                continue
            design_id = f"{category}__{item}"
            _try_add(designs, item_path, design_id, category)

    return designs


def _try_add(designs: dict, design_dir: str, design_id: str, category: str):
    """Try to add a design from design_dir if it has ground truth + RTL + TCL."""
    if design_id in designs:
        return
    tar_path = os.path.join(design_dir, "property.sva.tar.gz")
    if not os.path.exists(tar_path):
        return

    # Find FPV_*.tcl
    tcl_files = [f for f in os.listdir(design_dir)
                 if f.startswith("FPV_") and f.endswith(".tcl")]
    if not tcl_files:
        return

    module_name, clock, reset = parse_tcl(os.path.join(design_dir, tcl_files[0]))

    # Read RTL
    rtl = read_rtl_files(design_dir)
    if not rtl.strip():
        return

    # Extract ground truth
    gt_sva = extract_ground_truth_sva(tar_path)
    gt_count = count_assertions(gt_sva)

    designs[design_id] = {
        "design_id": design_id,
        "category": category,
        "module_name": module_name or design_id.split("__")[-1],
        "clock": clock,
        "reset": reset,
        "rtl": rtl,
        "design_dir": design_dir,
        "ground_truth_count": gt_count,
        "ground_truth_sva": gt_sva,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_test_designs() -> List[dict]:
    """Load all test designs from the AssertionBench dataset.

    Returns a list of design dicts sorted by RTL length (ascending complexity).
    """
    designs = {}

    # Scan both directories; verified_assertions first, Jasper_results supplements
    for subdir in ["verified_assertions", "Jasper_results"]:
        found = _find_designs_in(os.path.join(DATA_DIR, subdir), skip_icl=True)
        for did, info in found.items():
            if did not in designs:
                designs[did] = info

    result = list(designs.values())
    result.sort(key=lambda d: len(d["rtl"]))
    return result


def load_icl_examples(k: int = 5) -> List[dict]:
    """Load ICL examples for few-shot prompting.

    Returns up to k examples, each with {name, rtl, assertions_text}.
    """
    examples = []
    jasper_dir = os.path.join(DATA_DIR, "Jasper_results")

    for name in ICL_NAMES:
        if len(examples) >= k:
            break
        icl_dir = os.path.join(jasper_dir, name)
        if not os.path.isdir(icl_dir):
            continue

        # Read RTL (single .v file)
        v_files = [f for f in os.listdir(icl_dir) if f.endswith(".v")]
        if not v_files:
            continue
        with open(os.path.join(icl_dir, v_files[0])) as f:
            rtl = f.read()

        # Read assertions — try property_goldmine.sva first, then tar.gz
        assertions_text = ""
        goldmine = os.path.join(icl_dir, "property_goldmine.sva")
        if os.path.exists(goldmine):
            with open(goldmine) as f:
                raw = f.read()
            # Extract just assert lines for cleaner ICL
            asserts = re.findall(r"(?:property\s+\w+;\s*\n\s*@.*?endproperty\s*\n\s*\w+:\s*assert\s+property.*?;)",
                                 raw, re.DOTALL)
            if asserts:
                assertions_text = "\n\n".join(asserts)
            else:
                # Fallback: extract inline assert property statements
                inline = re.findall(r"assert\s+property\s*\(.*?\);", raw, re.DOTALL)
                assertions_text = "\n".join(inline) if inline else raw
        else:
            tar_path = os.path.join(icl_dir, "property.sva.tar.gz")
            if os.path.exists(tar_path):
                sva_content = extract_ground_truth_sva(tar_path)
                # Extract inline assertions
                inline = re.findall(r"assert\s+property\s*\(.*?\);", sva_content, re.DOTALL)
                # Limit to first 15 for ICL (some designs have thousands)
                assertions_text = "\n".join(inline[:15])

        if rtl.strip() and assertions_text.strip():
            examples.append({
                "name": name,
                "rtl": rtl.strip(),
                "assertions_text": assertions_text.strip(),
            })

    return examples


# ---------------------------------------------------------------------------
# CLI: print dataset summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading ICL examples...")
    icl = load_icl_examples(k=7)
    print(f"  Found {len(icl)} ICL examples:")
    for ex in icl:
        print(f"    {ex['name']}: RTL={len(ex['rtl'])} chars, "
              f"assertions={len(ex['assertions_text'])} chars")

    print("\nLoading test designs...")
    designs = load_test_designs()
    print(f"  Found {len(designs)} test designs:")
    for d in designs:
        print(f"    {d['design_id']}: module={d['module_name']} "
              f"clk={d['clock']} rst={d['reset']} "
              f"rtl={len(d['rtl'])} chars  gt={d['ground_truth_count']} assertions")
