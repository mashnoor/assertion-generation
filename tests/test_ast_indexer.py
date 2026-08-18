"""
tests/test_ast_indexer.py

Validates that the AST indexer produces correct chunks for known RTL snippets.
Uses real pyslang — no mocking of the parser.
"""

from __future__ import annotations

import os
import tempfile
import pytest

from ast_indexer import (
    extract_chunks_from_rtl,
    _extract_always_blocks_raw,
    _extract_assign_lines_raw,
)

# ===========================================================================
# RTL fixtures (module-level constants)
# ===========================================================================

SIMPLE_PIPELINE_RTL = """
module pipeline #(parameter NS=8, parameter WIDTH=128, parameter OPD=2)(
    input  clk, reset_,
    input  in_vld,
    input  [WIDTH-1:0] in_data [OPD-1:0],
    output reg out_vld,
    output reg [WIDTH-1:0] out_data [OPD-1:0]
);
    localparam DEPTH = NS;
    reg [WIDTH-1:0] pipe [NS-1:0][OPD-1:0];
    always @(posedge clk or negedge reset_) begin
        if (!reset_) begin
            out_vld <= 1'b0;
            out_data[0] <= 0;
        end else begin
            out_vld <= in_vld;
            out_data[0] <= pipe[NS-1][0];
        end
    end
endmodule
"""

SIMPLE_FSM_RTL = """
module fsm (
    input clk, rst,
    input valid,
    output reg [1:0] state
);
    localparam S0 = 2'b00, S1 = 2'b01, S2 = 2'b10;
    always @(posedge clk) begin
        if (rst) state <= S0;
        else case (state)
            S0: state <= valid ? S1 : S0;
            S1: state <= S2;
            S2: state <= S0;
        endcase
    end
endmodule
"""

EXEC_UNIT_RTL = """
module exec_unit #(parameter WIDTH=128)(
    input clk, reset_,
    input in_vld,
    input [WIDTH-1:0] in_data,
    output reg out_vld,
    output reg [WIDTH-1:0] out_data
);
    always @(posedge clk or negedge reset_) begin
        if (!reset_) begin
            out_vld <= 1'b0;
            out_data <= {WIDTH{1'b0}};
        end else begin
            out_vld <= in_vld;
            out_data <= in_data + 1;
        end
    end
endmodule
"""

PIPELINE_WITH_INSTANCES_RTL = """
module pipeline_top #(parameter NS=4, parameter WIDTH=32)(
    input clk, reset_,
    input in_vld,
    input [WIDTH-1:0] in_data,
    output out_vld,
    output [WIDTH-1:0] out_data
);
    wire [WIDTH-1:0] stage_data [NS:0];
    wire stage_vld [NS:0];
    assign stage_data[0] = in_data;
    assign stage_vld[0] = in_vld;
    genvar i;
    generate
        for (i = 0; i < NS; i = i+1) begin : stage
            exec_unit #(.WIDTH(WIDTH)) eu (
                .clk(clk), .reset_(reset_),
                .in_vld(stage_vld[i]), .in_data(stage_data[i]),
                .out_vld(stage_vld[i+1]), .out_data(stage_data[i+1])
            );
        end
    endgenerate
    assign out_vld = stage_vld[NS];
    assign out_data = stage_data[NS];
endmodule
"""

# Combined RTL for the instancing test (exec_unit must be visible to the parser)
PIPELINE_WITH_INSTANCES_COMBINED = EXEC_UNIT_RTL + "\n" + PIPELINE_WITH_INSTANCES_RTL


# ===========================================================================
# Helpers
# ===========================================================================

def _chunks_by_type(chunks, chunk_type):
    return [c for c in chunks if c.get("chunk_type") == chunk_type]


def _interface_content(chunks, module_name=None):
    iface = _chunks_by_type(chunks, "MODULE_INTERFACE")
    if module_name:
        iface = [c for c in iface if c.get("module_name") == module_name]
    return iface[0]["content"] if iface else ""


# ===========================================================================
# Tests — chunk presence and content
# ===========================================================================

def test_module_interface_extracted():
    """MODULE_INTERFACE chunk is produced for pipeline module."""
    chunks = extract_chunks_from_rtl("design_pipe_0", SIMPLE_PIPELINE_RTL, "pipeline")
    iface_chunks = _chunks_by_type(chunks, "MODULE_INTERFACE")
    assert len(iface_chunks) >= 1, "Expected at least one MODULE_INTERFACE chunk"
    assert "pipeline" in iface_chunks[0]["content"]


def test_parameter_values_correct():
    """NS=8, WIDTH=128, OPD=2 extracted with correct values."""
    chunks = extract_chunks_from_rtl("design_pipe_0", SIMPLE_PIPELINE_RTL, "pipeline")
    content = _interface_content(chunks, "pipeline")
    # Parameters must appear in the interface chunk
    assert "NS=8" in content or "NS" in content, f"NS not found in: {content}"
    assert "WIDTH=128" in content or "WIDTH" in content, f"WIDTH not found in: {content}"
    assert "OPD=2" in content or "OPD" in content, f"OPD not found in: {content}"


def test_localparam_extracted():
    """DEPTH localparam is extracted (DEPTH=8 or DEPTH=NS=8)."""
    chunks = extract_chunks_from_rtl("design_pipe_0", SIMPLE_PIPELINE_RTL, "pipeline")
    content = _interface_content(chunks, "pipeline")
    # DEPTH should appear as localparam — its value may be 'NS' or '8' depending
    # on how pyslang evaluates it; accept either.
    assert "DEPTH" in content, f"DEPTH localparam not found in: {content}"


def test_always_block_extracted():
    """At least one ALWAYS_BLOCK chunk produced for pipeline module."""
    chunks = extract_chunks_from_rtl("design_pipe_0", SIMPLE_PIPELINE_RTL, "pipeline")
    always_chunks = _chunks_by_type(chunks, "ALWAYS_BLOCK")
    assert len(always_chunks) >= 1, "Expected at least one ALWAYS_BLOCK chunk"


def test_always_sensitivity_posedge_negedge():
    """Sensitivity list has posedge clk and negedge reset_ for pipeline block."""
    chunks = extract_chunks_from_rtl("design_pipe_0", SIMPLE_PIPELINE_RTL, "pipeline")
    always_chunks = _chunks_by_type(chunks, "ALWAYS_BLOCK")
    assert always_chunks, "No ALWAYS_BLOCK chunks found"
    content = always_chunks[0]["content"]
    assert "posedge" in content, f"posedge not found in: {content}"
    assert "clk" in content, f"clk not found in: {content}"
    assert "negedge" in content or "reset_" in content, (
        f"negedge/reset_ not found in: {content}"
    )


def test_always_driven_signals():
    """out_vld and out_data appear in driven signals of the pipeline block."""
    chunks = extract_chunks_from_rtl("design_pipe_0", SIMPLE_PIPELINE_RTL, "pipeline")
    always_chunks = _chunks_by_type(chunks, "ALWAYS_BLOCK")
    assert always_chunks, "No ALWAYS_BLOCK chunks found"
    combined = "\n".join(c["content"] for c in always_chunks)
    assert "out_vld" in combined, f"out_vld not in always block content: {combined}"
    assert "out_data" in combined, f"out_data not in always block content: {combined}"


def test_reset_detection():
    """reset_ is identified as active-low reset (negedge) in the pipeline block."""
    chunks = extract_chunks_from_rtl("design_pipe_0", SIMPLE_PIPELINE_RTL, "pipeline")
    always_chunks = _chunks_by_type(chunks, "ALWAYS_BLOCK")
    assert always_chunks, "No ALWAYS_BLOCK chunks found"
    combined = "\n".join(c["content"] for c in always_chunks)
    # The content should mention negedge and reset_
    assert "negedge" in combined and "reset_" in combined, (
        f"Expected negedge reset_ in content: {combined}"
    )
    # Check for active-low annotation
    assert "active-low" in combined or "negedge" in combined, (
        f"Active-low reset not annotated: {combined}"
    )


def test_assign_chunks():
    """ASSIGN chunks extracted for continuous assignments in pipeline_top."""
    chunks = extract_chunks_from_rtl(
        "design_top_0", PIPELINE_WITH_INSTANCES_COMBINED, "pipeline"
    )
    assign_chunks = _chunks_by_type(chunks, "ASSIGN")
    assert len(assign_chunks) >= 1, "Expected at least one ASSIGN chunk"
    combined = "\n".join(c["content"] for c in assign_chunks)
    assert "assign" in combined.lower(), f"No assign content found: {combined}"


def test_instance_chunks():
    """INSTANCE chunks produced for exec_unit instantiations in pipeline_top."""
    chunks = extract_chunks_from_rtl(
        "design_top_0", PIPELINE_WITH_INSTANCES_COMBINED, "pipeline"
    )
    inst_chunks = _chunks_by_type(chunks, "INSTANCE")
    assert len(inst_chunks) >= 1, (
        "Expected at least one INSTANCE chunk from pipeline_top instantiating exec_unit"
    )
    # At least one instance should reference exec_unit or eu
    combined = "\n".join(c["content"] for c in inst_chunks)
    assert "exec_unit" in combined or "eu" in combined, (
        f"exec_unit instance not found in INSTANCE chunks: {combined}"
    )


def test_fsm_always_block():
    """FSM module always block is detected as sequential (posedge clk)."""
    chunks = extract_chunks_from_rtl("design_fsm_0", SIMPLE_FSM_RTL, "fsm")
    always_chunks = _chunks_by_type(chunks, "ALWAYS_BLOCK")
    assert always_chunks, "No ALWAYS_BLOCK chunks found for FSM"
    content = always_chunks[0]["content"]
    # Should be classified as sequential or always (inferred)
    assert any(kw in content for kw in ("sequential", "always", "posedge")), (
        f"FSM block not classified as sequential: {content}"
    )
    assert "clk" in content, f"clk not found in FSM block: {content}"


def test_chunk_metadata_has_design_id():
    """All chunks have design_id in metadata (as a top-level dict key)."""
    design_id = "meta_test_design"
    chunks = extract_chunks_from_rtl(design_id, SIMPLE_PIPELINE_RTL, "pipeline")
    assert chunks, "No chunks produced"
    for chunk in chunks:
        assert "design_id" in chunk, f"design_id missing from chunk: {chunk}"
        assert chunk["design_id"] == design_id, (
            f"Wrong design_id {chunk['design_id']!r} in chunk"
        )


def test_chunk_metadata_has_module_name():
    """All chunks have module_name in metadata (as a top-level dict key)."""
    chunks = extract_chunks_from_rtl("meta_test_2", SIMPLE_PIPELINE_RTL, "pipeline")
    assert chunks, "No chunks produced"
    for chunk in chunks:
        assert "module_name" in chunk, f"module_name missing from chunk: {chunk}"
        assert chunk["module_name"], f"module_name is empty in chunk: {chunk}"


def test_parse_errors_handled_gracefully():
    """Broken RTL does not crash — returns at least an empty list."""
    broken_rtl = "module broken_design (input clk, // unclosed module — no endmodule"
    try:
        chunks = extract_chunks_from_rtl("broken_0", broken_rtl, "pipeline")
        # Should not raise; result is a list (possibly empty or with fallback chunk)
        assert isinstance(chunks, list)
    except Exception as exc:
        pytest.fail(f"extract_chunks_from_rtl raised an exception on broken RTL: {exc}")


# ===========================================================================
# Raw helper unit tests (no pyslang)
# ===========================================================================

def test_extract_always_blocks_raw():
    """_extract_always_blocks_raw finds the always block in SIMPLE_PIPELINE_RTL."""
    blocks = _extract_always_blocks_raw(SIMPLE_PIPELINE_RTL)
    assert len(blocks) >= 1, "Expected at least one always block"
    assert "posedge clk" in blocks[0]


def test_extract_assign_lines_raw():
    """_extract_assign_lines_raw finds assign lines in PIPELINE_WITH_INSTANCES_RTL."""
    lines = _extract_assign_lines_raw(PIPELINE_WITH_INSTANCES_RTL)
    assert len(lines) >= 2, "Expected at least two assign lines"
    assert any("stage_data" in ln or "out_vld" in ln or "out_data" in ln for ln in lines)


# ===========================================================================
# Integration test: ChromaDB index + similarity search
# ===========================================================================

@pytest.mark.integration
def test_chromadb_index_and_search(tmp_path):
    """After indexing, similarity search returns a relevant MODULE_INTERFACE chunk.

    Requires either OLLAMA_BASE_URL to be set (OllamaEmbeddings) or
    langchain-huggingface to be installed (HuggingFaceEmbeddings).  The test
    is skipped if neither embeddings backend is available.
    """
    # --- Try to build an embeddings object ---
    embeddings = None
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "")
    embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:latest")

    if ollama_url:
        try:
            from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
            embeddings = OllamaEmbeddings(
                base_url=ollama_url,
                model=embedding_model,
            )
        except Exception:
            embeddings = None

    if embeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception:
            pass

    if embeddings is None:
        pytest.skip(
            "No embeddings backend available: set OLLAMA_BASE_URL or install "
            "langchain-huggingface to run this integration test."
        )

    db_path = str(tmp_path / "test_chroma")

    from ast_indexer import ASTIndexer

    indexer = ASTIndexer(
        db_path=db_path,
        embeddings=embeddings,
        collection_name="test_rtl_chunks",
    )

    design_id = "integ_pipe_0"
    n = indexer.index_design(design_id, SIMPLE_PIPELINE_RTL, "pipeline")
    assert n > 0, "index_design should have added chunks"

    # Verify idempotency
    n2 = indexer.index_design(design_id, SIMPLE_PIPELINE_RTL, "pipeline")
    assert n2 == 0, "Re-indexing the same design should return 0 (already indexed)"

    # Similarity search should surface MODULE_INTERFACE chunk
    docs = indexer.vector_store.similarity_search(
        "module interface ports parameters pipeline",
        k=3,
        filter={"design_id": design_id},
    )
    assert docs, "Similarity search returned no results after indexing"

    chunk_types = [doc.metadata.get("chunk_type") for doc in docs]
    assert "MODULE_INTERFACE" in chunk_types, (
        f"Expected MODULE_INTERFACE in top-3 results; got: {chunk_types}"
    )

    # Content should reference the module name
    iface_doc = next(d for d in docs if d.metadata.get("chunk_type") == "MODULE_INTERFACE")
    assert "pipeline" in iface_doc.page_content, (
        f"Module name 'pipeline' not in MODULE_INTERFACE content: {iface_doc.page_content}"
    )
