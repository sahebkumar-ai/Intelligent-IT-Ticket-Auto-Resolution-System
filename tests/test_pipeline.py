

from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)
from src.ticket_pipeline import TicketPipeline
import pytest


@pytest.fixture
def pipeline():

    return TicketPipeline(
        confidence_threshold=0.80
    )
def test_basic_ticket_processing(
    pipeline
):

    ticket = (
        "My VPN is not connecting "
        "to the company network."
    )

    result = pipeline.process_ticket(
        ticket_text=ticket,
        ticket_id="TEST-001"
    )

    assert result["success"] is True

    assert "classification" in result

    assert "resolution" in result

    assert "performance" in result

def test_classification_result(
    pipeline
):
    ticket = (
        "The office printer is offline "
        "and cannot print documents."
    )

    result = pipeline.process_ticket(
        ticket_text=ticket,
        ticket_id="TEST-002"
    )

    classification = result[
        "classification"
    ]

    assert classification[
        "category"
    ]

    assert isinstance(
        classification["category"],
        str
    )

    assert 0.0 <= (
        classification["confidence"]
    ) <= 1.0


def test_solution_recommendation(
    pipeline
):
   

    ticket = (
        "I cannot access the shared "
        "network folder."
    )

    result = pipeline.process_ticket(
        ticket_text=ticket,
        ticket_id="TEST-003"
    )

    resolution = result[
        "resolution"
    ]

    assert "solution" in resolution

    assert resolution[
        "solution"
    ]

    assert isinstance(
        resolution["steps"],
        list
    )
def test_ocr_text_integration(
    pipeline
):
    ticket_text = (
        "User cannot access application."
    )
    ocr_text = (
        "ERROR 503 SERVICE UNAVAILABLE "
        "APPLICATION SERVER TIMEOUT"
    )
    result = pipeline.process_ticket(
        ticket_text=ticket_text,
        ocr_text=ocr_text,
        ticket_id="TEST-004"
    )
    assert result["success"] is True
    processed_text = result[
        "input"
    ]["processed_text"]

    assert "error" in processed_text

    assert "503" in processed_text

def test_log_integration(
    pipeline
):
    ticket_text = (
        "Production application "
        "is not working."
    )

    log_text = (
        "ERROR database connection timeout "
        "connection refused"
    )

    result = pipeline.process_ticket(
        ticket_text=ticket_text,
        log_text=log_text,
        ticket_id="TEST-005"
    )

    assert result["success"] is True

    processed_text = result[
        "input"
    ]["processed_text"]

    assert "database" in processed_text

    assert "timeout" in processed_text

def test_empty_ticket(
    pipeline
):
    result = pipeline.process_ticket(
        ticket_text="",
        ticket_id="TEST-006"
    )

    assert result["success"] is False

    assert "message" in result


def test_whitespace_ticket(
    pipeline
):
    result = pipeline.process_ticket(
        ticket_text="     ",
        ticket_id="TEST-007"
    )

    assert result["success"] is False

def test_ticket_id(
    pipeline
):
    result = pipeline.process_ticket(
        ticket_text=(
            "Laptop is running very slowly."
        ),
        ticket_id="ENTERPRISE-12345"
    )

    assert result["success"] is True

    assert result[
        "ticket_id"
    ] == "ENTERPRISE-12345"

def test_processing_time(
    pipeline
):

    result = pipeline.process_ticket(
        ticket_text=(
            "My account is locked."
        ),
        ticket_id="TEST-009"
    )

    performance = result[
        "performance"
    ]

    assert (
        "processing_time_seconds"
        in performance
    )

    assert performance[
        "processing_time_seconds"
    ] >= 0

def test_automation_decision(
    pipeline
):
    result = pipeline.process_ticket(
        ticket_text=(
            "VPN connection is not working."
        ),
        ticket_id="TEST-010"
    )

    resolution = result[
        "resolution"
    ]

    assert (
        "automation_status"
        in resolution
    )

    assert (
        "requires_human_review"
        in resolution
    )

    assert isinstance(
        resolution[
            "requires_human_review"
        ],
        bool
    )
def test_multiple_ticket_types(
    pipeline
):
    tickets = [
        "VPN is not connecting.",
        "Outlook is not receiving emails.",
        "Printer is offline.",
        "Laptop is very slow.",
        "Database connection failed.",
    ]

    results = []

    for ticket in tickets:

        result = pipeline.process_ticket(
            ticket_text=ticket
        )

        results.append(
            result
        )

    assert len(results) == len(
        tickets
    )

    for result in results:

        assert result[
            "success"
        ] is True

        assert result[
            "classification"
        ]["category"]

        assert result[
            "resolution"
        ]["solution"]

def test_confidence_threshold(
    pipeline
):
    pipeline.confidence_threshold = 0.95

    result = pipeline.process_ticket(
        ticket_text=(
            "VPN is not connecting "
            "to the corporate network."
        )
    )

    confidence = result[
        "classification"
    ]["confidence"]

    requires_review = result[
        "resolution"
    ]["requires_human_review"]

    if confidence < 0.95:

        assert requires_review is True

def test_full_pipeline_output(
    pipeline
):
    result = pipeline.process_ticket(
        ticket_text=(
            "The production server "
            "is responding very slowly."
        ),
        ticket_id="PROD-001"
    )

    required_top_level_fields = [
        "success",
        "ticket_id",
        "input",
        "classification",
        "resolution",
        "performance"
    ]

    for field in required_top_level_fields:

        assert field in result

    required_classification_fields = [
        "category",
        "confidence"
    ]

    for field in required_classification_fields:

        assert field in result[
            "classification"
        ]

    required_resolution_fields = [
        "automation_status",
        "requires_human_review",
        "solution",
        "steps"
    ]

    for field in required_resolution_fields:

        assert field in result[
            "resolution"
        ]

