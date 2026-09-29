
from pathlib import Path
import sys

import pytest
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)

from src.classifier import TicketClassifier
@pytest.fixture
def classifier():

    return TicketClassifier()

def test_classifier_loads(classifier):
    
    assert classifier.model is not None

    assert classifier.vectorizer is not None

    assert classifier.label_encoder is not None

def test_vpn_ticket_prediction(classifier):
    

    ticket = (
        "User cannot connect to the "
        "company VPN from laptop."
    )

    result = classifier.predict(
        ticket
    )

    assert "category" in result

    assert "confidence" in result

    assert isinstance(
        result["category"],
        str
    )

    assert 0.0 <= result[
        "confidence"
    ] <= 1.0
def test_email_ticket_prediction(classifier):
    ticket = (
        "Outlook is not sending or "
        "receiving emails."
    )

    result = classifier.predict(
        ticket
    )

    assert isinstance(
        result["category"],
        str
    )

    assert result[
        "confidence"
    ] >= 0.0
def test_empty_ticket_rejected(classifier):
    

    with pytest.raises(
        ValueError
    ):

        classifier.predict(
            ""
        )

def test_top_k_predictions(classifier):
    

    ticket = (
        "The application cannot connect "
        "to the database."
    )

    results = classifier.predict_top_k(
        ticket,
        top_k=3
    )

    assert isinstance(
        results,
        list
    )

    assert len(results) <= 3

    for result in results:

        assert "category" in result

        assert "confidence" in result

        assert isinstance(
            result["category"],
            str
        )

        assert (
            0.0
            <= result["confidence"]
            <= 1.0
        )
def test_prediction_confidence(classifier):

    tickets = [
        "VPN is not connecting",
        "Laptop is very slow",
        "Printer is offline",
        "Database connection failed",
    ]

    for ticket in tickets:

        result = classifier.predict(
            ticket
        )

        confidence = result[
            "confidence"
        ]

        assert 0.0 <= confidence <= 1.0
def test_multiple_ticket_predictions(
    classifier
):

    tickets = [
        "I cannot access the shared folder.",
        "My account has been locked.",
        "The office printer is offline.",
        "The server CPU usage is very high.",
    ]

    results = []

    for ticket in tickets:

        result = classifier.predict(
            ticket
        )

        results.append(
            result
        )

    assert len(results) == len(
        tickets
    )

    for result in results:

        assert result[
            "category"
        ]

        assert (
            0.0
            <= result["confidence"]
            <= 1.0
        )



