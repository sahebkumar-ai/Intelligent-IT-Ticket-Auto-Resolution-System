
from sympy import python
from pathlib import Path
from typing import Optional
import time
BASE_DIR = Path(__file__).resolve().parent.parent
from src.preprocessing import (
    preprocess_ticket,
    combine_ticket_text,
)
from src.classifier import (
    TicketClassifier,
)
from src.solution_recommender import (
    SolutionRecommender,
)
class TicketPipeline:
    def __init__(
        self,
        confidence_threshold: float = 0.80
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.classifier = (
            TicketClassifier()
        )

        self.recommender = (
            SolutionRecommender()
        )

    def process_ticket(
        self,
        ticket_text: str,
        ocr_text: str = "",
        log_text: str = "",
        ticket_id: Optional[str] = None
    ) -> dict:
       
        start_time = time.perf_counter()
        combined_text = combine_ticket_text(
            ticket_text=ticket_text,
            ocr_text=ocr_text,
            log_text=log_text
        )

        if not combined_text:
            return {
                "success": False,
                "message": (
                    "The ticket does not contain "
                    "enough usable information."
                )
            }

        prediction = (
            self.classifier.predict(
                combined_text
            )
        )

        category = prediction[
            "category"
        ]

        confidence = prediction[
            "confidence"
        ]

        if confidence >= (
            self.confidence_threshold
        ):
            automation_status = (
                "Auto Recommendation"
            )

            requires_human_review = False

        else:
            automation_status = (
                "Human Review Required"
            )

            requires_human_review = True

        solution = (
            self.recommender.get_solution(
                category
            )
        )
        processing_time = (
            time.perf_counter()
            - start_time
        )

        result = {
            "success": True,

            "ticket_id": ticket_id,

            "input": {
                "original_text": ticket_text,
                "ocr_text": ocr_text,
                "log_text": log_text,
                "processed_text": combined_text
            },

            "classification": {
                "category": category,
                "confidence": round(
                    confidence,
                    4
                )
            },

            "resolution": {
                "automation_status": (
                    automation_status
                ),
                "requires_human_review": (
                    requires_human_review
                ),
                "solution_id": solution[
                    "solution_id"
                ],
                "problem": solution[
                    "problem"
                ],
                "solution": solution[
                    "solution"
                ],
                "steps": solution[
                    "steps"
                ]
            },

            "performance": {
                "processing_time_seconds": round(
                    processing_time,
                    4
                ),
                "target_response_time_seconds": 2.0,
                "within_target": (
                    processing_time < 2.0
                )
            }
        }

        return result

    def process_text_ticket(
        self,
        ticket_text: str,
        ticket_id: Optional[str] = None
    ) -> dict:
       
        return self.process_ticket(
            ticket_text=ticket_text,
            ticket_id=ticket_id
        )

    def process_ticket_with_ocr(
        self,
        ticket_text: str,
        ocr_text: str,
        ticket_id: Optional[str] = None
    ) -> dict:
        
        return self.process_ticket(
            ticket_text=ticket_text,
            ocr_text=ocr_text,
            ticket_id=ticket_id
        )

    def process_ticket_with_logs(
        self,
        ticket_text: str,
        log_text: str,
        ticket_id: Optional[str] = None
    ) -> dict:
        
        return self.process_ticket(
            ticket_text=ticket_text,
            log_text=log_text,
            ticket_id=ticket_id
        )


def create_pipeline(
    confidence_threshold: float = 0.80
) -> TicketPipeline:

    return TicketPipeline(
        confidence_threshold=(
            confidence_threshold
        )
    )


if __name__ == "__main__":

    print(
        "\n"
        + "=" * 65
    )

    print(
        "IT TICKET AUTO-RESOLUTION PIPELINE TEST"
    )

    print(
        "=" * 65
    )

    try:

        pipeline = TicketPipeline()

        sample_ticket = (
            "My VPN is not connecting "
            "to the company network."
        )

        result = (
            pipeline.process_text_ticket(
                ticket_text=sample_ticket,
                ticket_id="DEMO-001"
            )
        )

        print(
            "\nTicket:"
        )

        print(
            sample_ticket
        )

        print(
            "\nCategory:"
        )

        print(
            result["classification"][
                "category"
            ]
        )

        print(
            "\nConfidence:"
        )

        print(
            result["classification"][
                "confidence"
            ]
        )

        print(
            "\nAutomation Status:"
        )

        print(
            result["resolution"][
                "automation_status"
            ]
        )

        print(
            "\nRecommended Solution:"
        )

        print(
            result["resolution"][
                "solution"
            ]
        )

        print(
            "\nRecommended Steps:"
        )

        for index, step in enumerate(
            result["resolution"][
                "steps"
            ],
            start=1
        ):
            print(
                f"{index}. {step}"
            )

        print(
            "\nProcessing Time:"
        )

        print(
            result["performance"][
                "processing_time_seconds"
            ],
            "seconds"
        )

    except Exception as exc:

        print(
            "\nPipeline Error:"
        )

        print(
            str(exc)
        )

