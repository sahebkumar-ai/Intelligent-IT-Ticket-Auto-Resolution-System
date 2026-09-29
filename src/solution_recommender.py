

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


KNOWLEDGE_BASE_PATH = (
    BASE_DIR / "data" / "knowledge_base.csv"
)

class SolutionRecommender:
    

    def __init__(
        self,
        knowledge_base_path=KNOWLEDGE_BASE_PATH
    ):
        self.knowledge_base_path = Path(
            knowledge_base_path
        )

        self.knowledge_base = (
            self._load_knowledge_base()
        )

    def _load_knowledge_base(self):
        

        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: "
                f"{self.knowledge_base_path}"
            )

        data = pd.read_csv(
            self.knowledge_base_path,
            comment=chr(96)
        )

        required_columns = [
            "solution_id",
            "issue_category",
            "problem",
            "solution",
            "steps"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Knowledge base is missing columns: "
                + ", ".join(missing_columns)
            )

        return data

    def get_solution(
        self,
        issue_category: str
    ) -> dict:
        

        if not issue_category:
            return self._no_solution()

        category = str(
            issue_category
        ).strip()

        matches = self.knowledge_base[
            self.knowledge_base[
                "issue_category"
            ].str.lower()
            == category.lower()
        ]

        if matches.empty:
            return self._no_solution(
                category
            )

        record = matches.iloc[0]

        steps = self._parse_steps(
            record["steps"]
        )

        return {
            "solution_id": record[
                "solution_id"
            ],
            "issue_category": record[
                "issue_category"
            ],
            "problem": record[
                "problem"
            ],
            "solution": record[
                "solution"
            ],
            "steps": steps
        }

    def search_solutions(
        self,
        keyword: str,
        limit: int = 5
    ) -> list:
        

        if not keyword:
            return []

        keyword = str(
            keyword
        ).lower().strip()

        searchable_columns = [
            "issue_category",
            "problem",
            "solution"
        ]

        results = []

        for _, row in self.knowledge_base.iterrows():

            searchable_text = " ".join(
                str(row[column])
                for column in searchable_columns
            ).lower()

            if keyword in searchable_text:

                results.append(
                    {
                        "solution_id": row[
                            "solution_id"
                        ],
                        "issue_category": row[
                            "issue_category"
                        ],
                        "problem": row[
                            "problem"
                        ],
                        "solution": row[
                            "solution"
                        ],
                        "steps": self._parse_steps(
                            row["steps"]
                        )
                    }
                )

            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _parse_steps(steps) -> list:
        

        if pd.isna(steps):
            return []

        return [
            step.strip()
            for step in str(steps).split(";")
            if step.strip()
        ]

    @staticmethod
    def _no_solution(
        category: str = ""
    ) -> dict:
       

        return {
            "solution_id": None,
            "issue_category": category,
            "problem": "No matching solution found.",
            "solution": (
                "The ticket requires human support "
                "engineer review."
            ),
            "steps": [
                "Review the ticket details",
                "Check available system logs",
                "Escalate to the appropriate support team"
            ]
        }


def load_recommender():
    

    return SolutionRecommender()


if __name__ == "__main__":

    recommender = SolutionRecommender()

    result = recommender.get_solution(
        "VPN Connectivity"
    )

    print("\nSolution Recommendation")
    print("=" * 50)

    print(
        f"Category: "
        f"{result['issue_category']}"
    )

    print(
        f"Problem: "
        f"{result['problem']}"
    )

    print(
        f"Solution: "
        f"{result['solution']}"
    )

    print("\nSteps:")

    for number, step in enumerate(
        result["steps"],
        start=1
    ):
        print(
            f"{number}. {step}"
        )

