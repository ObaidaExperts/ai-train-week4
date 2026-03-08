import csv
import os
import logging
from typing import Any

logger = logging.getLogger("repository")

class ResultsRepository:
    """Repository to handle persistence of experiment results."""
    
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Initialize CSV if it doesn't exist or has wrong header."""
        header = [
            "Timestamp", "Request_ID", "Experiment_Type", "Model", "Prompt", "Response",
            "Input_Tokens", "Output_Tokens", "Cost_USD", "Status",
            "Temperature", "Top_P", "Top_K", "Logprobs"
        ]
        
        should_init = not os.path.exists(self.file_path)
        
        if not should_init:
            # Check if headers match
            try:
                with open(self.file_path, mode='r') as f:
                    actual_header = next(csv.reader(f))
                    if actual_header != header:
                        logger.warning(f"Header mismatch in {self.file_path}. Re-initializing.")
                        should_init = True
            except (StopIteration, Exception):
                should_init = True

        if should_init:
            logger.info(f"Initializing/Refreshing results file: {self.file_path}")
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

    def log_result(self, result: dict[str, Any]) -> None:
        """Append a new result record to the CSV using DictWriter."""
        header = [
            "Timestamp", "Request_ID", "Experiment_Type", "Model", "Prompt", "Response",
            "Input_Tokens", "Output_Tokens", "Cost_USD", "Status",
            "Temperature", "Top_P", "Top_K", "Logprobs"
        ]
        try:
            with open(self.file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                # Ensure all keys are present even if None
                row = {h: result.get(h, "") for h in header}
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to log result to {self.file_path}: {e}")
            raise

    def get_all_results(self) -> list[dict[str, Any]]:
        """Read all results from the CSV."""
        if not os.path.exists(self.file_path):
            return []
        
        results = []
        try:
            with open(self.file_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    results.append(row)
        except Exception as e:
            logger.error(f"Failed to read results from {self.file_path}: {e}")
            return []
            
        return results
