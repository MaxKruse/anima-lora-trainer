"""Create and update manifest.json tracking all permutations and statuses."""

import json
from pathlib import Path


class ManifestWriter:
    """Manages the manifest.json file for a matrix training run.

    Tracks all permutations, their statuses, and output files.
    """

    def __init__(self, manifest_path: str | Path, permutations: list[dict]):
        """Initialize manifest writer.

        Args:
            manifest_path: Path to the manifest.json file.
            permutations: List of permutation parameter dicts.
        """
        self.manifest_path = Path(manifest_path)
        self._data = self._load_or_create(permutations)
        # Always save on init to ensure manifest exists
        if not self.manifest_path.exists():
            self._save()

    def _load_or_create(self, permutations: list[dict]) -> dict:
        """Load existing manifest or create a new one.

        If the manifest file exists and has valid data, load it.
        Otherwise, create a new manifest with all permutations as pending.
        """
        if self.manifest_path.exists():
            try:
                content = self.manifest_path.read_text()
                data = json.loads(content)
                if "permutations" in data and len(data["permutations"]) == len(permutations):
                    return data
            except (json.JSONDecodeError, KeyError):
                pass

        # Create new manifest
        return {
            "permutations": [
                {
                    "index": i,
                    "params": perm,
                    "status": "pending",
                    "output_files": [],
                    "error": None,
                    "started_at": None,
                    "completed_at": None,
                }
                for i, perm in enumerate(permutations)
            ],
            "total": len(permutations),
            "completed": 0,
            "failed": 0,
        }

    def read(self) -> dict:
        """Read current manifest data."""
        return self._data

    def update_status(
        self,
        index: int,
        status: str,
        output_files: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        """Update the status of a permutation.

        Args:
            index: Index of the permutation.
            status: New status ('pending', 'running', 'completed', 'failed').
            output_files: List of output file paths (for completed).
            error: Error message (for failed).
        """
        import datetime

        perm = self._data["permutations"][index]
        old_status = perm["status"]
        perm["status"] = status

        if status == "running" and old_status == "pending":
            perm["started_at"] = datetime.datetime.now().isoformat()
        elif status in ("completed", "failed"):
            perm["completed_at"] = datetime.datetime.now().isoformat()

        if output_files is not None:
            perm["output_files"] = output_files
        if error is not None:
            perm["error"] = error

        # Update counts
        self._data["completed"] = sum(
            1 for p in self._data["permutations"] if p["status"] == "completed"
        )
        self._data["failed"] = sum(
            1 for p in self._data["permutations"] if p["status"] == "failed"
        )

        self._save()

    def get_permutation(self, index: int) -> dict:
        """Get a single permutation's data."""
        return self._data["permutations"][index]

    def get_pending_indices(self) -> list[int]:
        """Get indices of pending permutations."""
        return [
            i
            for i, p in enumerate(self._data["permutations"])
            if p["status"] == "pending"
        ]

    def get_running_indices(self) -> list[int]:
        """Get indices of running permutations."""
        return [
            i
            for i, p in enumerate(self._data["permutations"])
            if p["status"] == "running"
        ]

    def is_complete(self) -> bool:
        """Check if all permutations are done (completed or failed)."""
        return all(
            p["status"] in ("completed", "failed")
            for p in self._data["permutations"]
        )

    def _save(self) -> None:
        """Save manifest to disk."""
        self.manifest_path.write_text(json.dumps(self._data, indent=2))
