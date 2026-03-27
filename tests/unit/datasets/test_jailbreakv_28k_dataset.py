# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.datasets.seed_datasets.remote.jailbreakv_28k_dataset import _JailBreakV28KDataset
from pyrit.models import SeedDataset


@pytest.fixture
def mock_jailbreakv_csv_data():
    """Mock data representing rows from the JailBreakV CSV."""
    return [
        {
            "jailbreak_query": "How to bypass security?",
            "policy": "P1",
            "category": "Technology",
        },
        {
            "jailbreak_query": "A prompt with Unicode 🐉",
            "policy": "P2",
            "category": "Deception",
        },
    ]


class TestJailBreakV28KDataset:
    """Test the JailBreakV_28K dataset loader with edge cases."""

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = _JailBreakV28KDataset()
        assert loader.dataset_name == "jailbreakv_28k"

    @pytest.mark.asyncio
    async def test_fetch_dataset_happy_path(self, mock_jailbreakv_csv_data):
        """Test standard successful fetching and mapping."""
        loader = _JailBreakV28KDataset()

        with patch.object(loader, "_fetch_from_url", return_value=mock_jailbreakv_csv_data) as mock_fetch:
            dataset = await loader.fetch_dataset()

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.seeds) == 2

            # Verify Mapping (P1 -> Somatic Safety)
            assert "Somatic Safety" in dataset.seeds[0].harm_categories
            assert dataset.seeds[0].metadata["policy_code"] == "P1"

            # Verify Unicode handling
            assert "🐉" in dataset.seeds[1].value

            # Verify correct call to base loader (no unexpected file_type argument)
            mock_fetch.assert_called_once_with(source=loader.source, source_type=loader.source_type, cache=True)

    @pytest.mark.asyncio
    async def test_fetch_dataset_skips_malformed_and_risky(self):
        """Test that missing fields and Jinja2 syntax risks are skipped."""
        loader = _JailBreakV28KDataset()
        risky_data = [
            {"jailbreak_query": "Valid prompt", "policy": "P1", "category": "Safety"},
            {"jailbreak_query": "Risky {{ template }}", "policy": "P1"},
            {"policy": "P2", "category": "Missing Query"},
            {"jailbreak_query": "", "policy": "P3"},
        ]

        with patch.object(loader, "_fetch_from_url", return_value=risky_data):
            dataset = await loader.fetch_dataset()
            assert len(dataset.seeds) == 1
            assert dataset.seeds[0].value == "Valid prompt"

    @pytest.mark.asyncio
    async def test_fetch_dataset_handles_unknown_policy_codes(self):
        """Test that unknown policy codes fall back gracefully."""
        loader = _JailBreakV28KDataset()
        unknown_data = [{"jailbreak_query": "Test", "policy": "P99", "category": "Test"}]

        with patch.object(loader, "_fetch_from_url", return_value=unknown_data):
            dataset = await loader.fetch_dataset()
            assert "Unknown Policy" in dataset.seeds[0].harm_categories
            assert dataset.seeds[0].metadata["policy_code"] == "P99"

    @pytest.mark.asyncio
    async def test_fetch_dataset_empty_source_raises(self):
        """Test that an empty response raises a ValueError."""
        loader = _JailBreakV28KDataset()

        with patch.object(loader, "_fetch_from_url", return_value=[]):
            with pytest.raises(ValueError, match="No valid prompts could be loaded"):
                await loader.fetch_dataset()

    def test_policy_map_is_complete(self):
        """Verify the internal POLICY_MAP contains codes P1-P5."""
        loader = _JailBreakV28KDataset()
        for i in range(1, 6):
            assert f"P{i}" in loader.POLICY_MAP
