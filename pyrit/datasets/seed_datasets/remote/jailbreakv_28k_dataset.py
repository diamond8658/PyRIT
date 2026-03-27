# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _JailBreakV28KDataset(_RemoteDatasetLoader):
    """
    Loader for the JailBreakV_28K dataset (CSV Version V0.2).

    JailBreakV_28K is a large-scale benchmark for jailbreaking multimodal LLMs.
    This loader parses the CSV version of the dataset, extracting jailbreak queries
    and mapping them to safety policies.

    Reference: https://github.com/SaFo-Lab/JailBreakV_28K
    """

    # Mapping SaFo-Lab policy abbreviations to descriptive categories
    POLICY_MAP = {
        "P1": "Somatic Safety",
        "P2": "Propriety",
        "P3": "Privacy",
        "P4": "Public Interest",
        "P5": "Mental Health",
    }

    def __init__(
        self,
        *,
        source: str = (
            "https://raw.githubusercontent.com/SaFo-Lab/JailBreakV_28K/V0.2/JailBreakV_28K/JailBreakV_28K.csv"
        ),
        source_type: Literal["public_url", "file"] = "public_url",
    ):
        """
        Initialize the JailBreakV_28K dataset loader.

        Args:
            source: URL to the JailBreakV_28K CSV file.
            source_type: The type of source ('public_url' or 'file').
        """
        self.source = source
        self.source_type: Literal["public_url", "file"] = source_type

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "jailbreakv_28k"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch JailBreakV_28K dataset and return as SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing the JailBreakV_28K prompts.

        Raises:
            ValueError: If no valid prompts could be loaded from the source.
        """
        logger.info(f"Loading JailBreakV_28K CSV dataset from {self.source}")

        # The base loader infers CSV format from the .csv extension in the URL
        examples = self._fetch_from_url(
            source=self.source,
            source_type=self.source_type,
            cache=cache,
        )

        seed_prompts = []
        for example in examples:
            query = example.get("jailbreak_query", "")

            # Safety check: Skip empty prompts or those containing Jinja2 syntax
            if not query or "{{" in query or "{%" in query:
                continue

            seed_prompts.append(
                SeedPrompt(
                    value=str(query),
                    data_type="text",
                    dataset_name=self.dataset_name,
                    harm_categories=[
                        str(self.POLICY_MAP.get(example.get("policy"), "Unknown Policy")),
                        str(example.get("category", "Uncategorized")),
                    ],
                    metadata={
                        "policy_code": str(example.get("policy")),
                        "raw_category": str(example.get("category")),
                    },
                    description=f"JailBreakV_28K prompt (Policy: {example.get('policy')})",
                    source=self.source,
                )
            )

        if not seed_prompts:
            raise ValueError(f"No valid prompts could be loaded from {self.source}")

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from JailBreakV_28K")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)
