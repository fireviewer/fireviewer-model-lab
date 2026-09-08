from __future__ import annotations

from typing import Any

from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from torch.utils.data import DataLoader


class OptimizedGenericNonGeoSegmentationDataModule(GenericNonGeoSegmentationDataModule):
    """TerraTorch loader with the GPU-feeding options required by this trainer."""

    def __init__(
        self,
        *args: Any,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1")
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Any]]:
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
