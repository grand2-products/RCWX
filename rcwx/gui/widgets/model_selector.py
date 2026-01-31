"""Model selector widget."""

from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)


class ModelSelector(ctk.CTkFrame):
    """
    Model selection widget.

    Allows users to select and load RVC models.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_model_selected: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_model_selected = on_model_selected
        self._model_path: Optional[str] = None
        self._has_f0: Optional[bool] = None
        self._has_index: bool = False
        self._version: Optional[int] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="モデル選択",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 2))

        # Model dropdown
        self.model_var = ctk.StringVar(value="モデルを選択...")
        self.model_dropdown = ctk.CTkComboBox(
            self,
            variable=self.model_var,
            values=["モデルを選択..."],
            width=300,
            state="readonly",
            command=self._on_dropdown_change,
        )
        self.model_dropdown.grid(row=1, column=0, padx=10, pady=2, sticky="ew")

        # Browse button
        self.browse_btn = ctk.CTkButton(
            self,
            text="開く...",
            width=80,
            command=self._browse_model,
        )
        self.browse_btn.grid(row=1, column=1, padx=10, pady=2)

        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="状態: 未選択",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(2, 5))

        # Configure grid
        self.grid_columnconfigure(0, weight=1)

    def _browse_model(self) -> None:
        """Open file dialog to select a model."""
        filepath = filedialog.askopenfilename(
            title="RVCモデルを選択",
            filetypes=[
                ("PyTorchモデル", "*.pth"),
                ("すべてのファイル", "*.*"),
            ],
        )

        if filepath:
            self.set_model(filepath)

    def _on_dropdown_change(self, value: str) -> None:
        """Handle dropdown selection change."""
        if value != "モデルを選択..." and self._model_path:
            if self.on_model_selected:
                self.on_model_selected(self._model_path)

    def set_model(self, path: str) -> None:
        """
        Set the selected model.

        Args:
            path: Path to the model file
        """
        model_path = Path(path)
        if not model_path.exists():
            logger.error(f"Model file not found: {path}")
            return

        self._model_path = str(model_path)

        # Update dropdown
        model_name = model_path.stem
        self.model_var.set(model_name)
        self.model_dropdown.configure(values=[model_name])

        # Check for index file
        index_path = model_path.with_suffix(".index")
        if not index_path.exists():
            # Try looking in the same directory
            index_files = list(model_path.parent.glob("*.index"))
            self._has_index = len(index_files) > 0
        else:
            self._has_index = True

        # Update status (actual F0 detection happens after loading)
        self._update_status()

        # Notify callback
        if self.on_model_selected:
            self.on_model_selected(self._model_path)

    def _update_status(self) -> None:
        """Update the status label."""
        if self._model_path is None:
            self.status_label.configure(text="状態: 未選択", text_color="gray")
            return

        parts = []

        # Version info
        if self._version is not None:
            parts.append(f"RVC v{self._version}")

        # F0 info
        if self._has_f0 is not None:
            f0_text = "F0あり" if self._has_f0 else "F0なし"
            parts.append(f0_text)

        # Index info
        index_text = "Index: あり" if self._has_index else "Index: なし"
        parts.append(index_text)

        status_text = "状態: " + " | ".join(parts)
        self.status_label.configure(text=status_text, text_color="white")

    def set_model_info(self, has_f0: bool, version: int = 2) -> None:
        """
        Update model info after loading.

        Args:
            has_f0: Whether model supports F0
            version: RVC version (1 or 2)
        """
        self._has_f0 = has_f0
        self._version = version
        self._update_status()

    @property
    def model_path(self) -> Optional[str]:
        """Get the current model path."""
        return self._model_path
