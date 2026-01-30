"""Pitch control widget."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


class PitchControl(ctk.CTkFrame):
    """
    Pitch shift control widget.

    Allows users to adjust pitch shift and F0 mode.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_pitch_changed: Optional[Callable[[int], None]] = None,
        on_f0_mode_changed: Optional[Callable[[bool], None]] = None,
        on_f0_method_changed: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_pitch_changed = on_pitch_changed
        self.on_f0_mode_changed = on_f0_mode_changed
        self.on_f0_method_changed = on_f0_method_changed

        self._pitch: int = 0
        self._use_f0: bool = True

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Pitch shift section
        self.pitch_label = ctk.CTkLabel(
            self,
            text="ピッチシフト",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.pitch_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5))

        # Slider row
        self.min_label = ctk.CTkLabel(self, text="-24", font=ctk.CTkFont(size=10))
        self.min_label.grid(row=1, column=0, padx=(10, 5), pady=5)

        self.pitch_slider = ctk.CTkSlider(
            self,
            from_=-24,
            to=24,
            number_of_steps=48,
            width=250,
            command=self._on_slider_change,
        )
        self.pitch_slider.set(0)
        self.pitch_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.max_label = ctk.CTkLabel(self, text="+24", font=ctk.CTkFont(size=10))
        self.max_label.grid(row=1, column=2, padx=(5, 10), pady=5)

        # Current value display
        self.value_label = ctk.CTkLabel(
            self,
            text="現在値: 0 半音",
            font=ctk.CTkFont(size=12),
        )
        self.value_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        # Preset buttons
        self.preset_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.preset_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

        presets = [
            ("-12", -12),
            ("-5", -5),
            ("0", 0),
            ("+5", 5),
            ("+12", 12),
        ]

        for i, (label, value) in enumerate(presets):
            btn = ctk.CTkButton(
                self.preset_frame,
                text=label,
                width=50,
                command=lambda v=value: self._set_pitch(v),
            )
            btn.grid(row=0, column=i, padx=3, pady=5)

        # F0 mode section
        self.f0_label = ctk.CTkLabel(
            self,
            text="F0モード",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.f0_label.grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=(15, 5))

        self.f0_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.f0_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

        self.f0_var = ctk.StringVar(value="rmvpe")

        self.rmvpe_rb = ctk.CTkRadioButton(
            self.f0_frame,
            text="RMVPE (高品質)",
            variable=self.f0_var,
            value="rmvpe",
            command=self._on_f0_change,
        )
        self.rmvpe_rb.grid(row=0, column=0, padx=5, pady=5)

        self.fcpe_rb = ctk.CTkRadioButton(
            self.f0_frame,
            text="FCPE (低遅延)",
            variable=self.f0_var,
            value="fcpe",
            command=self._on_f0_change,
        )
        self.fcpe_rb.grid(row=0, column=1, padx=5, pady=5)

        self.none_rb = ctk.CTkRadioButton(
            self.f0_frame,
            text="なし",
            variable=self.f0_var,
            value="none",
            command=self._on_f0_change,
        )
        self.none_rb.grid(row=0, column=2, padx=5, pady=5)

        # Configure grid
        self.grid_columnconfigure(1, weight=1)

    def _on_slider_change(self, value: float) -> None:
        """Handle slider value change."""
        self._pitch = int(round(value))
        self._update_value_label()

        if self.on_pitch_changed:
            self.on_pitch_changed(self._pitch)

    def _set_pitch(self, value: int) -> None:
        """Set pitch to a specific value."""
        self._pitch = value
        self.pitch_slider.set(value)
        self._update_value_label()

        if self.on_pitch_changed:
            self.on_pitch_changed(self._pitch)

    def _update_value_label(self) -> None:
        """Update the current value label."""
        sign = "+" if self._pitch > 0 else ""
        self.value_label.configure(text=f"現在値: {sign}{self._pitch} 半音")

    def _on_f0_change(self) -> None:
        """Handle F0 mode change."""
        method = self.f0_var.get()
        self._use_f0 = method != "none"
        self._f0_method = method

        if self.on_f0_mode_changed:
            self.on_f0_mode_changed(self._use_f0)

        if self.on_f0_method_changed:
            self.on_f0_method_changed(method)

    def set_f0_enabled(self, enabled: bool) -> None:
        """Enable or disable F0 controls based on model support."""
        if enabled:
            self.rmvpe_rb.configure(state="normal")
            self.fcpe_rb.configure(state="normal")
            if self.f0_var.get() == "none":
                self.f0_var.set("rmvpe")
            self._use_f0 = True
        else:
            self.rmvpe_rb.configure(state="disabled")
            self.fcpe_rb.configure(state="disabled")
            self.f0_var.set("none")
            self._use_f0 = False

    def set_f0_method(self, method: str) -> None:
        """Set F0 method (rmvpe, fcpe, or none)."""
        self.f0_var.set(method)
        self._use_f0 = method != "none"
        self._f0_method = method

    @property
    def pitch(self) -> int:
        """Get current pitch shift."""
        return self._pitch

    @property
    def use_f0(self) -> bool:
        """Get current F0 mode."""
        return self._use_f0

    @property
    def f0_method(self) -> str:
        """Get current F0 method (rmvpe, fcpe, or none)."""
        return self.f0_var.get()
