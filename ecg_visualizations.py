"""
Advanced ECG Visualizations for CardioAI Pro
Provides comprehensive visualization capabilities for ECG analysis results
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class VisualizationStyle(str, Enum):
    """ECG visualization styles"""

    CLINICAL = "clinical"
    RESEARCH = "research"
    PRESENTATION = "presentation"
    DIAGNOSTIC = "diagnostic"


@dataclass
class ECGVisualizationConfig:
    """Configuration for ECG visualizations"""

    style: VisualizationStyle = VisualizationStyle.CLINICAL
    show_grid: bool = True
    show_annotations: bool = True
    show_measurements: bool = True
    show_interpretations: bool = True
    paper_speed: float = 25.0  # mm/s
    amplitude_scale: float = 10.0  # mm/mV
    figure_size: tuple[int, int] = (15, 10)
    dpi: int = 300
    color_scheme: str = "medical"


class ECGVisualizer:
    """Advanced ECG visualization system"""

    def __init__(self, config: ECGVisualizationConfig | None = None):
        self.config = config or ECGVisualizationConfig()
        self.lead_names = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        self.color_schemes = self._initialize_color_schemes()

    def _initialize_color_schemes(self) -> dict[str, dict[str, str]]:
        """Initialize color schemes for different visualization styles"""
        return {
            "medical": {
                "background": "#ffffff",
                "grid_major": "#ff0000",
                "grid_minor": "#ffcccc",
                "signal": "#000000",
                "annotations": "#0066cc",
                "measurements": "#009900",
                "abnormal": "#ff0000",
                "normal": "#008000",
            },
            "dark": {
                "background": "#1e1e1e",
                "grid_major": "#404040",
                "grid_minor": "#2a2a2a",
                "signal": "#ffffff",
                "annotations": "#66ccff",
                "measurements": "#66ff66",
                "abnormal": "#ff6666",
                "normal": "#66ff66",
            },
            "colorful": {
                "background": "#f8f9fa",
                "grid_major": "#dee2e6",
                "grid_minor": "#e9ecef",
                "signal": "#212529",
                "annotations": "#007bff",
                "measurements": "#28a745",
                "abnormal": "#dc3545",
                "normal": "#28a745",
            },
        }

    def create_standard_12_lead_plot(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 500.0,
        analysis_results: dict[str, Any] | None = None,
        title: str = "12-Lead ECG",
    ) -> plt.Figure:
        """
        Create standard 12-lead ECG plot with clinical formatting

        Args:
            ecg_signal: ECG signal array (12, N) or (N, 12)
            sampling_rate: Sampling rate in Hz
            analysis_results: Optional analysis results for annotations
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        if ecg_signal.shape[0] != 12:
            ecg_signal = ecg_signal.T

        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        colors = self.color_schemes[self.config.color_scheme]
        fig.patch.set_facecolor(colors["background"])

        gs = GridSpec(
            5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 0.8], hspace=0.3, wspace=0.2
        )

        duration = ecg_signal.shape[1] / sampling_rate
        time_axis = np.linspace(0, duration, ecg_signal.shape[1])

        lead_positions = [
            (0, 0),
            (0, 1),
            (0, 2),  # I, II, III
            (1, 0),
            (1, 1),
            (1, 2),  # aVR, aVL, aVF
            (2, 0),
            (2, 1),
            (2, 2),  # V1, V2, V3
            (3, 0),
            (3, 1),
            (3, 2),  # V4, V5, V6
        ]

        axes = []
        for i, (row, col) in enumerate(lead_positions):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

            ax.plot(time_axis, ecg_signal[i], color=colors["signal"], linewidth=1.2)

            if self.config.show_grid:
                self._add_medical_grid(ax, duration, colors)

            ax.text(
                0.02,
                0.95,
                self.lead_names[i],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
            )

            if analysis_results and self.config.show_annotations:
                self._add_lead_annotations(ax, i, analysis_results, time_axis, colors)

            ax.set_xlim(0, duration)
            ax.set_ylim(-2, 2)  # Typical ECG amplitude range
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

        rhythm_ax = fig.add_subplot(gs[4, :])
        rhythm_ax.plot(time_axis, ecg_signal[1], color=colors["signal"], linewidth=1.0)
        rhythm_ax.text(
            0.01,
            0.9,
            "Rhythm Strip (Lead II)",
            transform=rhythm_ax.transAxes,
            fontsize=10,
            fontweight="bold",
        )

        if self.config.show_grid:
            self._add_medical_grid(rhythm_ax, duration, colors)

        rhythm_ax.set_xlim(0, duration)
        rhythm_ax.set_ylim(-2, 2)
        rhythm_ax.set_xlabel("Time (seconds)")

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

        if analysis_results and self.config.show_interpretations:
            self._add_interpretation_text(fig, analysis_results)

        plt.tight_layout()
        return fig

    def create_interactive_plotly_visualization(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 500.0,
        analysis_results: dict[str, Any] | None = None,
        title: str = "Interactive 12-Lead ECG",
    ) -> go.Figure:
        """
        Create interactive Plotly visualization with advanced features

        Args:
            ecg_signal: ECG signal array (12, N) or (N, 12)
            sampling_rate: Sampling rate in Hz
            analysis_results: Optional analysis results for annotations
            title: Plot title

        Returns:
            Plotly Figure object
        """
        if ecg_signal.shape[0] != 12:
            ecg_signal = ecg_signal.T

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=self.lead_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(4)],
        )

        duration = ecg_signal.shape[1] / sampling_rate
        time_axis = np.linspace(0, duration, ecg_signal.shape[1])

        for i, lead_name in enumerate(self.lead_names):
            row = i // 3 + 1
            col = i % 3 + 1

            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=ecg_signal[i],
                    mode="lines",
                    name=lead_name,
                    line={"color": "black", "width": 1.5},
                    showlegend=False,
                    hovertemplate=f"<b>{lead_name}</b><br>"
                    + "Time: %{x:.3f}s<br>"
                    + "Amplitude: %{y:.3f}mV<br>"
                    + "<extra></extra>",
                ),
                row=row,
                col=col,
            )

            if analysis_results and self.config.show_annotations:
                self._add_plotly_annotations(
                    fig, i, analysis_results, time_axis, row, col
                )

        fig.update_layout(
            title={"text": title, "x": 0.5, "font": {"size": 16}},
            height=800,
            showlegend=False,
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        for i in range(1, 13):
            row = (i - 1) // 3 + 1
            col = (i - 1) % 3 + 1

            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                title_text="Time (s)" if row == 4 else "",
                row=row,
                col=col,
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                title_text="mV" if col == 1 else "",
                row=row,
                col=col,
            )

        return fig

    def create_feature_importance_heatmap(
        self,
        feature_importance: dict[str, float],
        lead_contributions: dict[str, float] | None = None,
        title: str = "ECG Feature Importance",
    ) -> go.Figure:
        """
        Create heatmap visualization of feature importance

        Args:
            feature_importance: Dictionary of feature names and importance scores
            lead_contributions: Optional lead-specific contributions
            title: Plot title

        Returns:
            Plotly Figure object
        """
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())

        fig = go.Figure(
            data=go.Heatmap(
                z=[importance_values],
                x=features,
                y=["Importance"],
                colorscale="RdYlBu_r",
                showscale=True,
                colorbar={"title": "Importance Score"},
                hovertemplate="Feature: %{x}<br>Importance: %{z:.3f}<extra></extra>",
            )
        )

        if lead_contributions:
            lead_names = list(lead_contributions.keys())
            lead_values = list(lead_contributions.values())

            fig.add_trace(
                go.Heatmap(
                    z=[lead_values],
                    x=lead_names,
                    y=["Lead Contribution"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar={"title": "Lead Contribution", "y": 0.3},
                    hovertemplate="Lead: %{x}<br>Contribution: %{z:.3f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Features/Leads",
            yaxis_title="",
            height=400 if not lead_contributions else 600,
        )

        return fig

    def create_attention_visualization(
        self,
        ecg_signal: np.ndarray,
        attention_weights: dict[str, list[float]],
        sampling_rate: float = 500.0,
        title: str = "ECG Attention Visualization",
    ) -> go.Figure:
        """
        Create visualization of attention weights overlaid on ECG signals

        Args:
            ecg_signal: ECG signal array (12, N) or (N, 12)
            attention_weights: Dictionary of lead names and attention weights
            sampling_rate: Sampling rate in Hz
            title: Plot title

        Returns:
            Plotly Figure object
        """
        if ecg_signal.shape[0] != 12:
            ecg_signal = ecg_signal.T

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=self.lead_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        duration = ecg_signal.shape[1] / sampling_rate
        time_axis = np.linspace(0, duration, ecg_signal.shape[1])

        for i, lead_name in enumerate(self.lead_names):
            row = i // 3 + 1
            col = i % 3 + 1

            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=ecg_signal[i],
                    mode="lines",
                    name=f"{lead_name}_signal",
                    line={"color": "black", "width": 1.5},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            if lead_name in attention_weights:
                attention = np.array(attention_weights[lead_name])
                if len(attention) == len(time_axis):
                    attention_normalized = (attention - attention.min()) / (
                        attention.max() - attention.min()
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=attention_normalized * 0.5
                            - 1.5,  # Scale and offset for visibility
                            mode="lines",
                            fill="tonexty",
                            name=f"{lead_name}_attention",
                            line={"color": "rgba(255, 0, 0, 0.3)", "width": 0},
                            fillcolor="rgba(255, 0, 0, 0.3)",
                            showlegend=False,
                            hovertemplate=f"<b>{lead_name} Attention</b><br>"
                            + "Time: %{x:.3f}s<br>"
                            + "Attention: %{y:.3f}<br>"
                            + "<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

        fig.update_layout(
            title={"text": title, "x": 0.5, "font": {"size": 16}},
            height=800,
            showlegend=False,
            hovermode="closest",
        )

        return fig

    def create_diagnostic_summary_plot(
        self, analysis_results: dict[str, Any], title: str = "ECG Diagnostic Summary"
    ) -> go.Figure:
        """
        Create comprehensive diagnostic summary visualization

        Args:
            analysis_results: Complete analysis results
            title: Plot title

        Returns:
            Plotly Figure object
        """
        analysis_results.get("primary_diagnosis", "Unknown")
        confidence = analysis_results.get("confidence", 0.0)
        detected_conditions = analysis_results.get("detected_conditions", {})
        clinical_urgency = analysis_results.get("clinical_urgency", "LOW")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Confidence Score",
                "Detected Conditions",
                "Clinical Urgency",
                "Risk Assessment",
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "pie"}],
            ],
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Confidence (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=1,
            col=1,
        )

        if detected_conditions:
            condition_names = list(detected_conditions.keys())[:10]  # Top 10
            condition_scores = [
                detected_conditions[name].get("confidence", 0)
                for name in condition_names
            ]

            fig.add_trace(
                go.Bar(
                    x=condition_scores,
                    y=condition_names,
                    orientation="h",
                    marker_color="lightblue",
                ),
                row=1,
                col=2,
            )

        urgency_colors = {"LOW": "green", "HIGH": "orange", "CRITICAL": "red"}
        urgency_values = {"LOW": 1, "HIGH": 2, "CRITICAL": 3}

        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=urgency_values.get(clinical_urgency, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Urgency: {clinical_urgency}"},
                gauge={
                    "axis": {"range": [0, 3]},
                    "bar": {"color": urgency_colors.get(clinical_urgency, "gray")},
                    "steps": [
                        {"range": [0, 1], "color": "green"},
                        {"range": [1, 2], "color": "orange"},
                        {"range": [2, 3], "color": "red"},
                    ],
                },
            ),
            row=2,
            col=1,
        )

        risk_factors = analysis_results.get("risk_factors", [])
        if risk_factors:
            fig.add_trace(
                go.Pie(
                    labels=risk_factors,
                    values=[1] * len(risk_factors),  # Equal weighting
                    hole=0.3,
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title={"text": title, "x": 0.5, "font": {"size": 16}},
            height=600,
            showlegend=False,
        )

        return fig

    def _add_medical_grid(
        self, ax: plt.Axes, duration: float, colors: dict[str, str]
    ) -> None:
        """Add medical-style grid to ECG plot"""
        if not self.config.show_grid:
            return

        major_time_interval = 0.2
        major_amplitude_interval = 0.5

        minor_time_interval = 0.04
        minor_amplitude_interval = 0.1

        for t in np.arange(0, duration, minor_time_interval):
            alpha = 0.3 if t % major_time_interval == 0 else 0.1
            color = (
                colors["grid_major"]
                if t % major_time_interval == 0
                else colors["grid_minor"]
            )
            ax.axvline(t, color=color, alpha=alpha, linewidth=0.5)

        for a in np.arange(-2, 2.1, minor_amplitude_interval):
            alpha = 0.3 if a % major_amplitude_interval == 0 else 0.1
            color = (
                colors["grid_major"]
                if a % major_amplitude_interval == 0
                else colors["grid_minor"]
            )
            ax.axhline(a, color=color, alpha=alpha, linewidth=0.5)

    def _add_lead_annotations(
        self,
        ax: plt.Axes,
        lead_idx: int,
        analysis_results: dict[str, Any],
        time_axis: np.ndarray,
        colors: dict[str, str],
    ) -> None:
        """Add annotations to individual lead plots"""
        if not self.config.show_annotations:
            return

        r_peaks = analysis_results.get("r_peaks", [])
        if r_peaks and lead_idx == 1:  # Show on Lead II
            for peak_idx in r_peaks:
                if peak_idx < len(time_axis):
                    ax.axvline(
                        time_axis[peak_idx],
                        color=colors["annotations"],
                        alpha=0.7,
                        linestyle="--",
                        linewidth=1,
                    )

        abnormalities = analysis_results.get("detected_conditions", {})
        if abnormalities:
            ax.axhspan(-2, 2, alpha=0.1, color=colors["abnormal"])

    def _add_plotly_annotations(
        self,
        fig: go.Figure,
        lead_idx: int,
        analysis_results: dict[str, Any],
        time_axis: np.ndarray,
        row: int,
        col: int,
    ) -> None:
        """Add annotations to Plotly subplots"""
        if not self.config.show_annotations:
            return

        r_peaks = analysis_results.get("r_peaks", [])
        if r_peaks and lead_idx == 1:  # Show on Lead II
            for peak_idx in r_peaks[:10]:  # Limit to first 10 for performance
                if peak_idx < len(time_axis):
                    fig.add_vline(
                        x=time_axis[peak_idx],
                        line_dash="dash",
                        line_color="blue",
                        opacity=0.7,
                        row=row,
                        col=col,
                    )

    def _add_interpretation_text(
        self, fig: plt.Figure, analysis_results: dict[str, Any]
    ) -> None:
        """Add interpretation text to the figure"""
        if not self.config.show_interpretations:
            return

        primary_diagnosis = analysis_results.get("primary_diagnosis", "Unknown")
        confidence = analysis_results.get("confidence", 0.0)
        clinical_urgency = analysis_results.get("clinical_urgency", "LOW")

        interpretation_text = f"""
        Primary Diagnosis: {primary_diagnosis}
        Confidence: {confidence:.1%}
        Clinical Urgency: {clinical_urgency}
        """

        fig.text(
            0.02,
            0.02,
            interpretation_text.strip(),
            transform=fig.transFigure,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
        )

    def save_visualization(
        self,
        fig: plt.Figure,
        filename: str,
        format: str = "png",
        dpi: int | None = None,
    ) -> str:
        """
        Save visualization to file

        Args:
            fig: Matplotlib figure
            filename: Output filename
            format: File format (png, pdf, svg)
            dpi: Resolution (uses config default if None)

        Returns:
            Path to saved file
        """
        dpi = dpi or self.config.dpi

        try:
            fig.savefig(
                filename,
                format=format,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Visualization saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
            raise

    def export_plotly_html(self, fig: go.Figure, filename: str) -> str:
        """
        Export Plotly figure to HTML

        Args:
            fig: Plotly figure
            filename: Output filename

        Returns:
            Path to saved file
        """
        try:
            fig.write_html(filename)
            logger.info(f"Interactive visualization saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export Plotly visualization: {e}")
            raise


def create_standard_ecg_report(
    ecg_signal: np.ndarray,
    analysis_results: dict[str, Any],
    sampling_rate: float = 500.0,
    patient_info: dict[str, str] | None = None,
) -> tuple[plt.Figure, go.Figure]:
    """
    Create comprehensive ECG report with both static and interactive visualizations

    Args:
        ecg_signal: ECG signal array
        analysis_results: Complete analysis results
        sampling_rate: Sampling rate in Hz
        patient_info: Optional patient information

    Returns:
        Tuple of (matplotlib figure, plotly figure)
    """
    visualizer = ECGVisualizer()

    title = "12-Lead ECG Analysis"
    if patient_info:
        patient_id = patient_info.get("patient_id", "Unknown")
        title = f"12-Lead ECG Analysis - Patient: {patient_id}"

    static_fig = visualizer.create_standard_12_lead_plot(
        ecg_signal, sampling_rate, analysis_results, title
    )

    interactive_fig = visualizer.create_interactive_plotly_visualization(
        ecg_signal, sampling_rate, analysis_results, title
    )

    return static_fig, interactive_fig


def create_interpretability_dashboard(
    ecg_signal: np.ndarray,
    analysis_results: dict[str, Any],
    interpretability_results: dict[str, Any],
    sampling_rate: float = 500.0,
) -> list[go.Figure]:
    """
    Create comprehensive interpretability dashboard

    Args:
        ecg_signal: ECG signal array
        analysis_results: Analysis results
        interpretability_results: Interpretability results from SHAP/LIME
        sampling_rate: Sampling rate in Hz

    Returns:
        List of Plotly figures for dashboard
    """
    visualizer = ECGVisualizer()
    figures = []

    main_fig = visualizer.create_interactive_plotly_visualization(
        ecg_signal,
        sampling_rate,
        analysis_results,
        "ECG Analysis with Interpretability",
    )
    figures.append(main_fig)

    feature_importance = interpretability_results.get("feature_importance", {})
    lead_contributions = interpretability_results.get("lead_contributions", {})

    if feature_importance:
        importance_fig = visualizer.create_feature_importance_heatmap(
            feature_importance, lead_contributions, "Feature Importance Analysis"
        )
        figures.append(importance_fig)

    attention_maps = interpretability_results.get("attention_maps", {})
    if attention_maps:
        attention_fig = visualizer.create_attention_visualization(
            ecg_signal, attention_maps, sampling_rate, "Attention Weight Visualization"
        )
        figures.append(attention_fig)

    summary_fig = visualizer.create_diagnostic_summary_plot(
        analysis_results, "Diagnostic Summary Dashboard"
    )
    figures.append(summary_fig)

    return figures
