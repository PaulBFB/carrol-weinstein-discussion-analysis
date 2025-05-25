import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from datetime import datetime, timedelta
import matplotlib.patches as patches
from collections import Counter


def load_and_process_data(csv_file_path: pathlib.Path):
    """
    Load and process the CSV data for time-based analysis.

    Args:
        csv_file_path (Path): Path to the CSV file

    Returns:
        pd.DataFrame: Processed data with sophisticated concepts only
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Clean the data
    df = df.dropna(subset=["speaker", "start_time", "end_time"])

    # Filter for sophisticated concepts only
    sophisticated_levels = ["Undergraduate", "Graduate", "High School/Undergraduate"]
    sophisticated_concepts = df[
        (df["education_level"].isin(sophisticated_levels))
        & (df["concept_name"].notna())
    ].copy()

    # Convert time to minutes for better readability
    sophisticated_concepts["start_minutes"] = sophisticated_concepts["start_time"] / 60
    sophisticated_concepts["end_minutes"] = sophisticated_concepts["end_time"] / 60

    return sophisticated_concepts


def create_timeline_visualization(df: pd.DataFrame, save_plot: bool = True):
    """
    Create a comprehensive timeline visualization showing concepts over time.

    Args:
        df (pd.DataFrame): Processed data with sophisticated concepts
        save_plot (boolean): whether to save the plot to local image file
    """
    # Set up the figure
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(
        "Speaker Topics Over Time - Sophisticated Concepts Only",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # Color scheme for speakers
    speaker_colors = {
        "Piers": "#3498db",  # Blue
        "Eric": "#e74c3c",  # Red
        "Sean": "#2ecc71",  # Green
    }

    # Education level markers
    education_markers = {
        "Graduate": "s",  # Square
        "Undergraduate": "o",  # Circle
        "High School/Undergraduate": "^",  # Triangle
    }

    # Education level sizes
    education_sizes = {
        "Graduate": 100,
        "Undergraduate": 60,
        "High School/Undergraduate": 40,
    }

    speakers = ["Piers", "Eric", "Sean"]

    # 1. Overall Timeline - All Speakers
    ax = axes[0]

    for speaker in speakers:
        speaker_data = df[df["speaker"] == speaker]
        if len(speaker_data) > 0:
            # Plot concept points
            for _, row in speaker_data.iterrows():
                marker = education_markers.get(row["education_level"], "o")
                size = education_sizes.get(row["education_level"], 60)
                ax.scatter(
                    row["start_minutes"],
                    speaker,
                    c=speaker_colors[speaker],
                    marker=marker,
                    s=size,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )

    ax.set_ylabel("Speaker")
    ax.set_title(
        "Concept Distribution Across All Speakers", fontsize=14, fontweight="bold"
    )
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, df["start_minutes"].max() + 2)

    # Add legend for education levels
    legend_elements = [
        plt.scatter(
            [],
            [],
            marker=marker,
            s=size,
            c="gray",
            label=f"{level}",
            edgecolors="black",
            linewidth=0.5,
        )
        for level, marker, size in zip(
            education_markers.keys(),
            education_markers.values(),
            education_sizes.values(),
        )
    ]
    ax.legend(handles=legend_elements, title="Education Level", loc="upper right")

    # 2-4. Individual Speaker Timelines
    for i, speaker in enumerate(speakers, 1):
        ax = axes[i]
        speaker_data = df[df["speaker"] == speaker].copy()

        if len(speaker_data) == 0:
            ax.text(
                0.5,
                0.5,
                f"No sophisticated concepts found for {speaker}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(
                f"{speaker} - Concept Timeline", fontsize=14, fontweight="bold"
            )
            continue

        # Create time bins for concept density
        max_time = df["start_minutes"].max()
        time_bins = np.linspace(0, max_time, 50)  # 50 bins

        # Calculate concept count per time bin
        concept_counts, bin_edges = np.histogram(
            speaker_data["start_minutes"], bins=time_bins
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot concept density as area chart
        ax.fill_between(
            bin_centers,
            concept_counts,
            alpha=0.3,
            color=speaker_colors[speaker],
            label="Concept Density",
        )
        ax.plot(bin_centers, concept_counts, color=speaker_colors[speaker], linewidth=2)

        # Overlay individual concepts as scatter points
        for _, row in speaker_data.iterrows():
            marker = education_markers.get(row["education_level"], "o")
            size = education_sizes.get(row["education_level"], 60)
            ax.scatter(
                row["start_minutes"],
                concept_counts.max() * 0.1,
                c=speaker_colors[speaker],
                marker=marker,
                s=size,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

        # Formatting
        ax.set_title(
            f"{speaker} - Concept Timeline ({len(speaker_data)} sophisticated concepts)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Concepts per Time Window")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_time + 2)

        # Add concept examples as annotations for peak moments
        if len(speaker_data) > 0:
            # Find peaks in concept density
            peak_indices = []
            for j in range(1, len(concept_counts) - 1):
                if (
                    concept_counts[j] > concept_counts[j - 1]
                    and concept_counts[j] > concept_counts[j + 1]
                    and concept_counts[j] > 1
                ):
                    peak_indices.append(j)

            # Annotate top 3 peaks with example concepts
            peak_values = [(concept_counts[idx], idx) for idx in peak_indices]
            peak_values.sort(reverse=True)

            for peak_val, peak_idx in peak_values[:3]:  # Top 3 peaks
                peak_time = bin_centers[peak_idx]
                # Find concepts near this time
                nearby_concepts = speaker_data[
                    (speaker_data["start_minutes"] >= peak_time - 1)
                    & (speaker_data["start_minutes"] <= peak_time + 1)
                ]["concept_name"].values

                if len(nearby_concepts) > 0:
                    example_concept = nearby_concepts[0][:25] + (
                        "..." if len(nearby_concepts[0]) > 25 else ""
                    )
                    ax.annotate(
                        f"{example_concept}",
                        xy=(peak_time, peak_val),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5
                        ),
                        fontsize=8,
                        ha="left",
                    )

    # Set common x-label
    axes[-1].set_xlabel("Time (minutes)", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if save_plot:
        plt.savefig("./topic_timeline.png")
    plt.show()


def create_concept_frequency_heatmap(df: pd.DataFrame, save_plot: bool = True):
    """
    Create a heatmap showing concept frequency over time periods.

    Args:
        df (pd.DataFrame): Processed data with sophisticated concepts
        save_plot (boolean): whether to save the plot to local image file
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Create time periods (10-minute windows)
    max_time = df["start_minutes"].max()
    time_periods = np.arange(0, max_time + 10, 10)  # 10-minute bins
    period_labels = [f"{int(i)}-{int(i + 10)}min" for i in time_periods[:-1]]

    # Create a matrix for heatmap
    speakers = ["Piers", "Eric", "Sean"]
    heatmap_data = np.zeros((len(speakers), len(period_labels)))

    for i, speaker in enumerate(speakers):
        speaker_data = df[df["speaker"] == speaker]

        for j, start_time in enumerate(time_periods[:-1]):
            end_time = time_periods[j + 1]
            concepts_in_period = len(
                speaker_data[
                    (speaker_data["start_minutes"] >= start_time)
                    & (speaker_data["start_minutes"] < end_time)
                ]
            )
            heatmap_data[i, j] = concepts_in_period

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        xticklabels=period_labels,
        yticklabels=speakers,
        annot=True,
        fmt="g",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Sophisticated Concepts"},
        ax=ax,
    )

    ax.set_title(
        "Concept Frequency Heatmap - 10-Minute Time Windows",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Speaker", fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plot:
        plt.savefig("./concept_frequency_heatmap.png")
    plt.show()


def create_cumulative_concepts_chart(df: pd.DataFrame, save_plot: bool = True):
    """
    Create a cumulative chart showing how concepts accumulate over time.

    Args:
        df (pd.DataFrame): Processed data with sophisticated concepts
        save_plot (boolean): whether to save the plot to local image file
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    speakers = ["Piers", "Eric", "Sean"]
    speaker_colors = {"Piers": "#3498db", "Eric": "#e74c3c", "Sean": "#2ecc71"}

    for speaker in speakers:
        speaker_data = df[df["speaker"] == speaker].copy()

        if len(speaker_data) > 0:
            # Sort by time
            speaker_data = speaker_data.sort_values("start_minutes")

            # Create cumulative count
            speaker_data["cumulative_concepts"] = range(1, len(speaker_data) + 1)

            # Plot cumulative line
            ax.plot(
                speaker_data["start_minutes"],
                speaker_data["cumulative_concepts"],
                color=speaker_colors[speaker],
                linewidth=3,
                label=f"{speaker} ({len(speaker_data)} total)",
                marker="o",
                markersize=4,
                alpha=0.8,
            )

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Cumulative Sophisticated Concepts", fontsize=12)
    ax.set_title(
        "Cumulative Sophisticated Concepts Over Time", fontsize=16, fontweight="bold"
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add vertical lines for major time markers
    max_time = df["start_minutes"].max()
    for time_marker in [15, 30, 45]:
        if time_marker < max_time:
            ax.axvline(x=time_marker, color="gray", linestyle="--", alpha=0.5)
            ax.text(
                time_marker,
                ax.get_ylim()[1] * 0.9,
                f"{time_marker}min",
                rotation=90,
                ha="right",
                va="top",
                alpha=0.7,
            )

    plt.tight_layout()
    if save_plot:
        plt.savefig("./cumulative_concepts.png")
    plt.show()


def generate_time_based_summary(df):
    """
    Generate a summary of time-based patterns.

    Args:
        df (pd.DataFrame): Processed data with sophisticated concepts
    """
    logger.info("=" * 60)
    logger.info("TIME-BASED ANALYSIS SUMMARY")
    logger.info("=" * 60)

    max_time_minutes = df["start_minutes"].max()
    total_concepts = len(df)

    logger.info(f"📊 OVERALL STATISTICS:")
    logger.info(f"  • Total Discussion Time: {max_time_minutes:.1f} minutes")
    logger.info(f"  • Total Sophisticated Concepts: {total_concepts}")
    logger.info(
        f"  • Average Concepts per Minute: {total_concepts / max_time_minutes:.2f}"
    )

    # Analyze activity by time periods
    logger.info(f"\n⏰ ACTIVITY BY TIME PERIODS:")

    # Early, middle, late discussion
    early_concepts = len(df[df["start_minutes"] <= max_time_minutes / 3])
    middle_concepts = len(
        df[
            (df["start_minutes"] > max_time_minutes / 3)
            & (df["start_minutes"] <= 2 * max_time_minutes / 3)
        ]
    )
    late_concepts = len(df[df["start_minutes"] > 2 * max_time_minutes / 3])

    logger.info(
        f"  • Early Discussion (0-{max_time_minutes / 3:.1f}min): {early_concepts} concepts"
    )
    logger.info(
        f"  • Middle Discussion ({max_time_minutes / 3:.1f}-{2 * max_time_minutes / 3:.1f}min): {middle_concepts} concepts"
    )
    logger.info(
        f"  • Late Discussion ({2 * max_time_minutes / 3:.1f}-{max_time_minutes:.1f}min): {late_concepts} concepts"
    )

    # Most active periods for each speaker
    logger.info(f"\n🎯 PEAK ACTIVITY PERIODS:")

    speakers = ["Piers", "Eric", "Sean"]
    for speaker in speakers:
        speaker_data = df[df["speaker"] == speaker]
        if len(speaker_data) > 0:
            # Find 5-minute window with most concepts
            best_window_start = 0
            max_concepts_in_window = 0

            for start_time in range(0, int(max_time_minutes), 1):  # Check every minute
                window_concepts = len(
                    speaker_data[
                        (speaker_data["start_minutes"] >= start_time)
                        & (speaker_data["start_minutes"] < start_time + 5)
                    ]
                )
                if window_concepts > max_concepts_in_window:
                    max_concepts_in_window = window_concepts
                    best_window_start = start_time

            logger.info(
                f"  • {speaker}: Most active at {best_window_start}-{best_window_start + 5}min ({max_concepts_in_window} concepts)"
            )

    # Concept sophistication over time
    logger.info(f"\n🎓 SOPHISTICATION TRENDS:")

    education_hierarchy = {
        "Graduate": 4,
        "Undergraduate": 3,
        "High School/Undergraduate": 2.5,
    }
    df["sophistication_score"] = df["education_level"].map(education_hierarchy)

    # Compare first half vs second half
    first_half = df[df["start_minutes"] <= max_time_minutes / 2]
    second_half = df[df["start_minutes"] > max_time_minutes / 2]

    if len(first_half) > 0 and len(second_half) > 0:
        first_half_avg = first_half["sophistication_score"].mean()
        second_half_avg = second_half["sophistication_score"].mean()

        logger.info(f"  • First Half Average Sophistication: {first_half_avg:.2f}/4.0")
        logger.info(
            f"  • Second Half Average Sophistication: {second_half_avg:.2f}/4.0"
        )

        if second_half_avg > first_half_avg:
            logger.info(
                f"  • 📈 Discussion became MORE sophisticated over time (+{second_half_avg - first_half_avg:.2f})"
            )
        elif first_half_avg > second_half_avg:
            logger.info(
                f"  • 📉 Discussion became LESS sophisticated over time (-{first_half_avg - second_half_avg:.2f})"
            )
        else:
            logger.info(f"  • ➡️ Sophistication remained consistent throughout")


# Main execution
if __name__ == "__main__":
    # Replace with your CSV file path

    csv_file_path = (
        pathlib.Path(__file__).parent / "data" / "topics_preprocessed_by_jack.csv"
    )

    try:
        logger.info("Loading and processing data...")
        df = load_and_process_data(csv_file_path)

        if len(df) == 0:
            logger.info("❌ No sophisticated concepts found in the data!")
            exit()

        logger.info(
            f"✅ Found {len(df)} sophisticated concepts across {df['speaker'].nunique()} speakers"
        )

        # Create visualizations
        logger.info("\n📈 Creating timeline visualization...")
        create_timeline_visualization(df)

        logger.info("\n🔥 Creating concept frequency heatmap...")
        create_concept_frequency_heatmap(df)

        logger.info("\n📊 Creating cumulative concepts chart...")
        create_cumulative_concepts_chart(df)

        # Generate summary
        generate_time_based_summary(df)

        logger.info(f"\n✅ All visualizations complete!")

    except FileNotFoundError:
        logger.warning(f"❌ Error: Could not find the file '{csv_file_path}'")
        logger.info(
            "Please make sure the CSV file is in the same directory as this script."
        )
    except Exception as e:
        logger.warning(f"❌ Error occurred during analysis: {str(e)}")
