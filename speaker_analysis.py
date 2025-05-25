import pathlib
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import loguru


logger = loguru.logger


def analyze_speaker_data(csv_file_path: pathlib.Path):
    """
    Analyze speaker data from CSV file to show speaking time and concept sophistication.
    
    Args:
        csv_file_path (str): Path to the CSV file
    
    Returns:
        dict: Analysis results for each speaker
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Clean the data - remove rows where speaker is null
    df = df.dropna(subset=['speaker'])
    
    # Create education level hierarchy for sophistication ranking
    education_hierarchy = {
        'Undergraduate': 3,
        'Graduate': 4,
        'High School/Undergraduate': 2.5  # Keep this as it's partially undergraduate
    }
    
    # Define levels to include (exclude high school and below)
    included_levels = ['Undergraduate', 'Graduate', 'High School/Undergraduate']
    
    # Initialize results dictionary
    results = {}
    
    # Get unique speakers
    speakers = df['speaker'].unique()
    
    logger.info("="*60)
    logger.info("SPEAKER ANALYSIS OVERVIEW")
    logger.info("="*60)
    logger.info("Note: Analysis excludes concepts at High School level and below")
    logger.info("Included education levels: Undergraduate, Graduate, High School/Undergraduate")
    logger.info("="*60)
    
    for speaker in speakers:
        # Filter data for current speaker
        speaker_data = df[df['speaker'] == speaker].copy()
        
        # Calculate UNIQUE speaking time by deduplicating time segments
        # Group by start_time and end_time to get unique segments
        unique_segments = speaker_data.groupby(['start_time', 'end_time'])['duration'].first()
        total_duration = unique_segments.sum()
        total_unique_segments = len(unique_segments)
        
        # Filter concepts to include only sophisticated levels
        sophisticated_concepts = speaker_data[
            (speaker_data['education_level'].isin(included_levels)) & 
            (speaker_data['concept_name'].notna())
        ].copy()
        
        total_concepts = len(sophisticated_concepts)
        unique_concepts = sophisticated_concepts['concept_name'].nunique()
        
        # Education level analysis (only for included levels)
        education_counts = sophisticated_concepts['education_level'].value_counts()
        
        # Calculate sophistication metrics
        if len(sophisticated_concepts) > 0:
            sophistication_scores = [education_hierarchy.get(level, 0) for level in sophisticated_concepts['education_level']]
            avg_sophistication = np.mean(sophistication_scores)
            max_sophistication = max(sophistication_scores)
        else:
            avg_sophistication = 0
            max_sophistication = 0
        
        # Get list of unique sophisticated concepts for reference
        concept_list = sophisticated_concepts['concept_name'].unique().tolist()
        
        # Store results
        results[speaker] = {
            'total_speaking_time': total_duration,
            'total_unique_segments': total_unique_segments,
            'avg_segment_duration': total_duration / total_unique_segments if total_unique_segments > 0 else 0,
            'total_sophisticated_concepts': total_concepts,
            'unique_sophisticated_concepts': unique_concepts,
            'education_level_breakdown': dict(education_counts),
            'avg_sophistication_score': avg_sophistication,
            'max_sophistication_score': max_sophistication,
            'concept_density': total_concepts / total_duration if total_duration > 0 else 0,
            'concept_list': concept_list[:10]  # First 10 concepts for reference
        }
        
        # logger.info detailed analysis for each speaker
        logger.info(f"\n{speaker.upper()}")
        logger.info("-" * 40)
        logger.info(f"Speaking Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        logger.info(f"Number of Unique Speaking Segments: {total_unique_segments}")
        logger.info(f"Average Segment Duration: {total_duration/total_unique_segments:.2f} seconds")
        
        logger.info(f"\nSophisticated Concept Analysis (Undergraduate level and above):")
        logger.info(f"  • Total Sophisticated Concepts: {total_concepts}")
        logger.info(f"  • Unique Sophisticated Concepts: {unique_concepts}")
        
        if total_concepts > 0:
            logger.info(f"  • Concept Density: {total_concepts/total_duration:.4f} concepts per second")
            logger.info(f"  • Average Sophistication Score: {avg_sophistication:.2f}/4.0")
            logger.info(f"  • Highest Sophistication Level: {max_sophistication}/4.0")
            
            logger.info(f"\nSophistication Level Breakdown:")
            for level, count in education_counts.items():
                score = education_hierarchy.get(level, 0)
                percentage = (count / total_concepts * 100)
                logger.info(f"  • {level}: {count} concepts ({percentage:.1f}%) [Score: {score}]")
            
            logger.info(f"\nExample Sophisticated Concepts:")
            for i, concept in enumerate(concept_list[:5]):  # Show first 5
                logger.info(f"  • {concept}")
        else:
            logger.info(f"  • No sophisticated concepts found for this speaker")
    
    return results

def create_visualizations(
        results,
        save_plot: bool = True
):
    """
    Create visualizations for the speaker analysis.
    
    Args:
        results (dict): Results from analyze_speaker_data function
        save_plot (bool): whether or not to store the plot
    """
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Speaker Analysis Dashboard - Sophisticated Concepts Only', fontsize=16, fontweight='bold')
    
    speakers = list(results.keys())
    
    # 1. Speaking Time Comparison
    speaking_times = [results[speaker]['total_speaking_time']/60 for speaker in speakers]  # Convert to minutes
    axes[0, 0].bar(speakers, speaking_times, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Total Speaking Time (Minutes)\n(Deduplicated)')
    axes[0, 0].set_ylabel('Minutes')
    for i, v in enumerate(speaking_times):
        axes[0, 0].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
    
    # 2. Sophisticated Concept Counts
    total_concepts = [results[speaker]['total_sophisticated_concepts'] for speaker in speakers]
    unique_concepts = [results[speaker]['unique_sophisticated_concepts'] for speaker in speakers]
    
    x = np.arange(len(speakers))
    width = 0.35
    axes[0, 1].bar(x - width/2, total_concepts, width, label='Total Concepts', color='orange', alpha=0.8)
    axes[0, 1].bar(x + width/2, unique_concepts, width, label='Unique Concepts', color='purple', alpha=0.8)
    axes[0, 1].set_title('Sophisticated Concepts\n(Undergraduate Level and Above)')
    axes[0, 1].set_ylabel('Number of Concepts')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(speakers)
    axes[0, 1].legend()
    
    # 3. Sophistication Scores
    avg_scores = [results[speaker]['avg_sophistication_score'] for speaker in speakers]
    max_scores = [results[speaker]['max_sophistication_score'] for speaker in speakers]
    
    axes[1, 0].bar(x - width/2, avg_scores, width, label='Average Score', color='gold', alpha=0.8)
    axes[1, 0].bar(x + width/2, max_scores, width, label='Max Score', color='crimson', alpha=0.8)
    axes[1, 0].set_title('Sophistication Scores\n(2.5-4.0 Scale)')
    axes[1, 0].set_ylabel('Sophistication Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(speakers)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 4.5)
    
    # 4. Concept Density
    concept_density = [results[speaker]['concept_density'] for speaker in speakers]
    axes[1, 1].bar(speakers, concept_density, color=['teal', 'orange', 'purple'])
    axes[1, 1].set_title('Sophisticated Concept Density\n(Concepts per Second)')
    axes[1, 1].set_ylabel('Concepts/Second')
    for i, v in enumerate(concept_density):
        if v > 0:
            axes[1, 1].text(i, v + max(concept_density)*0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_plot:
        plt.savefig('./plot.png')
    plt.show()

def generate_summary_report(results):
    """
    Generate a summary report with key insights.
    
    Args:
        results (dict): Results from analyze_speaker_data function
    """
    logger.info("\n" + "="*60)
    logger.info("SUMMARY REPORT - SOPHISTICATED CONCEPTS ONLY")
    logger.info("="*60)
    
    # Find speaker with most speaking time
    max_time_speaker = max(results.keys(), key=lambda x: results[x]['total_speaking_time'])
    max_time = results[max_time_speaker]['total_speaking_time']
    
    # Find speaker with highest concept density (among those with concepts)
    speakers_with_concepts = {k: v for k, v in results.items() if v['concept_density'] > 0}
    
    if speakers_with_concepts:
        max_density_speaker = max(speakers_with_concepts.keys(), key=lambda x: results[x]['concept_density'])
        max_density = results[max_density_speaker]['concept_density']
        
        # Find speaker with highest average sophistication
        max_soph_speaker = max(speakers_with_concepts.keys(), key=lambda x: results[x]['avg_sophistication_score'])
        max_soph = results[max_soph_speaker]['avg_sophistication_score']
        
        logger.info(f"🎯 KEY INSIGHTS:")
        logger.info(f"  • Most Speaking Time: {max_time_speaker} ({max_time/60:.1f} minutes)")
        logger.info(f"  • Highest Sophisticated Concept Density: {max_density_speaker} ({max_density:.4f} concepts/sec)")
        logger.info(f"  • Most Sophisticated Concepts: {max_soph_speaker} (avg score: {max_soph:.2f}/4.0)")
    else:
        logger.info(f"🎯 KEY INSIGHTS:")
        logger.info(f"  • Most Speaking Time: {max_time_speaker} ({max_time/60:.1f} minutes)")
        logger.info(f"  • No sophisticated concepts found in the analysis")
    
    # Calculate total statistics
    total_time = sum(results[speaker]['total_speaking_time'] for speaker in results)
    total_concepts = sum(results[speaker]['total_sophisticated_concepts'] for speaker in results)
    total_unique_concepts = len(set().union(*[set(results[speaker].get('concept_list', [])) for speaker in results]))
    
    logger.info(f"\n📊 OVERALL STATISTICS:")
    logger.info(f"  • Total Discussion Time: {total_time/60:.1f} minutes (deduplicated)")
    logger.info(f"  • Total Sophisticated Concepts Mentioned: {total_concepts}")
    logger.info(f"  • Total Unique Sophisticated Concepts: {total_unique_concepts}")
    if total_time > 0 and total_concepts > 0:
        logger.info(f"  • Average Sophisticated Concepts per Minute: {total_concepts/(total_time/60):.1f}")
    
    logger.info(f"\n📈 SOPHISTICATION BREAKDOWN:")
    all_education_counts = {}
    for speaker in results:
        for level, count in results[speaker]['education_level_breakdown'].items():
            all_education_counts[level] = all_education_counts.get(level, 0) + count
    
    if all_education_counts:
        for level, count in sorted(all_education_counts.items(), key=lambda x: {'Graduate': 4, 'Undergraduate': 3, 'High School/Undergraduate': 2.5}.get(x[0], 0), reverse=True):
            percentage = (count / total_concepts * 100) if total_concepts > 0 else 0
            logger.info(f"  • {level}: {count} concepts ({percentage:.1f}%)")
    
    logger.info(f"\n💡 NOTE: Analysis excludes High School level and below concepts")
    logger.info(f"   Only Undergraduate, Graduate, and High School/Undergraduate levels included")

# Main execution
if __name__ == "__main__":
    # Replace with your CSV file path
    
    file_path = pathlib.Path(__file__).parent / "data" / "topics_preprocessed_by_jack.csv"
    
    try:
        # Run the analysis
        results = analyze_speaker_data(file_path)
        
        # Create visualizations
        create_visualizations(results)
        
        # Generate summary report
        generate_summary_report(results)
        
        logger.info(f"\n✅ Analysis complete! Check the generated visualizations above.")
        
    except FileNotFoundError:
        logger.warning(f"❌ Error: Could not find the file '{file_path}'")
        logger.warning("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        logger.warning(f"❌ Error occurred during analysis: {str(e)}")
