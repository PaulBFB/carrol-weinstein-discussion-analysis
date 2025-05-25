import pathlib
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_speaker_data(csv_file_path):
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
        'Middle School': 1,
        'High School': 2,
        'High School/Undergraduate': 2.5,
        'Undergraduate': 3,
        'Graduate': 4,
        'General knowledge': 1.5  # Treating as basic level
    }
    
    # Initialize results dictionary
    results = {}
    
    # Get unique speakers
    speakers = df['speaker'].unique()
    
    print("="*60)
    print("SPEAKER ANALYSIS OVERVIEW")
    print("="*60)
    
    for speaker in speakers:
        # Filter data for current speaker
        speaker_data = df[df['speaker'] == speaker].copy()
        
        # Calculate total speaking time
        total_duration = speaker_data['duration'].sum()
        
        # Count total segments
        total_segments = len(speaker_data)
        
        # Analyze concepts mentioned
        concepts_data = speaker_data.dropna(subset=['concept_name'])
        total_concepts = len(concepts_data)
        unique_concepts = concepts_data['concept_name'].nunique()
        
        # Education level analysis
        education_counts = concepts_data['education_level'].value_counts().fillna(0)
        
        # Calculate sophistication metrics
        education_levels = concepts_data['education_level'].dropna()
        if len(education_levels) > 0:
            sophistication_scores = [education_hierarchy.get(level, 0) for level in education_levels]
            avg_sophistication = np.mean(sophistication_scores)
            max_sophistication = max(sophistication_scores)
        else:
            avg_sophistication = 0
            max_sophistication = 0
        
        # Store results
        results[speaker] = {
            'total_speaking_time': total_duration,
            'total_segments': total_segments,
            'avg_segment_duration': total_duration / total_segments if total_segments > 0 else 0,
            'total_concepts_mentioned': total_concepts,
            'unique_concepts': unique_concepts,
            'education_level_breakdown': dict(education_counts),
            'avg_sophistication_score': avg_sophistication,
            'max_sophistication_score': max_sophistication,
            'concept_density': total_concepts / total_duration if total_duration > 0 else 0
        }
        
        # Print detailed analysis for each speaker
        print(f"\n{speaker.upper()}")
        print("-" * 40)
        print(f"Speaking Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"Number of Speaking Segments: {total_segments}")
        print(f"Average Segment Duration: {total_duration/total_segments:.2f} seconds")
        print(f"Concept Density: {total_concepts/total_duration:.3f} concepts per second")
        print(f"\nConcept Analysis:")
        print(f"  • Total Concepts Mentioned: {total_concepts}")
        print(f"  • Unique Concepts: {unique_concepts}")
        print(f"  • Average Sophistication Score: {avg_sophistication:.2f}/4.0")
        print(f"  • Highest Sophistication Level: {max_sophistication}/4.0")
        
        print(f"\nSophistication Breakdown:")
        for level, count in education_counts.items():
            if pd.notna(level):
                score = education_hierarchy.get(level, 0)
                percentage = (count / total_concepts * 100) if total_concepts > 0 else 0
                print(f"  • {level}: {count} concepts ({percentage:.1f}%) [Score: {score}]")
    
    return results

def create_visualizations(results):
    """
    Create visualizations for the speaker analysis.
    
    Args:
        results (dict): Results from analyze_speaker_data function
    """
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Speaker Analysis Dashboard', fontsize=16, fontweight='bold')
    
    speakers = list(results.keys())
    
    # 1. Speaking Time Comparison
    speaking_times = [results[speaker]['total_speaking_time']/60 for speaker in speakers]  # Convert to minutes
    axes[0, 0].bar(speakers, speaking_times, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Total Speaking Time (Minutes)')
    axes[0, 0].set_ylabel('Minutes')
    for i, v in enumerate(speaking_times):
        axes[0, 0].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
    
    # 2. Concept Counts
    total_concepts = [results[speaker]['total_concepts_mentioned'] for speaker in speakers]
    unique_concepts = [results[speaker]['unique_concepts'] for speaker in speakers]
    
    x = np.arange(len(speakers))
    width = 0.35
    axes[0, 1].bar(x - width/2, total_concepts, width, label='Total Concepts', color='orange', alpha=0.8)
    axes[0, 1].bar(x + width/2, unique_concepts, width, label='Unique Concepts', color='purple', alpha=0.8)
    axes[0, 1].set_title('Concepts Mentioned')
    axes[0, 1].set_ylabel('Number of Concepts')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(speakers)
    axes[0, 1].legend()
    
    # 3. Sophistication Scores
    avg_scores = [results[speaker]['avg_sophistication_score'] for speaker in speakers]
    max_scores = [results[speaker]['max_sophistication_score'] for speaker in speakers]
    
    axes[1, 0].bar(x - width/2, avg_scores, width, label='Average Score', color='gold', alpha=0.8)
    axes[1, 0].bar(x + width/2, max_scores, width, label='Max Score', color='crimson', alpha=0.8)
    axes[1, 0].set_title('Sophistication Scores (1-4 Scale)')
    axes[1, 0].set_ylabel('Sophistication Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(speakers)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 4.5)
    
    # 4. Concept Density
    concept_density = [results[speaker]['concept_density'] for speaker in speakers]
    axes[1, 1].bar(speakers, concept_density, color=['teal', 'orange', 'purple'])
    axes[1, 1].set_title('Concept Density (Concepts per Second)')
    axes[1, 1].set_ylabel('Concepts/Second')
    for i, v in enumerate(concept_density):
        axes[1, 1].text(i, v + 0.0001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./plot')
    plt.show()

def generate_summary_report(results):
    """
    Generate a summary report with key insights.
    
    Args:
        results (dict): Results from analyze_speaker_data function
    """
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    # Find speaker with most speaking time
    max_time_speaker = max(results.keys(), key=lambda x: results[x]['total_speaking_time'])
    max_time = results[max_time_speaker]['total_speaking_time']
    
    # Find speaker with highest concept density
    max_density_speaker = max(results.keys(), key=lambda x: results[x]['concept_density'])
    max_density = results[max_density_speaker]['concept_density']
    
    # Find speaker with highest average sophistication
    max_soph_speaker = max(results.keys(), key=lambda x: results[x]['avg_sophistication_score'])
    max_soph = results[max_soph_speaker]['avg_sophistication_score']
    
    print(f"🎯 KEY INSIGHTS:")
    print(f"  • Most Speaking Time: {max_time_speaker} ({max_time/60:.1f} minutes)")
    print(f"  • Highest Concept Density: {max_density_speaker} ({max_density:.4f} concepts/sec)")
    print(f"  • Most Sophisticated Concepts: {max_soph_speaker} (avg score: {max_soph:.2f}/4.0)")
    
    # Calculate total statistics
    total_time = sum(results[speaker]['total_speaking_time'] for speaker in results)
    total_concepts = sum(results[speaker]['total_concepts_mentioned'] for speaker in results)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  • Total Discussion Time: {total_time/60:.1f} minutes")
    print(f"  • Total Concepts Discussed: {total_concepts}")
    print(f"  • Average Concepts per Minute: {total_concepts/(total_time/60):.1f}")

# Main execution
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file_path = pathlib.Path(__file__).parent / "data" / "topics_preprocessed_by_jack.csv"
    
    try:
        # Run the analysis
        results = analyze_speaker_data(csv_file_path)
        
        # Create visualizations
        create_visualizations(results)
        
        # Generate summary report
        generate_summary_report(results)
        
        print(f"\n✅ Analysis complete! Check the generated visualizations above.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{csv_file_path}'")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error occurred during analysis: {str(e)}")
