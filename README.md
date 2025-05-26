# YouTube Science Discussion Analyzer

A Python toolkit for analyzing physics and mathematics concepts discussed in YouTube video transcripts. This project extracts technical concepts from spoken dialogue, classifies them by education level, and generates comprehensive visualizations showing how sophisticated concepts are distributed among speakers over time.

## Features

- **Transcript Extraction**: Parse YouTube transcript HTML files and convert to structured CSV format
- **Concept Detection**: Automatically identify physics and mathematics concepts using local LLM (Ollama)
- **Education Level Classification**: Categorize concepts by sophistication (High School, Undergraduate, Graduate)
- **Speaker Analysis**: Compare speaking time, concept density, and sophistication levels across participants
- **Timeline Visualization**: Track how concepts emerge throughout the discussion
- **Comprehensive Reporting**: Generate detailed statistical summaries and insights

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) with `llama3.1:8b` model
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
# Install dependencies
uv sync

# Download required Ollama model
ollama pull llama3.1:8b
```

### Usage

1. **Extract YouTube Transcript**: Save the transcript HTML from YouTube and place in `data/` folder

2. **Process Transcript**:
   ```bash
   python get_transcript.py
   ```

3. **Analyze Concepts** (using Jupyter notebook):
   ```bash
   jupyter lab wrangle.ipynb
   ```

4. **Generate Visualizations**:
   ```bash
   # Speaker analysis with concept density and sophistication
   python speaker_analysis.py
   
   # Timeline analysis showing concepts over time
   python topics_over_time.py
   ```

## Output Examples

The toolkit generates several types of analysis:

### Speaker Analysis
- **Speaking time distribution** (deduplicated segments)
- **Concept density** (sophisticated concepts per second)
- **Sophistication scores** by education level
- **Comparative breakdowns** across participants

### Timeline Analysis
- **Concept emergence patterns** throughout the discussion
- **Activity heatmaps** in 10-minute windows  
- **Cumulative sophistication** progression
- **Peak activity identification** for each speaker

### Sample Insights
```
🔍 KEY INSIGHTS:
  • Most Speaking Time: Eric (15.2 minutes)
  • Highest Sophisticated Concept Density: Sean (0.0045 concepts/sec)
  • Most Sophisticated Concepts: Eric (avg score: 3.2/4.0)

📊 OVERALL STATISTICS:
  • Total Discussion Time: 59.4 minutes
  • Total Sophisticated Concepts Mentioned: 127
  • Average Sophisticated Concepts per Minute: 2.1
```

## Project Structure

```
├── get_transcript.py          # YouTube transcript extraction
├── parse_technical_terms.py   # LLM-based concept detection
├── speaker_analysis.py        # Speaker comparison analysis
├── topics_over_time.py       # Timeline and temporal analysis
├── wrangle.ipynb             # Data processing notebook
├── data/                     # Input transcripts and output CSVs
└── *.png                     # Generated visualization outputs
```

## Key Technologies

- **pandas** - Data manipulation and analysis
- **matplotlib/seaborn** - Statistical visualizations
- **ollama** - Local LLM integration for concept extraction
- **pydantic** - Structured data validation
- **BeautifulSoup** - HTML transcript parsing

## Development

```bash
# Format code
make format

# Run tests
make test

# Type checking
make mypy

# Package codebase for AI analysis
make pack
```

## Example Use Case

This project was developed to analyze a debate between physicist Sean Carroll and mathematician Eric Weinstein on Piers Morgan's show, automatically identifying when sophisticated physics concepts like "chiral fermions", "SU(3) symmetry", or "gauge theories" were mentioned and tracking their distribution across speakers and time.

## License

MIT License - see LICENSE file for details