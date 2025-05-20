# Napoleon Text Processing Project

A generalized historical text processing pipeline for analyzing, structuring, and organizing historical documents.

## Vision

Create a flexible text processing system that can:
- Analyze any historical text corpus
- Identify different voices, topics, and structural elements
- Extract meaningful patterns without hardcoded assumptions
- Support training of specialized language models

## Three-Stage Processing Pipeline

### Stage 1: Structural Parsing
- Extract raw text and preserve encoding
- Identify document structure (headers, sections, chapters)
- Remove modern publishing metadata
- Detect footnotes, endnotes, and editorial insertions
- Preserve document hierarchy

### Stage 2: Content Classification  
- Distinguish between voices (direct speech, quotations, narration)
- Classify content by type (description, reflection, dialogue)
- Detect stylistic patterns that indicate authorship
- Identify embedded documents (letters, speeches, journal entries)

### Stage 3: Semantic Organization
- Cluster content by topic across documents
- Build relationships between themes and concepts
- Create navigable semantic maps of the corpus
- Support intelligent retrieval by concept

## Iterative Development Plan

### Phase 1: Foundation (Current)
- ✅ Basic text ingestion and encoding detection
- ✅ Initial cleaning of obvious metadata
- ✅ Chunking for manageable processing
- ✅ Simple JSON output format

### Phase 2: Enhanced Structure Detection (Next)
- Improve boundary detection between metadata and content
- Detect natural section breaks within documents
- Identify quotations and embedded documents
- Build document structure maps

### Phase 3: Voice and Authorship Analysis
- Train classifiers to distinguish between voices
- Detect stylistic patterns indicating authorship
- Identify first-person vs third-person narratives
- Flag content with direct speech vs biographical material

### Phase 4: Semantic Understanding
- Implement topic modeling across the corpus
- Build relationships between related content
- Create topic graphs showing conceptual relationships
- Support content retrieval by semantic similarity

### Phase 5: Application-Specific Optimizations
- Fine-tune for specific historical figures or topics
- Optimize for training specialized language models
- Create APIs for intelligent content retrieval
- Build visualization tools for corpus exploration

## Immediate Next Steps

1. Enhance the metadata extraction with improved section boundary detection
2. Implement quotation and direct speech detection
3. Create a document structure analyzer for hierarchical content mapping
4. Develop a simple voice attribution classifier
5. Test the system across different types of historical documents

## Technologies

- Python for core processing
- NLTK and spaCy for NLP tasks
- LLMs (via Ollama) for intelligent text analysis
- Machine learning for classification and clustering
- Potential graph databases for semantic relationships

## Project Structure
napoleon/
├── venv/ # Virtual environment for isolated dependencies
├── corpus/ # Raw downloaded texts
├── segmented_corpus/ # Processed text segments
├── code/ # Python scripts
│ ├── inspect_project_gutenberg.py
│ └── napoleon_data_scraper.py
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Installation
1. Clone this repository:
git clone https://github.com/yourusername/napoleon.git
cd napoleon

2. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

3. Install required packages:
pip install -r requirements.txt


## Usage
### Inspect Project Gutenberg Structure
To verify the structure of the Gutenberg page:
python code/inspect_project_gutenberg.py


### Scrape Napoleon's Writings
To collect texts from Project Gutenberg:
python code/napoleon_data_scraper.py


## Future Enhancements
- Integrate more sources such as Wikisource and The Napoleon Series.
- Automate text cleaning and context extraction.

## License
This project is licensed under the MIT License.