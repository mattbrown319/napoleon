# Napoleon Text Processing Project

A generalized historical text processing pipeline for analyzing, structuring, and organizing historical documents.

## Vision

Create a flexible text processing system that can:
- Analyze any historical text corpus
- Identify different voices, topics, and structural elements
- Extract meaningful patterns without hardcoded assumptions
- Support training of specialized language models

## Current Status (2025-05-21)

We've implemented a hierarchical document analyzer that:
- Processes entire documents, not just samples
- Detects natural boundaries (content start/end, sections, chapters)
- Identifies quotations and embedded documents
- Maps relationships between document elements
- Creates a comprehensive structural map

**Performance Optimization**: Through extensive testing, we've discovered Ollama can support significant parallelism (5.35x speedup with 10 concurrent requests). Our implementation now uses:
- ThreadPoolExecutor with 8 workers by default
- Distributed processing across multiple Ollama instances
- Round-robin port assignment for load balancing
- Ability to utilize more system memory (180GB+ observed)

The distributed approach allows for significantly faster processing of long documents like the 172-section Napoleon biography.

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

### Phase 1: Foundation (Completed)
- ✅ Basic text ingestion and encoding detection
- ✅ Initial cleaning of obvious metadata
- ✅ Chunking for manageable processing
- ✅ Simple JSON output format

### Phase 2: Enhanced Structure Detection (Current)
- ✅ Improved boundary detection between metadata and content
- ✅ Detection of natural section breaks within documents
- ✅ Hierarchical document processing approach
- ✅ Complete document scanning
- ✅ Multi-instance distributed processing
- 🔄 Voice and authorship attribution

#### Phase 2 Implementation Details
Our hierarchical analyzer:

1. **Complete Document Analysis**
   - First analyzes document boundaries
   - Then scans the entire text for sections using regex patterns
   - Finally processes each section for deeper understanding
   - Identifies quotations and embedded documents throughout the entire text

2. **Multiple Detection Methods**
   - Fast regex scanning for structural elements
   - Targeted LLM analysis for semantic understanding
   - Fallback mechanisms when LLM queries timeout

3. **Relationship Mapping**
   - Tracks which quotes belong to which sections
   - Associates embedded documents with their containing sections
   - Creates a complete hierarchical representation

4. **Distributed Processing**
   - Runs multiple Ollama instances on different ports
   - Distributes section processing across instances
   - Uses round-robin assignment for load balancing
   - Achieves significantly better resource utilization

### Phase 3: Voice and Authorship Analysis (Next)
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

1. **Improve reliability of distributed processing**
   - Add retry mechanism with exponential backoff for EOF errors
   - Implement health checks for Ollama instances
   - Add progress tracking with time estimation
   - Fix duplicate data in markers and section titles

2. **Complete testing of hierarchical analyzer**
   - Test with Llama 3 70B model for improved performance
   - Analyze and visualize the resulting document structure

3. **Performance optimizations**
   - Implement batching for LLM queries to reduce API calls
   - Add selective processing for section topic identification
   - Optimize regex patterns for better detection accuracy

4. **Voice attribution enhancements**
   - Improve speaker detection for quotations
   - Add stylistic analysis for authorship identification
   - Implement cross-reference resolution for names and people

5. **Structured output formats**
   - Create formats optimized for model training
   - Design hierarchical JSON schema for complete document representation
   - Develop options for filtered outputs (e.g., only Napoleon's direct words)

## Key Design Decisions

1. **Hierarchical vs. Chunking**: We chose a hierarchical approach over chunking to preserve document structure and relationships.

2. **Complete Scanning**: We process the entire document rather than samples to ensure comprehensive coverage.

3. **Hybrid Detection**: We use regex for initial scanning and LLM for deeper analysis to balance efficiency and accuracy.

4. **Relationship Tracking**: We explicitly track relationships between document elements to support more sophisticated analysis.

5. **Distributed Processing**: We deploy multiple Ollama instances to better utilize system memory and increase throughput.

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
├── code/ 
│ ├── hierarchical_analyzer.py # New hierarchical document analysis
│ ├── structure_analyzer.py # Previous structure detection
│ ├── text_processor.py # Basic text processing
│ ├── start_ollama_instances.sh # Script to start multiple Ollama instances
│ └── napoleon_data_scraper.py # Data collection
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Usage
### Running Distributed Processing
To start multiple Ollama instances:
```bash
./code/start_ollama_instances.sh --ports 11434,11435,11436,11437
```

### Process Document Structure Hierarchically
To analyze the structure of a document using multiple Ollama instances:
```bash
python code/test_hierarchical.py --ports 11434,11435,11436,11437 path/to/file.txt
```

### Process Text (Original Method)
To process text files in a directory:
```bash
python code/process_texts.py input_directory output_directory
```


## Future Enhancements
- Integrate more sources such as Wikisource and The Napoleon Series.
- Automate text cleaning and context extraction.
- Build interactive visualization of document structure.
- Implement a dashboard for monitoring distributed processing.

## License
This project is licensed under the MIT License.