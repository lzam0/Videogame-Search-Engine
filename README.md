# README: IR Course Work

This project implements an Information Retrieval (IR) system that processes a collection of HTML documents related to video games. The system extracts metadata, applies weighting based on the content, and allows users to search through the documents using queries. The results are ranked by relevance, and both text-based tables and bar charts display the search results.

## Features

- **Reading HTML Files**: Processes HTML documents in a specified directory (`videogames`).
- **Content Extraction**: Extracts key information like developer, publisher, genre, and ESRB rating from each document.
- **Content Weighting**: Applies additional weight to important sections (developer, publisher, genre, and ESRB) to improve search relevance.
- **Search Queries**: Allows users to input queries to search through the documents, ranking the results by similarity.
- **Efficiency Testing**: Measures the time taken for processing and displays the efficiency of the system with a box plot.
- **Visualization**: Displays search results using tabular format and bar charts for better clarity.

## Installation

### Prerequisites

- **Python 3.x**

### Required Libraries

- `beautifulsoup4`
- `scikit-learn`
- `spacy`
- `nltk`
- `matplotlib`
- `regex`
- `tabulate`

You can install the required libraries using `pip`:

```bash
pip install beautifulsoup4 scikit-learn spacy nltk matplotlib regex tabulate
```

Additionally, install the spaCy English model and download NLTK stopwords:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt
```

Usage
To use the IR system, follow these steps:

Prepare the HTML Documents: Place your HTML documents in the videogames directory.

Run the Script: Execute the main script to process the documents and start the search interface.

Input Queries: Enter your search queries when prompted.

View Results: The system will display the search results in both tabular format and bar charts.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Thanks to the developers of the libraries used in this project.

Special thanks to the course instructors for their guidance and support.
