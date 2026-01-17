"""
Research Paper Synthesizer - Paper Sources

Implementations for fetching papers from various academic sources:
- Arxiv
- Semantic Scholar
- PubMed
- Local knowledge base
"""

from abc import ABC, abstractmethod
from typing import Optional
import urllib.parse
import time
import re

from .state import Paper, Author, PaperSource


class BasePaperSource(ABC):
    """Abstract base class for paper sources."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search for papers matching the query."""
        pass
    
    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by ID."""
        pass


class ArxivSource(BasePaperSource):
    """
    Arxiv paper source.
    
    Uses the Arxiv API to search and retrieve papers.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self._last_request = 0
    
    def _rate_limit(self):
        """Respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()
    
    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search Arxiv for papers."""
        try:
            import urllib.request
            import xml.etree.ElementTree as ET
            
            self._rate_limit()
            
            # Build query URL
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            
            # Fetch results
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            papers = []
            for entry in root.findall("atom:entry", ns):
                paper = self._parse_entry(entry, ns)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Arxiv search error: {e}")
            return []
    
    def _parse_entry(self, entry, ns) -> Optional[Paper]:
        """Parse an Arxiv entry into a Paper."""
        try:
            # Extract ID
            id_elem = entry.find("atom:id", ns)
            arxiv_url = id_elem.text if id_elem is not None else ""
            arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else arxiv_url
            
            # Extract title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Unknown"
            
            # Extract abstract
            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract authors
            authors = []
            for author_elem in entry.findall("atom:author", ns):
                name_elem = author_elem.find("atom:name", ns)
                if name_elem is not None:
                    affil_elem = author_elem.find("arxiv:affiliation", 
                        {"arxiv": "http://arxiv.org/schemas/atom"})
                    affiliation = affil_elem.text if affil_elem is not None else None
                    authors.append(Author(name=name_elem.text, affiliation=affiliation))
            
            # Extract year from published date
            published_elem = entry.find("atom:published", ns)
            year = 2024
            if published_elem is not None:
                year = int(published_elem.text[:4])
            
            # Extract PDF URL
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break
            
            return Paper(
                id=f"arxiv:{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                source=PaperSource.ARXIV,
                url=arxiv_url,
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
            )
            
        except Exception as e:
            print(f"Error parsing Arxiv entry: {e}")
            return None
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by Arxiv ID."""
        arxiv_id = paper_id.replace("arxiv:", "")
        papers = self.search(f"id:{arxiv_id}", max_results=1)
        return papers[0] if papers else None


class SemanticScholarSource(BasePaperSource):
    """
    Semantic Scholar paper source.
    
    Uses the Semantic Scholar API for academic paper search.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None, delay: float = 1.0):
        self.api_key = api_key
        self.delay = delay
        self._last_request = 0
    
    def _rate_limit(self):
        """Respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()
    
    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search Semantic Scholar for papers."""
        try:
            import urllib.request
            import json
            
            self._rate_limit()
            
            # Build query URL
            fields = "paperId,title,abstract,authors,year,url,citationCount,venue,externalIds"
            params = {
                "query": query,
                "limit": max_results,
                "fields": fields,
            }
            url = f"{self.BASE_URL}/paper/search?{urllib.parse.urlencode(params)}"
            
            # Build request
            req = urllib.request.Request(url)
            if self.api_key:
                req.add_header("x-api-key", self.api_key)
            
            # Fetch results
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            papers = []
            for item in data.get("data", []):
                paper = self._parse_result(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []
    
    def _parse_result(self, item: dict) -> Optional[Paper]:
        """Parse a Semantic Scholar result into a Paper."""
        try:
            # Extract authors
            authors = []
            for author in item.get("authors", []):
                authors.append(Author(name=author.get("name", "Unknown")))
            
            # Get external IDs
            external_ids = item.get("externalIds", {})
            
            return Paper(
                id=f"s2:{item.get('paperId', '')}",
                title=item.get("title", "Unknown"),
                abstract=item.get("abstract", "") or "",
                authors=authors,
                year=item.get("year", 2024) or 2024,
                source=PaperSource.SEMANTIC_SCHOLAR,
                url=item.get("url"),
                doi=external_ids.get("DOI"),
                arxiv_id=external_ids.get("ArXiv"),
                citations_count=item.get("citationCount", 0) or 0,
                venue=item.get("venue"),
            )
            
        except Exception as e:
            print(f"Error parsing Semantic Scholar result: {e}")
            return None
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by Semantic Scholar ID."""
        try:
            import urllib.request
            import json
            
            self._rate_limit()
            
            s2_id = paper_id.replace("s2:", "")
            fields = "paperId,title,abstract,authors,year,url,citationCount,venue,externalIds"
            url = f"{self.BASE_URL}/paper/{s2_id}?fields={fields}"
            
            req = urllib.request.Request(url)
            if self.api_key:
                req.add_header("x-api-key", self.api_key)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            return self._parse_result(data)
            
        except Exception as e:
            print(f"Error fetching paper {paper_id}: {e}")
            return None


class PubMedSource(BasePaperSource):
    """
    PubMed paper source.
    
    Uses the NCBI E-utilities API for biomedical literature search.
    """
    
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, delay: float = 0.35):  # NCBI allows ~3 requests/second
        self.delay = delay
        self._last_request = 0
    
    def _rate_limit(self):
        """Respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()
    
    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search PubMed for papers."""
        try:
            import urllib.request
            import xml.etree.ElementTree as ET
            
            self._rate_limit()
            
            # First, search for IDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "xml",
            }
            search_url = f"{self.SEARCH_URL}?{urllib.parse.urlencode(search_params)}"
            
            with urllib.request.urlopen(search_url, timeout=30) as response:
                search_data = response.read().decode('utf-8')
            
            # Parse IDs
            root = ET.fromstring(search_data)
            ids = [id_elem.text for id_elem in root.findall(".//Id")]
            
            if not ids:
                return []
            
            self._rate_limit()
            
            # Fetch details
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            }
            fetch_url = f"{self.FETCH_URL}?{urllib.parse.urlencode(fetch_params)}"
            
            with urllib.request.urlopen(fetch_url, timeout=30) as response:
                fetch_data = response.read().decode('utf-8')
            
            # Parse papers
            root = ET.fromstring(fetch_data)
            papers = []
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def _parse_article(self, article) -> Optional[Paper]:
        """Parse a PubMed article into a Paper."""
        try:
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None
            
            # Get PMID
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "unknown"
            
            # Get article info
            article_elem = medline.find(".//Article")
            if article_elem is None:
                return None
            
            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "Unknown"
            
            # Abstract
            abstract_elem = article_elem.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Authors
            authors = []
            for author_elem in article_elem.findall(".//Author"):
                last_name = author_elem.find("LastName")
                first_name = author_elem.find("ForeName")
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{first_name.text} {name}"
                    
                    affil_elem = author_elem.find(".//Affiliation")
                    affiliation = affil_elem.text if affil_elem is not None else None
                    authors.append(Author(name=name, affiliation=affiliation))
            
            # Year
            year = 2024
            pub_date = article_elem.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None:
                    year = int(year_elem.text)
            
            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            venue = journal_elem.text if journal_elem is not None else None
            
            # DOI
            doi = None
            for id_elem in article.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text
                    break
            
            return Paper(
                id=f"pubmed:{pmid}",
                title=title,
                abstract=abstract or "",
                authors=authors,
                year=year,
                source=PaperSource.PUBMED,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=doi,
                venue=venue,
            )
            
        except Exception as e:
            print(f"Error parsing PubMed article: {e}")
            return None
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by PubMed ID."""
        pmid = paper_id.replace("pubmed:", "")
        
        try:
            import urllib.request
            import xml.etree.ElementTree as ET
            
            self._rate_limit()
            
            fetch_params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
            }
            url = f"{self.FETCH_URL}?{urllib.parse.urlencode(fetch_params)}"
            
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode('utf-8')
            
            root = ET.fromstring(data)
            article = root.find(".//PubmedArticle")
            
            if article is not None:
                return self._parse_article(article)
            
            return None
            
        except Exception as e:
            print(f"Error fetching paper {paper_id}: {e}")
            return None


class MultiSourceSearch:
    """
    Unified search across multiple paper sources.
    """
    
    def __init__(
        self,
        enable_arxiv: bool = True,
        enable_semantic_scholar: bool = True,
        enable_pubmed: bool = True,
        semantic_scholar_api_key: Optional[str] = None,
    ):
        self.sources = {}
        
        if enable_arxiv:
            self.sources["arxiv"] = ArxivSource()
        
        if enable_semantic_scholar:
            self.sources["semantic_scholar"] = SemanticScholarSource(
                api_key=semantic_scholar_api_key
            )
        
        if enable_pubmed:
            self.sources["pubmed"] = PubMedSource()
    
    def search(
        self,
        query: str,
        max_results_per_source: int = 5,
        sources: Optional[list[str]] = None,
    ) -> list[Paper]:
        """
        Search across multiple sources.
        
        Args:
            query: Search query
            max_results_per_source: Max results from each source
            sources: Specific sources to search (None = all)
        
        Returns:
            Combined list of papers (deduplicated by title similarity)
        """
        all_papers = []
        seen_titles = set()
        
        sources_to_search = sources or list(self.sources.keys())
        
        for source_name in sources_to_search:
            if source_name not in self.sources:
                continue
            
            source = self.sources[source_name]
            
            try:
                papers = source.search(query, max_results=max_results_per_source)
                
                for paper in papers:
                    # Simple deduplication by normalized title
                    normalized_title = re.sub(r'\W+', '', paper.title.lower())
                    if normalized_title not in seen_titles:
                        seen_titles.add(normalized_title)
                        all_papers.append(paper)
                        
            except Exception as e:
                print(f"Error searching {source_name}: {e}")
        
        # Sort by relevance (using citations as proxy)
        all_papers.sort(key=lambda p: p.citations_count, reverse=True)
        
        return all_papers
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a paper by its ID (with source prefix)."""
        if paper_id.startswith("arxiv:"):
            return self.sources.get("arxiv", ArxivSource()).get_paper(paper_id)
        elif paper_id.startswith("s2:"):
            return self.sources.get("semantic_scholar", SemanticScholarSource()).get_paper(paper_id)
        elif paper_id.startswith("pubmed:"):
            return self.sources.get("pubmed", PubMedSource()).get_paper(paper_id)
        
        return None
