"""
Research tools for arXiv and Semantic Scholar API integration.
Provides parallel literature search and information extraction.
"""

import asyncio
from typing import List, Dict, Optional
import arxiv
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.retry import async_retry_with_exponential_backoff

logger = get_logger(__name__)


class ResearchTools:
    """
    Tools for academic literature search across multiple databases.

    Supports:
    - arXiv: Preprint repository for physics, CS, math, etc.
    - Semantic Scholar: AI-powered academic search engine
    """

    def __init__(self):
        """Initialize research API clients."""
        self.arxiv_client = arxiv.Client()

        # Initialize Semantic Scholar (API key is optional)
        if settings.semantic_scholar_api_key:
            self.s2_client = SemanticScholar(api_key=settings.semantic_scholar_api_key)
        else:
            self.s2_client = SemanticScholar()

        logger.info("Research tools initialized")

    def _clean_text(self, text: str) -> str:
        """
        Clean text data from APIs to prevent encoding issues.

        Handles:
        - Null bytes removal
        - UTF-8 encoding issues
        - Excessive whitespace
        - Length truncation to prevent token overflow

        Args:
            text: Raw text from API

        Returns:
            Cleaned text safe for LLM processing
        """
        if not text:
            return ""

        # Remove null bytes (can crash parsers)
        text = text.replace('\x00', '')

        # Fix common encoding issues
        try:
            # Encode to UTF-8 and decode, replacing errors
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            logger.warning(f"Encoding cleanup failed: {e}")
            return ""

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Truncate extremely long text (prevents token overflow)
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"
            logger.debug(f"Truncated text from {len(text)} to {max_length} chars")

        return text

    def _validate_paper_data(self, paper: Dict) -> bool:
        """
        Validate paper has required fields and data quality.

        Args:
            paper: Paper metadata dict

        Returns:
            True if paper passes validation
        """
        required_fields = ['title', 'abstract', 'source']

        # Check required fields exist
        if not all(field in paper for field in required_fields):
            logger.warning(f"Paper missing required fields: {paper.get('title', 'Unknown')}")
            return False

        # Check minimum content quality
        if len(paper['title']) < 10:
            logger.warning("Paper title too short")
            return False

        if len(paper['abstract']) < 50:
            logger.warning("Paper abstract too short")
            return False

        # Check for suspicious patterns (escaped bytes)
        if '\\x' in paper['abstract']:
            logger.warning("Paper contains escaped bytes")
            return False

        return True

    @async_retry_with_exponential_backoff(
        max_retries=3,
        initial_delay=2.0,
        exceptions=(Exception,)
    )
    async def search_arxiv(
        self,
        query: str,
        max_results: int = None
    ) -> List[Dict]:
        """
        Search arXiv API for papers.

        Args:
            query: Search query string
            max_results: Maximum number of results (default from settings)

        Returns:
            List of paper metadata dicts
        """
        max_results = max_results or settings.arxiv_max_results

        logger.info(f"Searching arXiv for: '{query}' (max {max_results} results)")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        try:
            for paper in self.arxiv_client.results(search):
                results.append({
                    'source': 'arxiv',
                    'title': self._clean_text(paper.title),
                    'authors': [self._clean_text(author.name) for author in paper.authors],
                    'abstract': self._clean_text(paper.summary),
                    'published': paper.published.isoformat() if paper.published else None,
                    'updated': paper.updated.isoformat() if paper.updated else None,
                    'arxiv_id': paper.entry_id,
                    'pdf_url': paper.pdf_url,
                    'categories': paper.categories,
                    'primary_category': paper.primary_category,
                    'comment': self._clean_text(paper.comment) if paper.comment else None,
                    'journal_ref': self._clean_text(paper.journal_ref) if paper.journal_ref else None,
                    'doi': paper.doi
                })
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            raise

        logger.info(f"Found {len(results)} papers on arXiv")
        return results

    @async_retry_with_exponential_backoff(
        max_retries=3,
        initial_delay=2.0,
        exceptions=(Exception,)
    )
    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search Semantic Scholar API for papers with timeout.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of paper metadata dicts
        """
        logger.info(f"Searching Semantic Scholar for: '{query}' (max {limit} results)")

        try:
            # Wrap in async timeout to prevent hanging (15 second timeout)
            async def do_search():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.s2_client.search_paper(
                        query,
                        limit=limit,
                        fields=['title', 'abstract', 'authors', 'year', 'citationCount',
                               'url', 'venue', 'publicationTypes', 'publicationDate',
                               'externalIds', 'influentialCitationCount']
                    )
                )

            # Add timeout of 5 seconds (fast-fail for slow API)
            results = await asyncio.wait_for(do_search(), timeout=5.0)

            papers = []
            for paper in results:
                papers.append({
                    'source': 'semantic_scholar',
                    'title': self._clean_text(paper.title) if paper.title else 'No title',
                    'authors': [self._clean_text(a.name) for a in paper.authors] if paper.authors else [],
                    'abstract': self._clean_text(paper.abstract) if paper.abstract else 'No abstract available',
                    'year': paper.year,
                    'citations': paper.citationCount if paper.citationCount else 0,
                    'influential_citations': paper.influentialCitationCount if hasattr(paper, 'influentialCitationCount') else 0,
                    'url': paper.url if paper.url else None,
                    's2_id': paper.paperId,
                    'venue': self._clean_text(paper.venue) if paper.venue else 'Unknown',
                    'publication_types': paper.publicationTypes if paper.publicationTypes else [],
                    'publication_date': paper.publicationDate if hasattr(paper, 'publicationDate') else None,
                    'external_ids': paper.externalIds if hasattr(paper, 'externalIds') else {}
                })

            logger.info(f"Found {len(papers)} papers on Semantic Scholar")
            return papers

        except asyncio.TimeoutError:
            logger.warning(f"Semantic Scholar search timed out for query: '{query}' - skipping")
            return []  # Return empty list on timeout
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []  # Return empty list on error instead of crashing

    async def parallel_search(
        self,
        queries: List[str],
        max_results_per_query: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Execute multiple searches in parallel across both databases.

        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query

        Returns:
            Dict with 'arxiv' and 'semantic_scholar' keys containing results
        """
        logger.info(f"Starting parallel search for {len(queries)} queries (arXiv only - S2 disabled due to performance)")

        # Create tasks for parallel execution (arXiv only)
        arxiv_tasks = [
            self.search_arxiv(q, max_results=max_results_per_query)
            for q in queries
        ]

        # DISABLED: Semantic Scholar is too slow/unreliable
        # s2_tasks = [
        #     self.search_semantic_scholar(q, limit=max_results_per_query)
        #     for q in queries
        # ]

        # Execute arXiv searches in parallel
        arxiv_results = await asyncio.gather(*arxiv_tasks, return_exceptions=True)

        # No S2 results
        s2_results = [[] for _ in queries]

        # Combine results by query
        combined = {
            'arxiv': {},
            'semantic_scholar': {}
        }

        for i, query in enumerate(queries):
            # Handle arXiv results (could be exception)
            if isinstance(arxiv_results[i], list):
                combined['arxiv'][query] = arxiv_results[i]
            else:
                logger.warning(f"arXiv search failed for '{query}': {arxiv_results[i]}")
                combined['arxiv'][query] = []

            # Handle Semantic Scholar results
            if isinstance(s2_results[i], list):
                combined['semantic_scholar'][query] = s2_results[i]
            else:
                logger.warning(f"S2 search failed for '{query}': {s2_results[i]}")
                combined['semantic_scholar'][query] = []

        total_papers = sum(len(papers) for papers in combined['arxiv'].values())
        total_papers += sum(len(papers) for papers in combined['semantic_scholar'].values())

        logger.info(f"Parallel search complete: {total_papers} total papers found")
        return combined

    def extract_key_findings(
        self,
        papers: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Extract and rank key findings from papers.

        Args:
            papers: List of paper metadata dicts
            top_k: Number of top papers to process

        Returns:
            List of key findings with metadata
        """
        findings = []

        # Sort papers by relevance/citations
        sorted_papers = sorted(
            papers,
            key=lambda p: (
                p.get('citations', 0) +
                p.get('influential_citations', 0) * 2  # Weight influential citations more
            ),
            reverse=True
        )[:top_k]

        for paper in sorted_papers:
            finding = {
                'title': paper['title'],
                'authors': paper['authors'][:3],  # First 3 authors
                'abstract': paper['abstract'],
                'source': paper['source'],
                'relevance_score': self._calculate_relevance(paper),
                'year': paper.get('year') or paper.get('published', '')[:4],
                'citations': paper.get('citations', 0),
                'url': paper.get('url') or paper.get('pdf_url')
            }
            findings.append(finding)

        logger.info(f"Extracted {len(findings)} key findings from {len(papers)} papers")
        return findings

    def _calculate_relevance(self, paper: Dict) -> float:
        """
        Calculate relevance score for a paper.

        Factors:
        - Citation count
        - Influential citations
        - Recency
        - Abstract quality

        Args:
            paper: Paper metadata dict

        Returns:
            Relevance score (0-10)
        """
        score = 0.0

        # Citation score (0-5 points)
        citations = paper.get('citations', 0)
        if citations > 100:
            score += 5.0
        elif citations > 50:
            score += 4.0
        elif citations > 10:
            score += 3.0
        elif citations > 0:
            score += 2.0

        # Recency score (0-3 points)
        year = paper.get('year')
        if year:
            try:
                year = int(year) if isinstance(year, int) else int(str(year)[:4])
                if year >= 2023:
                    score += 3.0
                elif year >= 2020:
                    score += 2.0
                elif year >= 2015:
                    score += 1.0
            except (ValueError, TypeError):
                pass

        # Abstract quality score (0-2 points)
        abstract = paper.get('abstract', '')
        if abstract and len(abstract) > 200:
            score += 2.0
        elif abstract and len(abstract) > 50:
            score += 1.0

        return min(score, 10.0)

    def deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Remove duplicate papers and validate data quality.

        Args:
            papers: List of paper metadata dicts

        Returns:
            Deduplicated list of validated papers
        """
        # First validate all papers
        valid_papers = [p for p in papers if self._validate_paper_data(p)]

        logger.info(f"Validated {len(valid_papers)}/{len(papers)} papers")

        # Then deduplicate
        seen_titles = set()
        unique_papers = []

        for paper in valid_papers:
            title_normalized = paper['title'].lower().strip()
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_papers.append(paper)

        logger.info(f"Deduplicated to {len(unique_papers)} unique valid papers")
        return unique_papers
