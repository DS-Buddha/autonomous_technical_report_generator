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
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'published': paper.published.isoformat() if paper.published else None,
                    'updated': paper.updated.isoformat() if paper.updated else None,
                    'arxiv_id': paper.entry_id,
                    'pdf_url': paper.pdf_url,
                    'categories': paper.categories,
                    'primary_category': paper.primary_category,
                    'comment': paper.comment,
                    'journal_ref': paper.journal_ref,
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
        Search Semantic Scholar API for papers.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of paper metadata dicts
        """
        logger.info(f"Searching Semantic Scholar for: '{query}' (max {limit} results)")

        try:
            results = self.s2_client.search_paper(
                query,
                limit=limit,
                fields=['title', 'abstract', 'authors', 'year', 'citationCount',
                       'url', 'venue', 'publicationTypes', 'publicationDate',
                       'externalIds', 'influentialCitationCount']
            )

            papers = []
            for paper in results:
                papers.append({
                    'source': 'semantic_scholar',
                    'title': paper.title if paper.title else 'No title',
                    'authors': [a.name for a in paper.authors] if paper.authors else [],
                    'abstract': paper.abstract if paper.abstract else 'No abstract available',
                    'year': paper.year,
                    'citations': paper.citationCount if paper.citationCount else 0,
                    'influential_citations': paper.influentialCitationCount if hasattr(paper, 'influentialCitationCount') else 0,
                    'url': paper.url if paper.url else None,
                    's2_id': paper.paperId,
                    'venue': paper.venue if paper.venue else 'Unknown',
                    'publication_types': paper.publicationTypes if paper.publicationTypes else [],
                    'publication_date': paper.publicationDate if hasattr(paper, 'publicationDate') else None,
                    'external_ids': paper.externalIds if hasattr(paper, 'externalIds') else {}
                })

            logger.info(f"Found {len(papers)} papers on Semantic Scholar")
            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            raise

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
        logger.info(f"Starting parallel search for {len(queries)} queries")

        # Create tasks for parallel execution
        arxiv_tasks = [
            self.search_arxiv(q, max_results=max_results_per_query)
            for q in queries
        ]
        s2_tasks = [
            self.search_semantic_scholar(q, limit=max_results_per_query)
            for q in queries
        ]

        # Execute in parallel
        all_tasks = arxiv_tasks + s2_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Separate results
        arxiv_results = results[:len(queries)]
        s2_results = results[len(queries):]

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
        Remove duplicate papers based on title similarity.

        Args:
            papers: List of paper metadata dicts

        Returns:
            Deduplicated list of papers
        """
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            title_normalized = paper['title'].lower().strip()
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_papers.append(paper)

        logger.info(f"Deduplicated {len(papers)} papers to {len(unique_papers)} unique")
        return unique_papers
