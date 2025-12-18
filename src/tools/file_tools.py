"""
File operation tools for markdown generation and management.
"""

from pathlib import Path
from typing import Optional, List
from datetime import datetime

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileTools:
    """
    Tools for file operations, especially markdown generation.
    """

    @staticmethod
    def save_markdown_report(
        content: str,
        topic: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save markdown content to a file.

        Args:
            content: Markdown content
            topic: Report topic (used for filename)
            output_dir: Output directory (default from settings)

        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path(settings.reports_output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from topic
        filename = FileTools._sanitize_filename(topic) + '.md'
        filepath = output_dir / filename

        # Write content
        try:
            filepath.write_text(content, encoding='utf-8')
            logger.info(f"Saved report to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = 100) -> str:
        """
        Convert a string to a safe filename.

        Args:
            name: Input string
            max_length: Maximum filename length

        Returns:
            Sanitized filename
        """
        # Replace spaces and special characters
        safe = name.lower()
        safe = safe.replace(' ', '_')
        safe = ''.join(c for c in safe if c.isalnum() or c in ('_', '-'))

        # Truncate if too long
        if len(safe) > max_length:
            safe = safe[:max_length]

        return safe or 'report'

    @staticmethod
    def format_markdown_section(
        title: str,
        content: str,
        level: int = 2
    ) -> str:
        """
        Format a markdown section with proper heading.

        Args:
            title: Section title
            content: Section content
            level: Heading level (1-6)

        Returns:
            Formatted markdown section
        """
        heading = '#' * level
        return f"{heading} {title}\n\n{content}\n\n"

    @staticmethod
    def format_code_block(
        code: str,
        language: str = 'python',
        caption: Optional[str] = None
    ) -> str:
        """
        Format a code block with syntax highlighting.

        Args:
            code: Code content
            language: Programming language
            caption: Optional caption/description

        Returns:
            Formatted markdown code block
        """
        block = f"```{language}\n{code}\n```"

        if caption:
            block = f"**{caption}**\n\n{block}"

        return block + "\n\n"

    @staticmethod
    def format_reference_list(papers: List[dict]) -> str:
        """
        Format a list of papers as markdown references.

        Args:
            papers: List of paper metadata dicts

        Returns:
            Formatted references section
        """
        references = ["## References\n"]

        for i, paper in enumerate(papers, 1):
            # Format authors
            authors = paper.get('authors', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            elif authors:
                author_str = ', '.join(authors)
            else:
                author_str = "Unknown"

            # Format year
            year = paper.get('year') or paper.get('published', '')[:4] or 'n.d.'

            # Format title
            title = paper.get('title', 'Untitled')

            # Format venue/source
            venue = paper.get('venue') or paper.get('journal_ref') or 'Preprint'

            # Format URL
            url = paper.get('url') or paper.get('pdf_url')

            # Build reference
            ref = f"[{i}] {author_str} ({year}). {title}. {venue}."

            if url:
                ref += f" Available at: {url}"

            references.append(ref)

        return '\n\n'.join(references) + '\n'

    @staticmethod
    def create_table_of_contents(sections: List[str]) -> str:
        """
        Create a table of contents from section titles.

        Args:
            sections: List of section titles

        Returns:
            Formatted table of contents
        """
        toc = ["## Table of Contents\n"]

        for i, section in enumerate(sections, 1):
            # Create anchor link
            anchor = section.lower().replace(' ', '-')
            anchor = ''.join(c for c in anchor if c.isalnum() or c == '-')

            toc.append(f"{i}. [{section}](#{anchor})")

        return '\n'.join(toc) + '\n\n'

    @staticmethod
    def add_metadata_header(
        topic: str,
        generated_date: Optional[datetime] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Create a metadata header for the report.

        Args:
            topic: Report topic
            generated_date: Generation timestamp
            metadata: Additional metadata

        Returns:
            Formatted metadata header
        """
        if generated_date is None:
            generated_date = datetime.now()

        header = [
            "---",
            f"title: \"{topic}\"",
            f"generated: {generated_date.strftime('%Y-%m-%d %H:%M:%S')}",
            "generator: Hybrid Agentic System"
        ]

        if metadata:
            for key, value in metadata.items():
                header.append(f"{key}: {value}")

        header.append("---\n")

        return '\n'.join(header) + '\n\n'

    @staticmethod
    def load_markdown_template(template_name: str) -> Optional[str]:
        """
        Load a markdown template file.

        Args:
            template_name: Template filename

        Returns:
            Template content or None
        """
        template_dir = Path(__file__).parent.parent / 'templates'
        template_path = template_dir / template_name

        if template_path.exists():
            try:
                return template_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to load template {template_name}: {e}")
                return None
        else:
            logger.warning(f"Template {template_name} not found")
            return None

    @staticmethod
    def ensure_directory_exists(path: Path) -> None:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
