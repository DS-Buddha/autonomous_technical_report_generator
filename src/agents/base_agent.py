"""
Base agent class with Google GenAI integration.
Provides common functionality for all specialized agents.
"""

from typing import Dict, Any, Optional, List
from google import genai
from google.genai import types

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.retry import retry_with_exponential_backoff

logger = get_logger(__name__)


class BaseAgent:
    """
    Base class for all agents in the hybrid agentic system.

    Provides:
    - Google GenAI client initialization
    - Common prompt management
    - Response generation with retry logic
    - Structured output parsing
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        model_tier: str = "standard",
        **kwargs
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent name
            system_prompt: System instruction for the agent
            model: Model name (optional, overrides tier selection)
            temperature: Sampling temperature (0.0-1.0)
            model_tier: Model tier for cost optimization
                - "fast": Cheap, fast models for simple tasks (Flash)
                - "standard": Balanced models for most tasks (Pro)
                - "advanced": Best models for complex reasoning (Pro with high temp)
            **kwargs: Additional configuration
        """
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_tier = model_tier

        # Select model based on tier if not explicitly provided
        if model:
            self.model = model
        else:
            self.model = self._select_model_by_tier(model_tier)

        # Initialize Google GenAI client
        self.client = genai.Client(api_key=settings.google_api_key)

        logger.info(
            f"Initialized {self.name} agent with model {self.model} "
            f"(tier: {model_tier})"
        )

    def _select_model_by_tier(self, tier: str) -> str:
        """
        Select appropriate model based on complexity tier.

        Model Costs (approximate per 1M tokens):
        - Flash: ~$0.075 (input) / $0.30 (output) - 80% cheaper
        - Pro:   ~$1.25 (input) / $5.00 (output) - Standard

        Args:
            tier: Complexity tier ("fast", "standard", "advanced")

        Returns:
            Model name
        """
        tier_models = {
            "fast": "gemini-1.5-flash",      # Cheap, fast for simple tasks
            "standard": "gemini-1.5-pro",    # Balanced for most tasks
            "advanced": "gemini-1.5-pro"     # Best available (same as standard for now)
        }

        model = tier_models.get(tier, tier_models["standard"])
        logger.debug(f"Selected {model} for tier '{tier}'")
        return model

    @retry_with_exponential_backoff(
        max_retries=3,
        initial_delay=1.0,
        exceptions=(Exception,)
    )
    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using Google GenAI.

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction override
            temperature: Optional temperature override
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        system = system_instruction or self.system_prompt
        temp = temperature if temperature is not None else self.temperature

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temp,
                    **kwargs
                )
            )

            if response.text:
                logger.debug(f"{self.name} generated response ({len(response.text)} chars)")
                return response.text
            else:
                logger.warning(f"{self.name} received empty response")
                return ""

        except Exception as e:
            logger.error(f"{self.name} generation error: {e}")
            raise

    def generate_json_response(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a JSON-structured response.

        Args:
            prompt: User prompt requesting JSON output
            schema: Optional JSON schema for validation
            **kwargs: Additional parameters

        Returns:
            Parsed JSON dict
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no markdown formatting."

        response_text = self.generate_response(json_prompt, **kwargs)

        # Try to extract JSON from response
        import json
        import re

        # Remove markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        # Clean up common issues
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)
            logger.debug(f"{self.name} successfully parsed JSON response")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"{self.name} JSON parsing error: {e}")
            logger.debug(f"Response text: {response_text[:200]}...")

            # Return error structure
            return {
                'error': 'Failed to parse JSON response',
                'raw_response': response_text[:500]
            }

    def run(
        self,
        task: str,
        context: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Run the agent on a task. To be overridden by subclasses.

        Args:
            task: Task description
            context: Optional context dict
            **kwargs: Additional parameters

        Returns:
            Task result (type depends on agent)
        """
        raise NotImplementedError(f"{self.name} must implement run() method")

    def format_context(self, context: Dict) -> str:
        """
        Format context dict as a readable string for prompts.

        Args:
            context: Context dictionary

        Returns:
            Formatted context string
        """
        if not context:
            return ""

        lines = ["Context:"]
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                lines.append(f"- {key}: {len(value)} items")
            else:
                lines.append(f"- {key}: {value}")

        return '\n'.join(lines)

    def __str__(self) -> str:
        return f"{self.name} (model: {self.model})"

    def __repr__(self) -> str:
        return f"BaseAgent(name='{self.name}', model='{self.model}')"
