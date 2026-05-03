"""
Patched version of openai_generic_client.py with robust JSON parsing and structured outputs
Fixes:
- "Extra data" error when XAI Grok-4 returns additional content after JSON
- ExtractedEntities validation errors via json_schema structured outputs
- Schema definition returns (GitHub #912)
"""
import json
import logging
import re
import typing
from copy import deepcopy
from typing import ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.client import LLMClient, get_extraction_language_instruction
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'



def extract_json_from_response(content: str) -> dict[str, typing.Any]:
    """
    Extract and parse the first valid JSON object from response content.

    Handles cases where LLM returns:
    - Pure JSON
    - JSON with leading/trailing text
    - JSON with markdown code blocks
    - Multiple JSON objects (takes first)

    Args:
        content: Raw response content from LLM

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    # Remove markdown code blocks if present
    content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)

    # Try direct parsing first (fastest path)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object boundaries with proper nesting
    brace_count = 0
    start_idx = None

    for i, char in enumerate(content):
        if char == '{':
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                # Found a complete JSON object
                json_str = content[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Continue searching
                    start_idx = None
                    continue

    # Last resort: regex search for JSON-like structure
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # If all else fails, raise original error
    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response. Content preview: {content[:200]}",
        content,
        0
    )


def normalize_entity_name_alias(
    parsed: dict[str, typing.Any], response_model: type[BaseModel] | None
) -> dict[str, typing.Any]:
    """Normalize provider aliases to `name` for ExtractedEntities responses.

    Some generic OpenAI-compatible providers may return `entity_name` or
    `entity` despite schema hints asking for `name`. This keeps ingestion
    resilient by mapping known aliases before downstream Pydantic validation.
    """
    model_name = getattr(response_model, '__name__', None)
    entities = parsed.get('extracted_entities')

    should_normalize = model_name == 'ExtractedEntities'

    if not should_normalize:
        if isinstance(entities, list):
            should_normalize = True
            logger.debug(
                'Entity alias normalization inferred from payload shape (response_model=%s)',
                model_name,
            )
        else:
            return parsed

    if not isinstance(entities, list):
        logger.debug(
            'Entity alias normalization skipped: expected list for extracted_entities, got %s',
            type(entities).__name__,
        )
        return parsed

    normalized_count = 0
    for entity in entities:
        if not isinstance(entity, dict) or 'name' in entity:
            continue

        if 'entity_name' in entity:
            entity['name'] = entity['entity_name']
            normalized_count += 1
        elif 'entity' in entity:
            entity['name'] = entity['entity']
            normalized_count += 1

    if normalized_count > 0:
        logger.debug(
            'Entity alias normalization applied for ExtractedEntities: normalized=%d total=%d',
            normalized_count,
            len(entities),
        )
    else:
        logger.debug(
            'Entity alias normalization invoked for ExtractedEntities but no aliases were found'
        )

    return parsed


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        response_model_name = getattr(response_model, '__name__', None)

        # NEW: Determine response_format based on response_model (Structured Outputs)
        response_format = {'type': 'json_object'}  # Default fallback
        if response_model:
            try:
                # Use json_schema for structured outputs (XAI Grok-4 compatible)
                schema = response_model.model_json_schema()
                json_schema_config = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': f'{response_model.__name__}_response',
                        'strict': True,
                        'schema': schema,
                    },
                }
                response_format = json_schema_config
            except Exception as schema_err:
                logger.warning(
                    'Failed to build json_schema for response_model=%s: %s. Falling back to json_object.',
                    response_model_name,
                    schema_err,
                )
                response_format = {'type': 'json_object'}


        raw_result = None
        parsed_before_normalization = None
        parsed_after_normalization = None

        try:
            # Try with determined response_format
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,  # Use passed max_tokens
                response_format=response_format,
            )
            raw_result = response.choices[0].message.content or ''

            # PATCHED: Use robust JSON extraction
            parsed_before_normalization = extract_json_from_response(raw_result)

            parsed_after_normalization = normalize_entity_name_alias(
                deepcopy(parsed_before_normalization), response_model
            )

            # NEW: Detect schema definition hallucination (GitHub #912 issue)
            if response_model and (
                '$defs' in parsed_after_normalization or '$def' in parsed_after_normalization
            ):
                logger.warning('LLM returned schema definition instead of data - triggering fallback')
                raise ValueError('LLM returned schema definition instead of data')

            return parsed_after_normalization

        except (openai.BadRequestError, json.JSONDecodeError, ValueError) as e:
            # FALLBACK: If json_schema fails, retry with json_object
            if response_model and response_format.get('type') == 'json_schema':
                logger.warning(
                    'json_schema failed (%s: %s); falling back to json_object',
                    e.__class__.__name__,
                    e,
                )
                try:
                    fallback_response = await self.client.chat.completions.create(
                        model=self.model or DEFAULT_MODEL,
                        messages=openai_messages,
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                        response_format={'type': 'json_object'},
                    )
                    fallback_result = fallback_response.choices[0].message.content or ''

                    fallback_parsed_before_normalization = extract_json_from_response(fallback_result)

                    fallback_parsed_after_normalization = normalize_entity_name_alias(
                        deepcopy(fallback_parsed_before_normalization), response_model
                    )
                    return fallback_parsed_after_normalization
                except Exception as fallback_err:
                    logger.error('Fallback to json_object also failed: %s', fallback_err)
                    raise e  # Propagate original error
            else:
                # Non-schema error or already in fallback
                raise

        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error('Error in generating LLM response: %s', e)
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        # CHANGED: Remove schema appending from prompt when using response_model
        # Let the API handle schema enforcement via json_schema response_format
        # Only add basic JSON reminder for non-structured calls
        if response_model is None:
            # For non-structured calls, add basic JSON reminder
            messages[-1].content += '\n\nRespond with a JSON object.'

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages,
                    response_model,
                    max_tokens=max_tokens,
                    model_size=model_size,
                    prompt_name=prompt_name,
                )
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                # Let OpenAI's client handle these retries
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f'The previous response attempt was invalid. '
                    f'Error type: {e.__class__.__name__}. '
                    f'Error details: {str(e)}. '
                    f'Please try again with a valid response, ensuring the output matches '
                    f'the expected format and constraints.'
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                logger.warning(
                    f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')
