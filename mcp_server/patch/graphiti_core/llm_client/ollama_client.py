import json
import logging
from typing import Any, Optional, Type
from pydantic import BaseModel

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts import Message

logger = logging.getLogger(name)

class OllamaGraphitiClient(OpenAIGenericClient):
    """
    Optimized LLM client for Ollama with Graphiti
    
    Features:
    - Simplifies Pydantic schemas for Ollama
    - Parses JSON responses robustly
    - Compatible with all Ollama models (DeepSeek, Qwen, Llama, etc.)
    """

def __init__(self, config: LLMConfig):
    super().__init__(config)
    logger.info(f"🦙 OllamaGraphitiClient initialisé: {config.model}")

def _create_example_from_schema(self, schema: dict) -> dict:
    """
    Creates a data example from a JSON schema

    Args:
        schema: Simplified JSON schema

    Returns:
        Example data conforming to the schema
    """
    if not isinstance(schema, dict):
        return {}

    properties = schema.get('properties', {})
    example = {}

    for key, value_schema in properties.items():
        if not isinstance(value_schema, dict):
            continue

        value_type = value_schema.get('type', 'string')

        if value_type == 'array':
            # Pour les arrays, créer un exemple avec un élément
            items_schema = value_schema.get('items', {})
            if isinstance(items_schema, dict):
                example[key] = [self._create_example_from_schema(items_schema)]
            else:
                example[key] = []
        elif value_type == 'object':
            # Pour les objets, récursion
            example[key] = self._create_example_from_schema(value_schema)
        elif value_type == 'string':
            example[key] = f"example_{key}"
        elif value_type == 'integer':
            example[key] = 0
        elif value_type == 'number':
            example[key] = 0.0
        elif value_type == 'boolean':
            example[key] = False
        else:
            example[key] = None

    return example

def _simplify_schema(self, schema: dict) -> dict:
    """
    Simplifies a Pydantic JSON schema for Ollama

    Transformations:
    - Removes $defs and resolves $ref
    - Flattens nested references
    - Keeps only type, properties, items, required

    Args:
        schema: Pydantic schema with $defs/$ref

    Returns:
        Simplified schema compatible with Ollama
    """
    if not isinstance(schema, dict):
        return schema

    # Extraire les définitions
    defs = schema.get('$defs', {})

    def resolve_ref(obj: Any, defs: dict) -> Any:
        """Résout récursivement les $ref"""
        if isinstance(obj, dict):
            # Si c'est une référence, la résoudre
            if '$ref' in obj:
                ref_path = obj['$ref']
                # Format: "#/$defs/EntityName"
                if ref_path.startswith('#/$defs/'):
                    def_name = ref_path.split('/')[-1]
                    if def_name in defs:
                        # Résoudre récursivement la définition
                        resolved = defs[def_name].copy()
                        return resolve_ref(resolved, defs)
                return obj

            # Résoudre récursivement les valeurs
            return {k: resolve_ref(v, defs) for k, v in obj.items() if k != '$defs'}

        elif isinstance(obj, list):
            return [resolve_ref(item, defs) for item in obj]

        return obj

    # Résoudre toutes les références
    simplified = resolve_ref(schema, defs)

    # Nettoyer les champs non nécessaires
    if isinstance(simplified, dict):
        # Garder uniquement les champs essentiels
        essential_keys = {'type', 'properties', 'items', 'required', 'description', 'title'}
        simplified = {k: v for k, v in simplified.items() if k in essential_keys}

    return simplified

async def generate_response(
    self,
    messages: list[Message],
    response_model: Type[BaseModel] | None = None,
    **kwargs
) -> dict | str:
    """
    Generates an LLM response with simplified schema for Ollama

    Args:
        messages: Conversation messages
        response_model: Pydantic model for structured output
        **kwargs: Additional arguments

    Returns:
        Response parsed according to response_model or str
    """
    # Si un response_model est fourni, simplifier le schéma
    if response_model is not None:
        try:
            # Obtenir le schéma Pydantic
            original_schema = response_model.model_json_schema()

            # Simplifier le schéma
            simplified_schema = self._simplify_schema(original_schema)

            # Créer un prompt clair pour Ollama avec EXEMPLE
            schema_str = json.dumps(simplified_schema, indent=2)

            # Créer un exemple pour clarifier
            example = self._create_example_from_schema(simplified_schema)
            example_str = json.dumps(example, indent=2)

            # Modifier le dernier message pour inclure le schéma simplifié ET un exemple
            if messages:
                last_message = messages[-1]
                enhanced_content = (
                    f"{last_message.content}\n\n"
                    f"CRITICAL: You MUST respond with ACTUAL DATA in JSON format, NOT the schema.\n\n"
                    f"Expected JSON structure:\n"
                    f"```json\n{schema_str}\n```\n\n"
                    f"EXAMPLE of a valid response (replace with real extracted data):\n"
                    f"```json\n{example_str}\n```\n\n"
                    f"STRICT REQUIREMENTS:\n"
                    f"1. Extract actual data from the text above\n"
                    f"2. Return ONLY the JSON data, NOT the schema definition\n"
                    f"3. Do NOT include $defs, $ref, properties, or other schema keywords\n"
                    f"4. Include all required fields with real values\n"
                    f"5. If no data found, use empty arrays []"
                )
                messages[-1] = Message(role=last_message.role, content=enhanced_content)

            logger.debug(f"📋 Schéma envoyé:\n{schema_str}")
            logger.debug(f"📋 Exemple envoyé:\n{example_str}")

        except Exception as e:
            logger.warning(f"⚠️ Erreur simplification schéma: {e}, utilisation standard")

    # ✨ IMPORTANT: Appeler sans response_model pour éviter que le parent réinjecte le schéma
    # On a déjà injecté notre version simplifiée dans le message
    try:
        response = await super().generate_response(messages, response_model=None, **kwargs)

        # Si response_model fourni, valider la réponse
        if response_model is not None and isinstance(response, dict):
            try:
                # Valider avec Pydantic
                validated = response_model(**response)
                logger.debug(f"✅ Réponse Ollama validée avec succès")
                return response
            except Exception as e:
                logger.error(f"❌ Validation Pydantic échouée: {e}")
                logger.error(f"Réponse reçue: {json.dumps(response, indent=2)}")
                # Retourner quand même pour investigation
                return response

        return response

    except json.JSONDecodeError as e:
        logger.error(f"❌ Erreur parsing JSON Ollama: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Erreur génération réponse Ollama: {e}")
        raise

def _extract_json_from_response(self, text: str) -> dict:
    """
    Extract JSON from an Ollama response that may contain markdown
    
    Handles cases:
    - ```json { ... } ```
    - Simple { ... }
    - Text before/after the JSON
    
    Args:
        text: Raw response text

    Returns:
        Extracted JSON dictionary
    """
    # Cas 1: JSON dans un bloc markdown
    if '```json' in text:
        start = text.find('```json') + 7
        end = text.find('```', start)
        if end != -1:
            json_str = text[start:end].strip()
            return json.loads(json_str)

    # Cas 2: JSON dans un bloc code simple
    if '```' in text:
        start = text.find('```') + 3
        end = text.find('```', start)
        if end != -1:
            json_str = text[start:end].strip()
            return json.loads(json_str)

    # Cas 3: JSON direct (trouver première { jusqu'à dernière })
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        json_str = text[first_brace:last_brace + 1]
        return json.loads(json_str)

    # Cas 4: Tout le texte est JSON
    return json.loads(text)
