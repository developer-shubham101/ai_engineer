"""
Metadata Generator Service

This module provides LLM-based metadata generation for document enrichment.
Uses existing LLM providers to generate summaries, keywords, themes, and entities.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from app.modules.core.metadata_models import SoftMetadata
from app.modules.llm.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


class IMetadataGenerator(ABC):
    """
    Interface for metadata generation services.
    """
    
    @abstractmethod
    async def generate_metadata(
        self, 
        text: str, 
        document_id: str,
        existing_metadata: Optional[Dict[str, Any]] = None
    ) -> SoftMetadata:
        """
        Generate soft metadata for a document.
        
        Args:
            text: The document text content
            document_id: Unique identifier for the document
            existing_metadata: Optional existing metadata for context
            
        Returns:
            SoftMetadata object with LLM-generated fields
        """
        pass


class LLMMetadataGenerator(IMetadataGenerator):
    """
    LLM-based metadata generator using existing LLM providers.
    """
    
    def __init__(self, llm_provider: ILLMProvider):
        """
        Initialize the metadata generator.
        
        Args:
            llm_provider: LLM provider instance (e.g., LocalLLMProvider)
        """
        self.llm_provider = llm_provider
        self.max_text_length = 4000  # Limit text to avoid context overflow
        
    async def generate_metadata(
        self, 
        text: str, 
        document_id: str,
        existing_metadata: Optional[Dict[str, Any]] = None
    ) -> SoftMetadata:
        """
        Generate soft metadata using LLM.
        
        The LLM is prompted to extract:
        - Summary (2-3 sentences)
        - Keywords (5-10 relevant terms)
        - Themes (3-5 main topics)
        - Entities (people, organizations, locations)
        """
        logger.info(f"Generating metadata for document: {document_id}")
        
        start_time = time.time()
        
        try:
            # Truncate text if too long to avoid context overflow
            truncated_text = self._truncate_text(text)
            
            # Build optimized prompt
            prompt = self._build_metadata_prompt(truncated_text, existing_metadata)
            
            # Call LLM with low temperature for consistency
            response = await self.llm_provider.generate(
                prompt=prompt,
                max_tokens=512,  # Enough for metadata but not excessive
                temperature=0.1,  # Low temperature for consistent extraction
            )
            
            # Parse LLM response into structured metadata
            soft_metadata = self._parse_llm_response(
                response.text,
                document_id
            )
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Generated metadata for {document_id} in {processing_time:.2f}ms"
            )
            
            return soft_metadata
            
        except Exception as e:
            logger.error(f"Failed to generate metadata for {document_id}: {e}")
            # Return minimal metadata on failure
            return self._create_fallback_metadata(text, document_id)
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to avoid context overflow.
        Takes first portion and last portion to capture both intro and conclusion.
        """
        if len(text) <= self.max_text_length:
            return text
        
        # Take 70% from start, 30% from end
        start_chars = int(self.max_text_length * 0.7)
        end_chars = int(self.max_text_length * 0.3)
        
        truncated = (
            text[:start_chars] + 
            "\n\n[... content truncated ...]\n\n" + 
            text[-end_chars:]
        )
        
        logger.debug(f"Truncated text from {len(text)} to {len(truncated)} characters")
        return truncated
    
    def _build_metadata_prompt(
        self, 
        text: str, 
        existing_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build token-optimized prompt for metadata extraction.
        """
        prompt = f"""Extract metadata from the following document.

Document:
{text}

Extract the following information in a structured format:

1. SUMMARY: Write a concise 2-3 sentence summary of the main content.

2. KEYWORDS: List 5-10 relevant keywords or key phrases (comma-separated).

3. THEMES: Identify 3-5 main themes or topics (comma-separated).

4. ENTITIES: Extract named entities:
   - PEOPLE: Names of people mentioned
   - ORGANIZATIONS: Companies, departments, or organizations
   - LOCATIONS: Places or locations mentioned

Format your response EXACTLY as follows:
SUMMARY: [your summary here]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
THEMES: [theme1, theme2, theme3, ...]
PEOPLE: [person1, person2, ...]
ORGANIZATIONS: [org1, org2, ...]
LOCATIONS: [location1, location2, ...]

If any category has no relevant entries, write "None".
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str, document_id: str) -> SoftMetadata:
        """
        Parse LLM response into SoftMetadata structure.
        """
        try:
            lines = response_text.strip().split('\n')
            
            summary = ""
            keywords = []
            themes = []
            entities = {"people": [], "organizations": [], "locations": []}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                elif line.startswith("KEYWORDS:"):
                    keywords_str = line.replace("KEYWORDS:", "").strip()
                    if keywords_str.lower() != "none":
                        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                elif line.startswith("THEMES:"):
                    themes_str = line.replace("THEMES:", "").strip()
                    if themes_str.lower() != "none":
                        themes = [t.strip() for t in themes_str.split(',') if t.strip()]
                elif line.startswith("PEOPLE:"):
                    people_str = line.replace("PEOPLE:", "").strip()
                    if people_str.lower() != "none":
                        entities["people"] = [p.strip() for p in people_str.split(',') if p.strip()]
                elif line.startswith("ORGANIZATIONS:"):
                    orgs_str = line.replace("ORGANIZATIONS:", "").strip()
                    if orgs_str.lower() != "none":
                        entities["organizations"] = [o.strip() for o in orgs_str.split(',') if o.strip()]
                elif line.startswith("LOCATIONS:"):
                    locs_str = line.replace("LOCATIONS:", "").strip()
                    if locs_str.lower() != "none":
                        entities["locations"] = [l.strip() for l in locs_str.split(',') if l.strip()]
            
            # Ensure we have at least a summary
            if not summary:
                summary = "Document content extracted."
            
            # Ensure we have at least some keywords
            if not keywords:
                keywords = ["document", "content"]
            
            # Ensure we have at least one theme
            if not themes:
                themes = ["general"]
            
            return SoftMetadata(
                summary=summary,
                keywords=keywords[:10],  # Limit to 10
                themes=themes[:5],  # Limit to 5
                entities=entities,
                llm_model=self.llm_provider.get_model_name(),
                confidence=0.8  # Default confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response for {document_id}: {e}")
            logger.debug(f"Raw response: {response_text}")
            return self._create_fallback_metadata("", document_id)
    
    def _create_fallback_metadata(self, text: str, document_id: str) -> SoftMetadata:
        """
        Create minimal fallback metadata when LLM generation fails.
        """
        # Extract simple keywords from text (first 100 words)
        words = text.split()[:100]
        keywords = list(set([
            w.strip('.,!?;:').lower() 
            for w in words 
            if len(w) > 4 and w.isalpha()
        ]))[:10]
        
        if not keywords:
            keywords = ["document", "content"]
        
        return SoftMetadata(
            summary=f"Document {document_id} - metadata generation failed, using fallback.",
            keywords=keywords,
            themes=["general"],
            entities={},
            llm_model=self.llm_provider.get_model_name(),
            confidence=0.3  # Low confidence for fallback
        )
