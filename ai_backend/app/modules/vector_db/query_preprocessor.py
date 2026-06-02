"""Query preprocessing for improved retrieval with spell correction and normalization."""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

logger = logging.getLogger(__name__)

# Terms that must never be spell-corrected.
# Matching is case-insensitive — add entries in any case.
PROTECTED_TERMS: frozenset[str] = frozenset({
    # Finance
    "PAT", "EBITDA", "IPO", "ROI", "CAGR", "OPEX", "CAPEX", "EBIT", "NAV", "NPA",
    # Technology
    "GPU", "CPU", "TPU", "API", "SDK", "CLI", "IDE", "VPN", "DNS", "CDN",
    "WiFi", "IoT", "SaaS", "PaaS", "IaaS", "CI", "CD", "R&D",
    # Auth / Security
    "RBAC", "SSO", "MFA", "JWT", "OAuth", "LDAP",
    # HR / Workplace
    "PTO", "WFH", "OOO", "RTO", "KPI",
    # Internal product / project codes
    "TBG", "TADS", "TOS", "TDL",
    # Tools
    "Jira", "Slack", "GitHub", "GitLab",
})

# Lowercase lookup set used at runtime (built once)
_PROTECTED_LOWER: frozenset[str] = frozenset(t.lower() for t in PROTECTED_TERMS)


class QueryType(Enum):
    """Query classification types."""
    FACTUAL = "factual"  # Who, what, when, where
    PROCEDURAL = "procedural"  # How to, steps, process
    POLICY = "policy"  # Rules, regulations, policies
    DEFINITION = "definition"  # What is, define, explain
    COMPARISON = "comparison"  # Difference, compare, vs
    TROUBLESHOOTING = "troubleshooting"  # Error, issue, problem, fix
    GENERAL = "general"  # Default


@dataclass
class ProcessedQuery:
    """Container for processed query variants."""
    original: str
    normalized: str
    corrected: Optional[str] = None
    expanded: Optional[str] = None
    rewritten: Optional[str] = None
    query_type: QueryType = QueryType.GENERAL
    all_variants: List[str] = None
    
    def __post_init__(self):
        """Build list of all unique query variants."""
        variants = [self.original, self.normalized]
        if self.corrected:
            variants.append(self.corrected)
        if self.expanded:
            variants.append(self.expanded)
        if self.rewritten:
            variants.append(self.rewritten)
        # Remove duplicates while preserving order
        self.all_variants = list(dict.fromkeys(variants))


class QueryPreprocessor:
    """Preprocesses queries with normalization, spell correction, and expansion."""
    
    def __init__(self):
        self.spell_checker = None
        if SpellChecker:
            try:
                self.spell_checker = SpellChecker()
                logger.info("Spell checker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize spell checker: {e}")
        else:
            logger.warning("pyspellchecker not installed. Spell correction disabled.")

        # Enhanced synonym/acronym dictionary
        self.expansions = {
            # Time off
            'pto': 'paid time off vacation leave',
            'ooo': 'out of office',
            'wfh': 'work from home remote',
            'rto': 'return to office',
            
            # Technology
            'aws': 'amazon web services cloud',
            'api': 'application programming interface endpoint',
            'rbac': 'role based access control permissions',
            'sso': 'single sign on authentication',
            'mfa': '2fa multi factor authentication',
            'ci cd': 'continuous integration deployment pipeline',
            
            # Business
            'hr': 'human resources personnel',
            'ceo': 'chief executive officer',
            'cto': 'chief technology officer',
            'cfo': 'chief financial officer',
            'kpi': 'key performance indicator metric',
            'roi': 'return on investment',
            'eod': 'end of day',
            'asap': 'as soon as possible urgent',
            
            # Common terms
            'docs': 'documentation documents',
            'info': 'information details',
            'config': 'configuration settings',
            'admin': 'administrator administration',
        }

        # Protect domain words from spell correction
        if self.spell_checker:
            domain_words = set(self.expansions.keys()) | {
                'rbac', 'pto', 'wfh', 'ooo', 'rto', 'sso', 'mfa',
                'onboarding', 'offboarding', 'payroll', 'reimbursement',
            } | _PROTECTED_LOWER
            self.spell_checker.word_frequency.load_words(domain_words)

        # Query classification patterns
        self.query_patterns = {
            QueryType.FACTUAL: [r'\bwho\b', r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', r'\bwhich\b'],
            QueryType.PROCEDURAL: [r'\bhow to\b', r'\bhow do\b', r'\bsteps\b', r'\bprocess\b', r'\bprocedure\b'],
            QueryType.POLICY: [r'\bpolicy\b', r'\bpolicies\b', r'\brule\b', r'\bregulation\b', r'\ballowed\b'],
            QueryType.DEFINITION: [r'\bwhat is\b', r'\bdefine\b', r'\bexplain\b', r'\bmeaning\b'],
            QueryType.COMPARISON: [r'\bdifference\b', r'\bcompare\b', r'\bvs\b', r'\bversus\b', r'\bbetter\b'],
            QueryType.TROUBLESHOOTING: [r'\berror\b', r'\bissue\b', r'\bproblem\b', r'\bfix\b', r'\btroubleshoot\b', r'\bfailed\b'],
        }
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify query type based on patterns.
        
        Helps optimize retrieval strategy and response generation.
        """
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.debug(f"Query classified as: {query_type.value}")
                    return query_type
        
        return QueryType.GENERAL
    
    def is_available(self) -> bool:
        """Check if spell checker is available."""
        return self.spell_checker is not None
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query text.
        
        - Lowercase
        - Remove extra whitespace
        - Remove special characters (keep alphanumeric, spaces, hyphens)
        """
        # Lowercase
        normalized = query.lower()
        
        # Remove special characters except alphanumeric, spaces, hyphens, apostrophes
        normalized = re.sub(r"[^a-z0-9\s\-']", ' ', normalized)
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def remove_repeated_chars(self, query: str) -> Optional[str]:
        """Collapse 3+ repeated characters to 2 (e.g. 'leeeeave' -> 'leeave')."""
        cleaned = re.sub(r'(.)\1{2,}', r'\1\1', query)
        return cleaned if cleaned != query else None

    def split_concatenated_words(self, query: str) -> Optional[str]:
        """Split run-together words using wordninja (e.g. 'leavepolicy' -> 'leave policy')."""
        try:
            import wordninja
        except ImportError:
            return None
        words = query.split()
        result, changed = [], False
        for word in words:
            if len(word) > 8 and word.isalpha():
                split = wordninja.split(word)
                if len(split) > 1:
                    result.extend(split)
                    changed = True
                    continue
            result.append(word)
        return ' '.join(result) if changed else None

    def _looks_broken(self, query: str) -> bool:
        """Return True if >30% of words are unknown — likely broken grammar."""
        if not self.spell_checker or len(query.split()) <= 3:
            return False
        unknown = self.spell_checker.unknown(query.split())
        return len(unknown) / len(query.split()) > 0.3

    def correct_spelling(self, query: str) -> Optional[str]:
        """
        Correct spelling errors in query.
        
        Returns corrected query if changes were made, None otherwise.
        """
        if not self.spell_checker:
            return None
        
        try:
            words = query.split()
            corrected_words = []
            has_corrections = False
            
            for word in words:
                # Skip short words (likely acronyms or valid short terms)
                if len(word) <= 2:
                    corrected_words.append(word)
                    continue

                # Never correct protected terms (case-insensitive)
                if word.lower() in _PROTECTED_LOWER:
                    corrected_words.append(word)
                    continue

                # Split digit-word combos before the digit-skip guard
                digit_split = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', word)
                digit_split = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', digit_split)
                if digit_split != word:
                    corrected_words.extend(digit_split.split())
                    has_corrections = True
                    continue

                # Skip words with hyphens or numbers (likely identifiers)
                if '-' in word or any(c.isdigit() for c in word):
                    corrected_words.append(word)
                    continue
                
                # Get correction
                correction = self.spell_checker.correction(word)
                
                if correction and correction != word:
                    corrected_words.append(correction)
                    has_corrections = True
                    logger.debug(f"Spell correction: '{word}' -> '{correction}'")
                else:
                    corrected_words.append(word)
            
            if has_corrections:
                return ' '.join(corrected_words)
            
            return None
            
        except Exception as e:
            logger.warning(f"Spell correction failed: {e}")
            return None
    
    def expand_query(self, query: str) -> Optional[str]:
        """
        Expand query with synonyms and acronym expansions.
        
        Returns expanded query if expansions found, None otherwise.
        """
        words = query.lower().split()
        expanded_words = []
        has_expansion = False
        
        for word in words:
            expanded_words.append(word)
            if word in self.expansions:
                expanded_words.append(self.expansions[word])
                has_expansion = True
                logger.debug(f"Expanded '{word}' -> '{self.expansions[word]}'")
        
        if has_expansion:
            return ' '.join(expanded_words)
        
        return None
    
    async def rewrite_with_llm(self, query: str, llm_provider=None) -> Optional[str]:
        """
        Rewrite query using LLM for better retrieval.
        
        Optional: Uses LLM to rephrase query for better semantic matching.
        """
        if not llm_provider:
            return None
        
        try:
            prompt = f"""Rewrite this search query to be more clear and specific for document retrieval.
Keep it concise (under 20 words). Only return the rewritten query, nothing else.

Original query: {query}

Rewritten query:"""
            
            response = await llm_provider.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            if response and response.text:
                rewritten = response.text.strip()
                if rewritten and rewritten != query:
                    logger.info(f"LLM rewrite: '{query}' -> '{rewritten}'")
                    return rewritten
            
            return None
            
        except Exception as e:
            logger.warning(f"LLM query rewrite failed: {e}")
            return None
    
    async def process_query(
        self,
        query: str,
        use_spell_correction: bool = True,
        use_expansion: bool = False,
        use_llm_rewrite: bool = False,
        llm_provider=None
    ) -> ProcessedQuery:
        """
        Process query with all available techniques.
        
        Args:
            query: Original user query
            use_spell_correction: Apply spell correction
            use_expansion: Expand acronyms and synonyms
            use_llm_rewrite: Use LLM to rewrite query
            llm_provider: LLM provider for rewriting
        
        Returns:
            ProcessedQuery with all variants and classification
        """
        # Classify query type
        query_type = self.classify_query(query)
        logger.info(f"Query type: {query_type.value}")

        # 1. Collapse repeated chars (leeeeave -> leeave)
        deduped = self.remove_repeated_chars(query)
        working = deduped if deduped else query

        # 2. Split run-together words (leavepolicy -> leave policy)
        segmented = self.split_concatenated_words(working)
        if segmented:
            working = segmented

        # 3. Spell correction on pre-processed query (before normalization strips apostrophes)
        corrected = None
        if use_spell_correction and self.is_available():
            corrected = self.correct_spelling(working)
            if corrected and corrected == working:
                corrected = None  # no real change

        # 4. Normalize (after spell correction so apostrophes are intact during correction)
        normalized = self.normalize_query(corrected if corrected else working)

        # 5. Expansion
        expanded = None
        if use_expansion:
            base_query = corrected if corrected else normalized
            expanded = self.expand_query(base_query)

        # 6. Auto LLM rewrite for broken grammar
        if not use_llm_rewrite and llm_provider and self._looks_broken(working):
            use_llm_rewrite = True
            logger.info("Auto-enabling LLM rewrite: query looks broken")

        # 7. LLM rewrite (optional, slower but more accurate)
        rewritten = None
        if use_llm_rewrite and llm_provider:
            rewritten = await self.rewrite_with_llm(query, llm_provider)
        
        return ProcessedQuery(
            original=query,
            normalized=normalized,
            corrected=corrected,
            expanded=expanded,
            rewritten=rewritten,
            query_type=query_type
        )
