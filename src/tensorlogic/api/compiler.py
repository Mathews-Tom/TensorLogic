"""Pattern compilation and caching for performance optimization.

This module implements pattern compilation with LRU caching to reduce parsing
overhead for repeated patterns. Patterns are parsed once and cached for reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from tensorlogic.api.parser import ParsedPattern, PatternParser
from tensorlogic.api.validation import PatternValidator
from tensorlogic.backends import TensorBackend

__all__ = ["CompiledPattern", "PatternCompiler", "get_global_compiler"]


@dataclass(frozen=True)
class CompiledPattern:
    """Compiled pattern with validated AST and metadata.

    A compiled pattern contains the parsed and validated AST along with
    metadata about free variables and predicates. Compiled patterns can
    be executed multiple times with different predicate/binding values.

    Attributes:
        pattern: Original pattern string
        parsed: Parsed pattern with AST
        cache_key: Cache key for this pattern
    """

    pattern: str
    parsed: ParsedPattern
    cache_key: str


class PatternCompiler:
    """Compile patterns to optimized execution plans with LRU caching.

    PatternCompiler caches parsed patterns to avoid re-parsing the same
    pattern string multiple times. This significantly improves performance
    when patterns are reused with different predicates or bindings.

    The compiler uses an LRU (Least Recently Used) cache to store parsed
    patterns. When the cache is full, the least recently used pattern is
    evicted to make room for new patterns.

    Cache Statistics:
        The compiler tracks cache hits and misses for performance monitoring.
        Use get_cache_stats() to retrieve statistics.

    Thread Safety:
        PatternCompiler instances are NOT thread-safe. Use separate instances
        per thread or implement external synchronization.

    Examples:
        >>> compiler = PatternCompiler(cache_size=128)
        >>> compiled = compiler.compile("forall x: P(x)", backend)
        >>> # Second call reuses cached parse result
        >>> compiled2 = compiler.compile("forall x: P(x)", backend)
        >>> stats = compiler.get_cache_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    """

    def __init__(self, cache_size: int = 128) -> None:
        """Initialize compiler with LRU cache.

        Args:
            cache_size: Maximum number of patterns to cache (default: 128)
                Larger cache sizes reduce cache misses but use more memory.

        Raises:
            ValueError: If cache_size is not positive
        """
        if cache_size <= 0:
            raise ValueError(f"Cache size must be positive, got {cache_size}")

        self.cache_size = cache_size
        self.parser = PatternParser()
        self.validator = PatternValidator()

        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0

        # Create cached parse function with specified cache size
        self._cached_parse = lru_cache(maxsize=cache_size)(self._parse_pattern)

    def _parse_pattern(self, pattern: str) -> ParsedPattern:
        """Parse pattern string (internal, cacheable).

        This method is wrapped with @lru_cache in __init__. Do not call directly.

        Args:
            pattern: Pattern string to parse

        Returns:
            Parsed pattern with AST
        """
        return self.parser.parse(pattern)

    def compile(
        self,
        pattern: str,
        backend: TensorBackend | None = None,
    ) -> CompiledPattern:
        """Compile pattern to optimized execution plan.

        Parses and validates the pattern string, caching the result for reuse.
        Subsequent calls with the same pattern string will use the cached
        parse result, avoiding expensive re-parsing.

        Note:
            Pattern validation requires predicates and bindings at execution
            time, so only parsing is cached. Validation happens at execution.

        Args:
            pattern: Pattern string to compile
                Examples: 'forall x: P(x)', 'exists y: P(x, y) and Q(y)'
            backend: Target backend (optional, for backend-specific optimizations)
                Currently unused but reserved for future optimization passes

        Returns:
            Compiled pattern ready for execution

        Raises:
            PatternSyntaxError: If pattern has invalid syntax

        Examples:
            >>> compiler = PatternCompiler()
            >>> compiled = compiler.compile("forall x: P(x)")
            >>> # Use compiled pattern with different predicates
            >>> result1 = execute(compiled, predicates={'P': pred1})
            >>> result2 = execute(compiled, predicates={'P': pred2})
        """
        # Check if pattern is in cache before parsing
        cache_info_before = self._cached_parse.cache_info()

        # Parse pattern (uses cache if available)
        parsed_pattern = self._cached_parse(pattern)

        # Update cache statistics
        cache_info_after = self._cached_parse.cache_info()
        if cache_info_after.hits > cache_info_before.hits:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        # Create cache key (for now just the pattern string)
        cache_key = pattern

        # Return compiled pattern
        return CompiledPattern(
            pattern=pattern,
            parsed=parsed_pattern,
            cache_key=cache_key,
        )

    def clear_cache(self) -> None:
        """Clear all cached patterns.

        Useful for testing or when memory needs to be reclaimed.
        Also resets cache statistics.
        """
        self._cached_parse.cache_clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for performance monitoring.

        Returns:
            Dictionary with cache statistics:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - size: Current cache size
                - maxsize: Maximum cache size
                - hit_rate: Cache hit rate (0.0 to 1.0)
                - total_requests: Total compile requests

        Examples:
            >>> compiler = PatternCompiler()
            >>> compiler.compile("forall x: P(x)")
            >>> compiler.compile("forall x: P(x)")  # Cache hit
            >>> stats = compiler.get_cache_stats()
            >>> assert stats['hits'] == 1
            >>> assert stats['hit_rate'] == 0.5
        """
        cache_info = self._cached_parse.cache_info()

        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": cache_info.currsize,
            "maxsize": cache_info.maxsize,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


# Global compiler instance for convenience
_global_compiler: PatternCompiler | None = None


def get_global_compiler() -> PatternCompiler:
    """Get global pattern compiler instance.

    Returns a singleton PatternCompiler instance shared across the application.
    This provides a convenient way to reuse compiled patterns without managing
    compiler instances manually.

    The global compiler uses default settings (cache_size=128).

    Returns:
        Global PatternCompiler instance

    Examples:
        >>> compiler = get_global_compiler()
        >>> compiled = compiler.compile("forall x: P(x)")
        >>> # Subsequent calls use the same global instance
        >>> compiler2 = get_global_compiler()
        >>> assert compiler is compiler2
    """
    global _global_compiler
    if _global_compiler is None:
        _global_compiler = PatternCompiler()
    return _global_compiler
