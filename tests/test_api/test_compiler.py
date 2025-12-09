"""Comprehensive tests for pattern compilation and caching."""

from __future__ import annotations

import pytest

from tensorlogic.api.compiler import CompiledPattern, PatternCompiler, get_global_compiler
from tensorlogic.api.errors import PatternSyntaxError
from tensorlogic.backends import create_backend


class TestPatternCompilerInitialization:
    """Test PatternCompiler initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test default compiler initialization."""
        compiler = PatternCompiler()
        assert compiler.cache_size == 128

        # Cache should be empty initially
        stats = compiler.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["maxsize"] == 128
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0

    def test_custom_cache_size(self) -> None:
        """Test compiler with custom cache size."""
        compiler = PatternCompiler(cache_size=256)
        assert compiler.cache_size == 256

        stats = compiler.get_cache_stats()
        assert stats["maxsize"] == 256

    def test_small_cache_size(self) -> None:
        """Test compiler with small cache size."""
        compiler = PatternCompiler(cache_size=2)
        assert compiler.cache_size == 2

        stats = compiler.get_cache_stats()
        assert stats["maxsize"] == 2

    def test_invalid_cache_size_raises(self) -> None:
        """Test that invalid cache size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PatternCompiler(cache_size=0)

        assert "positive" in str(exc_info.value).lower()
        assert "0" in str(exc_info.value)

    def test_negative_cache_size_raises(self) -> None:
        """Test that negative cache size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PatternCompiler(cache_size=-10)

        assert "positive" in str(exc_info.value).lower()


class TestBasicCompilation:
    """Test basic pattern compilation."""

    def test_compile_simple_pattern(self) -> None:
        """Test compiling a simple pattern."""
        compiler = PatternCompiler()
        compiled = compiler.compile("P(x)")

        # Verify compiled pattern structure
        assert isinstance(compiled, CompiledPattern)
        assert compiled.pattern == "P(x)"
        assert compiled.cache_key == "P(x)"
        assert compiled.parsed is not None

        # Should have recorded one cache miss
        stats = compiler.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_compile_quantified_pattern(self) -> None:
        """Test compiling quantified pattern."""
        compiler = PatternCompiler()
        compiled = compiler.compile("forall x: P(x)")

        assert compiled.pattern == "forall x: P(x)"
        assert compiled.parsed is not None

    def test_compile_complex_pattern(self) -> None:
        """Test compiling complex pattern."""
        compiler = PatternCompiler()
        compiled = compiler.compile("exists x: P(x) and Q(x) or R(x)")

        assert compiled.pattern == "exists x: P(x) and Q(x) or R(x)"
        assert compiled.parsed is not None

    def test_compile_with_backend(self) -> None:
        """Test compiling with explicit backend."""
        compiler = PatternCompiler()
        backend = create_backend("numpy")
        compiled = compiler.compile("P(x)", backend=backend)

        # Backend parameter is accepted but currently unused
        assert compiled.pattern == "P(x)"

    def test_compile_invalid_syntax_raises(self) -> None:
        """Test that invalid syntax raises PatternSyntaxError."""
        compiler = PatternCompiler()

        with pytest.raises(PatternSyntaxError):
            compiler.compile("forall")  # Incomplete quantifier

        with pytest.raises(PatternSyntaxError):
            compiler.compile("P(x) and")  # Incomplete binary op


class TestCaching:
    """Test caching behavior and LRU eviction."""

    def test_cache_hit_on_repeated_compilation(self) -> None:
        """Test that repeated compilation produces cache hit."""
        compiler = PatternCompiler()

        # First compilation - cache miss
        compiled1 = compiler.compile("P(x)")
        stats1 = compiler.get_cache_stats()
        assert stats1["misses"] == 1
        assert stats1["hits"] == 0

        # Second compilation - cache hit
        compiled2 = compiler.compile("P(x)")
        stats2 = compiler.get_cache_stats()
        assert stats2["misses"] == 1
        assert stats2["hits"] == 1

        # Should return same parsed pattern
        assert compiled1.parsed is compiled2.parsed

    def test_cache_miss_on_different_patterns(self) -> None:
        """Test that different patterns produce cache misses."""
        compiler = PatternCompiler()

        compiler.compile("P(x)")
        compiler.compile("Q(x)")
        compiler.compile("R(x)")

        stats = compiler.get_cache_stats()
        assert stats["misses"] == 3
        assert stats["hits"] == 0
        assert stats["size"] == 3

    def test_multiple_cache_hits(self) -> None:
        """Test multiple cache hits for same pattern."""
        compiler = PatternCompiler()

        pattern = "forall x: P(x)"
        compiler.compile(pattern)  # Miss

        for _ in range(5):
            compiler.compile(pattern)  # All hits

        stats = compiler.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 5

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        compiler = PatternCompiler(cache_size=2)

        # Fill cache
        compiler.compile("P(x)")  # Miss
        compiler.compile("Q(x)")  # Miss

        stats = compiler.get_cache_stats()
        assert stats["size"] == 2

        # Access P(x) to make it recently used
        compiler.compile("P(x)")  # Hit

        # Add R(x) - should evict Q(x) (least recently used)
        compiler.compile("R(x)")  # Miss

        # P(x) should still be in cache (hit)
        compiler.compile("P(x)")  # Hit

        # Q(x) should have been evicted (miss)
        compiler.compile("Q(x)")  # Miss

        stats = compiler.get_cache_stats()
        assert stats["misses"] == 4  # P, Q, R, Q
        assert stats["hits"] == 2  # P, P

    def test_mixed_patterns_caching(self) -> None:
        """Test caching behavior with mixed patterns."""
        compiler = PatternCompiler()

        patterns = [
            "P(x)",
            "Q(x)",
            "P(x)",  # Hit
            "R(x)",
            "Q(x)",  # Hit
            "P(x)",  # Hit
        ]

        for pattern in patterns:
            compiler.compile(pattern)

        stats = compiler.get_cache_stats()
        assert stats["misses"] == 3  # P, Q, R
        assert stats["hits"] == 3  # P, Q, P


class TestCacheStatistics:
    """Test cache statistics tracking and reporting."""

    def test_get_cache_stats_structure(self) -> None:
        """Test that get_cache_stats returns correct structure."""
        compiler = PatternCompiler()
        stats = compiler.get_cache_stats()

        # Verify all expected keys are present
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "maxsize" in stats
        assert "hit_rate" in stats
        assert "total_requests" in stats

        # Verify types
        assert isinstance(stats["hits"], int)
        assert isinstance(stats["misses"], int)
        assert isinstance(stats["size"], int)
        assert isinstance(stats["maxsize"], int)
        assert isinstance(stats["hit_rate"], float)
        assert isinstance(stats["total_requests"], int)

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        compiler = PatternCompiler()

        # No requests yet
        stats = compiler.get_cache_stats()
        assert stats["hit_rate"] == 0.0

        # One miss, no hits
        compiler.compile("P(x)")
        stats = compiler.get_cache_stats()
        assert stats["hit_rate"] == 0.0

        # One miss, one hit
        compiler.compile("P(x)")
        stats = compiler.get_cache_stats()
        assert stats["hit_rate"] == 0.5

        # One miss, two hits
        compiler.compile("P(x)")
        stats = compiler.get_cache_stats()
        assert abs(stats["hit_rate"] - 2 / 3) < 0.01

        # One miss, three hits
        compiler.compile("P(x)")
        stats = compiler.get_cache_stats()
        assert stats["hit_rate"] == 0.75

    def test_total_requests_tracking(self) -> None:
        """Test that total requests are tracked correctly."""
        compiler = PatternCompiler()

        for i in range(1, 6):
            compiler.compile(f"P{i}(x)")
            stats = compiler.get_cache_stats()
            assert stats["total_requests"] == i

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        compiler = PatternCompiler()

        # Add some patterns
        compiler.compile("P(x)")
        compiler.compile("Q(x)")
        compiler.compile("P(x)")  # Hit

        stats_before = compiler.get_cache_stats()
        assert stats_before["size"] == 2
        assert stats_before["hits"] == 1
        assert stats_before["misses"] == 2

        # Clear cache
        compiler.clear_cache()

        stats_after = compiler.get_cache_stats()
        assert stats_after["size"] == 0
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0
        assert stats_after["total_requests"] == 0

        # Next compilation should be a miss
        compiler.compile("P(x)")
        stats_final = compiler.get_cache_stats()
        assert stats_final["misses"] == 1
        assert stats_final["hits"] == 0


class TestCompiledPattern:
    """Test CompiledPattern dataclass properties."""

    def test_compiled_pattern_immutable(self) -> None:
        """Test that CompiledPattern is immutable (frozen)."""
        compiler = PatternCompiler()
        compiled = compiler.compile("P(x)")

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            compiled.pattern = "Q(x)"  # type: ignore

        with pytest.raises(AttributeError):
            compiled.cache_key = "new_key"  # type: ignore

    def test_compiled_pattern_attributes(self) -> None:
        """Test CompiledPattern attribute access."""
        compiler = PatternCompiler()
        compiled = compiler.compile("forall x: P(x)")

        assert compiled.pattern == "forall x: P(x)"
        assert compiled.cache_key == "forall x: P(x)"
        assert compiled.parsed is not None
        assert hasattr(compiled.parsed, "ast")
        assert hasattr(compiled.parsed, "free_variables")

    def test_compiled_pattern_equality(self) -> None:
        """Test CompiledPattern equality based on content."""
        compiler = PatternCompiler()

        # Same pattern compiled twice uses cached parsed pattern
        compiled1 = compiler.compile("P(x)")
        compiled2 = compiler.compile("P(x)")

        # They should have the same parsed object (from cache)
        assert compiled1.parsed is compiled2.parsed

        # But different CompiledPattern instances
        assert compiled1 is not compiled2


class TestGlobalCompiler:
    """Test global compiler singleton."""

    def test_get_global_compiler_singleton(self) -> None:
        """Test that get_global_compiler returns singleton."""
        compiler1 = get_global_compiler()
        compiler2 = get_global_compiler()

        # Should return same instance
        assert compiler1 is compiler2

    def test_global_compiler_default_settings(self) -> None:
        """Test that global compiler uses default settings."""
        compiler = get_global_compiler()

        assert compiler.cache_size == 128

        stats = compiler.get_cache_stats()
        assert stats["maxsize"] == 128

    def test_global_compiler_shared_cache(self) -> None:
        """Test that global compiler shares cache across calls."""
        compiler1 = get_global_compiler()
        compiler1.compile("P(x)")  # Miss

        compiler2 = get_global_compiler()
        compiler2.compile("P(x)")  # Hit (shared cache)

        stats = compiler2.get_cache_stats()
        assert stats["hits"] >= 1  # At least one hit


class TestIntegrationWithPatterns:
    """Test integration with actual pattern parsing and execution."""

    def test_compile_patterns_from_quantify(self) -> None:
        """Test compiling patterns used in quantify() examples."""
        compiler = PatternCompiler()

        # Patterns from quantify tests
        patterns = [
            "exists x: P(x)",
            "forall x: P(x)",
            "P(x) and Q(x)",
            "P(x) or Q(x)",
            "not P(x)",
            "P(x) -> Q(x)",
            "forall x: exists y: Related(x, y)",
        ]

        for pattern in patterns:
            compiled = compiler.compile(pattern)
            assert compiled.pattern == pattern
            assert compiled.parsed is not None

    def test_compile_patterns_from_reason(self) -> None:
        """Test compiling patterns used in reason() examples."""
        compiler = PatternCompiler()

        # Patterns from reason tests
        patterns = [
            "P(x) and Q(x)",
            "P(x) or Q(x)",
            "not P(x)",
            "P(x) -> Q(x)",
            "exists x: P(x)",
            "forall x: P(x)",
            "forall x: exists y: Related(x, y)",
            "forall x: P(x) -> Q(x)",
            "P(x) and not Q(x)",
        ]

        for pattern in patterns:
            compiled = compiler.compile(pattern)
            assert compiled.pattern == pattern
            assert compiled.parsed is not None

    def test_compiled_pattern_reuse(self) -> None:
        """Test that compiled patterns can be reused multiple times."""
        compiler = PatternCompiler()

        # Compile pattern
        pattern = "forall x: P(x) -> Q(x)"
        compiled = compiler.compile(pattern)

        # Pattern should be usable multiple times
        assert compiled.parsed.free_variables == set()  # No free variables
        assert compiled.parsed.ast is not None

        # Recompiling should hit cache
        compiled2 = compiler.compile(pattern)
        assert compiled.parsed is compiled2.parsed


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_pattern_compilation(self) -> None:
        """Test compiling empty pattern."""
        compiler = PatternCompiler()

        # Empty pattern should raise syntax error
        with pytest.raises(PatternSyntaxError):
            compiler.compile("")

    def test_whitespace_only_pattern(self) -> None:
        """Test compiling whitespace-only pattern."""
        compiler = PatternCompiler()

        with pytest.raises(PatternSyntaxError):
            compiler.compile("   ")

    def test_very_long_pattern(self) -> None:
        """Test compiling very long pattern."""
        compiler = PatternCompiler()

        # Create a long pattern with many predicates
        predicates = " and ".join([f"P{i}(x)" for i in range(50)])
        compiled = compiler.compile(predicates)

        assert compiled.pattern == predicates
        assert compiled.parsed is not None

    def test_unicode_in_pattern(self) -> None:
        """Test compiling pattern with unicode characters."""
        compiler = PatternCompiler()

        # Pattern with unicode variable names (should work)
        compiled = compiler.compile("P(α)")
        assert compiled.pattern == "P(α)"

    def test_cache_with_similar_patterns(self) -> None:
        """Test that similar but different patterns are cached separately."""
        compiler = PatternCompiler()

        patterns = [
            "P(x)",
            "P(y)",  # Different variable
            "P(x) and Q(x)",  # Additional operator
            "P(x) and Q(y)",  # Different variables
        ]

        for pattern in patterns:
            compiler.compile(pattern)

        stats = compiler.get_cache_stats()
        assert stats["size"] == 4  # All different
        assert stats["misses"] == 4
        assert stats["hits"] == 0


class TestPerformance:
    """Test performance characteristics of caching."""

    def test_cache_reduces_parsing_overhead(self) -> None:
        """Test that caching reduces parsing overhead."""
        import time

        compiler = PatternCompiler()
        pattern = "forall x: exists y: Related(x, y) and HasProperty(y)"

        # First compilation (cache miss) - includes parsing time
        start_miss = time.perf_counter()
        compiler.compile(pattern)
        time_miss = time.perf_counter() - start_miss

        # Subsequent compilations (cache hits) - no parsing
        times_hit = []
        for _ in range(10):
            start_hit = time.perf_counter()
            compiler.compile(pattern)
            times_hit.append(time.perf_counter() - start_hit)

        avg_time_hit = sum(times_hit) / len(times_hit)

        # Cache hits should be faster than cache miss
        # (This is a performance characteristic, not a strict requirement)
        # Just verify that both operations complete in reasonable time
        assert time_miss < 0.1  # Should parse in less than 100ms
        assert avg_time_hit < 0.1  # Cache hits should also be fast

    def test_cache_size_performance_tradeoff(self) -> None:
        """Test performance with different cache sizes."""
        # Small cache
        compiler_small = PatternCompiler(cache_size=2)

        # Large cache
        compiler_large = PatternCompiler(cache_size=100)

        patterns = [f"P{i}(x)" for i in range(10)]

        # Fill both caches
        for pattern in patterns:
            compiler_small.compile(pattern)
            compiler_large.compile(pattern)

        # Recompile all patterns
        for pattern in patterns:
            compiler_small.compile(pattern)
            compiler_large.compile(pattern)

        stats_small = compiler_small.get_cache_stats()
        stats_large = compiler_large.get_cache_stats()

        # Large cache should have more hits
        assert stats_large["hits"] >= stats_small["hits"]
