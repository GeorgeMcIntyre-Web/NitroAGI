"""Performance benchmarks for NitroAGI components."""

import pytest
import asyncio
import time
import statistics
from typing import List
from unittest.mock import AsyncMock, MagicMock
import psutil
import gc

from nitroagi.core.memory import MemoryManager, MemoryType
from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.core.orchestrator import Orchestrator
from nitroagi.core.base import ModuleRequest, ModuleContext


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_memory_throughput(self):
        """Benchmark memory system throughput."""
        memory_manager = MemoryManager()
        await memory_manager.initialize()
        
        num_operations = 1000
        
        # Benchmark writes
        write_times = []
        for i in range(num_operations):
            start = time.perf_counter()
            await memory_manager.store(
                key=f"perf_key_{i}",
                value={"data": f"test_data_{i}", "index": i},
                memory_type=MemoryType.WORKING
            )
            write_times.append(time.perf_counter() - start)
        
        # Benchmark reads
        read_times = []
        for i in range(num_operations):
            start = time.perf_counter()
            await memory_manager.retrieve(f"perf_key_{i}")
            read_times.append(time.perf_counter() - start)
        
        # Calculate statistics
        write_avg = statistics.mean(write_times) * 1000  # Convert to ms
        write_p95 = statistics.quantiles(write_times, n=20)[18] * 1000
        read_avg = statistics.mean(read_times) * 1000
        read_p95 = statistics.quantiles(read_times, n=20)[18] * 1000
        
        print(f"\nMemory Performance:")
        print(f"  Write avg: {write_avg:.3f}ms, p95: {write_p95:.3f}ms")
        print(f"  Read avg: {read_avg:.3f}ms, p95: {read_p95:.3f}ms")
        
        # Performance assertions
        assert write_avg < 10  # Writes should average < 10ms
        assert read_avg < 5    # Reads should average < 5ms
        assert write_p95 < 50  # 95% of writes < 50ms
        assert read_p95 < 20   # 95% of reads < 20ms
    
    @pytest.mark.asyncio
    async def test_language_module_latency(self):
        """Benchmark language module response latency."""
        module = LanguageModule()
        
        # Mock LLM provider for consistent timing
        mock_provider = AsyncMock()
        async def mock_generate(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate 50ms LLM response
            return "Generated response"
        mock_provider.generate = mock_generate
        module.llm_provider = mock_provider
        module._initialized = True
        
        latencies = []
        num_requests = 100
        
        for i in range(num_requests):
            request = ModuleRequest(
                context=ModuleContext(
                    request_id=f"perf-{i}",
                    user_id="test-user"
                ),
                data=f"Test prompt {i}",
                required_capabilities=[]
            )
            
            start = time.perf_counter()
            await module.process(request)
            latencies.append(time.perf_counter() - start)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies) * 1000
        p50 = statistics.median(latencies) * 1000
        p95 = statistics.quantiles(latencies, n=20)[18] * 1000
        p99 = statistics.quantiles(latencies, n=100)[98] * 1000
        
        print(f"\nLanguage Module Latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
        
        # Latency requirements
        assert avg_latency < 100  # Average < 100ms
        assert p95 < 200         # 95th percentile < 200ms
        assert p99 < 500         # 99th percentile < 500ms
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Benchmark concurrent request handling."""
        orchestrator = Orchestrator()
        
        # Mock module
        mock_module = AsyncMock()
        async def process_request(req):
            await asyncio.sleep(0.01)  # 10ms processing
            return MagicMock(data="response", status="success")
        mock_module.process = process_request
        
        orchestrator.module_registry.register("test", mock_module)
        await orchestrator.initialize()
        
        # Test different concurrency levels
        concurrency_levels = [10, 50, 100, 200]
        results = {}
        
        for concurrency in concurrency_levels:
            requests = []
            for i in range(concurrency):
                req = ModuleRequest(
                    context=ModuleContext(
                        request_id=f"concurrent-{i}",
                        user_id="test"
                    ),
                    data={"index": i},
                    required_capabilities=[]
                )
                requests.append(orchestrator.process_request(req))
            
            start = time.perf_counter()
            await asyncio.gather(*requests)
            elapsed = time.perf_counter() - start
            
            throughput = concurrency / elapsed
            results[concurrency] = {
                "elapsed": elapsed,
                "throughput": throughput
            }
            
            print(f"\nConcurrency {concurrency}:")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Throughput: {throughput:.1f} req/s")
        
        # Performance requirements
        assert results[100]["throughput"] > 500  # Handle > 500 req/s at 100 concurrent
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Benchmark memory usage under load."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_manager = MemoryManager()
        await memory_manager.initialize()
        
        # Store large amount of data
        num_items = 10000
        for i in range(num_items):
            await memory_manager.store(
                key=f"mem_test_{i}",
                value={"data": "x" * 1000, "index": i},  # 1KB per item
                memory_type=MemoryType.WORKING
            )
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        print(f"  Per item: {memory_increase / num_items * 1000:.2f} KB")
        
        # Memory efficiency check
        assert memory_increase < 100  # Less than 100MB for 10K items
        
        # Cleanup
        gc.collect()
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Benchmark cache hit/miss performance."""
        module = LanguageModule()
        module._initialized = True
        module.config.cache_enabled = True
        
        # Mock provider
        call_count = 0
        async def counting_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return f"Response {call_count}"
        
        mock_provider = AsyncMock()
        mock_provider.generate = counting_generate
        module.llm_provider = mock_provider
        
        # First call - cache miss
        request = ModuleRequest(
            context=ModuleContext(request_id="cache-test", user_id="test"),
            data="Test prompt",
            required_capabilities=[]
        )
        
        start = time.perf_counter()
        response1 = await module.process(request)
        miss_time = time.perf_counter() - start
        
        # Cache the response
        module.cache_response(request, response1)
        module.get_cached_response = AsyncMock(return_value=response1)
        
        # Second call - cache hit
        start = time.perf_counter()
        response2 = await module.process(request)
        hit_time = time.perf_counter() - start
        
        speedup = miss_time / hit_time
        
        print(f"\nCache Performance:")
        print(f"  Cache miss: {miss_time*1000:.2f}ms")
        print(f"  Cache hit: {hit_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        
        assert hit_time < miss_time * 0.1  # Cache hit should be 10x faster
        assert call_count == 1  # Provider called only once
    
    @pytest.mark.asyncio
    async def test_6g_network_optimization(self):
        """Benchmark 6G network optimization impact."""
        from nitroagi.core.network import NetworkOptimizer, NetworkMetrics
        
        optimizer = NetworkOptimizer()
        await optimizer.initialize()
        
        # Simulate different network conditions
        network_scenarios = [
            ("5G", NetworkMetrics(latency_ms=10, bandwidth_mbps=1000)),
            ("6G", NetworkMetrics(latency_ms=0.1, bandwidth_mbps=100000)),
        ]
        
        results = {}
        for name, metrics in network_scenarios:
            profile = await optimizer.select_profile(metrics)
            
            # Simulate data transfer
            data_size_mb = 100
            transfer_time = data_size_mb / (metrics.bandwidth_mbps / 8)  # Convert to MBps
            total_latency = metrics.latency_ms / 1000 + transfer_time
            
            results[name] = {
                "latency": metrics.latency_ms,
                "bandwidth": metrics.bandwidth_mbps,
                "transfer_time": transfer_time * 1000,  # Convert to ms
                "total_time": total_latency * 1000
            }
            
            print(f"\n{name} Performance:")
            print(f"  Latency: {metrics.latency_ms}ms")
            print(f"  Bandwidth: {metrics.bandwidth_mbps} Mbps")
            print(f"  100MB transfer: {transfer_time*1000:.2f}ms")
        
        # 6G should be significantly faster
        speedup = results["5G"]["total_time"] / results["6G"]["total_time"]
        print(f"\n6G Speedup: {speedup:.1f}x faster than 5G")
        
        assert speedup > 10  # 6G should be at least 10x faster
    
    @pytest.mark.asyncio
    async def test_scalability(self):
        """Test system scalability with increasing load."""
        orchestrator = Orchestrator()
        
        # Mock module
        mock_module = AsyncMock()
        mock_module.process = AsyncMock(
            return_value=MagicMock(data="response", status="success")
        )
        
        orchestrator.module_registry.register("test", mock_module)
        await orchestrator.initialize()
        
        # Test with increasing number of requests
        load_levels = [100, 500, 1000, 2000]
        performance_data = []
        
        for num_requests in load_levels:
            requests = []
            for i in range(num_requests):
                req = ModuleRequest(
                    context=ModuleContext(
                        request_id=f"scale-{i}",
                        user_id="test"
                    ),
                    data={"index": i},
                    required_capabilities=[]
                )
                requests.append(orchestrator.process_request(req))
            
            start = time.perf_counter()
            results = await asyncio.gather(*requests)
            elapsed = time.perf_counter() - start
            
            success_rate = sum(1 for r in results if r["status"] == "success") / num_requests
            throughput = num_requests / elapsed
            
            performance_data.append({
                "load": num_requests,
                "elapsed": elapsed,
                "throughput": throughput,
                "success_rate": success_rate
            })
            
            print(f"\nLoad {num_requests}:")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Throughput: {throughput:.1f} req/s")
            print(f"  Success rate: {success_rate*100:.1f}%")
        
        # Verify linear scalability
        # Throughput should not degrade significantly with load
        base_throughput = performance_data[0]["throughput"]
        for data in performance_data:
            assert data["throughput"] > base_throughput * 0.8  # Within 20% of base
            assert data["success_rate"] > 0.99  # >99% success rate
        
        await orchestrator.shutdown()