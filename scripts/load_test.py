#!/usr/bin/env python3
"""
Load Testing Script for NitroAGI NEXUS
"""

import asyncio
import aiohttp
import json
import time
import random
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import argparse
import sys


@dataclass
class TestResult:
    """Test result data"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    success: bool
    error: str = ""


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 10
    test_duration: int = 60  # seconds
    ramp_up_time: int = 10   # seconds
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    auth_token: str = ""


class LoadTester:
    """Load testing client for NEXUS"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.session = None
        self.running = False
        
        # Default endpoints if none provided
        if not self.config.endpoints:
            self.config.endpoints = [
                {"path": "/v1/nexus/status", "method": "GET", "weight": 20},
                {"path": "/v1/reasoning/abstract", "method": "POST", "weight": 15, 
                 "payload": {"input_data": {"data": [1, 2, 3, 4, 5]}}},
                {"path": "/v1/reasoning/solve-math", "method": "POST", "weight": 15,
                 "payload": {"problem": "Solve for x: 2x + 5 = 15"}},
                {"path": "/v1/learning/status", "method": "GET", "weight": 10},
                {"path": "/v1/modules/list", "method": "GET", "weight": 10},
                {"path": "/v1/nexus/execute", "method": "POST", "weight": 20,
                 "payload": {"goal": "Analyze the current system performance"}},
                {"path": "/v1/reasoning/creative-solve", "method": "POST", "weight": 10,
                 "payload": {"problem_description": "How to improve system efficiency?"}}
            ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        headers = {}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def make_request(self, endpoint: Dict[str, Any]) -> TestResult:
        """Make a single HTTP request"""
        start_time = time.time()
        url = f"{self.config.base_url}{endpoint['path']}"
        method = endpoint.get("method", "GET").upper()
        
        try:
            if method == "GET":
                async with self.session.get(url) as response:
                    await response.read()  # Ensure body is read
                    success = response.status < 400
                    
            elif method == "POST":
                payload = endpoint.get("payload", {})
                async with self.session.post(url, json=payload) as response:
                    await response.read()
                    success = response.status < 400
                    
            else:
                # Other methods
                async with self.session.request(method, url) as response:
                    await response.read()
                    success = response.status < 400
            
            response_time = time.time() - start_time
            
            return TestResult(
                endpoint=endpoint['path'],
                method=method,
                status_code=response.status,
                response_time=response_time,
                timestamp=datetime.now(),
                success=success
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint['path'],
                method=method,
                status_code=0,
                response_time=response_time,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
    
    def select_endpoint(self) -> Dict[str, Any]:
        """Select endpoint based on weights"""
        weights = [ep.get("weight", 1) for ep in self.config.endpoints]
        return random.choices(self.config.endpoints, weights=weights)[0]
    
    async def user_simulation(self, user_id: int):
        """Simulate a single user's behavior"""
        print(f"Starting user {user_id}")
        
        while self.running:
            endpoint = self.select_endpoint()
            result = await self.make_request(endpoint)
            self.results.append(result)
            
            # Random think time between requests (0.1 to 2 seconds)
            await asyncio.sleep(random.uniform(0.1, 2.0))
    
    async def ramp_up_users(self):
        """Gradually ramp up users"""
        tasks = []
        
        for i in range(self.config.concurrent_users):
            # Stagger user start times
            await asyncio.sleep(self.config.ramp_up_time / self.config.concurrent_users)
            
            if self.running:
                task = asyncio.create_task(self.user_simulation(i))
                tasks.append(task)
        
        return tasks
    
    async def run_load_test(self):
        """Run the complete load test"""
        print(f"Starting load test with {self.config.concurrent_users} users")
        print(f"Target URL: {self.config.base_url}")
        print(f"Test duration: {self.config.test_duration} seconds")
        print(f"Ramp-up time: {self.config.ramp_up_time} seconds")
        print("-" * 50)
        
        # Start the test
        self.running = True
        start_time = time.time()
        
        # Start ramp-up
        ramp_task = asyncio.create_task(self.ramp_up_users())
        
        # Start results monitoring
        monitor_task = asyncio.create_task(self.monitor_results())
        
        # Wait for test duration
        await asyncio.sleep(self.config.test_duration)
        
        # Stop the test
        self.running = False
        print("\nStopping load test...")
        
        # Cancel tasks
        ramp_task.cancel()
        monitor_task.cancel()
        
        try:
            await ramp_task
        except asyncio.CancelledError:
            pass
        
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Wait a bit for final results
        await asyncio.sleep(2)
        
        total_time = time.time() - start_time
        print(f"\nTest completed in {total_time:.2f} seconds")
        
        return self.generate_report()
    
    async def monitor_results(self):
        """Monitor and display real-time results"""
        last_count = 0
        
        while self.running:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            current_count = len(self.results)
            recent_results = self.results[last_count:current_count]
            
            if recent_results:
                success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
                avg_response_time = statistics.mean([r.response_time for r in recent_results])
                
                print(f"Requests: {current_count}, "
                      f"Success Rate: {success_rate:.2%}, "
                      f"Avg Response: {avg_response_time:.3f}s")
                
                last_count = current_count
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests
        
        # Response time statistics
        response_times = [r.response_time for r in self.results if r.success]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        # Requests per second
        if self.results:
            test_duration = (self.results[-1].timestamp - self.results[0].timestamp).total_seconds()
            rps = total_requests / max(test_duration, 1)
        else:
            rps = 0
        
        # Endpoint statistics
        endpoint_stats = {}
        for result in self.results:
            if result.endpoint not in endpoint_stats:
                endpoint_stats[result.endpoint] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_response_time": 0,
                    "response_times": []
                }
            
            stats = endpoint_stats[result.endpoint]
            stats["total"] += 1
            
            if result.success:
                stats["successful"] += 1
                stats["response_times"].append(result.response_time)
            else:
                stats["failed"] += 1
        
        # Calculate endpoint averages
        for endpoint, stats in endpoint_stats.items():
            if stats["response_times"]:
                stats["avg_response_time"] = statistics.mean(stats["response_times"])
                stats["success_rate"] = stats["successful"] / stats["total"]
            else:
                stats["avg_response_time"] = 0
                stats["success_rate"] = 0
            
            # Remove raw response times to reduce report size
            del stats["response_times"]
        
        # Error analysis
        error_types = {}
        for result in self.results:
            if not result.success:
                error_key = f"HTTP {result.status_code}" if result.status_code > 0 else result.error
                error_types[error_key] = error_types.get(error_key, 0) + 1
        
        return {
            "test_configuration": {
                "concurrent_users": self.config.concurrent_users,
                "test_duration": self.config.test_duration,
                "base_url": self.config.base_url,
                "endpoints_tested": len(self.config.endpoints)
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "requests_per_second": rps
            },
            "response_times": {
                "average": avg_response_time,
                "minimum": min_response_time,
                "maximum": max_response_time,
                "p95": p95_response_time,
                "p99": p99_response_time
            },
            "endpoint_statistics": endpoint_stats,
            "error_analysis": error_types
        }
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*60)
        print("LOAD TEST REPORT")
        print("="*60)
        
        # Test Configuration
        config = report["test_configuration"]
        print(f"Concurrent Users: {config['concurrent_users']}")
        print(f"Test Duration: {config['test_duration']}s")
        print(f"Base URL: {config['base_url']}")
        print(f"Endpoints Tested: {config['endpoints_tested']}")
        
        # Summary
        print("\nSUMMARY:")
        summary = report["summary"]
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Successful: {summary['successful_requests']}")
        print(f"  Failed: {summary['failed_requests']}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Requests/Second: {summary['requests_per_second']:.2f}")
        
        # Response Times
        print("\nRESPONSE TIMES:")
        times = report["response_times"]
        print(f"  Average: {times['average']:.3f}s")
        print(f"  Minimum: {times['minimum']:.3f}s")
        print(f"  Maximum: {times['maximum']:.3f}s")
        print(f"  95th Percentile: {times['p95']:.3f}s")
        print(f"  99th Percentile: {times['p99']:.3f}s")
        
        # Endpoint Statistics
        print("\nENDPOINT PERFORMANCE:")
        for endpoint, stats in report["endpoint_statistics"].items():
            print(f"  {endpoint}:")
            print(f"    Requests: {stats['total']}")
            print(f"    Success Rate: {stats['success_rate']:.2%}")
            print(f"    Avg Response: {stats['avg_response_time']:.3f}s")
        
        # Errors
        if report["error_analysis"]:
            print("\nERROR ANALYSIS:")
            for error, count in report["error_analysis"].items():
                print(f"  {error}: {count}")
        
        print("\n" + "="*60)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load test NitroAGI NEXUS")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration (seconds)")
    parser.add_argument("--rampup", type=int, default=10, help="Ramp-up time (seconds)")
    parser.add_argument("--token", help="Auth token")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = LoadTestConfig(
        base_url=args.url,
        concurrent_users=args.users,
        test_duration=args.duration,
        ramp_up_time=args.rampup,
        auth_token=args.token or ""
    )
    
    # Run load test
    async with LoadTester(config) as tester:
        report = await tester.run_load_test()
        
        # Print report
        tester.print_report(report)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Load test failed: {e}")
        sys.exit(1)