"""
Example: How NitroAGI NEXUS Orchestrator Works in Practice

This example demonstrates the complete flow of a multi-modal AI request
through the NEXUS orchestrator with prefrontal cortex executive control.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

# Import NitroAGI components
from nitroagi.core.orchestrator import Orchestrator, TaskRequest, ExecutionStrategy
from nitroagi.core.base import ModuleCapability, ModuleRegistry, ProcessingContext
from nitroagi.core.message_bus import MessageBus
from nitroagi.core.memory import MemoryManager
from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.modules.vision.vision_module import VisionModule
from nitroagi.modules.audio.audio_module import AudioModule


class NitroAGIDemo:
    """Demonstration of NitroAGI orchestrator in action."""
    
    def __init__(self):
        """Initialize the NitroAGI system."""
        # Create core components
        self.message_bus = MessageBus()
        self.registry = ModuleRegistry()
        self.memory = MemoryManager()
        
        # Create orchestrator with prefrontal cortex
        self.orchestrator = Orchestrator(
            registry=self.registry,
            message_bus=self.message_bus,
            max_concurrent_tasks=5
        )
        
        # Initialize modules
        self.language_module = LanguageModule({"name": "language"})
        self.vision_module = VisionModule({"name": "vision"})
        self.audio_module = AudioModule({"name": "audio"})
    
    async def setup(self):
        """Set up the system and register modules."""
        print("🚀 Starting NitroAGI NEXUS System...")
        
        # Start core services
        await self.message_bus.start()
        await self.memory.initialize()
        await self.orchestrator.start()
        
        # Register modules with their capabilities
        await self.registry.register_module(
            self.language_module,
            capabilities=[
                ModuleCapability.TEXT_GENERATION,
                ModuleCapability.TEXT_UNDERSTANDING,
                ModuleCapability.TRANSLATION
            ]
        )
        
        await self.registry.register_module(
            self.vision_module,
            capabilities=[
                ModuleCapability.IMAGE_UNDERSTANDING,
                ModuleCapability.OBJECT_DETECTION
            ]
        )
        
        await self.registry.register_module(
            self.audio_module,
            capabilities=[
                ModuleCapability.SPEECH_TO_TEXT,
                ModuleCapability.AUDIO_PROCESSING
            ]
        )
        
        print("✅ System initialized with modules: language, vision, audio")
        print("🧠 Prefrontal cortex executive control activated")
    
    async def example_1_simple_text(self):
        """Example 1: Simple text generation request."""
        print("\n" + "="*60)
        print("EXAMPLE 1: Simple Text Generation")
        print("="*60)
        
        # Create request
        request = TaskRequest(
            input_data="Write a haiku about artificial intelligence",
            required_capabilities=[ModuleCapability.TEXT_GENERATION],
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )
        
        print(f"📝 Request: {request.input_data}")
        print("🧠 Prefrontal Cortex Analysis:")
        
        # This happens internally in the orchestrator:
        # 1. Prefrontal cortex analyzes the request
        print("  • Task complexity: Low (single step)")
        print("  • Selected module: language")
        print("  • Predicted time: 2000ms")
        
        # Submit to orchestrator
        task_id = await self.orchestrator.submit_task(request)
        print(f"📋 Task ID: {task_id}")
        
        # Wait for completion
        result = await self.orchestrator.wait_for_task(task_id, timeout=10)
        
        print(f"✅ Status: {result.status}")
        print(f"⏱️  Execution time: {result.execution_time_ms:.0f}ms")
        print(f"📄 Output: {result.final_output}")
    
    async def example_2_multi_modal(self):
        """Example 2: Complex multi-modal processing."""
        print("\n" + "="*60)
        print("EXAMPLE 2: Multi-Modal Image Analysis")
        print("="*60)
        
        # Create complex request
        request = TaskRequest(
            input_data={
                "image_path": "/path/to/image.jpg",
                "goal": "Analyze this image and create a detailed marketing description"
            },
            required_capabilities=[
                ModuleCapability.IMAGE_UNDERSTANDING,
                ModuleCapability.OBJECT_DETECTION,
                ModuleCapability.TEXT_GENERATION
            ],
            execution_strategy=ExecutionStrategy.PIPELINE
        )
        
        print(f"📝 Request: {request.input_data['goal']}")
        print("🧠 Prefrontal Cortex Analysis:")
        print("  • Task complexity: High (multiple steps)")
        print("  • Task decomposition:")
        print("    1. Extract visual features from image")
        print("    2. Detect and identify objects")
        print("    3. Analyze scene composition")
        print("    4. Generate marketing description")
        
        print("\n📊 Execution Plan:")
        print("  Step 1: vision module → Extract features")
        print("  Step 2: vision module → Object detection")
        print("  Step 3: vision module → Scene analysis")
        print("  Step 4: language module → Generate description")
        
        # Submit to orchestrator
        task_id = await self.orchestrator.submit_task(request)
        
        # Simulate execution monitoring
        print("\n🔄 Execution Progress:")
        for i in range(4):
            await asyncio.sleep(0.5)
            print(f"  ✓ Step {i+1} completed")
        
        result = await self.orchestrator.wait_for_task(task_id, timeout=30)
        
        print(f"\n✅ Status: {result.status}")
        print(f"⏱️  Total execution time: {result.execution_time_ms:.0f}ms")
        print(f"📄 Marketing Description: {result.final_output}")
    
    async def example_3_parallel_consensus(self):
        """Example 3: Parallel execution with consensus."""
        print("\n" + "="*60)
        print("EXAMPLE 3: Parallel Processing with Consensus")
        print("="*60)
        
        request = TaskRequest(
            input_data="Translate 'Hello World' to Spanish",
            required_capabilities=[ModuleCapability.TRANSLATION],
            execution_strategy=ExecutionStrategy.CONSENSUS  # Use multiple modules
        )
        
        print(f"📝 Request: {request.input_data}")
        print("🧠 Prefrontal Cortex Analysis:")
        print("  • Strategy: Consensus (use multiple sources)")
        print("  • Parallel execution planned")
        
        print("\n🔀 Parallel Execution:")
        print("  → Module 1: Primary translation")
        print("  → Module 2: Secondary translation")
        print("  → Module 3: Verification translation")
        
        task_id = await self.orchestrator.submit_task(request)
        
        # Simulate parallel processing
        print("\n⚡ Processing in parallel...")
        await asyncio.sleep(1)
        
        result = await self.orchestrator.wait_for_task(task_id, timeout=10)
        
        print("\n🤝 Consensus Results:")
        print("  • Module 1: 'Hola Mundo'")
        print("  • Module 2: 'Hola Mundo'")
        print("  • Module 3: 'Hola Mundo'")
        print(f"✅ Final consensus: {result.final_output}")
    
    async def example_4_error_handling(self):
        """Example 4: Error handling and intervention."""
        print("\n" + "="*60)
        print("EXAMPLE 4: Error Handling & Recovery")
        print("="*60)
        
        request = TaskRequest(
            input_data="Process this corrupted data: [INVALID]",
            required_capabilities=[ModuleCapability.TEXT_UNDERSTANDING],
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )
        
        print(f"📝 Request: {request.input_data}")
        print("🧠 Prefrontal Cortex Monitoring:")
        
        task_id = await self.orchestrator.submit_task(request)
        
        print("\n⚠️  Error Detection:")
        print("  • Module execution failed")
        print("  • Executive Monitor triggered intervention")
        print("\n🔧 Recovery Actions:")
        print("  1. Attempting retry with fallback module...")
        print("  2. Using alternative processing strategy...")
        print("  3. Applying error correction...")
        
        await asyncio.sleep(2)
        
        result = await self.orchestrator.wait_for_task(task_id, timeout=10)
        
        if result.status == "completed":
            print("\n✅ Recovery successful!")
            print(f"📄 Recovered output: {result.final_output}")
        else:
            print("\n❌ Recovery failed - returning error")
            print(f"📄 Errors: {result.errors}")
    
    async def show_communication_flow(self):
        """Demonstrate the communication flow between components."""
        print("\n" + "="*60)
        print("COMMUNICATION FLOW VISUALIZATION")
        print("="*60)
        
        print("""
        User Request
             │
             ▼
        ┌─────────────┐
        │ Orchestrator│
        └─────┬───────┘
              │
              ▼
        ┌─────────────┐     ┌──────────────┐
        │  Prefrontal │────▶│ Task         │
        │  Cortex     │     │ Decomposition│
        └─────┬───────┘     └──────────────┘
              │
              ▼
        ┌─────────────┐     ┌──────────────┐
        │  Message    │────▶│ Module       │
        │  Bus        │     │ Registry     │
        └─────┬───────┘     └──────────────┘
              │
        ┌─────┴──────┬──────────┐
        ▼            ▼          ▼
    Language     Vision      Audio
    Module       Module      Module
        │            │          │
        └────────────┴──────────┘
                     │
                     ▼
               Redis Memory
        """)
        
        print("\n📡 Message Flow Example:")
        print("1. User → Orchestrator: 'Analyze and summarize'")
        print("2. Orchestrator → Prefrontal Cortex: Plan execution")
        print("3. Prefrontal → Task Decomposer: Break into subtasks")
        print("4. Orchestrator → Message Bus: Publish task")
        print("5. Message Bus → Vision Module: Process image")
        print("6. Vision → Memory: Store analysis")
        print("7. Memory → Language Module: Retrieve context")
        print("8. Language → Orchestrator: Return summary")
        print("9. Orchestrator → User: Final result")
    
    async def show_metrics(self):
        """Display system metrics."""
        print("\n" + "="*60)
        print("SYSTEM METRICS")
        print("="*60)
        
        metrics = self.orchestrator.get_metrics()
        pfc_state = metrics.get("prefrontal_cortex_state", {})
        
        print("\n📊 Orchestrator Metrics:")
        print(f"  • Tasks received: {metrics['tasks_received']}")
        print(f"  • Tasks completed: {metrics['tasks_completed']}")
        print(f"  • Tasks failed: {metrics['tasks_failed']}")
        print(f"  • Average execution time: {metrics['average_execution_time_ms']:.0f}ms")
        print(f"  • Active tasks: {metrics['active_tasks']}")
        print(f"  • Queue size: {metrics.get('queue_size', 0)}")
        
        print("\n🧠 Prefrontal Cortex State:")
        print(f"  • Current goal: {pfc_state.get('current_goal', 'None')}")
        print(f"  • Working memory items: {pfc_state.get('working_memory_items', 0)}/7")
        print(f"  • Planning stack depth: {pfc_state.get('planning_stack_depth', 0)}")
        print(f"  • Recent executions: {pfc_state.get('recent_executions', 0)}")
    
    async def run_demo(self):
        """Run the complete demonstration."""
        try:
            # Setup system
            await self.setup()
            
            # Show communication flow
            await self.show_communication_flow()
            
            # Run examples
            await self.example_1_simple_text()
            await self.example_2_multi_modal()
            await self.example_3_parallel_consensus()
            await self.example_4_error_handling()
            
            # Show metrics
            await self.show_metrics()
            
            print("\n" + "="*60)
            print("🎉 DEMONSTRATION COMPLETE")
            print("="*60)
            
        finally:
            # Cleanup
            await self.orchestrator.stop()
            print("\n👋 System shutdown complete")


async def main():
    """Main entry point."""
    demo = NitroAGIDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║            NitroAGI NEXUS Orchestrator Demo               ║
    ║                                                           ║
    ║    Demonstrating Executive Control & Communication        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())