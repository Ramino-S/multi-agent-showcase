import os
import sys
import asyncio
import uuid
from typing import Dict, Any, Optional, List

# Добавляем корень проекта в sys.path для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.orchestrator import agent_graph, HumanMessage
from langchain_core.callbacks import AsyncCallbackHandler

class NodeStatusCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.run_to_node = {}  # run_id -> node_name для узлов верхнего уровня
        self.parent_map = {}   # run_id -> parent_run_id

    def _has_active_node_ancestor(self, parent_run_id: Any) -> bool:
        curr = parent_run_id
        while curr is not None:
            if curr in self.run_to_node:
                return True
            curr = self.parent_map.get(curr)
        return False

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        self.parent_map[run_id] = parent_run_id
        
        if metadata and "langgraph_node" in metadata:
            node_name = metadata["langgraph_node"]
            if node_name in ["supervisor", "researcher", "coder", "writer", "validator"]:
                if not self._has_active_node_ancestor(parent_run_id):
                    self.run_to_node[run_id] = node_name
                    await self.queue.put({
                        "event": "node_start",
                        "node": node_name
                    })

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        self.parent_map.pop(run_id, None)
        
        if run_id in self.run_to_node:
            node_name = self.run_to_node.pop(run_id)
            
            latest_log = ""
            if isinstance(outputs, dict):
                logs = outputs.get("logs", [])
                if logs:
                    latest_log = logs[-1]
            if not latest_log:
                latest_log = f"{node_name.capitalize()} finished step."

            await self.queue.put({
                "event": "node_end",
                "node": node_name,
                "log": latest_log,
                "outputs": outputs
            })

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self.parent_map.pop(run_id, None)
        self.run_to_node.pop(run_id, None)

async def main():
    queue = asyncio.Queue()
    handler = NodeStatusCallbackHandler(queue)
    
    state = {
        "messages": [HumanMessage(content="Hello.")],
        "context": {"open_router_key": "mock-key", "language": "ru"},
        "model_config": {
            "supervisor": "deepseek/deepseek-v4-flash",
            "validator": "deepseek/deepseek-r1",
            "researcher": "google/gemini-3-flash",
            "coder": "deepseek/deepseek-v4-pro",
            "writer": "deepseek/deepseek-v4-flash"
        },
        "logs": []
    }
    
    async def run_graph():
        try:
            print("Invoking graph...")
            result = await agent_graph.ainvoke(
                state,
                config={
                    "configurable": {"thread_id": str(uuid.uuid4())},
                    "callbacks": [handler]
                }
            )
            print("Graph invocation finished.")
            await queue.put({"event": "completed", "result": result})
        except Exception as e:
            print(f"Graph invocation failed: {e}")
            await queue.put({"event": "failed", "error": str(e)})

    asyncio.create_task(run_graph())
    
    while True:
        item = await queue.get()
        print(f"Queue item: {item}")
        if item["event"] in ["completed", "failed"]:
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
