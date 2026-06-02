import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import json
import asyncio

from app.database import get_db, AsyncSessionLocal
from app.models import User, AgentSession
from app.schemas import AgentSessionResponse, AgentSessionCreate, AgentRunRequest
from app.auth import get_current_user
from app.agents.orchestrator import agent_graph, HumanMessage
from langchain_core.callbacks import AsyncCallbackHandler
from typing import Dict, Any, Optional

router = APIRouter(prefix="/sessions", tags=["Agent Sessions"])

# Локализованные логи для SSE-потока
LOG_MESSAGES = {
    "ru": {
        "start": "Запуск процесса выполнения...",
        "completed": "Выполнение успешно завершено.",
        "failed": "Произошла ошибка: {error}",
        "supervisor_desc": "Супервизор: Анализирую поставленную задачу и планирую шаги выполнения...",
        "researcher_desc": "Исследователь: Запуск агента. Приступаю к поиску и сбору информации...",
        "coder_desc": "Программист: Запуск агента. Приступаю к написанию скриптов и кодированию...",
        "writer_desc": "Писатель: Запуск агента. Приступаю к написанию и структурированию отчета...",
        "validator_desc": "Проверяющий: Запуск агента. Начинаю проверку и верификацию результатов...",
        "default_node_start": "Запуск агента {node_display}..."
    },
    "en": {
        "start": "Starting execution flow...",
        "completed": "Execution flow completed successfully.",
        "failed": "Error occurred: {error}",
        "supervisor_desc": "Supervisor: Analyzing the task and planning execution steps...",
        "researcher_desc": "Researcher: Starting agent. Beginning web search and information gathering...",
        "coder_desc": "Coder: Starting agent. Beginning scripting and code implementation...",
        "writer_desc": "Writer: Starting agent. Beginning drafting and structuring the report...",
        "validator_desc": "Validator: Starting agent. Beginning review and result verification...",
        "default_node_start": "Starting agent {node_display}..."
    }
}

NODE_TRANSLATIONS = {
    "ru": {
        "supervisor": "Супервизор",
        "researcher": "Исследователь",
        "coder": "Программист",
        "writer": "Писатель",
        "validator": "Проверяющий"
    },
    "en": {
        "supervisor": "Supervisor",
        "researcher": "Researcher",
        "coder": "Coder",
        "writer": "Writer",
        "validator": "Validator"
    }
}


@router.get("/", response_model=List[AgentSessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[AgentSession]:
    """Получить все сессии агента для текущего аутентифицированного пользователя."""
    result = await db.execute(
        select(AgentSession).where(AgentSession.user_id == current_user.id).order_by(AgentSession.created_at.desc())
    )
    return result.scalars().all()


@router.post("/", response_model=AgentSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_in: AgentSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> AgentSession:
    """Создать новую сессию/поток агента."""
    result = await db.execute(select(AgentSession).where(AgentSession.id == session_in.id))
    existing_session = result.scalars().first()
    if existing_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID already exists"
        )

    new_session = AgentSession(
        id=session_in.id,
        user_id=current_user.id,
        title=session_in.title,
        status="idle"
    )
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return new_session


class NodeStatusCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.run_to_node = {}  # run_id -> node_name
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


@router.post("/{session_id}/run")
async def run_agents(
    session_id: str,
    request: AgentRunRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> StreamingResponse:
    """Запустить граф мультиагентов с потоковым ответом (SSE)."""
    result = await db.execute(
        select(AgentSession).where(AgentSession.id == session_id, AgentSession.user_id == current_user.id)
    )
    session = result.scalars().first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    model_config = {
        "supervisor": "deepseek/deepseek-v4-flash",
        "validator": "deepseek/deepseek-r1",
        "researcher": "google/gemini-3-flash",
        "coder": "deepseek/deepseek-v4-pro",
        "writer": "deepseek/deepseek-v4-flash"
    }

    if request.universal_model:
        for k in model_config.keys():
            model_config[k] = request.universal_model.model
    elif request.model_overrides:
        for k, v in request.model_overrides.items():
            if k in model_config:
                model_config[k] = v.model

    session.status = "running"
    await db.commit()

    async def event_generator():
        language = request.language or "ru"
        queue = asyncio.Queue()
        handler = NodeStatusCallbackHandler(queue)
        
        state = {
            "messages": [HumanMessage(content=request.prompt)],
            "context": {
                "open_router_key": request.open_router_key,
                "tavily_key": request.tavily_key,
                "web_search_enabled": request.web_search_enabled,
                "language": language,
                "user_id": current_user.id,
                "session_id": session_id,
                "sandbox_mode": request.sandbox_mode or "lightweight"
            },
            "model_config": model_config,
            "logs": [LOG_MESSAGES.get(language, LOG_MESSAGES["en"])["start"]]
        }
        
        async def run_graph():
            try:
                result = await agent_graph.ainvoke(
                    state,
                    config={
                        "configurable": {"thread_id": session_id},
                        "callbacks": [handler]
                    }
                )
                
                final_res = ""
                if "messages" in result and result["messages"]:
                    ai_messages = [msg for msg in result["messages"][1:] if hasattr(msg, "content") and msg.content]
                    if ai_messages:
                        # Выбираем финальный ответ модели (наиболее длинное сообщение)
                        longest_msg = max(ai_messages, key=lambda m: len(m.content or ""))
                        final_res = longest_msg.content
                    else:
                        final_res = result["messages"][-1].content
                    
                await queue.put({
                    "event": "completed",
                    "result": final_res
                })
            except Exception as e:
                await queue.put({
                    "event": "failed",
                    "error": str(e)
                })

        task = asyncio.create_task(run_graph())
        
        try:
            init_msg = LOG_MESSAGES.get(language, LOG_MESSAGES["en"])["start"]
            yield f"data: {json.dumps({'log': init_msg})}\n\n"
            
            while True:
                item = await queue.get()
                
                if item["event"] == "node_start":
                    translations = NODE_TRANSLATIONS.get(language, NODE_TRANSLATIONS["en"])
                    node_display = translations.get(item["node"], item["node"].capitalize())
                    
                    msg_dict = LOG_MESSAGES.get(language, LOG_MESSAGES["en"])
                    start_msg = msg_dict.get(
                        f"{item['node']}_desc", 
                        msg_dict["default_node_start"].format(node_display=node_display)
                    )
                    
                    payload = {
                        "node": item["node"],
                        "log": start_msg,
                        "status": "running"
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    
                elif item["event"] == "node_end":
                    payload = {
                        "node": item["node"],
                        "log": item["log"],
                        "status": "completed_node"
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    
                elif item["event"] == "completed":
                    async with AsyncSessionLocal() as fresh_db:
                        res = await fresh_db.execute(
                            select(AgentSession).where(AgentSession.id == session_id)
                        )
                        fresh_session = res.scalars().first()
                        if fresh_session:
                            fresh_session.status = "completed"
                            await fresh_db.commit()
                    
                    comp_msg = LOG_MESSAGES.get(language, LOG_MESSAGES["en"])["completed"]
                    yield f"data: {json.dumps({'status': 'completed', 'log': comp_msg, 'result': item['result']})}\n\n"
                    break
                    
                elif item["event"] == "failed":
                    async with AsyncSessionLocal() as fresh_db:
                        res = await fresh_db.execute(
                            select(AgentSession).where(AgentSession.id == session_id)
                        )
                        fresh_session = res.scalars().first()
                        if fresh_session:
                            fresh_session.status = "failed"
                            await fresh_db.commit()
                    
                    fail_msg = LOG_MESSAGES.get(language, LOG_MESSAGES["en"])["failed"].format(error=item["error"])
                    yield f"data: {json.dumps({'status': 'failed', 'log': fail_msg})}\n\n"
                    break
                    
        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")
