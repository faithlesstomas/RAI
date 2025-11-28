"""
Chat page for RAI WebUI.
"""
import json
import uuid
from typing import Any, Dict, List, Optional

import httpx
import websockets
from nicegui import ui, app
from returns.result import Success

from layout import layout, main_content
from storage import storage
from config import API_BASE_URL
from utils import safe_api_call

HTTP_OK = 200

@ui.page('/chat')
@layout
async def chat_page() -> None:
    """Render the chat page with Chain Configuration."""

    # Session Management
    if 'session_id' not in app.storage.user:
        app.storage.user['session_id'] = str(uuid.uuid4())
    
    current_session_id = app.storage.user['session_id']
    
    # State
    available_agents: Dict[str, Any] = {}
    chain_steps: List[str] = [] # List of agent IDs
    
    # UI References
    chat_container = None
    steps_container = None
    
    # --- Data Loading ---
    async def load_agents() -> None:
        nonlocal available_agents
        async def fetch() -> Any:
            async with httpx.AsyncClient() as client:
                return await client.get(f"{API_BASE_URL}/agents/", follow_redirects=True)
        
        result = await safe_api_call(fetch(), "Error loading agents")
        if isinstance(result, Success):
            resp = result.unwrap()
            if resp.status_code == HTTP_OK:
                available_agents = resp.json()
                update_steps_ui()

    # --- UI Helpers ---
    def add_step(agent_id: Optional[str] = None) -> None:
        default_agent = agent_id if agent_id else (list(available_agents.keys())[0] if available_agents else None)
        if default_agent:
            chain_steps.append(default_agent)
            update_steps_ui()

    def remove_step(index: int) -> None:
        if 0 <= index < len(chain_steps):
            chain_steps.pop(index)
            update_steps_ui()

    def update_step(index: int, value: str) -> None:
        if 0 <= index < len(chain_steps):
            chain_steps[index] = value

    def update_steps_ui() -> None:
        if not steps_container:
            return
            
        steps_container.clear()
        with steps_container:
            if not chain_steps:
                 ui.label("No steps defined. Add an agent to start.").classes("text-sm text-gray-500 italic")
            
            for i, agent_id in enumerate(chain_steps):
                with ui.row().classes('w-full items-center gap-2 mb-2'):
                    ui.label(f"{i+1}.").classes('text-gray-400 font-bold')
                    
                    # Agent Selector
                    select = ui.select(
                        options=list(available_agents.keys()), 
                        value=agent_id,
                        on_change=lambda e, idx=i: update_step(idx, e.value)
                    ).classes('flex-grow').props('dark dense outlined options-dense')
                    
                    # Remove Button
                    ui.button(icon='close', on_click=lambda _, idx=i: remove_step(idx)).props('flat round dense color=red size=sm')

    # --- Chat Logic ---
    async def send_message() -> None:
        text = text_input.value
        if not text:
            return
        
        if not chain_steps:
            ui.notify("Please add at least one agent to the chain.", type="warning")
            return

        text_input.value = ''
        
        # Render User Message
        with chat_container:
            render_message('User', text, True)
            
            # Placeholder for AI
            response_row = ui.row().classes('w-full justify-start items-start gap-4 q-py-md border-b border-slate-800/50')
            with response_row:
                ui.avatar(icon='smart_toy', color='primary', text_color='white').classes('shadow-sm')
                response_col = ui.column().classes('flex-grow min-w-0')
                with response_col:
                    ui.label('RAI').classes('text-xs text-primary q-mb-xs font-bold')
                    spinner = ui.spinner(type='dots', color='primary')
                    response_content = ui.markdown('').classes('w-full prose prose-invert max-w-none hidden')

        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        
        # Build Chain Config
        chain_configs = []
        for agent_id in chain_steps:
            agent_config = available_agents.get(agent_id)
            if agent_config:
                chain_configs.append({
                    "model": agent_config.get('model'),
                    "backend": agent_config.get('backend', 'ollama'),
                    "system_prompt": agent_config.get('system', '')
                })
        
        payload = {
            "chain_input": text,
            "chain_configs": chain_configs,
            "session_id": current_session_id
        }
        
        # WebSocket Communication
        server_url = storage.get_settings().get('server_url', 'http://127.0.0.1:8000')
        ws_url = server_url.replace('http', 'ws').replace('https', 'wss') + '/ws/v1/chat'
        
        try:
            async with websockets.connect(ws_url) as websocket:
                await websocket.send(json.dumps(payload))
                
                response_text = ""
                try:
                    spinner.delete()
                    response_content.classes(remove='hidden')
                except Exception:
                    pass
                
                async for message in websocket:
                    data = json.loads(message)
                    if data.get("type") == "response":
                        content = data.get("payload", {}).get("content", "")
                        response_text += content
                        response_content.set_content(response_text)
                        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
                        break # Assume single response for now
                    
                    if data.get("type") == "error":
                        ui.notify(f"Error: {data.get('detail')}", type='negative')
                        break
                
                # Save to history (mock)
                storage.add_chat_message(current_session_id, {'name': 'User', 'content': text, 'sent': True})
                storage.add_chat_message(current_session_id, {'name': 'RAI', 'content': response_text, 'sent': False})

        except Exception as ex:
            ui.notify(f"Connection failed: {ex}", type='negative')
            try:
                spinner.delete()
            except: pass

    def render_message(name: str, content: str, sent: bool) -> None:
        if sent: # User
            with ui.row().classes('w-full justify-end items-start gap-4 q-py-md'):
                with ui.column().classes('items-end max-w-3xl'):
                    ui.label('You').classes('text-xs text-slate-400 q-mb-xs font-bold')
                    ui.label(content).classes('text-base bg-slate-800 text-slate-200 px-4 py-2 rounded-lg whitespace-pre-wrap')
                ui.avatar(icon='person', color='slate-700', text_color='white').classes('shadow-sm')
        else: # AI
            with ui.row().classes('w-full justify-start items-start gap-4 q-py-md border-b border-slate-800/50'):
                ui.avatar(icon='smart_toy', color='primary', text_color='white').classes('shadow-sm')
                with ui.column().classes('flex-grow min-w-0'):
                    ui.label('RAI').classes('text-xs text-primary q-mb-xs font-bold')
                    ui.markdown(content).classes('w-full prose prose-invert max-w-none')

    # --- Layout Construction ---
    with main_content():
        with ui.row().classes('w-full h-[calc(100vh-100px)] gap-4 no-wrap'):
            
            # LEFT: Chat Area
            with ui.column().classes('flex-grow h-full relative'):
                # Chat History
                chat_container = ui.column().classes('w-full flex-grow overflow-y-auto q-pr-md pb-24')
                with chat_container:
                    # Load existing messages
                    for msg in storage.get_chat_history(current_session_id):
                        render_message(msg['name'], msg['content'], msg['sent'])

                # Input Area (Floating at bottom of this column)
                with ui.row().classes('absolute-bottom w-full bg-[#0f172a] p-4 border-t border-slate-700 gap-2'):
                    text_input = ui.input(placeholder='Message...').classes('flex-grow').props('dark outlined rounded')
                    text_input.on('keydown.enter', send_message)
                    ui.button(icon='send', on_click=send_message).props('round color=primary unelevated')

            # RIGHT: Chain Configuration
            with ui.column().classes('w-80 h-full bg-slate-900 border-l border-slate-700 p-4'):
                ui.label('Execution Chain').classes('text-lg font-bold text-white q-mb-md')
                
                # Session ID
                ui.input('Session ID', value=current_session_id).props('dark outlined readonly dense').classes('w-full q-mb-lg')
                
                # Steps Header
                with ui.row().classes('w-full justify-between items-center q-mb-sm'):
                    ui.label('Chain Steps').classes('text-sm font-bold text-gray-400')
                    ui.button(icon='add', on_click=lambda: add_step()).props('flat round dense color=primary size=sm')
                
                # Steps List
                steps_container = ui.column().classes('w-full gap-2 overflow-y-auto flex-grow')
                
                # Initial empty state or default
                update_steps_ui()

    # Initial Load
    await load_agents()
