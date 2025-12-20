"""
History page for RAI WebUI.
"""
from typing import Any, Dict, List, Optional
import httpx
from nicegui import ui
from returns.result import Success
from layout import layout, main_content
from config import API_BASE_URL
from utils import safe_api_call

HTTP_OK = 200

@ui.page('/history')
@layout
async def history_page() -> None: # noqa: PLR0915
    """Render the history page with Master-Detail layout."""
    
    # State
    selected_session_id: Optional[str] = None
    sessions_data: List[Dict[str, Any]] = []
    
    # UI References
    session_list_container = None
    replay_container = None
    
    async def load_sessions() -> None:
        nonlocal sessions_data
        async def fetch() -> httpx.Response:
            async with httpx.AsyncClient() as client:
                return await client.get(f"{API_BASE_URL}/history/sessions", follow_redirects=True)
        
        result = await safe_api_call(fetch(), "Error loading history")
        if isinstance(result, Success):
            resp = result.unwrap()
            if resp.status_code == HTTP_OK:
                data = resp.json()
                sessions_data = data.get("sessions", [])
                render_session_list()

    async def load_session_details(session_id: str) -> None:
        nonlocal selected_session_id
        selected_session_id = session_id
        render_session_list() # Update active state
        
        replay_container.clear()
        
        async def fetch() -> httpx.Response:
            async with httpx.AsyncClient() as client:
                return await client.get(f"{API_BASE_URL}/history/sessions/{session_id}", follow_redirects=True)

        result = await safe_api_call(fetch(), f"Error loading session {session_id}")
        
        with replay_container:
            if isinstance(result, Success):
                resp = result.unwrap()
                if resp.status_code == HTTP_OK:
                    data = resp.json()
                    messages = data.get("messages", [])
                    
                    ui.label(f"Session: {session_id}").classes('text-xl font-bold text-white q-mb-lg')
                    
                    if not messages:
                        ui.label("No messages in this session.").classes('text-gray-500 italic')
                    
                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        is_user = (role == "user")
                        
                        if is_user:
                            with ui.row().classes('w-full justify-end items-start gap-4 q-py-md'):
                                with ui.column().classes('items-end max-w-3xl'):
                                    ui.label('You').classes('text-xs text-slate-400 q-mb-xs font-bold')
                                    ui.label(content).classes('text-base bg-slate-800 text-slate-200 px-4 py-2 rounded-lg whitespace-pre-wrap')
                                ui.avatar(icon='person', color='slate-700', text_color='white').classes('shadow-sm')
                        else:
                            with ui.row().classes('w-full justify-start items-start gap-4 q-py-md border-b border-slate-800/50'):
                                ui.avatar(icon='smart_toy', color='primary', text_color='white').classes('shadow-sm')
                                with ui.column().classes('flex-grow min-w-0'):
                                    ui.label('RAI').classes('text-xs text-primary q-mb-xs font-bold')
                                    ui.markdown(content).classes('w-full prose prose-invert max-w-none')
                else:
                    ui.label(f"Failed to load details: {resp.status_code}").classes('text-red')
            else:
                ui.label("Failed to load details.").classes('text-red')

    def render_session_list() -> None:
        session_list_container.clear()
        with session_list_container:
            if not sessions_data:
                ui.label("No history found.").classes('text-gray-500 italic')
                return
                
            for session in sessions_data:
                s_id = session.get("id")
                summary = session.get("summary", "No summary")
                timestamp = session.get("timestamp", "")
                
                is_selected = (s_id == selected_session_id)
                bg_color = 'bg-slate-700' if is_selected else 'bg-slate-800'
                
                with ui.card().classes(f'w-full p-3 mb-2 cursor-pointer {bg_color} hover:bg-slate-700').on('click', lambda _, s=s_id: load_session_details(s)):
                    ui.label(summary).classes('font-bold text-white text-sm truncate')
                    with ui.row().classes('justify-between w-full mt-1'):
                        ui.label(s_id[:8]).classes('text-xs text-gray-400')
                        ui.label(timestamp).classes('text-xs text-gray-500')

    # --- Layout Construction ---
    with main_content():
        ui.label('Session History').classes('text-2xl font-bold q-mb-md text-white')
        
        with ui.splitter(value=30).classes('w-full h-[calc(100vh-150px)] border border-slate-700 rounded-lg') as splitter:
            
            # LEFT: Session List
            with splitter.before:
                session_list_container = ui.column().classes('w-full h-full p-4 bg-slate-900 overflow-y-auto')
            
            # RIGHT: Replay
            with splitter.after:
                replay_container = ui.column().classes('w-full h-full p-6 bg-slate-800 overflow-y-auto')
                with replay_container:
                    ui.label("Select a session to view details.").classes('text-gray-500 italic')

    # Initial Load
    await load_sessions()
