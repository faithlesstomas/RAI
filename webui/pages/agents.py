"""
Agents management page for RAI WebUI.
"""
from typing import Any, Dict, List, Optional
import httpx
from nicegui import ui
from returns.result import Success
from layout import layout, main_content
from config import API_BASE_URL
from utils import safe_api_call

HTTP_OK = 200

@ui.page('/agents')
@layout
async def agents_page() -> None: # noqa: PLR0915
    """Render the agents page with Master-Detail layout."""
    
    # State
    selected_agent_id: Optional[str] = None
    agents_data: Dict[str, Any] = {}
    available_models: List[str] = []
    
    # UI Elements references
    agent_list_container = None
    editor_container = None
    
    # Form Elements
    name_input = None
    model_select = None
    backend_select = None
    ollama_host_input = None
    system_input = None
    tools_input = None # Simple string input for now, comma separated
    
    async def load_models(backend: str = "ollama") -> None:
        nonlocal available_models
        async def fetch() -> httpx.Response:
            async with httpx.AsyncClient() as client:
                return await client.get(f"{API_BASE_URL}/models/{backend}", follow_redirects=True)
        
        result = await safe_api_call(fetch(), f"Error loading models for {backend}")
        if isinstance(result, Success):
            resp = result.unwrap()
            if resp.status_code == HTTP_OK:
                data = resp.json()
                available_models = data.get("models", [])
                if model_select:
                    model_select.options = available_models
                    model_select.update()

    async def load_agents() -> None:
        nonlocal agents_data
        async def fetch() -> httpx.Response:
            async with httpx.AsyncClient() as client:
                return await client.get(f"{API_BASE_URL}/agents/", follow_redirects=True)

        result = await safe_api_call(fetch(), "Error loading agents")
        if isinstance(result, Success):
            resp = result.unwrap()
            if resp.status_code == HTTP_OK:
                agents_data = resp.json()
                render_agent_list()

    def select_agent(agent_id: str) -> None:
        nonlocal selected_agent_id
        selected_agent_id = agent_id
        config = agents_data.get(agent_id, {})
        
        # Populate form
        if name_input:
            name_input.value = agent_id
            name_input.disable() # Cannot edit ID of existing agent
        
        if model_select:
            model_select.value = config.get("model", "")
            
        if backend_select:
            backend_select.value = config.get("backend", "ollama")
            
        if ollama_host_input:
            ollama_host_input.value = config.get("ollama_host", "http://127.0.0.1:11434")
            ollama_host_input.visible = (backend_select.value == "ollama")
            
        if system_input:
            system_input.value = config.get("system", "")
            
        if tools_input:
            tools_input.value = ", ".join(config.get("tools", []))
            
        render_agent_list() # To update active state styling

    def new_agent() -> None:
        nonlocal selected_agent_id
        selected_agent_id = None
        
        if name_input:
            name_input.value = ""
            name_input.enable()
            
        if model_select:
            model_select.value = available_models[0] if available_models else ""
            
        if backend_select:
            backend_select.value = "ollama"
            
        if ollama_host_input:
            ollama_host_input.value = "http://127.0.0.1:11434"
            ollama_host_input.visible = True
            
        if system_input:
            system_input.value = "You are a helpful AI assistant."
            
        if tools_input:
            tools_input.value = ""
            
        render_agent_list()

    async def save_agent() -> None:
        if not name_input.value:
            ui.notify("Agent ID is required", type="negative")
            return
            
        agent_id = name_input.value
        
        payload = {
            "model": model_select.value,
            "backend": backend_select.value,
            "ollama_host": ollama_host_input.value,
            "system": system_input.value,
            "tools": [t.strip() for t in tools_input.value.split(",") if t.strip()]
        }
        
        async def post() -> httpx.Response: # noqa: PLR0911
            async with httpx.AsyncClient() as client:
                # Determine if create or update based on if it existed
                # Actually API handles both via POST/PUT but let's use POST for create/update logic if simple
                # The API has POST for create and PUT for update.
                if selected_agent_id: # Update
                     return await client.put(f"{API_BASE_URL}/agents/{agent_id}", json=payload)
                else: # Create
                     return await client.post(f"{API_BASE_URL}/agents/{agent_id}", json=payload)

        result = await safe_api_call(post(), "Error saving agent")
        if isinstance(result, Success):
            resp = result.unwrap()
            if resp.status_code == HTTP_OK:
                ui.notify(f"Agent '{agent_id}' saved!", type='positive')
                await load_agents()
                select_agent(agent_id)
            else:
                ui.notify(f"Failed to save agent: {resp.text}", type='negative')

    async def delete_agent_action() -> None:
        if not selected_agent_id:
            return
            
        async def delete() -> httpx.Response:
            async with httpx.AsyncClient() as client:
                return await client.delete(f"{API_BASE_URL}/agents/{selected_agent_id}")

        result = await safe_api_call(delete(), "Error deleting agent")
        if isinstance(result, Success):
            resp = result.unwrap()
            if resp.status_code == HTTP_OK:
                ui.notify(f"Agent '{selected_agent_id}' deleted.", type='positive')
                await load_agents()
                new_agent()
            else:
                ui.notify(f"Failed to delete agent: {resp.text}", type='negative')

    def render_agent_list() -> None:
        agent_list_container.clear()
        with agent_list_container:
            ui.button('+ New Agent', on_click=new_agent).classes('w-full q-mb-md').props('color=primary')
            
            for agent_id, config in agents_data.items():
                is_selected = (agent_id == selected_agent_id)
                bg_color = 'bg-slate-700' if is_selected else 'bg-slate-800'
                
                with ui.card().classes(f'w-full p-3 mb-2 cursor-pointer {bg_color} hover:bg-slate-700').on('click', lambda _, a=agent_id: select_agent(a)):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('smart_toy', size='20px').classes('text-gray-400')
                        ui.label(agent_id).classes('font-bold text-white')
                    ui.label(config.get('model', 'Unknown')).classes('text-xs text-gray-400 ml-7')

    # --- Layout Construction ---
    with main_content():
        ui.label('Agent Configuration').classes('text-2xl font-bold q-mb-md text-white')
        
        with ui.splitter(value=30).classes('w-full h-[calc(100vh-150px)] border border-slate-700 rounded-lg') as splitter:
            
            # LEFT: Agent List
            with splitter.before:
                agent_list_container = ui.column().classes('w-full h-full p-4 bg-slate-900 overflow-y-auto')
            
            # RIGHT: Editor
            with splitter.after:
                editor_container = ui.column().classes('w-full h-full p-6 bg-slate-800 overflow-y-auto')
                with editor_container:
                    # Header
                    with ui.row().classes('w-full justify-between items-center q-mb-lg'):
                        ui.label('Agent Editor').classes('text-xl font-bold text-white')
                        ui.button('Delete', icon='delete', color='negative', on_click=delete_agent_action).props('flat dense')
                    
                    # Form
                    name_input = ui.input('Agent ID').classes('w-full q-mb-md').props('dark outlined')
                    
                    with ui.row().classes('w-full gap-4 q-mb-md'):
                        model_select = ui.select([], label='Model').classes('w-1/2').props('dark outlined')
                        backend_select = ui.select(['ollama', 'openai', 'anthropic'], label='Backend', value='ollama',
                                                 on_change=lambda e: update_backend_ui(e.value)).classes('w-1/2').props('dark outlined')
                    
                    ollama_host_input = ui.input('Ollama Host', value='http://127.0.0.1:11434').classes('w-full q-mb-md').props('dark outlined')
                    
                    system_input = ui.textarea('System Prompt').classes('w-full q-mb-md h-48').props('dark outlined')
                    
                    tools_input = ui.input('Tools (comma separated)').classes('w-full q-mb-lg').props('dark outlined')
                    
                    ui.button('Save Agent', icon='save', on_click=save_agent).classes('w-full').props('color=primary size=lg')

    def update_backend_ui(backend: str) -> None:
        if ollama_host_input:
            ollama_host_input.visible = (backend == 'ollama')
        # Trigger model reload for backend
        # Note: In a real app we'd await this, but in a callback we might need to create a task
        # For now, let's just trigger it and hope for the best or use ui.timer
        # ui.timer(0.1, lambda: load_models(backend), once=True) 
        # Better:
        # asyncio.create_task(load_models(backend))
        pass 

    # Initial Load
    await load_models()
    await load_agents()
