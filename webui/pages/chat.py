from nicegui import ui, app
from layout import layout, main_content
from storage import storage
from theme import apply_theme
from config import API_BASE_URL
from utils import safe_api_call
from returns.result import Success, Failure
import httpx
import json
import websockets

import uuid

@ui.page('/chat')
@layout
async def chat_page():
    
    # Session Management
    if 'session_id' not in app.storage.user:
        app.storage.user['session_id'] = str(uuid.uuid4())
    
    session_id = app.storage.user['session_id']

    # Right Drawer for History (Must be top-level)
    with ui.right_drawer(value=False).classes('bg-slate-900 border-l border-slate-700') as history_drawer:
        ui.label('Chat History').classes('text-xl font-bold p-4 text-white')
        history_container = ui.column().classes('w-full')
        
        def load_history_list():
            history_container.clear()
            sessions = storage.get_sessions()
            print(f"DEBUG: Loaded sessions: {list(sessions.keys())}") # Debug logging
            # Sort by last message timestamp (mock logic for now, just reverse insertion order)
            for sess_id, msgs in reversed(list(sessions.items())):
                print(f"DEBUG: Session {sess_id} has {len(msgs)} messages")
                if not msgs: continue
                
                # Summary is first few words of first user message
                first_user_msg = next((m['content'] for m in msgs if m.get('sent')), "New Chat")
                summary = (first_user_msg[:30] + '...') if len(first_user_msg) > 30 else first_user_msg
                
                with history_container:
                    with ui.card().classes('w-full p-2 bg-slate-800 hover:bg-slate-700 cursor-pointer').on('click', lambda s=sess_id: load_session(s)):
                        ui.label(summary).classes('text-sm text-white font-bold')
                        ui.label(sess_id[:8]).classes('text-xs text-gray-400')

        def load_session(sess_id):
            app.storage.user['session_id'] = sess_id
            ui.navigate.to('/chat')

        load_history_list()

    with main_content():
        # --- Header ---
        with ui.row().classes('w-full justify-between items-center q-mb-md max-w-4xl mx-auto'):
            ui.label('Chat with Agents').classes('text-2xl font-bold text-primary')

            # Model/Chain Selection & History Toggle
            with ui.row().classes('items-center gap-2'):
                 ui.button(icon='history', on_click=history_drawer.toggle).props('flat round color=white').tooltip('Chat History')
                 
                 # We need to fetch this async, so we use a container and update it
                 selection_container = ui.row().classes('items-center gap-2')
        
        # Reactive variable for selected option
        selected_option = ui.select({}, value=None).classes('w-64 hidden') # Hidden initially

        async def load_options():
            try:
                import httpx
                API_BASE_URL = "http://127.0.0.1:8000/api/v1"
                
                options = {}
                
                async with httpx.AsyncClient() as client:
                    # Fetch Agents
                    agents_resp = await client.get(f"{API_BASE_URL}/agents/", follow_redirects=True)
                    if agents_resp.status_code == 200:
                        agents = agents_resp.json()
                        for name, config in agents.items():
                            options[f"agent:{name}"] = f"Agent: {name}"
                    
                    # Fetch Chains (Placeholder for now as API doesn't list chains yet)
                    # chains = storage.get_chains() 
                    # if chains:
                    #     for c in chains:
                    #         options[f"chain:{c['name']}"] = f"Chain: {c['name']}"
                
                if not options:
                    selection_container.clear()
                    with selection_container:
                        ui.label('No agents found.').classes('text-red')
                    return

                # Default selection
                default_value = list(options.keys())[0]
                
                selection_container.clear()
                with selection_container:
                    # Re-create select to populate options
                    selected_option.options = options
                    selected_option.value = default_value
                    selected_option.classes(remove='hidden')
                    selected_option.move(selection_container)

            except Exception as e:
                ui.notify(f"Error loading agents: {e}", type='negative')

        # Trigger load
        await load_options()

        # --- Chat Area ---
        # Use a scroll area or just a column with padding at bottom for the fixed footer
        chat_container = ui.column().classes('w-full max-w-4xl mx-auto flex-grow q-mb-xl pb-24 px-4') 
        
        def render_message(name, content, sent):
            # Distinct layout for User vs AI
            if sent: # User
                with ui.row().classes('w-full justify-end items-start gap-4 q-py-md'):
                    with ui.column().classes('items-end max-w-3xl'):
                        ui.label('You').classes('text-xs text-slate-400 q-mb-xs font-bold')
                        ui.label(content).classes('text-base bg-slate-800 text-slate-200 px-4 py-2 rounded-lg whitespace-pre-wrap')
                    ui.avatar(icon='person', color='slate-700', text_color='white').classes('shadow-sm')
            else: # AI (RAI)
                with ui.row().classes('w-full justify-start items-start gap-4 q-py-md border-b border-slate-800/50'):
                    ui.avatar(icon='smart_toy', color='primary', text_color='white').classes('shadow-sm')
                    with ui.column().classes('flex-grow min-w-0'): # min-w-0 needed for flex child to shrink
                        ui.label('RAI').classes('text-xs text-primary q-mb-xs font-bold')
                        # Markdown content - ensure it handles code blocks well
                        ui.markdown(content).classes('w-full prose prose-invert max-w-none [&_pre]:bg-slate-900 [&_pre]:p-4 [&_pre]:rounded-md [&_code]:text-pink-300')

        # Load History
        with chat_container:
            for msg in storage.get_chat_history(session_id):
                render_message(msg['name'], msg['content'], msg['sent'])

        # --- WebSocket Logic ---
        async def send_message(e=None): # e is optional for manual calls
            text = text_input.value
            if not text:
                return
            
            text_input.value = ''
            
            # Display User Message
            with chat_container:
                render_message('User', text, True)
                
                # Placeholder for response
                response_row = ui.row().classes('w-full justify-start items-start gap-4 q-py-md border-b border-slate-800/50')
                with response_row:
                    ui.avatar(icon='smart_toy', color='primary', text_color='white').classes('shadow-sm')
                    response_col = ui.column().classes('flex-grow min-w-0')
                    with response_col:
                        ui.label('RAI').classes('text-xs text-primary q-mb-xs font-bold')
                        spinner = ui.spinner(type='dots', color='primary')
                        response_content = ui.markdown('').classes('w-full prose prose-invert max-w-none [&_pre]:bg-slate-900 [&_pre]:p-4 [&_pre]:rounded-md [&_code]:text-pink-300 hidden')

            # Scroll to bottom
            ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
            
            # Save User Message
            storage.add_chat_message(session_id, {'name': 'User', 'content': text, 'sent': True})
            
            # Prepare Payload
            if not selected_option.value:
                ui.notify("No agent selected", type='warning')
                return

            selection_type, name = selected_option.value.split(':', 1)
            
            chain_configs = []
            if selection_type == 'chain':
                # TODO: Implement chain fetching from API
                pass
            else: # agent
                # We need to fetch the specific agent config again or store it. 
                # For simplicity, we'll fetch it or rely on the server to handle "agent_name" in the future.
                # But current API expects full config in chain_configs.
                # Let's fetch it quickly.
                async def fetch_agent_config_api():
                    async with httpx.AsyncClient() as client:
                        return await client.get(f"{API_BASE_URL}/agents/{name}", follow_redirects=True)

                result = await safe_api_call(fetch_agent_config_api(), "Error fetching agent config")
                
                if isinstance(result, Success):
                    resp = result.unwrap()
                    if resp.status_code == 200:
                        agent_config = resp.json()
                        chain_configs.append({
                            "model": agent_config.get('model'),
                            "backend": agent_config.get('backend', 'ollama'),
                            "system_prompt": agent_config.get('system', '')
                        })
                    else:
                        ui.notify(f"Failed to fetch agent config: {resp.status_code}", type='negative')
                        return
                else:
                    return # Error handled by safe_api_call

            payload = {
                "chain_input": text,
                "chain_configs": chain_configs,
                "session_id": app.storage.user['session_id']
            }

            # Send via WebSocket
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
                            break # Exit loop after receiving the full response (non-streaming)
                                
                        elif data.get("type") == "error":
                            ui.notify(f"Error: {data.get('detail')}", type='negative')
                            break
                    
                    # Save AI Response
                    storage.add_chat_message(session_id, {'name': 'RAI', 'content': response_text, 'sent': False})
                            
            except Exception as ex:
                try:
                    spinner.delete()
                except Exception:
                    pass
                ui.notify(f"Connection failed: {ex}", type='negative')
                with response_col:
                    ui.label(f"Error: {ex}").classes('text-red')

        # --- Input Area (Sticky Footer) ---
        # ui.footer cannot be used here because layout wraps us in a column.
        # We use a fixed positioned row instead.
        with ui.row().classes('fixed-bottom w-full bg-[#0f172a]/90 backdrop-blur-md border-t border-slate-700 p-4 z-50'):
            with ui.row().classes('w-full max-w-4xl mx-auto items-center gap-4'):
                # New Chat Button
                def new_chat():
                    # Generate new session ID
                    new_session_id = str(uuid.uuid4())
                    app.storage.user['session_id'] = new_session_id
                    
                    # Clear UI
                    chat_container.clear()
                    
                    # Notify
                    ui.notify('New chat started')
                    
                    # Reload page to ensure clean state (optional but safer for now)
                    ui.navigate.to('/chat')
                
                ui.button(icon='delete_sweep', on_click=new_chat).props('round flat color=grey').tooltip('New Chat')

                # Input
                text_input = ui.input(placeholder='Message RAI...').props('rounded outlined bg-color=slate-800 text-white').classes('flex-grow').on('keydown.enter', send_message)
                
                # Send Button
                ui.button(icon='send', on_click=send_message).props('round color=primary unelevated')
