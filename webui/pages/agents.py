import httpx
from nicegui import ui
from layout import layout, main_content
from config import API_BASE_URL
from utils import safe_api_call
from returns.result import Success, Failure

@ui.page('/agents')
@layout
async def agents_page():
    with main_content():
        ui.label('Agent Manager').classes('text-2xl font-bold q-mb-md')

        # Container for agent cards
        agents_container = ui.element('div').classes('grid grid-cols-3 w-full gap-4')

        async def load_agents():
            agents_container.clear()
            
            async def fetch():
                async with httpx.AsyncClient() as client:
                    return await client.get(f"{API_BASE_URL}/agents/", follow_redirects=True)

            result = await safe_api_call(fetch(), "Error loading agents")
            
            if isinstance(result, Success):
                response = result.unwrap()
                if response.status_code == 200:
                    agents = response.json()
                    if not agents:
                        with agents_container:
                            ui.label('No agents found. Create one!').classes('text-gray-500 italic col-span-3 text-center')
                    
                    for agent_id, config in agents.items():
                        with agents_container:
                            with ui.card().classes('minimal-card flex flex-col gap-2'):
                                with ui.row().classes('items-center justify-between w-full'):
                                    ui.label(agent_id).classes('text-lg font-bold')
                                    ui.icon('smart_toy', size='20px').classes('text-gray-400')
                                
                                ui.separator().classes('bg-gray-800')
                                
                                with ui.row().classes('gap-2 text-sm text-gray-400'):
                                    ui.icon('memory', size='16px')
                                    ui.label(config.get('model', 'Unknown Model'))
                                
                                with ui.row().classes('gap-2 text-sm text-gray-400'):
                                    ui.icon('dns', size='16px')
                                    ui.label(config.get('backend', 'Unknown Backend'))

                                ui.label(config.get('system', '')[:100] + '...').classes('text-xs text-gray-500 mt-2')

                                with ui.row().classes('mt-auto justify-end gap-2'):
                                    ui.button(icon='edit', on_click=lambda e, id=agent_id: ui.notify(f'Edit {id} (Coming Soon)')).props('flat round dense size=sm')
                                    ui.button(icon='delete', color='negative', on_click=lambda e, id=agent_id: delete_agent(id)).props('flat round dense size=sm')

                else:
                    ui.notify(f"Failed to load agents: {response.status_code}", type='negative')

        async def create_agent(name, model, system):
            async def post():
                async with httpx.AsyncClient() as client:
                    payload = {
                        "model": model,
                        "system": system,
                        "backend": "ollama" # Default for now
                    }
                    return await client.post(f"{API_BASE_URL}/agents/{name}", json=payload)

            result = await safe_api_call(post(), "Error creating agent")
            
            if isinstance(result, Success):
                response = result.unwrap()
                if response.status_code == 200:
                    ui.notify(f"Agent '{name}' created!", type='positive')
                    create_dialog.close()
                    await load_agents()
                else:
                    ui.notify(f"Failed to create agent: {response.text}", type='negative')

        async def delete_agent(agent_id):
            async def delete():
                async with httpx.AsyncClient() as client:
                    return await client.delete(f"{API_BASE_URL}/agents/{agent_id}")

            result = await safe_api_call(delete(), "Error deleting agent")

            if isinstance(result, Success):
                response = result.unwrap()
                if response.status_code == 200:
                    ui.notify(f"Agent '{agent_id}' deleted.", type='positive')
                    await load_agents()
                else:
                    ui.notify(f"Failed to delete agent: {response.text}", type='negative')

        # Create Agent Dialog
        with ui.dialog() as create_dialog, ui.card().classes('minimal-card w-96'):
            ui.label('Create New Agent').classes('text-lg font-bold mb-4')
            name_input = ui.input('Agent ID (Name)').classes('w-full')
            model_input = ui.input('Model (e.g. gemma:2b)').classes('w-full')
            system_input = ui.textarea('System Prompt').classes('w-full')
            
            with ui.row().classes('justify-end w-full mt-4'):
                ui.button('Cancel', on_click=create_dialog.close).props('flat color=grey')
                ui.button('Create', on_click=lambda: create_agent(name_input.value, model_input.value, system_input.value)).props('flat color=primary')

        # Floating Action Button to add agent
        ui.button(icon='add', on_click=create_dialog.open).classes('fixed bottom-8 right-8 rounded-full shadow-lg').props('fab color=accent')

        # Initial Load
        await load_agents()
