import httpx
from nicegui import ui
from layout import layout, main_content
from config import API_BASE_URL

@ui.page('/dashboard')
@layout
async def dashboard_page():
    with main_content():
        ui.label('Dashboard').classes('text-2xl font-bold q-mb-md')

        # Stat Cards Container
        # We use a row of cards. We need to update them dynamically.
        
        # Create reactive values or just update labels directly
        active_agents_label = ui.label('Loading...').classes('text-2xl font-bold')
        total_sessions_label = ui.label('Loading...').classes('text-2xl font-bold')
        uptime_label = ui.label('N/A').classes('text-2xl font-bold') # Server doesn't expose uptime yet
        
        with ui.element('div').classes('grid grid-cols-4 w-full gap-4 mb-8'):
            with ui.card().classes('minimal-card flex flex-row items-center gap-4'):
                ui.icon('smart_toy', size='32px').classes('text-indigo-500')
                with ui.column().classes('gap-0'):
                    ui.label('Active Agents').classes('text-gray-400 text-xs uppercase tracking-wider')
                    active_agents_label.move(ui.element('div')) # Move label here

            with ui.card().classes('minimal-card flex flex-row items-center gap-4'):
                ui.icon('history', size='32px').classes('text-green-500')
                with ui.column().classes('gap-0'):
                    ui.label('Total Sessions').classes('text-gray-400 text-xs uppercase tracking-wider')
                    total_sessions_label.move(ui.element('div'))

            with ui.card().classes('minimal-card flex flex-row items-center gap-4'):
                ui.icon('timer', size='32px').classes('text-blue-500')
                with ui.column().classes('gap-0'):
                    ui.label('Uptime').classes('text-gray-400 text-xs uppercase tracking-wider')
                    uptime_label.move(ui.element('div'))

            with ui.card().classes('minimal-card flex flex-row items-center gap-4'):
                ui.icon('memory', size='32px').classes('text-purple-500')
                with ui.column().classes('gap-0'):
                    ui.label('Memory').classes('text-gray-400 text-xs uppercase tracking-wider')
                    ui.label('N/A').classes('text-2xl font-bold')

        ui.label('Recent Activity').classes('text-xl font-bold q-mb-sm')
        activity_container = ui.column().classes('w-full gap-2')

        async def refresh_dashboard():
            try:
                async with httpx.AsyncClient() as client:
                    # Fetch Agents
                    agents_resp = await client.get(f"{API_BASE_URL}/agents/", follow_redirects=True)
                    if agents_resp.status_code == 200:
                        agents_data = agents_resp.json()
                        active_agents_label.set_text(str(len(agents_data)))
                    
                    # Fetch History
                    history_resp = await client.get(f"{API_BASE_URL}/history/sessions")
                    if history_resp.status_code == 200:
                        history_data = history_resp.json()
                        sessions = history_data.get('sessions', [])
                        total_sessions_label.set_text(str(len(sessions)))
                        
                        # Update Activity Feed
                        activity_container.clear()
                        with activity_container:
                            for session in sessions[:5]: # Show last 5
                                with ui.card().classes('minimal-card w-full'):
                                    with ui.row().classes('items-center gap-4'):
                                        ui.icon('chat_bubble', size='20px').classes('text-gray-500')
                                        with ui.column().classes('gap-0'):
                                            ui.label(f"Session: {session.get('id', 'Unknown')}").classes('font-bold')
                                            ui.label(session.get('summary', 'No summary')).classes('text-sm text-gray-400')
                                        ui.label(session.get('timestamp', '')).classes('ml-auto text-xs text-gray-600')

            except Exception as e:
                ui.notify(f"Error refreshing dashboard: {e}", type='negative')

        # Initial Load
        await refresh_dashboard()
