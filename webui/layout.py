from nicegui import ui
from theme import apply_theme

def menu_item(label: str, target: str, icon: str):
    """Creates a minimalist menu item."""
    # Check if active (simple path check - in a real app might need more robust routing check)
    # For now, we rely on NiceGUI's active-class handling if we used ui.link, but for ui.item we style manually
    # or let the user click.
    
    with ui.item(on_click=lambda: ui.navigate.to(target)).classes('rai-menu-item clickable v-ripple'):
        with ui.item_section().props('avatar').classes('min-w-0 q-pr-sm'):
            ui.icon(icon, size='20px')
        with ui.item_section():
            ui.label(label).classes('text-sm font-medium')

def layout(page_func):
    async def wrapper():
        apply_theme()
        
        # Minimalist Header (Hidden or very subtle)
        # REMOVED: ui.header caused layout issues (empty space).
        # Mobile menu button (only visible on small screens) - placed absolutely
        ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat round dense').classes('lg:hidden fixed top-4 left-4 z-50 text-gray-400')

        # Sidebar
        with ui.left_drawer(value=True).classes('rai-sidebar no-shadow').props('width=240 behavior="desktop" bordered') as left_drawer:
            # App Logo / Title Area
            with ui.row().classes('items-center q-px-md q-py-lg'):
                ui.icon('token', size='24px').classes('text-indigo-500')
                ui.label('RAI').classes('text-lg font-bold tracking-tight q-ml-sm')
            
            # Navigation
            with ui.column().classes('q-mt-md'):
                ui.label('WORKSPACE').classes('text-xs font-bold text-gray-600 q-px-md q-mb-sm tracking-wider')
                menu_item('Dashboard', '/dashboard', 'dashboard')
                menu_item('Chat', '/chat', 'chat_bubble_outline')
                menu_item('Agents', '/agents', 'smart_toy')
                menu_item('Chains', '/chains', 'hub') # Changed icon to 'hub' for chains/flow
                
                ui.separator().classes('q-my-md bg-gray-800')
                
                ui.label('SETTINGS').classes('text-xs font-bold text-gray-600 q-px-md q-mb-sm tracking-wider')
                menu_item('Configuration', '/settings', 'tune')

        # Main Content Area
        # We removed the wrapping columns here to allow pages to define top-level elements (like drawers).
        # Pages should use `with main_content():` to apply the standard container styling.
        
        # Await the page function if it is async
        result = page_func()
        if hasattr(result, '__await__'):
            await result

    return wrapper

def main_content():
    """Standard container for page content."""
    # Outer container for centering and background
    with ui.column().classes('w-full h-screen no-wrap items-center bg-[#0f172a]'): # Slate 900
        # Inner container for max width and padding
        return ui.column().classes('w-full max-w-5xl h-full q-px-md q-py-none')
