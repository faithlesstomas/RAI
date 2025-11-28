"""
Layout configuration for RAI WebUI.
"""
from typing import Any, Callable, Coroutine
from nicegui import ui
from theme import apply_theme

def menu_item(label: str, target: str, icon: str) -> None:
    """Creates a minimalist menu item."""
    # Check if active (simple path check - in a real app might need more robust routing check)
    # For now, we rely on NiceGUI's active-class handling if we used ui.link,
    # but for ui.item we style manually or let the user click.

    with ui.item(on_click=lambda: ui.navigate.to(target)).classes('rai-menu-item clickable v-ripple'):
        with ui.item_section().props('avatar').classes('min-w-0 q-pr-sm'):
            ui.icon(icon, size='20px')
        with ui.item_section():
            ui.label(label).classes('text-sm font-medium')

def layout(page_func: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., Coroutine[Any, Any, None]]:
    """Decorator to apply the standard layout to a page."""
    async def wrapper() -> None:
        apply_theme()

        # Minimalist Header (Hidden or very subtle)
        # REMOVED: ui.header caused layout issues (empty space).
        # Mobile menu button (only visible on small screens) - placed absolutely
        # Sidebar
        with ui.left_drawer(value=True).classes('rai-sidebar no-shadow').props(
            'width=240 behavior="desktop" bordered'
        ) as left_drawer:
            # App Logo / Title Area
            with ui.row().classes('bg-slate-900 border-b border-slate-700 h-16 items-center px-4 w-full'):
                ui.button(icon='menu', on_click=left_drawer.toggle).props('flat round color=white')
                ui.label('RAI Assistant').classes('text-xl font-bold text-white ml-2')

            # Navigation
            with ui.column().classes('q-mt-md'):
                ui.label('WORKSPACE').classes(
                    'text-xs font-bold text-gray-600 q-px-md q-mb-sm tracking-wider'
                )
                menu_item('Playground', '/chat', 'chat_bubble_outline')
                menu_item('Agents', '/agents', 'smart_toy')
                menu_item('History', '/history', 'history')

                ui.separator().classes('q-my-md bg-gray-800')

                # Server Status (Mock for now, could be real later)
                with ui.row().classes('absolute-bottom w-full p-4 items-center gap-2'):
                    ui.icon('circle', size='12px').classes('text-green-500')
                    ui.label('Server Online').classes('text-xs text-gray-400')

        # Mobile menu button (only visible on small screens) - placed absolutely
        ui.button(
            on_click=left_drawer.toggle,
            icon='menu'
        ).props('flat round dense').classes('lg:hidden fixed top-4 left-4 z-50 text-gray-400')

        # Main Content Area
        # We removed the wrapping columns here to allow pages to define top-level elements (like drawers).
        # Pages should use `with main_content():` to apply the standard container styling.

        # Await the page function if it is async
        result = page_func()
        if hasattr(result, '__await__'):
            await result

    return wrapper

def main_content() -> Any: # noqa: ANN401
    """Standard container for page content."""
    # Outer container for centering and background
    with ui.column().classes('w-full h-screen no-wrap items-center bg-[#0f172a]'): # Slate 900
        # Inner container for max width and padding
        return ui.column().classes('w-full max-w-5xl h-full q-px-md q-py-none')
