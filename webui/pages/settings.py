from nicegui import ui
from layout import layout
from storage import storage

@ui.page('/settings')
@layout
def settings_page():
    ui.label('Settings').classes('text-h4 q-mb-md')
    
    with ui.card().classes('w-full max-w-2xl'):
        ui.label('Server Configuration').classes('text-h6')
        
        server_url = ui.input('RAI Server URL', value=storage.get_settings().get('server_url', 'http://127.0.0.1:8000')) \
            .classes('w-full')
        
        def save():
            settings = storage.get_settings()
            settings['server_url'] = server_url.value
            storage.save_settings(settings)
            ui.notify('Settings saved!', type='positive')

        ui.button('Save', on_click=save).classes('q-mt-md')
