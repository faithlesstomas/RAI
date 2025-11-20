from nicegui import ui
from layout import layout
from storage import storage

@ui.page('/chains')
@layout
def chains_page():
    ui.label('Chains').classes('text-h4 q-mb-md')

    chains_container = ui.column().classes('w-full max-w-4xl')

    def refresh_chains():
        chains_container.clear()
        chains = storage.get_chains()
        with chains_container:
            if not chains:
                ui.label('No chains created yet.').classes('text-grey italic')
            for chain in chains:
                with ui.card().classes('w-full q-mb-sm'):
                    with ui.row().classes('items-center justify-between w-full'):
                        with ui.column():
                            ui.label(chain['name']).classes('text-h6')
                            agent_names = [step['agent'] for step in chain['steps']]
                            ui.label(" -> ".join(agent_names)).classes('text-caption text-grey')
                        with ui.row():
                            ui.button(icon='edit', on_click=lambda c=chain: open_editor(c)).props('flat round dense')
                            ui.button(icon='delete', color='negative', on_click=lambda c=chain: delete_chain(c)).props('flat round dense')

    def delete_chain(chain):
        storage.delete_chain(chain['name'])
        refresh_chains()
        ui.notify(f"Chain '{chain['name']}' deleted.")

    # --- Editor Dialog ---
    editor_dialog = ui.dialog()

    def open_editor(chain=None):
        editor_dialog.clear()
        available_agents = [a['name'] for a in storage.get_agents()]
        
        if not available_agents:
            ui.notify('Create some agents first!', type='warning')
            return

        with editor_dialog, ui.card().classes('w-full max-w-2xl'):
            ui.label('Edit Chain' if chain else 'New Chain').classes('text-h6')
            
            name = ui.input('Name', value=chain['name'] if chain else '').classes('w-full')
            
            steps_container = ui.column().classes('w-full q-my-md')
            current_steps = chain['steps'] if chain else [] # List of dicts: {'agent': 'name'}

            def render_steps():
                steps_container.clear()
                with steps_container:
                    for i, step in enumerate(current_steps):
                        with ui.row().classes('items-center w-full'):
                            ui.label(f"Step {i+1}:").classes('q-mr-sm')
                            sel = ui.select(available_agents, value=step['agent']).classes('flex-grow')
                            def update_step(val, idx=i):
                                current_steps[idx]['agent'] = val
                            sel.on_value_change(lambda e, idx=i: update_step(e.value, idx))
                            
                            ui.button(icon='delete', color='negative', on_click=lambda idx=i: remove_step(idx)).props('flat round dense')

            def add_step():
                current_steps.append({'agent': available_agents[0]})
                render_steps()

            def remove_step(idx):
                current_steps.pop(idx)
                render_steps()

            render_steps()
            ui.button('Add Step', icon='add', on_click=add_step).props('flat')

            with ui.row().classes('w-full justify-end q-mt-md'):
                ui.button('Cancel', on_click=editor_dialog.close).props('flat')
                def save():
                    if not name.value:
                        ui.notify('Name is required', type='negative')
                        return
                    if not current_steps:
                        ui.notify('Add at least one step', type='negative')
                        return
                    
                    new_data = {
                        'name': name.value,
                        'steps': current_steps
                    }
                    storage.save_chain(new_data)
                    editor_dialog.close()
                    refresh_chains()
                    ui.notify('Chain saved!')
                ui.button('Save', on_click=save)

        editor_dialog.open()

    ui.button('Create New Chain', icon='add', on_click=lambda: open_editor(None)).classes('q-mb-md')
    refresh_chains()
