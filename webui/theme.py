"""
Theme configuration for RAI WebUI.
"""
from nicegui import ui

def apply_theme() -> None:
    """Applies global theme settings for the 'Modern Slate' design."""
    # Set primary colors to match the "Slate" aesthetic
    ui.colors(
        primary='#6366f1',   # Indigo-500 (Brand Color)
        secondary='#94a3b8', # Slate-400 (Muted text)
        accent='#38bdf8',    # Sky-400 (Secondary accents)
        dark='#0f172a',      # Slate-900 (Background)
        positive='#22c55e',  # Green-500
        negative='#ef4444',  # Red-500
        info='#3b82f6',      # Blue-500
        warning='#f59e0b'    # Amber-500
    )

    # Custom CSS for the modern slate look
    ui.add_head_html('''
        <style>
            :root {
                --rai-bg: #0f172a;        /* Slate 900 */
                --rai-surface: #1e293b;   /* Slate 800 */
                --rai-border: #334155;    /* Slate 700 */
                --rai-primary: #6366f1;   /* Indigo 500 */
                --rai-text-main: #f8fafc; /* Slate 50 */
                --rai-text-muted: #94a3b8;/* Slate 400 */
            }

            body {
                background-color: var(--rai-bg);
                color: var(--rai-text-main);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                -webkit-font-smoothing: antialiased;
            }

            /* Minimalist Card */
            .minimal-card {
                background: var(--rai-surface);
                border: 1px solid var(--rai-border);
                border-radius: 12px; /* Slightly softer corners */
                padding: 20px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }

            /* Clean Inputs */
            .q-field__control {
                border-radius: 8px !important;
                background: var(--rai-surface) !important; /* Ensure inputs match surface */
            }
            .q-field--outlined .q-field__control:before {
                border: 1px solid var(--rai-border);
            }
            .q-field--outlined.q-field--focused .q-field__control:after {
                border-width: 1px;
                border-color: var(--rai-primary);
            }
            .q-field__native, .q-field__prefix, .q-field__suffix, .q-field__input {
                color: var(--rai-text-main) !important;
            }
            .q-field__label {
                color: var(--rai-text-muted) !important;
            }

            /* Sidebar Styling */
            .rai-sidebar {
                background-color: var(--rai-bg);
                border-right: 1px solid var(--rai-border);
            }

            .rai-menu-item {
                color: var(--rai-text-muted);
                border-radius: 8px;
                transition: all 0.2s ease;
                margin: 4px 12px;
                padding: 8px 12px;
            }
            .rai-menu-item:hover {
                color: var(--rai-text-main);
                background-color: var(--rai-surface);
            }
            .rai-menu-item.q-item--active {
                color: var(--rai-primary);
                background-color: rgba(99, 102, 241, 0.1); /* Indigo with opacity */
                font-weight: 600;
            }

            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: transparent;
            }
            ::-webkit-scrollbar-thumb {
                background: #334155; /* Slate 700 */
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #475569; /* Slate 600 */
            }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    ''')
