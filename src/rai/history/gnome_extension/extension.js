import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const INTERFACE = `
<node>
  <interface name="org.richai.History">
    <method name="GetSnapshot">
      <arg type="b" direction="out" name="locked"/>
      <arg type="b" direction="out" name="idle"/>
      <arg type="i" direction="out" name="workspace"/>
      <arg type="s" direction="out" name="application_id"/>
      <arg type="s" direction="out" name="title"/>
      <arg type="i" direction="out" name="pid"/>
    </method>
    <signal name="ActiveWindowChanged">
      <arg type="s" name="application_id"/>
      <arg type="s" name="title"/>
      <arg type="i" name="pid"/>
    </signal>
    <signal name="WorkspaceChanged"><arg type="i" name="workspace"/></signal>
  </interface>
</node>`;

export default class RaiHistoryExtension extends Extension {
    enable() {
        this._dbus = Gio.DBusExportedObject.wrapJSObject(INTERFACE, this);
        this._dbus.export(Gio.DBus.session, '/org/richai/History');
        this._owner = Gio.bus_own_name_on_connection(
            Gio.DBus.session,
            'org.richai.History',
            Gio.BusNameOwnerFlags.NONE,
            null,
            null,
        );
        this._windowSignal = global.display.connect(
            'notify::focus-window', () => this._emitWindow());
        this._workspaceSignal = global.workspace_manager.connect(
            'active-workspace-changed', () => this._emitWorkspace());
        this._emitWindow();
        this._emitWorkspace();
    }

    disable() {
        if (this._windowSignal)
            global.display.disconnect(this._windowSignal);
        if (this._workspaceSignal)
            global.workspace_manager.disconnect(this._workspaceSignal);
        if (this._owner)
            Gio.bus_unown_name(this._owner);
        if (this._dbus)
            this._dbus.unexport();
        this._windowSignal = null;
        this._workspaceSignal = null;
        this._owner = null;
        this._dbus = null;
    }

    GetSnapshot() {
        const [applicationId, title, pid] = this._window();
        const monitor = global.backend.get_core_idle_monitor();
        return [
            Boolean(Main.sessionMode.isLocked),
            monitor.get_idletime() >= 60000,
            global.workspace_manager.get_active_workspace_index(),
            applicationId,
            title,
            pid,
        ];
    }

    _window() {
        const window = global.display.focus_window;
        if (!window)
            return ['', '', 0];
        const applicationId = window.get_gtk_application_id()
            || window.get_wm_class_instance()
            || window.get_wm_class()
            || '';
        return [applicationId.toLowerCase(), window.get_title() || '', window.get_pid() || 0];
    }

    _emitWindow() {
        if (!this._dbus)
            return;
        const [applicationId, title, pid] = this._window();
        this._dbus.emit_signal(
            'ActiveWindowChanged', new GLib.Variant('(ssi)', [applicationId, title, pid]));
    }

    _emitWorkspace() {
        if (!this._dbus)
            return;
        const index = global.workspace_manager.get_active_workspace_index();
        this._dbus.emit_signal('WorkspaceChanged', new GLib.Variant('(i)', [index]));
    }
}
