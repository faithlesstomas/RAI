;;; rai-client.el --- Load the RAI Emacs client example -*- lexical-binding: t; -*-

;;; Commentary:

;; Compatibility entry point for the maintained client in clients/emacs.
;;
;; 1. Start RAI with: `uv run rai serve`
;; 2. Load this file with: `M-x load-file`
;; 3. Configure authentication as described in clients/emacs/README.md.
;; 4. Run: `M-x rai-chat`
;;
;; The old example used the legacy provider-owned IPC `run` request and embedded
;; one model configuration in the editor.  The current client talks to the
;; provider-neutral RAI Assistant API and keeps its wire compatibility boundary
;; in rai-transport.el.

;;; Code:

(let* ((example-directory
        (file-name-directory (or load-file-name buffer-file-name)))
       (client-directory
        (expand-file-name "../clients/emacs" example-directory)))
  (add-to-list 'load-path client-directory))

(require 'rai)

(provide 'rai-client)
;;; rai-client.el ends here
