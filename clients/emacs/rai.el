;;; rai.el --- Rich AI Assistant client for Emacs -*- lexical-binding: t; -*-

;; Copyright (C) 2026 RAI contributors

;; Author: RAI contributors
;; Version: 0.1.0
;; Package-Requires: ((emacs "28.1"))
;; Keywords: tools, ai
;; URL: https://gitlab.com/tk-lab1/ai/rai

;;; Commentary:

;; `rai-chat' opens a lightweight client for the RAI-owned Assistant runtime.
;; `rai-ask-with-context' adds an explicit, bounded editor observation.  The
;; transport and response adapter live in rai-transport.el so an evolving RAI
;; API does not leak into the UI and context modules.

;;; Code:

(require 'button)
(require 'json)
(require 'subr-x)
(require 'rai-transport)
(require 'rai-context)
(require 'rai-repl)

(defcustom rai-default-session-id nil
  "Session ID used by new RAI chat buffers.

When nil, the server assigns an ID and the client retains it in the buffer."
  :type '(choice (const :tag "Server generated" nil) string)
  :group 'rai)

(defcustom rai-default-data-class "LOCAL"
  "Data class attached to Assistant turns from Emacs."
  :type '(choice (const "PUBLIC") (const "LOCAL") (const "PRIVATE"))
  :group 'rai)

(defcustom rai-chat-buffer-name "*RAI Assistant*"
  "Name of the default RAI chat buffer."
  :type 'string
  :group 'rai)

(defface rai-chat-user-face
  '((t :inherit font-lock-keyword-face :weight bold))
  "Face for user prompts in a RAI chat buffer."
  :group 'rai)

(defface rai-chat-assistant-face
  '((t :inherit font-lock-doc-face :weight bold))
  "Face for Assistant labels in a RAI chat buffer."
  :group 'rai)

(defvar-local rai-session-id nil
  "RAI Assistant session associated with the current chat buffer.")

(defvar-local rai--last-turn-id nil
  "Most recent Assistant turn identifier in the current chat buffer.")

(defvar-local rai--last-manifest-id nil
  "Most recent context manifest identifier in the current chat buffer.")

(defvar-local rai--pending-request nil
  "URL response buffer for the request currently in flight.")

(defvar rai-chat-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "a") #'rai-ask)
    (define-key map (kbd "c") #'rai-ask-with-context)
    (define-key map (kbd "g") #'rai-show-last-context)
    (define-key map (kbd "m") #'rai-show-memories)
    (define-key map (kbd "s") #'rai-show-sessions)
    (define-key map (kbd "k") #'rai-cancel-request)
    (define-key map (kbd "q") #'quit-window)
    map)
  "Keymap for `rai-chat-mode'.")

(define-derived-mode rai-chat-mode special-mode "RAI-Chat"
  "Major mode for a provider-neutral Rich AI Assistant conversation."
  (setq-local truncate-lines nil)
  (setq-local rai-session-id rai-default-session-id))

(defun rai--chat-buffer ()
  "Return the default initialized RAI chat buffer."
  (let ((buffer (get-buffer-create rai-chat-buffer-name)))
    (with-current-buffer buffer
      (unless (derived-mode-p 'rai-chat-mode)
        (rai-chat-mode)))
    buffer))

(defun rai--insert-chat (buffer role text &optional metadata)
  "Insert ROLE and TEXT into chat BUFFER, followed by optional METADATA."
  (when (buffer-live-p buffer)
    (with-current-buffer buffer
      (let ((inhibit-read-only t)
            (at-end (= (point) (point-max))))
        (goto-char (point-max))
        (unless (= (point-min) (point-max))
          (insert "\n"))
        (insert (propertize (concat role "\n")
                            'face (if (string= role "You")
                                      'rai-chat-user-face
                                    'rai-chat-assistant-face)))
        (insert text "\n")
        (when metadata
          (insert (propertize metadata 'face 'shadow) "\n"))
        (when at-end (goto-char (point-max)))))))

(defun rai--request-error (chat-buffer error)
  "Render normalized transport ERROR in CHAT-BUFFER."
  (when (buffer-live-p chat-buffer)
    (with-current-buffer chat-buffer
      (setq rai--pending-request nil))
    (rai--insert-chat chat-buffer "RAI error"
                      (rai-transport-error-message error))))

(defun rai--request-success (chat-buffer response)
  "Render normalized Assistant RESPONSE in CHAT-BUFFER."
  (when (buffer-live-p chat-buffer)
    (with-current-buffer chat-buffer
      (setq rai--pending-request nil
            rai-session-id (or (plist-get response :session-id) rai-session-id)
            rai--last-turn-id (plist-get response :turn-id)
            rai--last-manifest-id (plist-get response :manifest-id)))
    (rai--insert-chat
     chat-buffer "RAI" (or (plist-get response :content) "")
     (format "session=%s  manifest=%s"
             (or (plist-get response :session-id) "unknown")
             (or (plist-get response :manifest-id) "none")))))

(defun rai--submit (display-prompt wire-prompt)
  "Display DISPLAY-PROMPT and submit WIRE-PROMPT to RAI."
  (let ((chat-buffer (rai--chat-buffer)))
    (with-current-buffer chat-buffer
      (when (and rai--pending-request
                 (buffer-live-p rai--pending-request))
        (user-error "A RAI request is already in progress"))
      (rai--insert-chat chat-buffer "You" display-prompt)
      (let ((session rai-session-id)
            (reply-to rai--last-turn-id))
        (setq rai--pending-request
              (rai-transport-turn
               wire-prompt
               :session-id session
               :reply-to-turn-id reply-to
               :data-class rai-default-data-class
               :on-success (lambda (response)
                             (rai--request-success chat-buffer response))
               :on-error (lambda (error)
                           (rai--request-error chat-buffer error))))))
    (pop-to-buffer chat-buffer)))

;;;###autoload
(defun rai-chat (&optional new-session)
  "Open the RAI chat buffer.

With prefix argument NEW-SESSION, clear only the client-side conversation view
and ask the server to allocate a new session on the next request."
  (interactive "P")
  (let ((buffer (rai--chat-buffer)))
    (when new-session
      (with-current-buffer buffer
        (when (and rai--pending-request (buffer-live-p rai--pending-request))
          (user-error "Cancel the active request before starting a new session"))
        (let ((inhibit-read-only t))
          (erase-buffer))
        (setq rai-session-id nil
              rai--last-turn-id nil
              rai--last-manifest-id nil)))
    (pop-to-buffer buffer)))

;;;###autoload
(defun rai-ask (prompt)
  "Send PROMPT to the RAI Assistant without ambient editor content."
  (interactive (list (read-from-minibuffer "Ask RAI: ")))
  (when (string-empty-p (string-trim prompt))
    (user-error "Prompt must not be empty"))
  (rai--submit prompt prompt))

;;;###autoload
(defun rai-ask-with-context (prompt)
  "Send PROMPT with explicit bounded context from the current buffer."
  (interactive (list (read-from-minibuffer "Ask RAI with editor context: ")))
  (when (string-empty-p (string-trim prompt))
    (user-error "Prompt must not be empty"))
  ;; Build context before opening the chat buffer so the source is unambiguous.
  (let ((wire-prompt (rai-context-render-for-prompt prompt)))
    (rai--submit prompt wire-prompt)))

(defun rai-cancel-request ()
  "Cancel the current client-side HTTP request.

The server remains responsible for its own cancellation and terminal audit
semantics; killing the response buffer only disconnects this client."
  (interactive)
  (if (and rai--pending-request (buffer-live-p rai--pending-request))
      (progn
        (kill-buffer rai--pending-request)
        (setq rai--pending-request nil)
        (rai--insert-chat (current-buffer) "RAI" "Request cancelled locally."))
    (message "No RAI request is in progress")))

(defun rai--show-json (title payload)
  "Display decoded JSON PAYLOAD in a read-only buffer named TITLE."
  (let ((buffer (get-buffer-create title)))
    (with-current-buffer buffer
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert (json-encode payload))
        (condition-case nil
            (json-pretty-print-buffer)
          (json-parse-error nil))
        (goto-char (point-min))
        (special-mode)))
    (display-buffer buffer)))

(defun rai--show-error-message (error)
  "Show transport ERROR in the echo area."
  (message "RAI: %s" (rai-transport-error-message error)))

(defun rai-show-last-context ()
  "Display the exact server-side context package for the latest reply."
  (interactive)
  (unless rai--last-manifest-id
    (user-error "This chat has no context manifest yet"))
  (rai-transport-get-context
   rai--last-manifest-id
   (lambda (payload) (rai--show-json "*RAI Context*" payload))
   #'rai--show-error-message))

(defun rai-show-sessions ()
  "Display resumable RAI Assistant sessions."
  (interactive)
  (rai-transport-get
   'sessions
   (lambda (payload) (rai--show-json "*RAI Sessions*" payload))
   #'rai--show-error-message))

(defun rai-show-memories ()
  "Display active RAI Assistant memories."
  (interactive)
  (rai-transport-get
   'memories
   (lambda (payload) (rai--show-json "*RAI Memories*" payload))
   #'rai--show-error-message))

(defun rai-show-memory-operations ()
  "Display the RAI memory admission and mutation trace."
  (interactive)
  (rai-transport-get
   'memory-operations
   (lambda (payload) (rai--show-json "*RAI Memory Operations*" payload))
   #'rai--show-error-message))

(defun rai-memory-diagnostics ()
  "Display RAI Assistant memory projection diagnostics."
  (interactive)
  (rai-transport-get
   'memory-diagnostics
   (lambda (payload) (rai--show-json "*RAI Memory Diagnostics*" payload))
   #'rai--show-error-message))

(defun rai-ping ()
  "Check whether the local RAI daemon health endpoint responds."
  (interactive)
  (rai-transport-request
   "GET" "/health" :authenticated nil
   :on-success (lambda (_payload) (message "RAI daemon is available"))
   :on-error #'rai--show-error-message))

(provide 'rai)
;;; rai.el ends here
