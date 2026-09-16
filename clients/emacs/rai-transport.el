;;; rai-transport.el --- HTTP transport for Rich AI -*- lexical-binding: t; -*-

;; Copyright (C) 2026 RAI contributors

;; Author: RAI contributors
;; Version: 0.1.0
;; Package-Requires: ((emacs "28.1"))
;; Keywords: tools, ai
;; URL: https://gitlab.com/tk-lab1/ai/rai

;;; Commentary:

;; This file is the deliberately small compatibility boundary between Emacs
;; and the evolving RAI Assistant HTTP API.  UI code should call the public
;; functions in this file instead of depending on endpoint paths or wire keys.

;;; Code:

(require 'auth-source)
(require 'cl-lib)
(require 'json)
(require 'subr-x)
(require 'url)
(require 'url-http)

(defvar url-http-response-status)

(defgroup rai nil
  "A local-first Rich AI client."
  :group 'tools
  :prefix "rai-")

(defcustom rai-base-url "http://127.0.0.1:8000"
  "Base URL of the local RAI daemon.

The value must not contain credentials."
  :type 'string
  :group 'rai)

(defcustom rai-auth-source-host "rai-local"
  "Host used to find the RAI API token with `auth-source-search'."
  :type 'string
  :group 'rai)

(defcustom rai-auth-source-user "rai"
  "User used to find the RAI API token with `auth-source-search'."
  :type 'string
  :group 'rai)

(defcustom rai-api-token-file
  (let* ((runtime-home (getenv "XDG_RUNTIME_DIR"))
         (cache-home (or (getenv "XDG_CACHE_HOME")
                         (expand-file-name "~/.cache"))))
    (if runtime-home
        (expand-file-name "rai/api-token" runtime-home)
      (expand-file-name "rai/run/api-token" cache-home)))
  "Path of the protected token file created by the local RAI daemon."
  :type 'file
  :group 'rai)

(defcustom rai-api-token-function #'rai-transport-default-token
  "Function returning the RAI API token, or nil when unavailable.

The default checks RAI_API_TOKEN, RAI's protected runtime token file and then
`auth-source'.  The token is sent in an X-RAI-Token header and is never put in
a URL."
  :type 'function
  :group 'rai)

(defcustom rai-api-endpoints
  '((turn . "/api/v1/assistant/turn")
    (sessions . "/api/v1/assistant/sessions")
    (memories . "/api/v1/assistant/memories")
    (memory-operations . "/api/v1/assistant/memory-operations")
    (memory-diagnostics . "/api/v1/assistant/diagnostics/memory"))
  "Named RAI Assistant endpoints used by the client.

Keeping paths here isolates the rest of the package from API migrations."
  :type '(alist :key-type symbol :value-type string)
  :group 'rai)

(defcustom rai-turn-response-adapter #'rai-transport-adapt-turn-v1
  "Function converting a successful turn response to a stable plist.

The function receives decoded JSON and should return at least :content.  This
is the intended compatibility seam while the Assistant API is evolving."
  :type 'function
  :group 'rai)

(cl-defstruct (rai-transport-error
               (:constructor rai-transport-error-create))
  "A normalized transport or HTTP failure."
  status
  message
  payload)

(defun rai-json-get (object key &optional default)
  "Get KEY from decoded JSON OBJECT, returning DEFAULT when absent.

Both string and symbol alist keys are accepted so this also works with clients
that bind the legacy `json-key-type'."
  (let* ((string-key (if (symbolp key) (symbol-name key) key))
         (symbol-key (if (symbolp key) key (intern-soft key)))
         (string-cell (and (listp object) (assoc string-key object)))
         (symbol-cell (and symbol-key (listp object) (assq symbol-key object))))
    (cond (string-cell (cdr string-cell))
          (symbol-cell (cdr symbol-cell))
          (t default))))

(defun rai-transport-default-token ()
  "Return the configured local RAI API token without logging it."
  (let* ((environment-token (getenv "RAI_API_TOKEN"))
         (file-token
          (when (and rai-api-token-file
                     (file-regular-p rai-api-token-file)
                     (file-readable-p rai-api-token-file)
                     (let ((modes (file-modes rai-api-token-file)))
                       (and modes (zerop (logand modes #o077)))))
            (string-trim
             (with-temp-buffer
               (insert-file-contents-literally rai-api-token-file)
               (buffer-string))))))
    (if (not (string-empty-p (or environment-token "")))
        environment-token
      (if (not (string-empty-p (or file-token "")))
          file-token
        (let* ((entry (car (auth-source-search
                            :host rai-auth-source-host
                            :user rai-auth-source-user
                            :max 1
                            :require '(:secret))))
               (secret (plist-get entry :secret))
               (token (cond ((functionp secret) (funcall secret))
                            ((stringp secret) secret))))
          (unless (string-empty-p (or token "")) token))))))

(defun rai-transport-endpoint (name)
  "Return the configured path for endpoint NAME."
  (or (alist-get name rai-api-endpoints)
      (error "Unknown RAI endpoint: %s" name)))

(defun rai-transport--url (path)
  "Join `rai-base-url' and PATH."
  (concat (string-remove-suffix "/" rai-base-url)
          (if (string-prefix-p "/" path) path (concat "/" path))))

(defun rai-transport--request-id ()
  "Create an opaque request identifier suitable for retries."
  (format "emacs-%x-%x" (floor (* 1000000 (float-time))) (random most-positive-fixnum)))

(defun rai-transport--json-body ()
  "Decode the JSON body in the current URL response buffer."
  (goto-char (or (and (boundp 'url-http-end-of-headers)
                      url-http-end-of-headers)
                 (point-min)))
  (skip-chars-forward "\r\n")
  (if (eobp)
      nil
    (condition-case nil
        (json-parse-buffer :object-type 'alist
                           :array-type 'list
                           :null-object nil
                           :false-object nil)
      (json-parse-error
       (buffer-substring-no-properties (point) (point-max))))))

(defun rai-transport--error-message (status payload)
  "Build a safe message for HTTP STATUS and decoded PAYLOAD."
  (let* ((detail (rai-json-get payload "detail"))
         (message (or (rai-json-get detail "message")
                      (and (stringp detail) detail)
                      (rai-json-get payload "message"))))
    (or message (format "RAI request failed with HTTP status %s" status))))

(defun rai-transport--handle-response (url-status on-success on-error)
  "Handle URL-STATUS and call ON-SUCCESS or ON-ERROR.

This function runs in the response buffer created by `url-retrieve'."
  (let ((response-buffer (current-buffer)))
    (unwind-protect
        (let ((transport-failure (plist-get url-status :error)))
          (if transport-failure
              (funcall on-error
                       (rai-transport-error-create
                        :status nil
                        :message (format "RAI transport failure: %s"
                                         transport-failure)
                        :payload nil))
            (let* ((status (or url-http-response-status 0))
                   (payload (rai-transport--json-body)))
              (if (and (>= status 200) (< status 300))
                  (funcall on-success payload)
                (funcall on-error
                         (rai-transport-error-create
                          :status status
                          :message (rai-transport--error-message status payload)
                          :payload payload))))))
      (when (buffer-live-p response-buffer)
        (kill-buffer response-buffer)))))

(cl-defun rai-transport-request
    (method path &key data on-success on-error (authenticated t))
  "Send an asynchronous METHOD request to PATH.

DATA is JSON-encoded when non-nil.  ON-SUCCESS receives decoded JSON and
ON-ERROR receives a `rai-transport-error'.  Return the URL response buffer, or
nil when the request could not be started."
  (let ((token (and authenticated (funcall rai-api-token-function))))
    (if (and authenticated (string-empty-p (or token "")))
        (progn
          (funcall on-error
                   (rai-transport-error-create
                    :status nil
                    :message (concat "No RAI API token. Set RAI_API_TOKEN or "
                                     "add an auth-source entry for "
                                     rai-auth-source-user "@" rai-auth-source-host)
                    :payload nil))
          nil)
      (let ((url-request-method method)
            (url-request-data (and data (encode-coding-string
                                         (json-encode data) 'utf-8)))
            (url-request-extra-headers
             (append '(("Accept" . "application/json")
                       ("Content-Type" . "application/json; charset=utf-8"))
                     (when token `(("X-RAI-Token" . ,token))))))
        (url-retrieve (rai-transport--url path)
                      #'rai-transport--handle-response
                      (list on-success on-error)
                      t t)))))

(cl-defun rai-transport-turn
    (prompt &key session-id reply-to-turn-id request-id
            (data-class "LOCAL") on-success on-error)
  "Submit PROMPT as one bounded Assistant turn.

The callback receives a normalized plist rather than the wire representation."
  (let ((payload `(("prompt" . ,prompt)
                   ("request_id" . ,(or request-id
                                         (rai-transport--request-id)))
                   ("data_class" . ,data-class))))
    (when session-id
      (setq payload (append payload `(("session_id" . ,session-id)))))
    (when reply-to-turn-id
      (setq payload (append payload
                            `(("reply_to_turn_id" . ,reply-to-turn-id)))))
    (rai-transport-request
     "POST" (rai-transport-endpoint 'turn)
     :data payload
     :on-success (lambda (response)
                   (funcall on-success
                            (funcall rai-turn-response-adapter response)))
     :on-error on-error)))

(defun rai-transport-adapt-turn-v1 (response)
  "Normalize a current v1 Assistant turn RESPONSE."
  (list :session-id (rai-json-get response "session_id")
        :request-id (rai-json-get response "request_id")
        :turn-id (rai-json-get response "turn_id")
        :content (rai-json-get response "content" "")
        :status (rai-json-get response "status")
        :manifest-id (rai-json-get response "manifest_id")
        :admitted-memory-ids (rai-json-get response "admitted_memory_ids")
        :memory-operation-ids (rai-json-get response "memory_operation_ids")
        :wire-response response))

(defun rai-transport-get (endpoint on-success on-error)
  "GET named ENDPOINT and call ON-SUCCESS or ON-ERROR."
  (rai-transport-request "GET" (rai-transport-endpoint endpoint)
                         :on-success on-success :on-error on-error))

(defun rai-transport-get-context (manifest-id on-success on-error)
  "Fetch the exact context package for MANIFEST-ID."
  (rai-transport-request
   "GET"
   (format "/api/v1/assistant/contexts/%s"
           (url-hexify-string manifest-id))
   :on-success on-success :on-error on-error))

(provide 'rai-transport)
;;; rai-transport.el ends here
