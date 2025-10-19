;; Guile Scheme example client for the proposed IPC API v2.
;; This client demonstrates how to use the new `run` command.

;; Note: This script requires Guile modules for sockets and JSON.
;; - (ice-9 networking sockets) is standard.
;; - (json) is available in recent Guile versions. If not available,
;;   you may need to install a library like `guile-json`.

(use-modules (ice-9 networking sockets))
(use-modules (json))
(use-modules (srfi srfi-1))

(define socket-file "/tmp/rai-ipc.sock")

;; Low-level function to send a command and receive a response
(define (run-ipc-command payload)
  (let* ((sock (socket AF_UNIX SOCK_STREAM 0))
         (request-obj `(("request_id" . ,(symbol->string (uuid))) ; Guile doesn't have uuid built-in, this is a placeholder
                       ("command" . "run")
                       ("payload" . ,payload)))
         (request-str (string-append (json->string request-obj) "\n")))
    (let-values (((port-r port-w) (socket-ports sock))) 
      (connect sock AF_UNIX socket-file)
      (display request-str port-w)
      (force-output port-w)
      (let ((response-str (get-line port-r))) 
        (close-port port-r)
        (close-port port-w)
        (if (not (eof-object? response-str))
            (string->json response-str)
            '(("status" . "error") ("error_message" . "No response from server.")))))))

;; --- Agent Configuration Definitions (using association lists) ---

(define summarizer-config
  '(("agent_class" . "AgentAgno")
    ("backend" . "ollama")
    ("model" . "gemma3:4b") ; You can change this to any model you have installed
    ("system_prompt" . "You are an expert summarizer. Condense the given text into a few key points.")))

(define translator-config
  '(("agent_class" . "AgentAgno")
    ("backend" . "ollama")
    ("model" . "gemma3:4b") ; You can change this to any model you have installed
    ("system_prompt" . "You are a translator. Translate the given text into Polish.")))

;; --- Client-Side Function Definitions ---

(define (agent-summarizer text)
  (display "--- Calling Summarizer Agent ---\n")
  (let* ((payload `(("input" . ,text)
                   ("chain" . (,(list->vector (list summarizer-config))))))
         (response (run-ipc-command payload)))
    (let ((status (assoc-ref response "status")))
      (if (string=? status "success")
          (assoc-ref (assoc-ref response "payload") "content")
          (error "Summarizer agent failed: " (assoc-ref response "error_message"))))))

(define (agent-translator text)
  (display "--- Calling Translator Agent ---\n")
  (let* ((payload `(("input" . ,text)
                   ("chain" . (,(list->vector (list translator-config))))))
         (response (run-ipc-command payload)))
    (let ((status (assoc-ref response "status")))
      (if (string=? status "success")
          (assoc-ref (assoc-ref response "payload") "content")
          (error "Translator agent failed: " (assoc-ref response "error_message"))))))

(define (run-server-side-chain text)
  (display "\n--- Running Full Chain on Server-Side ---\n")
  (let* ((payload `(("input" . ,text)
                   ("chain" . ,(list->vector (list summarizer-config translator-config)))))
         (response (run-ipc-command payload)))
    (let ((status (assoc-ref response "status")))
      (if (string=? status "success")
          (assoc-ref (assoc-ref response "payload") "content")
          (error "Server-side chain failed: " (assoc-ref response "error_message"))))))

;; --- Main Execution ---

(define long-text
  "The new IPC architecture is based on a stateless `run` command. This command accepts a chain of agent configurations, allowing for flexible and dynamic execution of AI tasks. The server processes the chain sequentially, passing the output of one agent as the input to the next.")

(display (string-append "Initial text:\n" long-text "\n\n"))

;; --- Example 1: Client-side composition ---
(display "--- Example 1: Client-Side Composition (Two API calls) ---\n")
(let* ((summary (agent-summarizer long-text)))
  (display (string-append "Intermediate Summary:\n" summary "\n\n"))
  (let ((final-translation (agent-translator summary)))
    (display (string-append "Final Result (from client-side chain):\n" final-translation "\n"))))

;; --- Example 2: Server-side composition ---
(display "\n--- Example 2: Server-Side Composition (One API call) ---\n")
(let ((final-result-server (run-server-side-chain long-text)))
  (display (string-append "Final Result (from server-side chain):\n" final-result-server "\n")))
