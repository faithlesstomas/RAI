(use-modules (json)
             (ice-9 match)
             (ice-9 pretty-print)
             (ice-9 textual-ports)
             (ice-9 rdelim))

(define socket-file "/tmp/rai-ipc.sock")

;; Funkcja do wysyłania komend do serwera IPC
(define (run-ipc-command payload)
  (let* ((sock (socket PF_UNIX SOCK_STREAM 0))
         (addr (make-socket-address AF_UNIX socket-file))
         (request-obj `(("request_id" . "guile-client-repl")
                       ("command" . "run")
                       ("payload" . ,payload)))
         (request-str (string-append (scm->json-string request-obj) "\n")))
    (connect sock addr)
    (display request-str sock)
    (force-output sock)
    (let ((response-str (get-line sock)))
      (close-port sock)
      (if (not (eof-object? response-str))
          (json->scm (open-input-string response-str))
          '(("status" . "error") ("error_message" . "No response from server."))))))

;; Domyślna konfiguracja agenta
(define default-agent-config
  '(("agent_class" . "AgentAgno")
    ("backend" . "ollama")
    ("model" . "gemma3:4b")
    ("system_prompt" . "You are a helpful assistant.")))

;; Funkcja do obsługi komend REPL
(define (handle-repl-command line agent-config)
  (let* ((parts (string-split line #\space))
         (command (car parts))
         (args (cdr parts)))
    (cond
     ((string=? command ",h")
      (display "Available commands:\n")
      (display "  ,h                 Show this help message.\n")
      (display "  ,config            Show the current agent configuration.\n")
      (display "  ,set <key> <value> Set a configuration value.\n")
      (display "  ,rai <prompt>      Send a prompt to the RAI server.\n")
      (display "  ,q                 Exit the client.\n")
      agent-config)
     ((string=? command ",config")
      (pretty-print agent-config)
      agent-config)
     ((string=? command ",set")
      (if (< (length args) 2)
          (begin (display "Error: ,set requires a key and a value.\n") agent-config)
          (let* ((key (car args))
                 (value (string-join (cdr args) " "))
                 (new-pair (cons key value))
                 (existing-pair (assoc key agent-config)))
            (if existing-pair
                (map (lambda (p) (if (string=? (car p) key) new-pair p)) agent-config)
                (cons new-pair agent-config)))))
     ((string=? command ",rai")
      (if (null? args)
          (begin (display "Error: ,rai requires a prompt.\n") agent-config)
          (begin
            (let* ((input-prompt (string-join args " "))
                   (payload `(("input" . ,input-prompt)
                             ("chain" . ,(list->vector (list agent-config)))))
                   (response (run-ipc-command payload)))
              (let ((status (assoc-ref response "status")))
                (if (string=? status "success")
                    (display (string-append (assoc-ref (assoc-ref response "payload") "content") "\n"))
                    (display (string-append "Error: " (assoc-ref response "error_message") "\n")))))
            agent-config)))
     (else
      (display (string-append "Unknown command: " command "\n"))
      agent-config))))

;; Główna pętla REPL
(define (main-repl)
  (display "Welcome to RAI Guile Client!\n")
  (display "Type ,h for a list of commands.\n")
  (let loop ((agent-config default-agent-config))
    (display "rai> ")
    (force-output)
    (let ((input (read-line)))
      (cond
       ((eof-object? input) (newline) (display "Exiting.\n"))
       ((string=? input ",q") (display "Exiting.\n"))
       ((string-prefix? "," input)
        (loop (handle-repl-command input agent-config)))
       (else
        (begin ; New begin block
         ;; Evaluate Scheme expression
         (catch #t
           (lambda ()
             (let* ((expr (read (open-input-string input)))
                   (result (primitive-eval expr (current-module))))
               (if (not (eq? result (void)))
                   (begin (pretty-print result) (newline))))) ; Display result if not void
           (lambda (key . args)
             (display (string-append "Error evaluating Scheme: " (format #f "~a" key) "\n")))))
         (loop agent-config))))))

;; Uruchomienie pętli
(main-repl)
