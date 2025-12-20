;;; rai-client.el --- An Emacs Lisp client for the RAI IPC server -*- lexical-binding: t -*-

;;; Commentary:

;;; This file provides a simple Emacs Lisp client to interact with the
;;; RAI IPC server.

;;; Usage Instructions:
;;; 1. Run the server in a terminal: `rai serve-ipc`
;;; 2. Load this file in Emacs: `M-x load-file` and select `rai-client.el`
;;; 3. Run the client: `M-x rai-chat`
;;; 4. Type your prompt and press Enter.
;;; 5. The response will appear in the `*rai-chat*` buffer.

;;; Code:

(require 'json)
(require 'subr-x)

(defconst rai-ipc-socket-path "/tmp/rai-ipc.sock"
  "The path to the RAI IPC Unix socket.")

(defun rai--send-and-receive (prompt)
  "Construct a full JSON payload and send it to the IPC server.

This function takes a PROMPT, builds the entire request object,
including metadata and the agent chain, before sending it to the
server.  It then waits for and returns the raw string response."
  (with-temp-buffer
    (let* ((proc-name "rai-ipc-client")
           (payload `((request_id . "emacs-client-request")
                       (command . "run")
                       (payload . ((input . ,prompt)
                                   ;; Use a simple agent configuration as default.
                                   (chain . [((agent_class . "AgentAgno")
                                              (backend . "ollama")
                                              (model . "gemma3:4b")
                                              (system_prompt . "You are a helpful assistant."))]
                                             )))))
           (request-str (concat (json-encode payload) "
"))
           (process (make-network-process
                     :name proc-name
                     :server nil
                     :family 'local
                     :service rai-ipc-socket-path
                     :sentinel (lambda (proc _msg) (kill-buffer (process-buffer proc)))
                     :filter (lambda (proc string) (setq rai--response-accumulator (concat rai--response-accumulator string)))
                     :buffer (current-buffer))))
      (setq rai--response-accumulator "")
      (unwind-protect
          (progn
            (process-send-string process request-str)
            (process-send-eof process)
            ;; Wait for the process to finish (with a 30s timeout).
            (let ((start-time (float-time)))
              (while (and (process-live-p process) (< (- (float-time) start-time) 30))
                (sit-for 0.1)))
            rai--response-accumulator)
        (when (process-live-p process)
          (delete-process process))))))(defun rai-chat (prompt)
  "Send a PROMPT to the RAI server and display the result in the *rai-chat* buffer."
  (interactive "sPrompt: ")
  (let* ((raw-response (rai--send-and-receive prompt))
         (json-response (json-read-from-string raw-response))
         ;; Extract the actual content from the JSON response.
         (content (cdr (assoc 'content (cdr (assoc 'payload json-response))))))
    (with-current-buffer (get-buffer-create "*rai-chat*")
      (goto-char (point-max))
      (unless (eq (point-min) (point-max))
        (insert "

"))
      (insert (format ">> %s

" prompt))
      (insert content)
      (pop-to-buffer (current-buffer)))))

(provide 'rai-client)
;;; rai-client.el ends here
