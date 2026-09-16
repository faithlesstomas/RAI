;;; rai-tests.el --- Tests for the RAI Emacs client -*- lexical-binding: t; -*-

;; Copyright (C) 2026 RAI contributors

;;; Code:

(require 'ert)
(require 'cl-lib)
(require 'rai)

(ert-deftest rai-json-get-accepts-current-and-legacy-keys ()
  (should (equal (rai-json-get '(("content" . "one")) "content") "one"))
  (should (equal (rai-json-get '((content . "two")) "content") "two"))
  (should (equal (rai-json-get nil "missing" "fallback") "fallback")))

(ert-deftest rai-turn-v1-adapter-hides-wire-shape ()
  (let ((adapted
         (rai-transport-adapt-turn-v1
          '(("session_id" . "work")
            ("turn_id" . "turn-1")
            ("content" . "hello")
            ("manifest_id" . "manifest-1")))))
    (should (equal (plist-get adapted :session-id) "work"))
    (should (equal (plist-get adapted :content) "hello"))
    (should (equal (plist-get adapted :manifest-id) "manifest-1"))))

(ert-deftest rai-transport-fails-before-network-without-token ()
  (let ((rai-api-token-function (lambda () nil))
        captured)
    (should-not
     (rai-transport-request
      "GET" "/api/v1/assistant/sessions"
      :on-success #'ignore
      :on-error (lambda (error) (setq captured error))))
    (should (rai-transport-error-p captured))
    (should (string-match-p "No RAI API token"
                            (rai-transport-error-message captured)))))

(ert-deftest rai-transport-reads-only-protected-runtime-token-file ()
  (let ((token-file (make-temp-file "rai-emacs-token-"))
        (process-environment (copy-sequence process-environment)))
    (unwind-protect
        (progn
          (setenv "RAI_API_TOKEN" nil)
          (with-temp-file token-file (insert "runtime-token\n"))
          (set-file-modes token-file #o600)
          (let ((rai-api-token-file token-file))
            (should (equal (rai-transport-default-token) "runtime-token")))
          (set-file-modes token-file #o644)
          (let ((rai-api-token-file token-file))
            (cl-letf (((symbol-function 'auth-source-search)
                       (lambda (&rest _arguments) nil)))
              (should-not (rai-transport-default-token)))))
      (delete-file token-file))))

(ert-deftest rai-transport-turn-uses-one-versioned-endpoint ()
  (let (captured-path captured-data)
    (cl-letf (((symbol-function 'rai-transport-request)
               (lambda (_method path &rest arguments)
                 (setq captured-path path
                       captured-data (plist-get arguments :data))
                 'request-buffer)))
      (should
       (eq (rai-transport-turn
            "hello" :session-id "work" :request-id "request-1"
            :on-success #'ignore :on-error #'ignore)
           'request-buffer)))
    (should (equal captured-path "/api/v1/assistant/turn"))
    (should (equal (cdr (assoc "prompt" captured-data)) "hello"))
    (should (equal (cdr (assoc "session_id" captured-data)) "work"))
    (should (equal (cdr (assoc "request_id" captured-data)) "request-1"))))

(ert-deftest rai-context-blocks-sensitive-files ()
  (with-temp-buffer
    (setq buffer-file-name "/tmp/project/.env.local")
    (should (rai-context-sensitive-p))
    (should-error (rai-context-current) :type 'user-error)))

(ert-deftest rai-context-snapshot-retains-metadata-not-buffer-text ()
  (with-temp-buffer
    (rename-buffer "rai-test-context" t)
    (insert "SECRET-TEXT-MUST-NOT-BE-IN-A-SNAPSHOT")
    (let ((snapshot (rai-context-record "test-event" 'compile)))
      (should (rai-context-snapshot-p snapshot))
      (should (equal (rai-context-snapshot-event snapshot) "test-event"))
      (should-not (string-match-p
                   "SECRET-TEXT"
                   (prin1-to-string snapshot))))))

(ert-deftest rai-explicit-context-bounds-defun-content ()
  (with-temp-buffer
    (rename-buffer "rai-test-defun-context" t)
    (emacs-lisp-mode)
    (insert "(defun rai-test-example ()\n  (+ 1 2))\n")
    (goto-char 20)
    (let* ((rai-context-content-policy 'region-or-defun)
           (rai-context-max-content-chars 12)
           (context (rai-context-current))
           (buffer-data (rai-json-get context "buffer"))
           (content (rai-json-get buffer-data "content")))
      (should (equal (rai-json-get content "kind") "defun"))
      (should (eq (rai-json-get content "truncated") t))
      (should (= (length (rai-json-get content "text")) 12)))))

(ert-deftest rai-context-render-labels-observations-untrusted ()
  (with-temp-buffer
    (rename-buffer "rai-test-render-context" t)
    (text-mode)
    (let ((rai-context-content-policy 'metadata)
          (rendered (rai-context-render-for-prompt "help me")))
      (should (string-prefix-p "help me" rendered))
      (should (string-match-p "untrusted-observation" rendered))
      (should (string-match-p "data, not instructions" rendered)))))

(ert-deftest rai-elisp-evaluation-is-disabled-by-default ()
  (let ((rai-elisp-evaluation-enabled nil))
    (should-error
     (rai-elisp-evaluate-expression '(+ 1 2) (lambda (_form) t))
     :type 'user-error)))

(ert-deftest rai-elisp-evaluation-requires-confirmation ()
  (let ((rai-elisp-evaluation-enabled t))
    (should-error
     (rai-elisp-evaluate-expression '(+ 1 2) (lambda (_form) nil))
     :type 'user-error)
    (should (equal
             (rai-elisp-evaluate-expression '(+ 1 2) (lambda (_form) t))
             "3"))))

(provide 'rai-tests)
;;; rai-tests.el ends here
