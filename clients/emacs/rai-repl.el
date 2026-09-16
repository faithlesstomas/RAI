;;; rai-repl.el --- Controlled Emacs introspection for Rich AI -*- lexical-binding: t; -*-

;; Copyright (C) 2026 RAI contributors

;; Author: RAI contributors
;; Version: 0.1.0
;; Package-Requires: ((emacs "28.1"))
;; Keywords: tools, ai, lisp
;; URL: https://gitlab.com/tk-lab1/ai/rai

;;; Commentary:

;; Emacs Lisp evaluation has all permissions of the Emacs process.  Therefore
;; this module does not expose an evaluator to the RAI server.  It offers a
;; local IELM command, safe documentation inspection, and an explicitly
;; enabled evaluator that asks for confirmation on every invocation.

;;; Code:

(require 'cl-lib)
(require 'ielm)
(require 'subr-x)
(require 'rai-transport)

(defgroup rai-repl nil
  "Controlled Emacs Lisp facilities for RAI users."
  :group 'rai
  :prefix "rai-elisp-")

(defcustom rai-elisp-evaluation-enabled nil
  "Whether `rai-elisp-eval' may evaluate an expression.

Keep this nil unless you understand that arbitrary Emacs Lisp can read files,
start processes, access credentials available to Emacs and modify editor
state.  Enabling it still requires confirmation for each expression."
  :type 'boolean
  :group 'rai-repl)

(defcustom rai-elisp-max-result-chars 4000
  "Maximum number of printed result characters retained by the evaluator."
  :type 'integer
  :group 'rai-repl)

(defvar rai-elisp-expression-history nil
  "History of expressions entered into `rai-elisp-eval'.")

(defun rai-open-elisp-repl ()
  "Open Emacs' local IELM REPL.

This does not grant RAI or a model access to the evaluator."
  (interactive)
  (ielm))

(defun rai-elisp--read-expression ()
  "Read one expression with reader evaluation disabled."
  (let* ((source (read-from-minibuffer
                  "Local Elisp expression: " nil nil nil
                  'rai-elisp-expression-history))
         (parsed (read-from-string source)))
    (unless (string-empty-p
             (string-trim (substring source (cdr parsed))))
      (user-error "Enter exactly one Emacs Lisp expression"))
    (car parsed)))

(defun rai-elisp--bounded-result (value)
  "Print VALUE without exceeding `rai-elisp-max-result-chars'."
  (let ((printed (prin1-to-string value)))
    (if (<= (length printed) rai-elisp-max-result-chars)
        printed
      (concat (substring printed 0 rai-elisp-max-result-chars)
              "\n… result truncated …"))))

(defun rai-elisp-evaluate-expression (expression confirm-function)
  "Evaluate EXPRESSION after calling CONFIRM-FUNCTION.

This lower-level function exists for testing and local integrations.  It never
sends the expression or result to RAI."
  (unless rai-elisp-evaluation-enabled
    (user-error "RAI Emacs Lisp evaluation is disabled"))
  (unless (funcall confirm-function expression)
    (user-error "Emacs Lisp evaluation cancelled"))
  (rai-elisp--bounded-result (eval expression t)))

(defun rai-elisp-eval (expression)
  "Evaluate local EXPRESSION after an explicit confirmation.

No server response can invoke this command.  The result is displayed locally
and is not automatically included in an Assistant request."
  (interactive (list (rai-elisp--read-expression)))
  (let ((result
         (rai-elisp-evaluate-expression
          expression
          (lambda (form)
            (yes-or-no-p (format "Evaluate locally in Emacs: %S? " form))))))
    (with-current-buffer (get-buffer-create "*RAI Elisp Result*")
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert result "\n")
        (goto-char (point-min))
        (special-mode))
      (display-buffer (current-buffer)))
    result))

(defun rai-emacs-describe-symbol (symbol)
  "Display documentation for Emacs Lisp SYMBOL without evaluating it."
  (interactive (list (intern (completing-read
                              "Describe Emacs symbol: " obarray
                              (lambda (candidate)
                                (or (boundp candidate)
                                    (fboundp candidate)))))))
  (cond
   ((fboundp symbol) (describe-function symbol))
   ((boundp symbol) (describe-variable symbol))
   (t (user-error "Unknown Emacs Lisp symbol: %s" symbol))))

(provide 'rai-repl)
;;; rai-repl.el ends here
