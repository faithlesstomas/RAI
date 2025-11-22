(import (scheme base)
        (scheme write)
        (hoot ffi)
        (rai-api)
        (dom))

;; JS Interop Helpers
(define-foreign js-keys "Object" "keys" (ref null extern) -> (ref null extern))
(define-foreign js-array-length "Array.prototype" "length" (ref null extern) -> i32)
(define-foreign js-array-ref "Reflect" "get" (ref null extern) i32 -> (ref null extern))
(define-foreign js-object-ref "Reflect" "get" (ref null extern) (ref string) -> (ref null extern))
(define-foreign js-string->string "String" "toString" (ref null extern) -> (ref string))

(define (js-array->list arr)
  (let ((len (js-array-length arr)))
    (let loop ((i 0) (acc '()))
      (if (< i len)
          (loop (+ i 1) (cons (js-array-ref arr i) acc))
          (reverse acc)))))

;; HTML Tag Helpers (SXML generators)
(define (tag name . children) `(,name ,@children))
(define (div . c) (apply tag 'div c))
(define (h1 . c) (apply tag 'h1 c))
(define (h2 . c) (apply tag 'h2 c))
(define (ul . c) (apply tag 'ul c))
(define (li . c) (apply tag 'li c))
(define (span . c) (apply tag 'span c))
(define (button attrs . c) `(button ,attrs ,@c))
(define (label attrs . c) `(label ,attrs ,@c))
(define (input attrs) `(input ,attrs))
(define (br) '(br))

;; Global State
(define *agents* '())

(define (refresh-ui)
  (render-app "app" app-root))

(define (fetch-agents!)
  (rai-api-get "/agents/"
               (lambda (data)
                 (let* ((keys (js-array->list (js-keys data)))
                        (agents (map (lambda (k)
                                       (cons k (js-object-ref data k)))
                                     keys)))
                   (set! *agents* agents)
                   (refresh-ui)))))

(define (delete-agent-action agent-id)
  (rai-api-delete (string-append "/agents/" agent-id)
                  (lambda (resp)
                    (fetch-agents!))))

(define (agent-display agent-pair)
  (let* ((agent-id (car agent-pair))
         (config (cdr agent-pair))
         (model (js-object-ref config "model")))
    (li
     (span agent-id " (" model ") ")
     (button `(@ (click ,(lambda (e) (delete-agent-action agent-id)))) "Delete"))))

(define (agent-list-component)
  (ul
   (map agent-display *agents*)))

(define (add-agent-form-component)
  (div
   (label '(@ (for "new-agent-id")) "Agent ID: ")
   (input '(@ (id "new-agent-id") (type "text") (placeholder "my-agent")))
   (br)
   (label '(@ (for "new-agent-model")) "Model: ")
   (input '(@ (id "new-agent-model") (type "text") (value "gemma2:latest")))
   (br)
   (label '(@ (for "new-agent-system")) "System Prompt: ")
   (input '(@ (id "new-agent-system") (type "text") (value "You are a helpful AI assistant.")))
   (br)
   (button
    `(@ (click ,(lambda (e)
                  (let* ((id-input (get-element-by-id "new-agent-id"))
                         (model-input (get-element-by-id "new-agent-model"))
                         (sys-input (get-element-by-id "new-agent-system"))
                         (new-id (element-value id-input))
                         (new-model (element-value model-input))
                         (new-system (element-value sys-input)))
                    (if (and (not (string=? new-id "")) (not (string=? new-model "")))
                        (begin
                          (rai-api-post (string-append "/agents/" new-id)
                                        `(("model" . ,new-model)
                                          ("system" . ,new-system))
                                        (lambda (resp)
                                          (set-element-value! id-input "")
                                          (fetch-agents!))))
                        #f)))))
    "Add Agent")))

(define (app-root)
  (div
   (h2 "Agent List")
   (agent-list-component)
   (h2 "Add New Agent")
   (add-agent-form-component)))

(define (main)
  (fetch-agents!))

(main)
