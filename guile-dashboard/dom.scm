(define-module (dom)
  #:use-module (hoot ffi)
  #:use-module (ice-9 match)
  #:export (sxml->dom render-app get-element-by-id element-value set-element-value!))

(define-foreign make-element "document" "createElement" (ref string) -> (ref null extern))
(define-foreign make-text-node "document" "createTextNode" (ref string) -> (ref null extern))
(define-foreign append-child! "Node.prototype" "appendChild" (ref null extern) (ref null extern) -> (ref null extern))
(define-foreign set-attribute! "Element.prototype" "setAttribute" (ref null extern) (ref string) (ref string) -> none)
(define-foreign get-element-by-id "document" "getElementById" (ref string) -> (ref null extern))
;; replaceChildren is variadic in JS, but Hoot FFI might need explicit handling or a wrapper.
;; For simplicity, let's clear and append.
(define-foreign set-inner-html! "Element.prototype" "innerHTML" (ref null extern) (ref string) -> none)
(define-foreign add-event-listener! "EventTarget.prototype" "addEventListener" (ref null extern) (ref string) (ref null extern) -> none)
(define-foreign element-value "HTMLInputElement.prototype" "value" (ref null extern) -> (ref string))
(define-foreign set-element-value! "HTMLInputElement.prototype" "value" (ref null extern) (ref string) -> none)
(define-foreign event-target "Event.prototype" "target" (ref null extern) -> (ref null extern))

;; Helper to clear children
(define (clear-children! elem)
  (set-inner-html! elem ""))

(define (sxml->dom exp)
  (match exp
    ((? string? str) (make-text-node str))
    ((? number? num) (make-text-node (number->string num)))
    (((? symbol? tag) . body)
     (let ((elem (make-element (symbol->string tag))))
       (let loop ((nodes body))
         (match nodes
           (() elem)
           ((('@ . attrs) . rest)
            (for-each (lambda (attr)
                        (match attr
                          ((name value)
                           (cond
                            ((eq? name 'click) (add-event-listener! elem "click" value))
                            ((eq? name 'onclick) (add-event-listener! elem "click" value))
                            (else (set-attribute! elem (symbol->string name) (if (string? value) value (format #f "~a" value))))))))
                      attrs)
            (loop rest))
           ((child . rest)
            (append-child! elem (sxml->dom child))
            (loop rest))))
       elem))))

(define (render-app root-id sxml)
  (let ((root (get-element-by-id root-id)))
    (clear-children! root)
    (append-child! root (sxml->dom sxml))))
