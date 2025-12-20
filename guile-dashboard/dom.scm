(define-module (dom)
  #:use-module (hoot ffi)
  #:use-module (ice-9 match)
  #:export (sxml->dom render-app get-element-by-id element-value set-element-value!))

(define-foreign make-element "window" "createElementHelper" (ref string) -> (ref null extern))
(define-foreign make-text-node "window" "createTextNodeHelper" (ref string) -> (ref null extern))
(define-foreign append-child! "window" "appendChildHelper" (ref null extern) (ref null extern) -> (ref null extern))
(define-foreign set-attribute! "window" "setAttributeHelper" (ref null extern) (ref string) (ref string) -> none)
(define-foreign get-element-by-id "window" "getElementByIdHelper" (ref string) -> (ref null extern))
;; replaceChildren is variadic in JS, but Hoot FFI might need explicit handling or a wrapper.
;; For simplicity, let's clear and append.
(define-foreign set-inner-html! "window" "setInnerHTML" (ref null extern) (ref string) -> none)
(define-foreign add-event-listener! "window" "addEventListenerHelper" (ref null extern) (ref string) (ref null extern) -> none)
(define-foreign element-value "window" "getElementValue" (ref null extern) -> (ref string))
(define-foreign set-element-value! "window" "setElementValue" (ref null extern) (ref string) -> none)

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
           ('() elem)
           ((('@ . attrs) . rest)
            (for-each (lambda (attr)
                        (match attr
                          ((name value)
                           (cond
                            ((eq? name 'click) (add-event-listener! elem "click" (procedure->external value)))
                            ((eq? name 'onclick) (add-event-listener! elem "click" (procedure->external value)))
                            (else (set-attribute! elem (symbol->string name) (if (string? value) value (format #f "~a" value))))))))
                      attrs)
            (loop rest))
           ((child . rest)
            (unless (null? child)
              (append-child! elem (sxml->dom child)))
            (loop rest))))
       elem))))

(define (render-app root-id sxml)
  (let ((root (get-element-by-id root-id)))
    (clear-children! root)
    (append-child! root (sxml->dom sxml))))
