(define-module (rai-api)
  #:use-module (hoot ffi)
  #:export (rai-api-get rai-api-post rai-api-put rai-api-delete json-stringify make-js-object))

(define-foreign fetch "window" "fetch" (ref string) (ref null extern) -> (ref null extern))
(define-foreign response-json "window" "jsonHelper" (ref null extern) -> (ref null extern))
(define-foreign then "window" "thenHelper" (ref null extern) (ref null extern) -> (ref null extern))
(define-foreign json-stringify "JSON" "stringify" (ref null extern) -> (ref string))

(define-foreign new-object "window" "newObject" -> (ref null extern))
(define-foreign new-array "window" "newArray" -> (ref null extern))
(define-foreign set-property-string! "window" "setPropertyHelper" (ref null extern) (ref string) (ref string) -> none)
(define-foreign set-property-object! "window" "setPropertyHelper" (ref null extern) (ref string) (ref null extern) -> none)

(define (make-js-object alist)
  (let ((obj (new-object)))
    (for-each (lambda (pair)
                (let ((key (car pair))
                      (val (cdr pair)))
                  (cond
                   ((string? val) (set-property-string! obj key val))
                   ((null? val) (set-property-object! obj key (new-array)))
                   (else (set-property-object! obj key val)))))
              alist)
    obj))

(define-foreign get-hostname "window" "getHostnameHelper" -> (ref string))

(define (get-base-url)
  (string-append "http://" (get-hostname) ":8000/api/v1"))

(define (make-request method path body callback)
  (let* ((url (string-append (get-base-url) path))
         (headers (make-js-object '(("Content-Type" . "application/json"))))
         (opts-alist `(("method" . ,method)
                       ("headers" . ,headers)))
         (opts (make-js-object (if body
                                   (cons `("body" . ,body) opts-alist)
                                   opts-alist))))
    (then (fetch url opts)
          (procedure->external
           (lambda (response)
             (then (response-json response)
                   (procedure->external callback)))))))

(define (rai-api-get path callback)
  (make-request "GET" path #f callback))

(define (rai-api-post path json-alist callback)
  ;; Convert scheme alist to JS object then stringify
  (let ((js-obj (make-js-object json-alist)))
    (make-request "POST" path (json-stringify js-obj) callback)))

(define (rai-api-put path json-alist callback)
  (let ((js-obj (make-js-object json-alist)))
    (make-request "PUT" path (json-stringify js-obj) callback)))

(define (rai-api-delete path callback)
  (make-request "DELETE" path #f callback))
