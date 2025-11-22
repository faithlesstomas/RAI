(define-module (rai-api)
  #:use-module (hoot ffi)
  #:export (rai-api-get rai-api-post rai-api-delete))

(define-foreign fetch "window" "fetch" (ref string) (ref null extern) -> (ref null extern))
(define-foreign response-json "Response.prototype" "json" (ref null extern) -> (ref null extern))
(define-foreign then "Promise.prototype" "then" (ref null extern) (ref null extern) -> (ref null extern))
(define-foreign json-stringify "JSON" "stringify" (ref null extern) -> (ref string))

(define-foreign new-object "Object" "new" -> (ref null extern))
(define-foreign set-property! "Reflect" "set" (ref null extern) (ref string) (ref null extern) -> i32)

(define (make-js-object alist)
  (let ((obj (new-object)))
    (for-each (lambda (pair)
                (set-property! obj (car pair) (cdr pair)))
              alist)
    obj))

(define *base-url* "http://127.0.0.1:8000/api/v1")

(define (make-request method path body callback)
  (let* ((url (string-append *base-url* path))
         (headers (make-js-object '(("Content-Type" . "application/json"))))
         (opts-alist `(("method" . ,method)
                       ("headers" . ,headers)))
         (opts (make-js-object (if body
                                   (cons `("body" . ,body) opts-alist)
                                   opts-alist))))
    (then (fetch url opts)
          (lambda (response)
            (then (response-json response)
                  callback)))))

(define (rai-api-get path callback)
  (make-request "GET" path #f callback))

(define (rai-api-post path json-alist callback)
  ;; Convert scheme alist to JS object then stringify
  (let ((js-obj (make-js-object json-alist)))
    (make-request "POST" path (json-stringify js-obj) callback)))

(define (rai-api-delete path callback)
  (make-request "DELETE" path #f callback))
