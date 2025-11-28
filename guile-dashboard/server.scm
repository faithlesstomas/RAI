(use-modules (web server)
             (web request)
             (web response)
             (web uri)
             (ice-9 binary-ports)
             (ice-9 match)
             (rnrs bytevectors)
             (srfi srfi-1))

(define (get-content-type filename)
  (cond
   ((string-suffix? ".html" filename) '(text/html))
   ((string-suffix? ".js" filename) '(application/javascript))
   ((string-suffix? ".wasm" filename) '(application/wasm))
   ((string-suffix? ".css" filename) '(text/css))
   (else '(application/octet-stream))))

(define (handler request body)
  (let* ((path (uri-path (request-uri request)))
         (method (request-method request)))
    
    (cond
     ;; Log endpoint
     ((and (eq? method 'POST) (string=? path "/log"))
      (let ((log-msg (utf8->string body)))
        (format #t "\x1b[33m[BROWSER]\x1b[0m ~a~%" log-msg)
        (values (build-response #:code 200) #vu8())))
     
     ;; Static files
     ((eq? method 'GET)
      (let* ((clean-path (if (string=? path "/") "index.html" (substring path 1)))
             ;; Basic security: prevent .. in path
             (safe-path (if (string-contains clean-path "..") #f clean-path)))
        
        (if (and safe-path (file-exists? safe-path) (not (file-is-directory? safe-path)))
            (let ((content (call-with-input-file safe-path get-bytevector-all)))
              (values (build-response #:code 200
                                      #:headers `((content-type . ,(get-content-type safe-path))))
                      content))
            (begin
              (format #t "404: ~a~%" path)
              (values (build-response #:code 404) "Not Found")))))
     
     (else
      (values (build-response #:code 405) "Method Not Allowed")))))

(display "Starting Guile Web Server on port 8080...\n")
(run-server handler 'http '(#:port 8080))
