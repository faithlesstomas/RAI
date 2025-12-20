(use-modules (web server)
             (web request)
             (web response)
             (web uri)
             (ice-9 binary-ports)
             (ice-9 match)
             (json)
             (rnrs bytevectors)
             (srfi srfi-1))

;; Disable output buffering for logs
(setvbuf (current-output-port) 'none)
(setvbuf (current-error-port) 'none)

(define (get-content-type filename)
  (cond
   ((string-suffix? ".html" filename) '(text/html))
   ((string-suffix? ".js" filename) '(application/javascript))
   ((string-suffix? ".wasm" filename) '(application/wasm))
   ((string-suffix? ".css" filename) '(text/css))
   (else '(application/octet-stream))))

(define (format-ip addr)
  (cond
   ((number? addr) (inet-ntop AF_INET addr))
   ((pair? addr) (format-ip (cdr addr))) ;; Handle (family . address) pair
   ((not addr) "-")
   (else (format #f "~a" addr))))

(define (get-log-level-value level-str)
  (let ((up (string-upcase (or level-str "INFO"))))
    (cond
     ((string=? up "DEBUG") 0)
     ((string=? up "INFO") 1)
     ((string=? up "WARNING") 2)
     ((string=? up "ERROR") 3)
     (else 1)))) ;; Default to INFO

(define *current-log-level* (get-log-level-value (getenv "LOG_LEVEL")))

(define (should-log? msg-level-str)
  (>= (get-log-level-value msg-level-str) *current-log-level*))

(define (log-transaction request response)
  (let* ((meta (request-meta request))
         (addr (assoc-ref meta 'remote-addr))
         (ip (format-ip addr))
         (method (request-method request))
         (path (uri-path (request-uri request)))
         (headers (request-headers request))
         (user-agent (or (assoc-ref headers 'user-agent) "-"))
         (code (response-code response))
         (time (strftime "%H:%M:%S" (localtime (current-time))))
         ;; Classify request level
         (is-api (string-prefix? "/api" path))
         (level (cond
                 ((>= code 500) "ERROR")
                 ((>= code 400) "WARNING")
                 (is-api "INFO")
                 (else "DEBUG"))))
    
    (when (should-log? level)
      (format #t "[~a] [SERVER] [~a] ~a - ~a ~a (~a) -> ~a~%" time level ip method path user-agent code))))

(define (dispatch-request request body)
  (let* ((path (uri-path (request-uri request)))
         (method (request-method request)))
    
    (cond
     ;; Log endpoint
     ((and (eq? method 'POST) (string=? path "/log"))
      (let* ((json-body (utf8->string body))
             (log-data (catch 'json-invalid
                              (lambda () (json-string->scm json-body))
                              (lambda _ #f))))
        (when log-data
          (let ((level (assoc-ref log-data "level"))
                (msg (assoc-ref log-data "message"))
                (time (strftime "%H:%M:%S" (localtime (current-time)))))
            (when (should-log? level)
              (format #t "[~a] [BROWSER] [~a] ~a~%" time (string-upcase level) msg)))))
      (values (build-response #:code 200) #vu8()))
     
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

(define (handler request body)
  (let ((path (uri-path (request-uri request))))
    (call-with-values (lambda () (dispatch-request request body))
      (lambda (response response-body)
        (unless (and (string=? path "/log") (not (should-log? "DEBUG")))
          (log-transaction request response))
        (values response response-body)))))

(define (shutdown-handler signum)
  (format #t "\nReceived signal ~a. Shutting down...\n" signum)
  (exit 0))

(sigaction SIGINT shutdown-handler)
(sigaction SIGTERM shutdown-handler)
(sigaction SIGHUP shutdown-handler)

(define (write-pid-file)
  (call-with-output-file "server.pid"
    (lambda (port)
      (display (getpid) port))))

(write-pid-file)

(display "Starting Guile Web Server on port 8080 (0.0.0.0)...\n")
(run-server handler 'http `(#:port 8080 #:addr ,(inet-pton AF_INET "0.0.0.0")))
