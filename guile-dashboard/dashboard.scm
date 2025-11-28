(import (scheme base)
        (scheme write)
        (hoot ffi)
        (rai-api)
        (dom))

;; JS Interop Helpers
(define-foreign js-keys "window" "getKeysHelper" (ref null extern) -> (ref null extern))
(define-foreign js-reflect-get "Reflect" "get" (ref null extern) (ref string) -> (ref null extern))

(define (js-array-length arr)
  (let ((len-obj (js-reflect-get arr "length")))
    ;; We need to convert the JS number object to a Scheme integer.
    ;; Hoot might handle this automatically for i32 return types if it was a direct call,
    ;; but here we get an extern ref.
    ;; Let's assume for now we can cast or use a helper.
    ;; Actually, let's try to define a foreign function that does the property access and return i32 directly if possible,
    ;; OR use a JS helper.
    ;; Simpler: Use a JS helper for array length to avoid complex casting in Scheme for now.
    (js-array-length-helper arr)))

(define-foreign js-array-length-helper "window" "getArrayLength" (ref null extern) -> i32)
(define-foreign js-array-ref "window" "getArrayElementHelper" (ref null extern) i32 -> (ref null extern))
(define-foreign js-object-ref "window" "getPropertyHelper" (ref null extern) (ref string) -> (ref null extern))
(define-foreign js-string->string "window" "String" (ref null extern) -> (ref string))
(define-foreign js-alert "window" "alertHelper" (ref string) -> none)
(define-foreign js-response-ok? "window" "responseOkHelper" (ref null extern) -> i32)

(define (js-array->list arr)
  (let ((len (js-array-length arr)))
    (let loop ((i 0) (acc '()))
      (if (< i len)
          (loop (+ i 1) (cons (js-array-ref arr i) acc))
          (reverse acc)))))

;; HTML Tag Helpers (SXML generators)
(define (tag name . children) `(,name ,@children))
(define (div attrs . c) `(div ,attrs ,@c))
(define (h1 . c) (apply tag 'h1 c))
(define (h2 . c) (apply tag 'h2 c))
(define (h3 . c) (apply tag 'h3 c))
(define (ul . c) (apply tag 'ul c))
(define (li . c) (apply tag 'li c))
(define (span . c) (apply tag 'span c))
(define (button attrs . c) `(button ,attrs ,@c))
(define (label attrs . c) `(label ,attrs ,@c))
(define (input attrs) `(input ,attrs))
(define (textarea attrs . c) `(textarea ,attrs ,@c))
(define (br) '(br))

;; Global State
(define *current-view* 'agents)
(define *agents* '())
(define *selected-agent* #f)
(define *sidebar-open* #f)
(define *is-editing* #f)
(define *is-saving* #f)

(define-foreign fetch "window" "fetch" (ref string) (ref null extern) -> (ref null extern))

(define (log-message level msg)
  (let* ((payload (make-js-object `(("level" . ,level)
                                    ("message" . ,msg))))
         (body (json-stringify payload))
         (opts (make-js-object `(("method" . "POST")
                                 ("body" . ,body)))))
    (fetch "/log" opts)))

(define (log-info msg) (log-message "info" msg))
(define (log-warning msg) (log-message "warning" msg))
(define (log-error msg) (log-message "error" msg))
(define (log-debug msg) (log-message "debug" msg))

(define (refresh-ui)
  (log-debug (string-append "UI Refresh. View: " (symbol->string *current-view*)))
  (render-app "app" (app-root)))

(define (toggle-sidebar!)
  (set! *sidebar-open* (not *sidebar-open*))
  (refresh-ui))

(define (close-sidebar!)
  (set! *sidebar-open* #f)
  (refresh-ui))

(define (set-view! view)
  (set! *current-view* view)
  (set! *sidebar-open* #f) ;; Close on view change
  (set! *is-editing* #f)   ;; Reset editing state
  (refresh-ui))

(define (get-view-title)
  (cond
    ((eq? *current-view* 'agents) "RAI - Agents")
    ((eq? *current-view* 'chat) "RAI - Playground")
    ((eq? *current-view* 'history) "RAI - History")
    (else "RAI Assistant")))

;; --- API Calls ---

(define (fetch-agents!)
  (log-info "Fetching agents list...")
  (rai-api-get "/agents/"
               (lambda (data)
                 (let* ((keys (js-array->list (js-keys data)))
                        (agents (map (lambda (k)
                                       (let ((k-str (js-string->string k)))
                                         (cons k-str (js-object-ref data k-str))))
                                     keys)))
                   (set! *agents* agents)
                   (refresh-ui)))))

(define-foreign stop-propagation "window" "stopPropagationHelper" (ref null extern) -> none)

(define (delete-agent-action agent-id e)
  (log-info (string-append "Action: Delete agent " agent-id))
  (stop-propagation e)
  (when (and *selected-agent* (string=? (car *selected-agent*) agent-id))
    (set! *selected-agent* #f)
    (set! *is-editing* #f))
  (rai-api-delete (string-append "/agents/" agent-id)
                  (lambda (resp)
                    (fetch-agents!))))

;; --- Components ---

(define (mobile-header)
  (div '(@ (class "mobile-header"))
       (button `(@ (class "menu-btn") (click ,(lambda (e) (toggle-sidebar!)))) "☰")
       (span '(@ (style "font-weight: bold; font-size: 1.1rem;")) (get-view-title))
       (div '(@ (style "width: 24px;")) "") ;; Spacer
       ))

(define (sidebar)
  (div `(@ (class ,(string-append "sidebar" (if *sidebar-open* " open" ""))))
       (div '(@ (class "sidebar-title")) "RAI Assistant")
       (div `(@ (class "nav-item") (click ,(lambda (e) (set-view! 'chat)))) "Playground")
       (div `(@ (class "nav-item") (click ,(lambda (e) (set-view! 'agents)))) "Agents")
       (div `(@ (class "nav-item") (click ,(lambda (e) (set-view! 'history)))) "History")))

;; Agents View
(define (agent-list-item agent-pair)
  (let* ((agent-id (car agent-pair))
         (config (cdr agent-pair))
         (model (js-string->string (js-object-ref config "model")))
         (is-selected (and *selected-agent* (string=? agent-id (car *selected-agent*)))))
    (div `(@ (class "card") 
             (style ,(string-append "cursor: pointer;" (if is-selected "border: 2px solid #3b82f6;" "")))
             (click ,(lambda (e) 
                       (set! *selected-agent* agent-pair)
                       (set! *is-editing* #t) ;; Switch to editor on mobile
                       (refresh-ui))))
         (div '(@ (style "display: flex; justify-content: space-between; align-items: center;"))
              (span '(@ (style "font-weight: bold; color: white;")) agent-id)
              (button `(@ (class "btn btn-danger") 
                          (click ,(lambda (e) 
                                    ;; Stop propagation would be nice here but we don't have it easily yet.
                                    ;; However, since this button is inside the card which has a click handler,
                                    ;; the card's handler might fire too if we are not careful.
                                    ;; In standard DOM, events bubble.
                                    ;; Ideally we need stopPropagation().
                                    ;; For now, let's rely on the fact that we are re-rendering.
                                    ;; BUT, the card click handler sets *is-editing* to #t.
                                    ;; We need to prevent that.
                                    ;; Since we can't easily stop propagation without a helper,
                                    ;; let's add a helper or hack it.
                                    ;; Actually, let's just make sure delete action doesn't set editing mode.
                                    ;; But the card click WILL fire because of bubbling.
                                    ;; We need stopPropagation.
                                    (delete-agent-action agent-id e)))) "Delete"))
         (div '(@ (style "font-size: 0.8em; color: #94a3b8;")) model))))

;; Global State for Models
(define *available-models* '())

(define (fetch-models!)
  (rai-api-get "/models/"
               (lambda (data)
                 (let* ((models-obj (js-object-ref data "models"))
                        ;; Assuming models is an array of strings or objects with 'id'
                        ;; Based on API, it returns { "models": [ ... ] }
                        ;; Let's assume simple list of strings for now or handle objects if needed.
                        ;; Actually, let's just fetch for 'ollama' backend by default or all.
                        ;; The /models endpoint returns a list.
                        (len (js-array-length models-obj))
                        (models-list '()))
                   (let loop ((i 0))
                     (if (< i len)
                         (let ((m (js-array-ref models-obj i)))
                           ;; If m is object with id, get id. If string, use string.
                           ;; Assuming string for simplicity or checking type.
                           ;; Let's assume it's a list of strings as per typical simple API, 
                           ;; or objects. Let's try to treat as string first.
                           (set! models-list (cons (js-string->string m) models-list))
                           (loop (+ i 1)))))
                   (set! *available-models* (reverse models-list))
                   (refresh-ui)))))

(define (agents-view)
  (let* ((sel-id (if *selected-agent* (car *selected-agent*) ""))
         (sel-config (if *selected-agent* (cdr *selected-agent*) #f))
         (sel-model (if sel-config (js-string->string (js-object-ref sel-config "model")) "gemma2:latest"))
         (sel-backend (if sel-config (js-string->string (js-object-ref sel-config "backend")) "ollama"))
         (sel-system (if sel-config (js-string->string (js-object-ref sel-config "system")) "You are a helpful assistant.")))
    (div `(@ (class ,(string-append "split-view" (if *is-editing* " mobile-editing" ""))))
         ;; Left: List
         (div '(@ (class "split-left"))
              (h2 "Agents")
              (button `(@ (class "btn btn-primary") (style "width: 100%; margin-bottom: 10px;") 
                          (click ,(lambda (e) 
                                    (log-debug "Action: Clicked New Agent")
                                    (set! *selected-agent* #f) 
                                    (set! *is-editing* #t) ;; Switch to editor on mobile
                                    (refresh-ui)))) "+ New Agent")
              (if (null? *agents*)
                  (div '(@ (style "color: #94a3b8; font-style: italic;")) "No agents found or loading...")
                  (apply div '() (map agent-list-item *agents*))))
         
         ;; Right: Editor
         (div '(@ (class "split-right"))
              (div '(@ (class "mobile-only-nav"))
                   (button `(@ (class "btn") (style "margin-bottom: 10px; background: #334155; color: white;")
                               (click ,(lambda (e) (set! *is-editing* #f) (refresh-ui)))) "← Back to List"))
              (h2 (if *selected-agent* (string-append "Edit Agent: " sel-id) "New Agent"))
              (div '(@ (class "form-group"))
                   (label '(@ (for "agent-id")) "Agent ID")
                   (input `(@ (id "agent-id") (class "form-control") (placeholder "e.g. coder") (value ,sel-id)
                            ,@(if *selected-agent* '((disabled "true")) '()))))
              (div '(@ (class "form-group"))
                   (label '(@ (for "agent-backend")) "Backend")
                   (div '(@ (class "select-wrapper"))
                        (apply tag 'select `(@ (id "agent-backend") (class "form-control"))
                               (map (lambda (b) 
                                      `(option (@ (value ,b) ,@(if (string=? b sel-backend) '((selected "selected")) '())) ,b))
                                    '("ollama" "openai" "anthropic" "gemini" "groq")))))
              (div '(@ (class "form-group"))
                   (label '(@ (for "agent-model")) "Model")
                   (div '(@ (class "select-wrapper"))
                        (if (null? *available-models*)
                            (input `(@ (id "agent-model") (class "form-control") (value ,sel-model))) ;; Fallback to input if no models loaded
                            (apply tag 'select `(@ (id "agent-model") (class "form-control"))
                                   (map (lambda (m) 
                                          `(option (@ (value ,m) ,@(if (string=? m sel-model) '((selected "selected")) '())) ,m))
                                        *available-models*)))))
              (div '(@ (class "form-group"))
                   (label '(@ (for "agent-system")) "System Prompt")
                   (textarea `(@ (id "agent-system") (class "form-control") (rows "5")) sel-system))

              (button `(@ (class "btn btn-primary")
                          (click ,(lambda (e)
                                    (log-debug "Action: Clicked Save Agent")
                                    (if *is-saving*
                                        (js-alert "Saving in progress...")
                                        (let* ((id-input (get-element-by-id "agent-id"))
                                               (backend-input (get-element-by-id "agent-backend"))
                                               (model-input (get-element-by-id "agent-model"))
                                               (sys-input (get-element-by-id "agent-system"))
                                               (new-id (element-value id-input))
                                               (new-backend (element-value backend-input))
                                               (new-model (element-value model-input))
                                               (new-system (element-value sys-input)))
                                          (if (and (not (string=? new-id "")) (not (string=? new-model "")))
                                              (cond
                                               ((and (not *selected-agent*) 
                                                     (let loop ((lst *agents*))
                                                       (if (null? lst) #f
                                                           (if (string=? (caar lst) new-id) #t (loop (cdr lst))))))
                                                (begin
                                                  (log-info (string-append "Validation Error: Agent ID '" new-id "' already exists."))
                                                  (js-alert (string-append "Agent ID '" new-id "' already exists. Please choose a different ID."))))
                                               (else
                                                (let ((payload `(("backend" . ,new-backend)
                                                                 ("model" . ,new-model)
                                                                 ("system" . ,new-system)
                                                                 ("tools" . ())))
                                                      (callback (lambda (resp)
                                                                  (set! *is-saving* #f)
                                                                  (let* ((status-ref (js-object-ref resp "status"))
                                                                         (status (if (string? status-ref) 
                                                                                     status-ref 
                                                                                     (js-string->string status-ref))))
                                                                    (if (string=? status "success")
                                                                        (begin
                                                                          (log-info (string-append "Agent saved: " new-id))
                                                                          (fetch-agents!)
                                                                          (set! *is-editing* #f)
                                                                          (refresh-ui))
                                                                        (let ((err-msg (js-string->string (json-stringify resp))))
                                                                          (log-error (string-append "Error saving agent: " err-msg))
                                                                          (js-alert (string-append "Failed to save agent. Server response: " err-msg))))))))
                                                  (if *selected-agent*
                                                      (rai-api-put (string-append "/agents/" new-id) payload callback)
                                                      (rai-api-post (string-append "/agents/" new-id) payload callback)))))
                                              (js-alert "Agent ID and Model are required.")))))))

                      (span "Save Agent"))))))

;; Chat View (Placeholder)
(define (chat-view)
  (div '(@ (class "split-view"))
       (div '(@ (class "split-left"))
            (h2 "Chat")
            (div '(@ (class "card")) "Welcome to RAI Playground. (Guile implementation in progress)"))
       (div '(@ (class "split-right"))
            (h2 "Chain Config")
            (div '(@ (class "card")) "Chain steps configuration..."))))

;; History View (Placeholder)
(define (history-view)
  (div '(@ (class "split-view"))
       (div '(@ (class "split-left"))
            (h2 "Sessions")
            (div '(@ (class "card")) "Session 1"))
       (div '(@ (class "split-right"))
            (h2 "Replay")
            (div '(@ (class "card")) "Select a session..."))))

(define (content)
  (cond
   ((eq? *current-view* 'agents) (agents-view))
   ((eq? *current-view* 'chat) (chat-view))
   ((eq? *current-view* 'history) (history-view))
   (else (div '() "Unknown view"))))

(define (app-root)
  (div '(@ (class "container"))
       (mobile-header)
       (div `(@ (class ,(string-append "sidebar-overlay" (if *sidebar-open* " open" "")))
                (click ,(lambda (e) (close-sidebar!)))) "")
       (sidebar)
       (div '(@ (class "main-content"))
            (content))))

(define (main)
  (refresh-ui)
  (fetch-agents!)
  (fetch-models!))

(main)
