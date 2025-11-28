.PHONY: all clean serve stop-serve start-serve log

all: app.wasm

app.wasm: dashboard.scm rai-api.scm dom.scm
	guix shell --pure -m manifest.scm -- guild compile-wasm -L . -o app.wasm --bundle dashboard.scm

serve: all
	guix shell --pure -E LOG_LEVEL -m manifest.scm -- guile server.scm

start-serve: all
	@echo "Starting server in background..."
	@nohup guix shell --pure -E LOG_LEVEL -m manifest.scm -- guile server.scm > server.log 2>&1 &
	@echo "Server started. Logs: server.log"

log:
	tail -f server.log

clean:
	rm -f app.wasm app.js server.pid server.log

stop-serve:
	@if [ -f server.pid ]; then \
		PID=`cat server.pid`; \
		echo "Stopping server (PID $$PID)..."; \
		kill $$PID 2>/dev/null || true; \
		sleep 1; \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Server did not stop, forcing kill..."; \
			kill -9 $$PID 2>/dev/null || true; \
		fi; \
		rm -f server.pid; \
		echo "Server stopped."; \
	else \
		echo "No server.pid found."; \
	fi
