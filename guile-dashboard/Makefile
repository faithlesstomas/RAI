.PHONY: all clean serve

all: app.wasm

app.wasm: dashboard.scm rai-api.scm dom.scm
	guix shell --pure -m manifest.scm -- guild compile-wasm -L . -o app.wasm --bundle dashboard.scm

serve: all
	guix shell --pure -m manifest.scm -- guile server.scm

clean:
	rm -f app.wasm app.js
