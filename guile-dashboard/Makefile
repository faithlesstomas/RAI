.PHONY: all clean serve

all: app.wasm

app.wasm: dashboard.scm rai-api.scm dom.scm
	guix shell --pure -m manifest.scm -- guild compile-wasm -L . -o app.wasm --bundle dashboard.scm

serve: all
	python3 -m http.server -b 127.0.0.1 8080

clean:
	rm -f app.wasm app.js
