EMACS ?= emacs
ELISP := rai-transport.el rai-context.el rai-repl.el rai.el

.PHONY: test compile clean

test:
	$(EMACS) --batch -Q -L . -L test -l rai-tests.el -f ert-run-tests-batch-and-exit

compile:
	$(EMACS) --batch -Q -L . -f batch-byte-compile $(ELISP)

clean:
	rm -f *.elc test/*.elc
