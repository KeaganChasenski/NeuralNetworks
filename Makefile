VENV := venv

install: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

venv: $(VENV)/bin/activate

XOR: venv
	./$(VENV)/bin/python XOR.py 

Classifier: venv
	./$(VENV)/bin/python Classifier.py

clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	find . -type f -name '*.pyc' -delete
	rm -rf *.gif

.PHONY: all venv run clean