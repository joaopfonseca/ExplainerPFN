.PHONY: todos
SRC_DIR := .

# Fetch all TODOs in repository
todos:
	@echo "Fetching all TODOs in the repository..."
	@grep -r --color=always -n "TODO" $(SRC_DIR) --exclude="*.ipynb" --exclude="Makefile" --exclude="*.pyc" --exclude="*.sample" || echo "No TODOs found."

notes:
	@echo "Fetching all notes in the repository..."
	@grep -r --color=always -n "NOTE" $(SRC_DIR) --exclude="*.ipynb" --exclude="Makefile" --exclude="*.pyc" --exclude="*.sample" || echo "No notes found."
