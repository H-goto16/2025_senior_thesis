.PHONY:  setup backend frontend dev clean server setup-parallel setup-venv test test-backend test-frontend train-model view-training-stats clean-training-data fix-expo-router test-fix latex latex-bib latex-clean latex-docker

PYTHON_COMMAND=python3
PIP_COMMAND=pip3
PNPM_COMMAND=pnpm

setup:
	aqua i
	make setup-parallel
	make setup-venv

setup-parallel:
	cd frontend && ${PNPM_COMMAND} install &
	${PYTHON_COMMAND} -m venv .venv &
	wait

setup-venv:
	. .venv/bin/activate && ${PYTHON_COMMAND} -m pip install -r backend/requirements.txt

server:
	cd backend/src && . ../../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

deploy-server:
	cd backend/src && . ../../.venv/bin/activate && gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

deploy-server-public:
	cd backend/src && . ../../.venv/bin/activate && gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 &
	cd frontend && bash -c 'pnpm exec localtunnel --port 8000 --subdomain dish-detection-api'

deploy-server-cloudflare:
	cd backend/src && . ../../.venv/bin/activate && gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 &
	cloudflared tunnel --url http://localhost:8000

frontend:
	cd frontend && ${PNPM_COMMAND} run start

frontend-web:
	cd frontend && ${PNPM_COMMAND} run web -- --port 8081

capture-screens:
	# Requires frontend web to be running on http://localhost:8081
	cd frontend && APP_BASE_URL=http://localhost:8081 ${PNPM_COMMAND} run capture-screens

fix-expo-router:
	@echo "🔧 Fixing expo-router path issue..."
	cd frontend && find node_modules/.pnpm/expo-router@5.1.0_*/node_modules/expo-router -name "_ctx*.js" -exec sed -i 's|"../../../../src/app"|"../../../../../src/app"|g' {} \;
	@echo "🧹 Clearing caches..."
	cd frontend && rm -rf .expo .metro-cache node_modules/.cache ~/.expo
	@echo "✅ expo-router fix completed!"

dev:
	make fix-expo-router
	make server & make frontend

test:
	make test-backend & make test-frontend

test-backend:
	cd backend && . ../.venv/bin/activate && pytest

test-frontend:
	cd frontend && ${PNPM_COMMAND} test

train-model:
	@echo "Starting model fine-tuning..."
	curl -X POST "http://localhost:8000/training/start" \
	  -H "accept: application/json" \
	  -H "Content-Type: application/json"

view-training-stats:
	@echo "Fetching training data statistics..."
	curl -X GET "http://localhost:8000/training/data/stats" \
	  -H "accept: application/json" | python3 -m json.tool

clean-training-data:
	@echo "Cleaning training data directory..."
	cd backend/src && rm -rf training_data/
	@echo "Training data cleared."

venv-clean:
	rm -rf .venv

stop:
	pkill -f "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
	pkill -f "pnpm run start"
	pkill -f "gunicorn main:app"
	pkill -f "localtunnel"
	pkill -f "cloudflared"

test-fix:
	@echo "🧪 Testing expo-router fix..."
	cd frontend && timeout 30 pnpm web --port 8081 || echo "Test completed"

# --- LaTeX build ---
LATEX_DIR=LaTex
LATEX_MAIN=Goto.tex

latex:
	cd $(LATEX_DIR) && uplatex -interaction=nonstopmode $(LATEX_MAIN) && uplatex -interaction=nonstopmode $(LATEX_MAIN) && dvipdfmx Goto.dvi

latex-bib:
	cd $(LATEX_DIR) && uplatex -interaction=nonstopmode $(LATEX_MAIN) && upbibtex Goto && uplatex -interaction=nonstopmode $(LATEX_MAIN) && uplatex -interaction=nonstopmode $(LATEX_MAIN) && dvipdfmx Goto.dvi

latex-clean:
	cd $(LATEX_DIR) && rm -f *.aux *.log *.toc *.lof *.lot *.out *.bbl *.blg *.dvi *.synctex.gz *.fls *.fdb_latexmk

latex-docker:
	docker run --rm -v $(PWD)/$(LATEX_DIR):/work -w /work paperist/alpine-texlive-ja:latest sh -c 'tlmgr update --self --verify-repo=none || true && tlmgr install --verify-repo=none algorithmicx algorithms pxjahyper || true && uplatex -interaction=nonstopmode $(LATEX_MAIN) && upbibtex Goto || true && uplatex -interaction=nonstopmode $(LATEX_MAIN) && uplatex -interaction=nonstopmode $(LATEX_MAIN) && dvipdfmx Goto.dvi'
