.PHONY:  setup backend frontend dev clean server setup-parallel setup-venv test test-backend test-frontend train-model view-training-stats clean-training-data fix-expo-router test-fix

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
