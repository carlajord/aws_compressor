#!/usr/bin/env bash
# Bootstrap script for EC2 (Amazon Linux 2023, t3.medium).
# Run as ec2-user via Session Manager after launching the instance.
#
# Usage:
#   bash bootstrap.sh <S3_BUCKET> <AWS_REGION>
#
# The pipeline (S3 → DuckDB) runs without LLM credentials.
# To enable the Chat page, update .env afterwards:
#   echo 'OPENAI_API_KEY=sk-xxx' >> ~/aws_compressor/.env
#   echo 'OPENAI_BASE_URL=https://your-endpoint/v1' >> ~/aws_compressor/.env
#   sudo systemctl restart streamlit
#
# Example:
#   bash bootstrap.sh oid-compressor-csantiago3 us-east-1

set -euo pipefail

BUCKET="${1:?Usage: bootstrap.sh <BUCKET> <REGION>}"
REGION="${2:?}"

APP_DIR="/home/ec2-user/aws_compressor"
REPO_URL="https://github.com/csantiago3/aws_compressor.git"

echo "=== 1/6 System packages ==="
sudo dnf install -y git python3.11 python3.11-pip gcc python3.11-devel

echo "=== 2/6 Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

echo "=== 3/6 Clone repo ==="
if [ -d "$APP_DIR" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

echo "=== 4/6 Install Python dependencies ==="
uv sync

echo "=== 5/6 Write .env ==="
cat > "$APP_DIR/.env" <<EOF
OID_ENV=aws
OID_AWS_REGION=${REGION}
OID_S3_BUCKET=${BUCKET}
OID_RAW_PREFIX=raw/
OID_DB_PATH=data/oid.duckdb
OID_TRACES_DB_PATH=data/traces.duckdb
OID_LLM_BACKEND=openai
OID_LLM_MODEL=openai.gpt-oss-120b
OID_AGENT_MAX_STEPS=8
OID_SQL_ROW_LIMIT=500
OID_ZSCORE_THRESHOLD=5.0
OID_IFOREST_CONTAMINATION=0.01
# Add these after bootstrap to enable the Chat page:
# OPENAI_API_KEY=your-key
# OPENAI_BASE_URL=https://your-endpoint/v1
EOF
chmod 600 "$APP_DIR/.env"

echo "=== 6/6 Run pipeline (S3 → DuckDB) ==="
mkdir -p "$APP_DIR/data"
cd "$APP_DIR"
uv run python -m oidpipe.pipeline --aws

echo ""
echo "=== Bootstrap complete ==="
echo "Dashboards are ready. To start Streamlit:"
echo "  cd $APP_DIR && uv run streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
echo ""
echo "To install as a service:"
echo "  sudo cp infra/streamlit.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload && sudo systemctl enable --now streamlit"
echo ""
echo "To enable Chat (text-to-SQL), add your OpenAI-compatible API credentials to .env:"
echo "  echo 'OPENAI_API_KEY=sk-xxx' >> $APP_DIR/.env"
echo "  echo 'OPENAI_BASE_URL=https://your-endpoint/v1' >> $APP_DIR/.env"
echo "  sudo systemctl restart streamlit"
