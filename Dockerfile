# Dashboard image for DigitalOcean App Platform (or any Docker host).
# Keys come from environment variables set in the DigitalOcean app settings,
# never from files baked into the image.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
EXPOSE 8080
# Auth0 config: paste the whole secrets.toml into a STREAMLIT_SECRETS env var.
CMD mkdir -p .streamlit && if [ -n "$STREAMLIT_SECRETS" ]; then printf "%s" "$STREAMLIT_SECRETS" > .streamlit/secrets.toml; fi && streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
