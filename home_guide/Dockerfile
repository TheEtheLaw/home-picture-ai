# Home guide: a small Python service that runs the iptv-org EPG grabber (Node.js). The grabber's code isn't built in:
# the service fetches it on its first start and brings it up to date each day (its sites' ways change as the sites do).
FROM node:22-bookworm-slim

ENV PYTHONUNBUFFERED=1 NPM_CONFIG_UPDATE_NOTIFIER=false NPM_CONFIG_FUND=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 git ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY server.py /app/

CMD ["python3", "-u", "/app/server.py"]
