FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WARO_CLI_BINARY=/usr/local/bin/waro

WORKDIR /app

COPY .local/bin/waro /usr/local/bin/waro
RUN chmod +x /usr/local/bin/waro && /usr/local/bin/waro --version

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
