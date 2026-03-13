FROM python:3.11-slim

# Evita logs “buffered”
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# copia o certificado corporativo e atualiza o trust store
COPY certs/corp-root.crt /usr/local/share/ca-certificates/corp-root.crt
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# garante que bibliotecas Python/requests/boto3 usem o bundle do sistema
ENV AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Dependências do sistema (se precisar de certificados/ssl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Porta do container
EXPOSE 8080

# Em produção: gunicorn
# app:create_app precisa existir e retornar Flask app (no seu app.py já existe create_app)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:create_app()", "--workers", "2", "--threads", "4", "--timeout", "120"]