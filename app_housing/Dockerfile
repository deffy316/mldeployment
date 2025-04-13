FROM python:3.9-slim

WORKDIR /app_housing

COPY app_housing/ /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9001

CMD ["python", "app.py"]
