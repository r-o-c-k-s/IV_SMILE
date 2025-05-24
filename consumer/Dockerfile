FROM python:3.10
WORKDIR /app
COPY main.py . 
RUN pip install kafka-python psycopg2-binary python-dotenv
RUN pip install pandas numpy sqlalchemy

CMD ["python", "main.py"]
