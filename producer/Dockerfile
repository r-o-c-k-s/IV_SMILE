FROM python:3.10
WORKDIR /app
COPY main.py . 
RUN pip install ibapi kafka-python
CMD ["python", "main.py"]
