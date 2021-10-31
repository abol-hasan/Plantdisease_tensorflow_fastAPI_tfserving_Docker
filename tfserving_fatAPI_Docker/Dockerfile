FROM python:3.9-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./fastapi_TF /code/fastapi_TF
COPY . .
EXPOSE 8501
CMD ["uvicorn", "fastapi_TF.tf-serving-fastapi:app", "--host", "0.0.0.0", "--port", "80"]
