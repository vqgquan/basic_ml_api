FROM python:3.12

WORKDIR /code

# Copy entire project into container
COPY ./requirements.txt /code/

# Install dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code/

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
