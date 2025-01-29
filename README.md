# Link for ML page

https://thecleverprogrammer.com/2022/02/01/waiter-tips-prediction-with-machine-learning/

## To start nootebook

Install requirements:

```shell
pip install -r requirements.txt
```
Run command in notebook folder:
```shell
cd notebook; jupyter lab 
```

## Start FastAPI

```shell
fastapi dev app.py
```
Endpoints:
* /models
* /continue-train
* /predict

## Create sandbox

```shell
docker network create ci-cd
docker compose up
docker network inspect ci-cd
```

## Create docker image
```shell
docker build -t waiter-tips .
```

## Ansible

```shell
ansible-playbook -i inventory.yaml playbook.yaml
```

## Login to ssh

```shell
ssh ansible@localhost -p 2222
```

