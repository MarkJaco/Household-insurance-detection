FROM python:3.7-bullseye

ENV http_proxy="http://proxylb.prod.iaa.si-aws.cloud:80"
ENV https_proxy="http://proxylb.prod.iaa.si-aws.cloud:80"
ENV no_proxy = "localhost,127.0.0.1,169.254.170.2,169.254.169.254,.athena.eu-central-1.amazonaws.com,codebuild.eu-central-1.amazonaws.com,codecommit.eu-central-1.amazonaws.com,ec2.eu-central-1.amazonaws.com,ec2messages.eu-central-1.amazonaws.com,api.ecr.eu-central-1.amazonaws.com,dkr.ecr.eu-central-1.amazonaws.com,elasticfilesystem.eu-central-1.amazonaws.com,elasticloadbalancing.eu-central-1.amazonaws.com,git-codecommit.eu-central-1.amazonaws.com,kms.eu-central-1.amazonaws.com,logs.eu-central-1.amazonaws.com,redshift.eu-central-1.amazonaws.com,s3.eu-central-1.amazonaws.com,api.sagemaker.eu-central-1.amazonaws.com,runtime.sagemaker.eu-central-1.amazonaws.com,sns.eu-central-1.amazonaws.com,sqs.eu-central-1.amazonaws.com,ssm.eu-central-1.amazonaws.com,ssmmessages.eu-central-1.amazonaws.com,sts.eu-central-1.amazonaws.com,secretsmanager.eu-central-1.amazonaws.com,git.system.local,pki.system.local,system.local,.system.local,si-aws.cloud,.si-aws.cloud,172.16.0.0/12,10.0.0.0/8,192.168.0.0/16"

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -U pip

COPY . /app
RUN pip install -r app/requirements.txt

WORKDIR /app
EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
