# Wine-Prediction
Training Setup:
Create an S3 bucket and upload the training and validation data set.

EMR Installation:
To begin with, navigate to the analytics section in the AWS dashboard and click on EMR. Next, select "Create Cluster". Give a name to the cluster and choose "emr-5.36.0". Then, select the option "Spark: Spark 2.4.8 on Hadoop 2.10.1 YARN and Zeppelin 0.10.0". Under the "Number of instances" column, select 4 instances and choose a previously created C2 key pair.

Prediction without Docker:
•	Creating EC2 Instance
Step 1: Under Compute Column in the AWS Management Console Click EC2
Step 2: Select the AMI of your choice. Amazon Linux 2 AMI is usually preferred.
Launch EC2 Instance
•	Installing Libraries on EC2 Instance
sudo yum update -y
python –version
pip install pandas
pip install numpy
pip install -U scikit-learn
pip install findspark
pip install sklearn
Install Spark and hadoop
wget http://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz
sudo tar -zxvf spark-3.0.0-bin-hadoop2.7.tgz
•	Running your Application in EC2
Copy the wine2.py file to the Ec2 instance and spark-submit wine2.py to run the code in ec2.


Prediction with Docker:
•	Installation and Building Docker
Follow above steps (Prediction without-Docker) and update sudo apt update -y
•	Install Docker “sudo yum install docker -y”
•	Start the Service by using “sudo service docker start”
•	Add ec2-user to docker group by “sudo usermod -a -G docker ec2-user”
•	Build Dockerfile “sudo docker build . -f Dockerfile -t wine”
•	Pushing and Pulling created Image to DockerHub by login DockerHub and by command “docker pull pvatsal/wine:latest”

Running Application with Docker
Run: sudo docker pull pvatsal/wine:latest
docker run -it pvatsal/wine:latest s3//mywineproject/ValidationDataset.csv
