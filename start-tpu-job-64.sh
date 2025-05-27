### Setup
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"'
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='pip install -U keras keras-hub tensorflow-cpu datasets' 
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='python3 -c "import jax; jax.distributed.initialize(); print(jax.device_count()); print(jax.local_device_count())"'
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='pip install gcsfs proto-plus==1.24.0.dev1'
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`; echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list'
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc'
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='sudo apt-get update; sudo apt-get install gcsfuse'
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='mkdir -p /home/<your user id>/<your mount point>; gcsfuse --implicit-dirs gs://<your staging bucket> /home/<your user id>/<your mount point>'
### Copy required files and start job
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='gsutil cp gs://<your staging bucket>/vocab/* . '
gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='gsutil cp gs://<your staging bucket>/scripts/* . '
nohup gcloud compute tpus tpu-vm ssh <tpu cluster name> --project <your project id> --zone=<your zone> --worker=all  --command='python3 colmo2-tpu-pretrain-64.py ' & disown