# Setup

## Cloud Machine Recommendation
**We recommend using a cloud machine with high-performance GPUs for running these experiments.** We use **Hyperstack H100** for optimal performance. Other cloud options include:
- AWS EC2 (Deep Learning Base OSS Nvidia Driver GPU AMI - Ubuntu 24.04)
- Google Cloud Platform with A100/H100 GPUs
- Lambda Labs
- Paperspace

## Quick Setup on Cloud Machine
Run the setup script from `server.sh` to automatically install dependencies and configure the environment:

```bash
# Run the server.sh setup script
wget https://raw.githubusercontent.com/DDantalion/CSE232B/main/server.sh
sudo su
bash server.sh
```
