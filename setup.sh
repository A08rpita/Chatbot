echo "Setting up the Flask Currency Converter project..."


echo "Updating the system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y


echo "Installing Python and pip..."
sudo apt-get install python3 python3-pip -y


echo "Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate


echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt


ec
